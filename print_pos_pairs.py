#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
print_pos_pairs.py
------------------
Print ALL positive pairs (same group) with their cosine similarity scores.
Optionally sort and/or save to CSV.

Usage examples:
  python print_pos_pairs.py --csv dataset/00train_test/test.csv --base_dir dataset/00train_test
  python print_pos_pairs.py --csv dataset/00train_test/train.csv --checkpoint runs/repvit_embed_512_best.pt --sort desc
  python print_pos_pairs.py --csv dataset/00train_test/test.csv --base_dir dataset/00train_test --save_csv runs/pos_pairs.csv

Requires: torch, timm, pillow, numpy
"""

import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    import torch
    import timm
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception as e:
    print("[Error] Please install dependencies: torch, timm, pillow, numpy")
    print("       pip install torch timm pillow numpy")
    print("Exception:", repr(e))
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="Print ALL POS pairs (same group) with cosine scores.")
    ap.add_argument("--csv", required=True, type=str, help="CSV file with 'path,group' (header optional)")
    ap.add_argument("--base_dir", default="", type=str, help="Optional prefix for relative paths")
    ap.add_argument("--model", default="repvit_m1_0", type=str, help="timm model name")
    ap.add_argument("--checkpoint", default="", type=str, help="optional checkpoint to load")
    ap.add_argument("--image_size", default=224, type=int, help="override image size if needed")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="device")
    ap.add_argument("--batch_size", default=32, type=int, help="batch size for embedding extraction")
    ap.add_argument("--sort", default="none", choices=["none","asc","desc"], help="sort POS pairs by cosine")
    ap.add_argument("--save_csv", default="", type=str, help="optional path to save POS pairs")
    ap.add_argument("--max_print", default=5000, type=int, help="limit on stdout prints (CSV always full)")
    return ap.parse_args()


def fix_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    elif name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif name == "mps":
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def read_rows(csv_path: str, base_dir: str="") -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    import csv as _csv
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        sniff_header = _csv.Sniffer().has_header(sample)
        rdr = _csv.reader(f)
        header = None
        if sniff_header:
            header = next(rdr, None)
        if header:
            h = [c.strip().lower() for c in header]
            p_idx = h.index("path") if "path" in h else 0
            g_idx = h.index("group") if "group" in h else 1
        else:
            p_idx, g_idx = 0, 1
        for line in rdr:
            if not line:
                continue
            p = line[p_idx].strip()
            g = line[g_idx].strip() if len(line) > g_idx else "0"
            if base_dir and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            rows.append((p, g))
    ok = [(p, g) for (p, g) in rows if os.path.exists(p)]
    missing = len(rows) - len(ok)
    if missing > 0:
        print(f"[Warn] {missing} files not found; using {len(ok)} images.")
    return ok


def load_model(model_name: str, checkpoint: str, device: torch.device):
    model = timm.create_model(model_name, pretrained=(checkpoint == ""))
    model.eval().to(device)
    if checkpoint:
        print(f"[Info] loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    return model


def build_transform(model, image_size: int):
    cfg = resolve_data_config({}, model=model)
    if image_size and image_size > 0:
        cfg = {**cfg, "input_size": (3, image_size, image_size)}
    return create_transform(**cfg)


def forward_features_embedding(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            feats = model.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            if feats.ndim == 4:
                feats = feats.mean(dim=(2,3))
            out = feats
        else:
            out = model(x)
        if out.ndim > 2:
            out = out.flatten(1)
        return out


def extract_embeddings(model, tfm, rows: List[Tuple[str,str]], device: torch.device, batch_size: int=32):
    from PIL import Image
    embs = []
    groups = []
    paths  = []
    total = len(rows)

    def chunks(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    for chunk_rows in chunks(rows, batch_size):
        imgs = []
        kept = []
        for p, g in chunk_rows:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[Warn] open fail: {p} ({e})")
                continue
            x = tfm(img)
            imgs.append(x)
            kept.append((p,g))
        if not imgs:
            continue
        batch = torch.stack(imgs, 0).to(device)
        z = forward_features_embedding(model, batch).detach().cpu().numpy().astype(np.float32)
        embs.append(z)
        groups.extend([g for _, g in kept])
        paths.extend([p for p, _ in kept])
        print(f"\r[Embed] {len(paths)}/{total} done", end="")
        sys.stdout.flush()
    print()
    if not embs:
        raise RuntimeError("No embeddings extracted.")
    embs = np.concatenate(embs, 0)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return embs, np.array(groups), np.array(paths)


def main():
    args = parse_args()
    device = fix_device(args.device)
    print(f"[Info] device = {device}")

    rows = read_rows(args.csv, args.base_dir)
    if len(rows) < 2:
        print("[Error] Need at least 2 valid images.")
        sys.exit(2)

    model = load_model(args.model, args.checkpoint, device)
    tfm = build_transform(model, args.image_size)

    print(f"[Info] extracting embeddings for {len(rows)} images...")
    embs, groups, paths = extract_embeddings(model, tfm, rows, device, batch_size=args.batch_size)
    print(f"[Info] embeddings: {embs.shape}")

    sims = embs @ embs.T
    np.fill_diagonal(sims, -2.0)

    same = (groups[:, None] == groups[None, :])
    np.fill_diagonal(same, False)

    iu = np.triu_indices(len(groups), k=1)
    S = sims[iu]
    Y = same[iu]

    pos_idxs = np.flatnonzero(Y)
    pos_pairs = [(S[g], iu[0][g], iu[1][g]) for g in pos_idxs]

    # sort if requested
    if args.sort != "none":
        reverse = (args.sort == "desc")
        pos_pairs.sort(key=lambda t: t[0], reverse=reverse)

    print(f"\n=== POS pairs (same group) total: {len(pos_pairs)} ===")
    to_print = pos_pairs[:args.max_print]
    for sim, i, j in to_print:
        print(f"sim={sim:.4f} | POS | group[i]={groups[i]} | group[j]={groups[j]}\n"
              f"  a: {paths[i]}\n  b: {paths[j]}")
    if len(pos_pairs) > args.max_print:
        print(f"... (truncated, printed first {args.max_print} of {len(pos_pairs)})")

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a_path","a_group","b_path","b_group","cosine"])
            for sim, i, j in pos_pairs:
                w.writerow([paths[i], groups[i], paths[j], groups[j], f"{sim:.6f}"])
        print(f"\n[Saved] wrote {len(pos_pairs)} POS pairs to {args.save_csv}")

    print("\n[Done]")
    

if __name__ == "__main__":
    main()
