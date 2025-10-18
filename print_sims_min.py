#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
print_sims_min.py
-----------------
Minimal standalone script:
- Read a CSV of "path,group" (header optional; extra columns ignored)
- Compute L2-normalized embeddings with a timm model (default: repvit_m1_0)
- Build cosine similarity matrix (diagonal excluded)
- Print a quick view of cosine values for same-group (POS) and different-group (NEG):
    * counts + basic stats
    * first N values for POS/NEG (randomly sampled)
    * top-K hard negatives (highest cosine among NEG)
    * top-K hard positives (lowest cosine among POS)

Usage:
  python print_sims_min.py --csv dataset/00train_test/test.csv
  python print_sims_min.py --csv dataset/00train_test/train.csv --model repvit_m1_0 --topk 5
  python print_sims_min.py --csv dataset/00train_test/test.csv --checkpoint runs/repvit_embed_512_best.pt

Requires: torch, timm, pillow, numpy
"""

import argparse
import csv
import os
import sys
import random
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
    ap = argparse.ArgumentParser(description="Quickly print cosine sims for same/diff groups.")
    ap.add_argument("--csv", required=True, type=str, help="CSV file with 'path,group'")
    ap.add_argument("--base_dir", default="", type=str, help="Optional prefix for relative paths")
    ap.add_argument("--model", default="repvit_m1_0", type=str, help="timm model name")
    ap.add_argument("--checkpoint", default="", type=str, help="optional checkpoint to load")
    ap.add_argument("--image_size", default=224, type=int, help="override image size if needed")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="device")
    ap.add_argument("--batch_size", default=32, type=int, help="batch size for embedding extraction")
    ap.add_argument("--seed", default=42, type=int, help="random seed")
    ap.add_argument("--samples", default=20, type=int, help="how many POS/NEG samples to print")
    ap.add_argument("--topk", default=10, type=int, help="how many hard examples to print")
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
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        sniff_header = csv.Sniffer().has_header(sample)
        rdr = csv.reader(f)
        header = None
        if sniff_header:
            header = next(rdr, None)
        if header:
            # try to locate "path" and "group"
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

    # remove missing
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


def safe_forward(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(x)
    # try to coerce to (B,D)
    if isinstance(out, dict):
        for k in ["emb", "feat", "features", "logits"]:
            if k in out:
                out = out[k]
                break
        else:
            out = list(out.values())[0]
    elif isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim > 2:
        out = out.flatten(1)
    return out


def extract_embeddings(model, tfm, rows: List[Tuple[str,str]], device: torch.device, batch_size: int=32):
    embs = []
    groups = []
    paths  = []
    total = len(rows)

    def chunks(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    for chunk_rows in chunks(rows, batch_size):
        imgs = []
        for p, g in chunk_rows:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[Warn] open fail: {p} ({e})")
                continue
            x = tfm(img)
            imgs.append(x)
        if not imgs:
            continue
        batch = torch.stack(imgs, 0).to(device)
        z = safe_forward(model, batch).detach().cpu().numpy().astype(np.float32)
        embs.append(z)
        groups.extend([g for _, g in chunk_rows])
        paths.extend([p for p, _ in chunk_rows])
        print(f"\r[Embed] {len(paths)}/{total} done", end="")
        sys.stdout.flush()
    print()
    embs = np.concatenate(embs, 0)
    # L2 norm
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return embs, np.array(groups), np.array(paths)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    print(f"[Info] embeddings shape: {embs.shape}")

    # cosine similarity matrix
    sims = embs @ embs.T
    np.fill_diagonal(sims, -2.0)  # exclude self

    # same-group mask (exclude diagonal already)
    same = (groups[:, None] == groups[None, :])
    np.fill_diagonal(same, False)

    # use upper triangle to avoid duplicates
    iu = np.triu_indices(len(groups), k=1)
    pair_sims = sims[iu]
    pair_same = same[iu]

    pos = pair_sims[pair_same]
    neg = pair_sims[~pair_same]

    def stats(arr):
        if arr.size == 0:
            return "count=0"
        return ("count={n} mean={m:.4f} std={s:.4f} min={a:.4f} p25={p1:.4f} "
                "p50={p2:.4f} p75={p3:.4f} max={b:.4f}").format(
            n=arr.size, m=arr.mean(), s=arr.std(), a=arr.min(),
            p1=np.percentile(arr,25), p2=np.percentile(arr,50),
            p3=np.percentile(arr,75), b=arr.max()
        )

    print("\n=== Summary ===")
    print("POS (same group):", stats(pos))
    print("NEG (diff group):", stats(neg))

    # Random samples to "see" values
    def sample_show(arr, name, k):
        print(f"\n{name} samples (up to {k}):")
        if arr.size == 0:
            print("  (none)")
            return
        idx = np.random.choice(arr.size, size=min(k, arr.size), replace=False)
        vals = np.sort(arr[idx])
        print(" ", ", ".join(f"{v:.4f}" for v in vals))

    sample_show(pos, "POS", args.samples)
    sample_show(neg, "NEG", args.samples)

    # Hard cases
    K = min(args.topk, pos.size if pos.size>0 else 0)
    if K > 0:
        # lowest similarities among positives
        pos_idx = np.argpartition(pos, K-1)[:K]
        pos_sorted = pos[pos_idx][np.argsort(pos[pos_idx])]
        print(f"\nHard POS (lowest {K}):\n " + ", ".join(f"{v:.4f}" for v in pos_sorted))
    else:
        print("\nHard POS: (none)")

    K = min(args.topk, neg.size if neg.size>0 else 0)
    if K > 0:
        # highest similarities among negatives
        neg_idx = np.argpartition(-neg, K-1)[:K]
        neg_sorted = neg[neg_idx][np.argsort(-neg[neg_idx])]
        print(f"\nHard NEG (highest {K}):\n " + ", ".join(f"{v:.4f}" for v in neg_sorted))
    else:
        print("\nHard NEG: (none)")

    print("\n[Done] Printed cosine similarity overview.")
    

if __name__ == "__main__":
    main()
