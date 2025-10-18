#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_sims.py
--------------
Standalone analyzer to compute and inspect cosine similarities between image embeddings.

Usage (examples):
  python debug_sims.py --csv dataset/00train_test/test.csv --model repvit_m1_0 --device auto --topk 5
  python debug_sims.py --csv dataset/00train_test/train.csv --checkpoint runs/repvit_embed_512_best.pt --topk 10
  python debug_sims.py --csv dataset/00train_test/test.csv --save_csv runs/sims_debug.csv

CSV format:
  - Preferred: with header "path,group" (additional columns are ignored).
  - Or: two columns without header: first is image path, second is group id/name.
  - Paths can be relative; you can supply --base_dir to prefix them.

What this prints:
  - Summary stats for positive pairs (same group) and negative pairs (different group)
  - Hard positives (lowest similarity) and hard negatives (highest similarity)
  - Top-K nearest neighbors for a few anchor images
  - Optionally dumps sampled pairwise sims to CSV

Author: (your friendly assistant)
"""

import argparse
import csv
import os
import sys
import math
import random
from typing import List, Tuple, Optional

import numpy as np

try:
    import torch
    from PIL import Image
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception as e:
    print("[Error] Missing packages. Please ensure you have: torch, timm, pillow, numpy.")
    print("Exception:", repr(e))
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="Compute and inspect cosine similarities between image embeddings.")
    ap.add_argument("--csv", type=str, required=True, help="CSV file with image paths and group ids")
    ap.add_argument("--base_dir", type=str, default="", help="Optional base directory to prefix to relative paths")
    ap.add_argument("--model", type=str, default="repvit_m1_0", help="timm model name")
    ap.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint to load (fine-tuned weights)")
    ap.add_argument("--image_size", type=int, default=224, help="Image input size override if needed")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"], help="Compute device")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    ap.add_argument("--num_workers", type=int, default=0, help="(unused) for future dataloader speedup")
    ap.add_argument("--topk", type=int, default=5, help="K for nearest neighbors / hard sample display")
    ap.add_argument("--anchors", type=int, default=5, help="How many random anchors to display neighbors for")
    ap.add_argument("--save_csv", type=str, default="", help="Optionally save sampled pairwise similarities to CSV")
    ap.add_argument("--max_pairs_csv", type=int, default=200000, help="Max pairs to dump if save_csv is set")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
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
    """
    Returns: list of (path, group)
    Accepts header with 'path' and 'group' or two columns without header.
    Extra columns are ignored.
    """
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        sniff = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)
        rdr = csv.reader(f)
        header = None
        if sniff:
            header = next(rdr, None)
        if header:
            # Normalize header to lower-case
            h = [c.strip().lower() for c in header]
            try:
                p_idx = h.index("path")
                g_idx = h.index("group")
            except ValueError:
                # Fallback: assume first two columns are path, group
                p_idx, g_idx = 0, 1
            for line in rdr:
                if not line: 
                    continue
                p = line[p_idx].strip()
                g = line[g_idx].strip() if len(line) > g_idx else "0"
                if base_dir and not os.path.isabs(p):
                    p = os.path.join(base_dir, p)
                rows.append((p, g))
        else:
            # No header; read as path, group
            for line in rdr:
                if not line:
                    continue
                p = line[0].strip()
                g = line[1].strip() if len(line) > 1 else "0"
                if base_dir and not os.path.isabs(p):
                    p = os.path.join(base_dir, p)
                rows.append((p, g))
    # Basic sanity
    ok = []
    for p, g in rows:
        if not os.path.exists(p):
            print(f"[Warn] Missing file: {p}")
        else:
            ok.append((p, g))
    if len(ok) < len(rows):
        print(f"[Info] {len(rows)-len(ok)} files missing; proceeding with {len(ok)} existing images.")
    return ok


def load_model(model_name: str, checkpoint: str, device: torch.device):
    model = timm.create_model(model_name, pretrained=(checkpoint == ""))
    model.eval()
    model.to(device)
    if checkpoint:
        print(f"[Info] Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            # try common keys
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                # maybe it's already a state dict
                state = ckpt
        else:
            state = ckpt
        # flexible load
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Load] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
        if len(missing) < 20 and len(unexpected) < 20 and (missing or unexpected):
            print("  missing:", missing)
            print("  unexpected:", unexpected)
    return model


def build_transform(model, image_size: int):
    cfg = resolve_data_config({}, model=model)
    # Override size if requested
    if image_size and image_size > 0:
        cfg = {**cfg, "input_size": (3, image_size, image_size)}
    tfm = create_transform(**cfg)
    return tfm


def safe_forward(model, x):
    """Try to get an embedding-like tensor from model(x)."""
    with torch.no_grad():
        out = model(x)
    # Handle common return types
    if isinstance(out, dict):
        for k in ["emb", "feat", "features", "logits"]:
            if k in out:
                out = out[k]
                break
        else:
            # take the first value
            out = list(out.values())[0]
    elif isinstance(out, (tuple, list)):
        out = out[0]
    # Flatten to (B, D)
    if out.ndim > 2:
        out = torch.flatten(out, start_dim=1)
    return out


def extract_embeddings(model, tfm, rows: List[Tuple[str,str]], device: torch.device, batch_size: int=32):
    embs = []
    groups = []
    paths  = []

    def chunk(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    total = len(rows)
    for chunk_rows in chunk(rows, batch_size):
        imgs = []
        for p, g in chunk_rows:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[Warn] Failed to open {p}: {e}")
                continue
            x = tfm(img)
            imgs.append(x)
        if not imgs:
            continue
        batch = torch.stack(imgs, 0).to(device)
        z = safe_forward(model, batch)  # (B, D)
        z = z.detach().cpu().numpy().astype(np.float32)
        embs.append(z)
        groups.extend([g for _, g in chunk_rows])
        paths.extend([p for p, _ in chunk_rows])
        done = len(paths)
        print(f"\r[Embed] {done}/{total} done", end="")
        sys.stdout.flush()
    print()  # newline
    if not embs:
        raise RuntimeError("No embeddings extracted (check images/paths).")
    embs = np.concatenate(embs, axis=0)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = embs / norms
    return embs, np.array(groups), np.array(paths)


def summarize(arr: np.ndarray, name: str):
    if arr.size == 0:
        print(f"[{name}] count=0")
        return
    arr = arr.astype(np.float64, copy=False)
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)
    print(f"[{name}] count={arr.size}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}  max={arr.max():.4f}")


def compute_sims(embs: np.ndarray) -> np.ndarray:
    sims = embs @ embs.T  # cosine since embs are L2-normalized
    np.fill_diagonal(sims, -2.0)  # exclude self-match safely
    return sims


def hard_pairs(groups: np.ndarray, paths: np.ndarray, sims: np.ndarray, topk: int):
    N = len(groups)
    same = (groups[:, None] == groups[None, :])
    np.fill_diagonal(same, False)

    iu = np.triu_indices(N, k=1)
    S = sims[iu]
    Y = same[iu]

    s_pos = S[Y]
    s_neg = S[~Y]

    print("\n=== Cosine Similarity Summary (diagonal excluded) ===")
    summarize(s_pos, "POS (same group)")
    summarize(s_neg, "NEG (diff group)")

    # Hard positives: lowest similarities
    if s_pos.size > 0:
        kpos = min(topk, s_pos.size)
        idx_pos = np.argpartition(s_pos, kpos-1)[:kpos]
        # Map back to global pair indices among POS
        pos_global = np.flatnonzero(Y)[idx_pos]
        # Sort ascending
        pos_global = pos_global[np.argsort(S[pos_global])]
    else:
        pos_global = np.array([], dtype=int)

    # Hard negatives: highest similarities
    if s_neg.size > 0:
        kneg = min(topk, s_neg.size)
        idx_neg = np.argpartition(-s_neg, kneg-1)[:kneg]
        neg_global = np.flatnonzero(~Y)[idx_neg]
        # Sort descending
        neg_global = neg_global[np.argsort(-S[neg_global])]
    else:
        neg_global = np.array([], dtype=int)

    def show(global_indices, title):
        print(f"\n--- {title} ---")
        if global_indices.size == 0:
            print("  (none)")
            return
        for g in global_indices.tolist():
            i = iu[0][g]
            j = iu[1][g]
            s = S[g]
            y = Y[g]
            print(f"sim={s:.4f} | {'POS' if y else 'NEG'} | group[i]={groups[i]} | group[j]={groups[j]}\n"
                  f"  a: {paths[i]}\n  b: {paths[j]}")

    show(pos_global, f"Hard POS (lowest sim) top-{len(pos_global)}")
    show(neg_global, f"Hard NEG (highest sim) top-{len(neg_global)}")

    return iu, S, Y


def show_neighbors(sims: np.ndarray, groups: np.ndarray, paths: np.ndarray, anchors: int, topk: int, seed: int):
    N = len(groups)
    rng = random.Random(seed)
    chosen = rng.sample(range(N), k=min(max(1, anchors), N))
    print("\n=== Random anchors: Top-K nearest neighbors ===")
    for i in chosen:
        row = sims[i].copy()
        # already -2.0 on diagonal
        nn_idx = np.argpartition(-row, min(topk, N-1)-1)[:min(topk, N-1)]
        nn_idx = nn_idx[np.argsort(-row[nn_idx])]
        print(f"\n[Anchor #{i}] group={groups[i]}  path={paths[i]}")
        for rank, j in enumerate(nn_idx, 1):
            rel = "SAME" if groups[i] == groups[j] else "DIFF"
            print(f"  #{rank:<2d} sim={row[j]:.4f} | {rel} | {paths[j]}")


def maybe_dump_pairs(save_csv: str, iu, S, Y, groups: np.ndarray, paths: np.ndarray, max_pairs: int, seed: int):
    if not save_csv:
        return
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    idx_all = np.arange(S.shape[0])
    if len(idx_all) > max_pairs:
        rng = np.random.default_rng(seed)
        idx_all = rng.choice(idx_all, size=max_pairs, replace=False)
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a_path","a_group","b_path","b_group","cosine","same_group"])
        for g in idx_all.tolist():
            i = iu[0][g]
            j = iu[1][g]
            w.writerow([paths[i], groups[i], paths[j], groups[j], f"{S[g]:.6f}", int(Y[g])])
    print(f"\n[Saved] dumped {len(idx_all)} pairs to {save_csv}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = fix_device(args.device)
    print(f"[Info] device = {device}")

    rows = read_rows(args.csv, base_dir=args.base_dir)
    if len(rows) < 2:
        print("[Error] Need at least 2 images to compute similarities.")
        sys.exit(2)

    model = load_model(args.model, args.checkpoint, device)
    tfm = build_transform(model, args.image_size)

    print(f"[Info] extracting embeddings for {len(rows)} images...")
    embs, groups, paths = extract_embeddings(model, tfm, rows, device, batch_size=args.batch_size)
    print(f"[Info] embeddings shape = {embs.shape} (L2-normalized)")

    if embs.shape[0] > 12000:
        print("[Warn] N is large. The full similarity matrix is O(N^2) memory/time. Proceed with caution.")

    sims = compute_sims(embs)
    iu, S, Y = hard_pairs(groups, paths, sims, topk=args.topk)
    show_neighbors(sims, groups, paths, anchors=args.anchors, topk=args.topk, seed=args.seed)
    maybe_dump_pairs(args.save_csv, iu, S, Y, groups, paths, args.max_pairs_csv, args.seed)

    print("\n[Done] Similarity analysis finished.")

if __name__ == "__main__":
    main()
