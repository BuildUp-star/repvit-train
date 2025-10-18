#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_pairs.py
--------------
Count how many positive (same group) and negative (different group) pairs exist,
based purely on the CSV's group labels. No model is required.

Usage:
  python count_pairs.py --csv dataset/00train_test/test.csv
  python count_pairs.py --csv dataset/00train_test/train.csv

CSV format:
  - With header: must contain "path,group" (extra columns ignored). If header missing, first two columns are used.
"""

import argparse
import csv
from collections import Counter
from math import comb

def parse_args():
    ap = argparse.ArgumentParser(description="Count POS/NEG pairs from CSV groups.")
    ap.add_argument("--csv", required=True, type=str, help="CSV with 'path,group' (header optional)")
    return ap.parse_args()

def read_groups(csv_path):
    groups = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        sniff_header = csv.Sniffer().has_header(sample)
        rdr = csv.reader(f)
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
            # path = line[p_idx].strip()  # not needed
            g = line[g_idx].strip() if len(line) > g_idx else "0"
            groups.append(g)
    return groups

def main():
    args = parse_args()
    groups = read_groups(args.csv)
    N = len(groups)
    if N < 2:
        print(f"N={N}. Need at least 2 rows.")
        return
    cnt = Counter(groups)
    pos = sum(comb(c, 2) for c in cnt.values() if c >= 2)
    tot = comb(N, 2)
    neg = tot - pos

    print("=== Pair Count Summary ===")
    print(f"Images (N): {N}")
    print(f"Groups   : {len(cnt)}")
    print("Per-group counts:")
    # show sorted by count desc
    for g, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {g}: {c}")
    print("---")
    print(f"Total pairs     C(N,2): {tot}")
    print(f"POS pairs (same group): {pos}")
    print(f"NEG pairs (diff group): {neg}")

if __name__ == "__main__":
    main()
