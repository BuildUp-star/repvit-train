# Split & Pairing for Embedding Training

This dataset was auto-split by **subfolder groups** per top-level class, using an 80/20 split by groups.
- Top-level classes: the first directory under root (e.g., `corridor/`, `home/`, `market/`, `office/`, `restaurant/`, `station/`).
- A *group* is the second-level subfolder (e.g., `station/001`), which contains visually similar frames.
- Train/Test split uses whole groups to avoid leakage.

Generated files:
- `train.csv`: columns = [path, class, group]
- `test.csv`: columns = [path, class, group]
- `pairs_pos.csv`: positive pairs (same class & group)
- `pairs_neg.csv`: negative pairs (different groups; mix of intra-class and cross-class)
- `triplets.csv`: (anchor, positive, negative) triplets suitable for triplet / contrastive losses

**Notes**
- Positive pairs mainly use adjacent frames plus a few spaced pairs per group to increase diversity.
- Negative pairs sample fixed counts per anchor: up to 4 from other groups in the same class + up to 2 cross-class.
- Sampling is deterministic (seed=42) for reproducibility.
- Modify the code to adjust ratios or counts if you want more or fewer pairs/triplets.

**Typical usage**
- Contrastive/BCE pairs: load `pairs_pos.csv` and `pairs_neg.csv` together; label=1 for positives, 0 for negatives.
- Triplet loss: use `triplets.csv` with (anchor, positive, negative).

