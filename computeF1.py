import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# === 1. 读取 CSV ===
df = pd.read_csv("runs/sims_debug.csv")

# === 2. 提取相似度和标签 ===
cosine = df["cosine"].values
labels = df["same_group"].values

# === 3. 遍历阈值范围 ===
thr_list = np.linspace(0.1, 0.99, 200)
best_f1, best_thr, best_prec, best_recall = 0, 0, 0, 0

for thr in thr_list:
    preds = (cosine > thr).astype(int)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
        best_prec, best_recall = prec, recall

print(f"Best threshold = {best_thr:.3f}")
print(f"Precision = {best_prec:.3f}, Recall = {best_recall:.3f}, F1 = {best_f1:.3f}")
