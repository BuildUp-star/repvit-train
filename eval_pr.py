# eval_pr.py
import argparse, os, csv, math
from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def read_items(csv_path: str, base_dir: str = "") -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, 1):
            p = row.get('path') or row.get('image') or row.get('img') or ''
            g = row.get('group') or row.get('label') or row.get('cls') or ''
            if not p: 
                continue
            full = os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
            if not os.path.exists(full):
                print(f"[Warn] missing file: {full}")
                continue
            rows.append((full, str(g)))
    if not rows:
        raise ValueError("No valid rows found in CSV.")
    return rows

@torch.no_grad()
def extract_embeddings(model, files, transform, device, batch_size=64):
    xs = []
    for i in range(0, len(files), batch_size):
        paths = files[i:i+batch_size]
        ims = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            ims.append(transform(im))
        x = torch.stack(ims, 0).to(device)
        # 直接用 model(x) 的输出维度（logits 或者你训练的头的维度）
        out = model(x)
        if out.ndim > 2:
            out = out.flatten(1)
        out = F.normalize(out, p=2, dim=-1)
        xs.append(out.cpu())
    Z = torch.cat(xs, 0)  # [N, D]
    return Z

def compute_metrics(emb, groups, thr: float, max_pairs: int = 0):
    """
    emb: [N,D] (L2-normalized), groups: list[str]
    thr: cosine threshold, pred_same = (cos > thr)
    """
    N = emb.size(0)
    sims = emb @ emb.T                  # [N,N]
    sims.fill_diagonal_(-1.0)          # 避免自比对被选中

    # 采样或全量两两组合的上三角索引
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    if max_pairs and max_pairs < idx_i.numel():
        # 随机下采样，保证大数据也能跑
        sel = torch.randperm(idx_i.numel())[:max_pairs]
        idx_i, idx_j = idx_i[sel], idx_j[sel]

    cosv = sims[idx_i, idx_j]          # [M]
    pred_same = cosv > thr

    g = torch.tensor([hash(x) for x in groups], dtype=torch.long)
    true_same = (g[idx_i] == g[idx_j])

    TP = int(((pred_same == True)  & (true_same == True)).sum().item())
    FP = int(((pred_same == True)  & (true_same == False)).sum().item())
    TN = int(((pred_same == False) & (true_same == False)).sum().item())
    FN = int(((pred_same == False) & (true_same == True)).sum().item())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "pairs": idx_i.numel(),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with columns: path,group')
    ap.add_argument('--base_dir', default='', help='Prefix for relative paths in CSV')
    ap.add_argument('--model_name', default='repvit_m1_0')
    ap.add_argument('--checkpoint', default='', help='optional .pt/.pth to load (model head unchanged)')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--max_pairs', type=int, default=0, help='limit number of pairs for speed (0 = all)')
    args = ap.parse_args()

    # device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = args.device

    # data
    rows = read_items(args.csv, args.base_dir)
    files = [p for p,_ in rows]
    groups = [g for _,g in rows]
    print(f"[Info] images = {len(files)} | unique groups = {len(set(groups))}")

    # model & transform
    model = timm.create_model(args.model_name, pretrained=True).to(device).eval()
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        # 尝试严格/非严格加载
        try:
            model.load_state_dict(ckpt, strict=True)
        except Exception:
            model.load_state_dict(ckpt, strict=False)
        print(f"[Info] loaded checkpoint: {args.checkpoint}")

    cfg = resolve_data_config({'input_size': (3, args.image_size, args.image_size)}, model=model)
    transform = create_transform(**cfg)

    # embeddings
    with torch.no_grad():
        emb = extract_embeddings(model, files, transform, device, args.batch_size)
    print(f"[Info] embeddings shape = {tuple(emb.shape)}  (D = {emb.shape[1]})")

    # metrics @ threshold
    res = compute_metrics(emb, groups, thr=args.threshold, max_pairs=args.max_pairs)
    print("\n===== Metrics @ threshold {:.3f} =====".format(args.threshold))
    print(f"pairs={res['pairs']}  TP={res['TP']} FP={res['FP']} TN={res['TN']} FN={res['FN']}")
    print("precision={:.4f}  recall={:.4f}  f1={:.4f}".format(res['precision'], res['recall'], res['f1']))

if __name__ == "__main__":
    main()
