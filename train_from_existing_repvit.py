#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse, random, time
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm  # 需要: pip install timm

# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize(z): return F.normalize(z, dim=-1)

# ----------------- datasets -----------------
class PairDataset(Dataset):
    def __init__(self, pos_csv, neg_csv, transform):
        import csv, os
        self.samples = []
        # 以 CSV 所在目录为基准，解析相对路径
        base = None
        if pos_csv and os.path.exists(pos_csv):
            base = os.path.dirname(os.path.abspath(pos_csv))
        if not base and neg_csv and os.path.exists(neg_csv):
            base = os.path.dirname(os.path.abspath(neg_csv))
        if not base:
            base = "."

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(base, p)

        if pos_csv and os.path.exists(pos_csv):
            with open(pos_csv, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    self.samples.append((_resolve(r['a']), _resolve(r['b']), 1.0))
        if neg_csv and os.path.exists(neg_csv):
            with open(neg_csv, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    self.samples.append((_resolve(r['a']), _resolve(r['b']), 0.0))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        a, b, y = self.samples[i]
        ia = Image.open(a).convert('RGB')
        ib = Image.open(b).convert('RGB')
        if self.transform:
            ia = self.transform(ia)
            ib = self.transform(ib)
        return ia, ib, torch.tensor([y], dtype=torch.float32)

class TripletDataset(Dataset):
    def __init__(self, csv_path, transform):
        import csv, os
        base = os.path.dirname(os.path.abspath(csv_path))

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(base, p)

        with open(csv_path, 'r', encoding='utf-8') as f:
            self.rows = [(_resolve(r['anchor']), _resolve(r['positive']), _resolve(r['negative']))
                         for r in csv.DictReader(f)]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        a, p, n = self.rows[i]
        ia = Image.open(a).convert('RGB')
        ip = Image.open(p).convert('RGB')
        inn = Image.open(n).convert('RGB')
        if self.transform:
            ia = self.transform(ia)
            ip = self.transform(ip)
            inn = self.transform(inn)
        return ia, ip, inn

# ----------------- model wrap -----------------
class RepViTWithHead(nn.Module):
    """ 复用已创建的 timm 模型，移除分类头，接 512-d embedding head """
    def __init__(self, backbone: nn.Module, embed_dim=512, mlp=False):
        super().__init__()
        self.backbone = backbone

        # 移除分类头 + 启用全局池化，确保 forward 输出为 (B, feat_dim)
        if hasattr(self.backbone, 'reset_classifier'):
            self.backbone.reset_classifier(num_classes=0, global_pool='avg')
        feat_dim = getattr(self.backbone, 'num_features', None)
        if feat_dim is None:
            # 兜底：跑一次 dummy 推理推断维度
            with torch.no_grad():
                z = self.backbone(torch.zeros(1,3,224,224))
                feat_dim = z.shape[-1]

        if mlp:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, embed_dim),
            )
        else:
            self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        f = self.backbone(x)          # (B, feat_dim)，已全局池化
        z = self.head(f)              # (B, embed_dim)
        return normalize(z)           # 归一化，便于余弦相似

def freeze_by_ratio(module: nn.Module, ratio: float):
    """按参数次序冻结前 ratio 比例（0~1）。不依赖具体层名，通用且稳妥。"""
    params = [p for p in module.parameters()]
    cutoff = int(len(params) * ratio)
    for i, p in enumerate(params):
        p.requires_grad = (i >= cutoff)

# ----------------- losses -----------------
class BCEPairLoss(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
        self.crit = nn.BCEWithLogitsLoss()
    def forward(self, za, zb, y):
        logit = (za * zb).sum(-1) * self.scale
        return self.crit(logit, y)

# ----------------- eval: 组内最近邻召回 -----------------
def read_items(csv_path):
    """用于评测：读取 test.csv 的 (path, class/group) 项，并把相对路径解析到 CSV 所在目录"""
    import csv, os
    rows = []
    base = os.path.dirname(os.path.abspath(csv_path))

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((_resolve(r['path']), f"{r['class']}/{r['group']}"))
    return rows

@torch.no_grad()
def eval_group_recall(model, test_csv, image_size, device):
    if not os.path.exists(test_csv): return None
    rows = read_items(test_csv)
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    embs, groups = [], []
    model.eval()
    for path, gid in rows:
        x = tfm(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
        z = model(x).cpu().numpy()[0]
        embs.append(z); groups.append(gid)
    embs = np.stack(embs, 0)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-9)
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    nn_idx = sims.argmax(1)
    hits = sum(1 for i,j in enumerate(nn_idx) if groups[i]==groups[j])
    return hits/len(rows)

# ----------------- train -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', type=str, default='dataset/00train_test')
    ap.add_argument('--method', type=str, choices=['triplet','pairs-bce'], default='triplet')
    ap.add_argument('--model_name', type=str, default='repvit_m1_0')
    ap.add_argument('--embed_dim', type=int, default=512)
    ap.add_argument('--freeze_ratio', type=float, default=0.8, help='冻结前多少比例参数 (0~1)')
    ap.add_argument('--mlp_head', action='store_true', help='使用两层 MLP 作为 head')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--save', type=str, default='runs/repvit_embed_512.pt')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every_sec', type=float, default=10.0, help='每隔多少秒打印一次训练进度')
    ap.add_argument('--patience', type=int, default=3, help='Eval 指标若连续多少个 epoch 不提升则早停')
    ap.add_argument('--target_recall', type=float, default=None, help='达到该group-NN recall@1阈值（0~1）则提前停止')
    ap.add_argument('--save_best', type=str, default='runs/repvit_embed_512_best.pt', help='最佳模型单独保存路径')

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1) 复用你指定的 timm 模型
    backbone = timm.create_model(args.model_name, pretrained=True).eval()
    # 移除分类头 + 开池化
    if hasattr(backbone, 'reset_classifier'):
        backbone.reset_classifier(num_classes=0, global_pool='avg')

    model = RepViTWithHead(backbone, embed_dim=args.embed_dim, mlp=args.mlp_head).to(device)

    # 2) 冻结前面一部分参数
    freeze_by_ratio(model.backbone, args.freeze_ratio)
    # head 全部训练
    for p in model.head.parameters():
        p.requires_grad = True

    # 3) 数据
    tf_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomResizedCrop(args.image_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    if args.method == 'triplet':
        tri_csv = os.path.join(args.csv_dir, 'triplets.csv')
        ds = TripletDataset(tri_csv, tf_train)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        criterion = nn.TripletMarginLoss(margin=0.2, p=2.0)
    else:
        pos_csv = os.path.join(args.csv_dir, 'pairs_pos.csv')
        neg_csv = os.path.join(args.csv_dir, 'pairs_neg.csv')
        ds = PairDataset(pos_csv, neg_csv, tf_train)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        criterion = BCEPairLoss(scale=20.0)

    # 4) 优化器
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 5) 训练 + 心跳日志 + 早停
    best_rec = -1.0
    epochs_no_improve = 0

    for ep in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        last_log = t0
        loss_acc = 0.0
        it_cnt = 0
        seen_images = 0

        for it, batch in enumerate(dl, start=1):
            optim.zero_grad(set_to_none=True)
            bsz = batch[0].size(0)  # 当前 batch 大小
            with torch.cuda.amp.autocast(enabled=args.fp16):
                if args.method == 'triplet':
                    xa, xp, xn = [t.to(device) for t in batch]
                    za, zp, zn = model(xa), model(xp), model(xn)
                    loss = criterion(za, zp, zn)
                else:
                    xa, xb, y = batch
                    xa, xb, y = xa.to(device), xb.to(device), y.to(device)
                    za, zb = model(xa), model(xb)
                    loss = criterion(za, zb, y)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            # 累计指标
            it_cnt += 1
            seen_images += bsz
            loss_acc += loss.item()

            # === 心跳日志：每 args.log_every_sec 秒输出一次 ===
            now = time.time()
            if now - last_log >= args.log_every_sec:
                elapsed = now - t0
                avg_loss = loss_acc / it_cnt
                ips = seen_images / max(elapsed, 1e-9)  # images per second
                # 预估本 epoch ETA
                total_it = len(dl)
                eta_epoch = (elapsed / max(it_cnt,1)) * (total_it - it_cnt)
                print(f"[Epoch {ep} | {it_cnt}/{total_it}] "
                      f"avg_loss={avg_loss:.4f}  "
                      f"speed={ips:.1f} img/s  "
                      f"eta_epoch={eta_epoch:.1f}s")
                last_log = now

        # --- 一个 epoch 结束 ---
        dt = time.time() - t0
        epoch_loss = loss_acc / max(it_cnt,1)
        print(f"[Epoch {ep} DONE] loss={epoch_loss:.4f}  time={dt:.1f}s")

        # 轻量评测（组内最近邻召回）
        test_csv = os.path.join(args.csv_dir, 'test.csv')
        rec = eval_group_recall(model, test_csv, args.image_size, device)
        if rec is not None:
            print(f"[Eval] group-NN recall@1 = {rec*100:.2f}%")
            # 保存最佳
            if rec > best_rec:
                best_rec = rec
                epochs_no_improve = 0
                torch.save({'epoch': ep, 'model': model.state_dict(), 'args': vars(args)}, args.save_best)
                print(f"[Best Improved] saved best to {args.save_best}")
            else:
                epochs_no_improve += 1

            # 目标达成就提前停止
            if args.target_recall is not None and rec >= args.target_recall:
                print(f"[EarlyStop] 达到目标 recall {args.target_recall*100:.2f}% ，提前停止。")
                break

            # 早停：连续若干 epoch 无提升
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[EarlyStop] 连续 {args.patience} 个 epoch 指标未提升，提前停止。")
                break

        # 常规保存（每个 epoch）
        torch.save({'epoch': ep, 'model': model.state_dict(), 'args': vars(args)}, args.save)
        print(f"[Saved] {args.save}")

    print("Done.")
    if best_rec >= 0:
        print(f"[Summary] best recall@1 = {best_rec*100:.2f}% | best_model: {args.save_best}")


if __name__ == "__main__":
    main()
