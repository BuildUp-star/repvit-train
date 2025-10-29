#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse, random, time
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from contextlib import nullcontext
# --- 可选：DirectML (Qualcomm Elite X) 支持 ---
# 如果你在 Windows on ARM / Snapdragon X Elite 上安装了 `pip install torch-directml`
# 这里会自动启用 DML 设备以使用笔记本 GPU 训练；否则忽略。
try:
    import torch_directml as _dml
    _DML_AVAILABLE = True
except Exception:
    _DML_AVAILABLE = False
    _dml = None
import timm  # 需要: pip install timm

# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize(z): return F.normalize(z, dim=-1)

# ----------------- datasets -----------------
# -------- SupCon dataset (CSV: path,class,group) --------
class TransformTwice:
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, img): return self.tfm(img), self.tfm(img)

class SupConCSVDataset(Dataset):
    """
    从 train.csv 读取每张图片的 (path, class, group)，
    返回两次随机增强的 view1, view2，以及整数化的 group_id
    """
    def __init__(self, csv_path, transform):
        base = os.path.dirname(os.path.abspath(csv_path))
        def _resolve(p): return p if os.path.isabs(p) else os.path.join(base, p)

        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append((_resolve(r['path']), f"{r['class']}/{r['group']}"))

        # 将 "class/group" 映射成连续的整数 id
        gids = sorted({g for _, g in rows})
        self.gid_map = {g:i for i,g in enumerate(gids)}
        self.items = [(p, self.gid_map[g]) for p, g in rows]
        self.tfm2 = TransformTwice(transform)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, gid = self.items[i]
        img = Image.open(p).convert('RGB')
        v1, v2 = self.tfm2(img)
        return v1, v2, torch.tensor(gid, dtype=torch.long)


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
        
def l2_normalize(x, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))

class RepViTWithLogitHead(nn.Module):
    """
    不改动/不重置 backbone（timm RepViT），直接在 logits 后面接 512-d embedding head。
    适用于你的“logits dim = 1000, features dim = 448”的场景。
    """
    def __init__(self, backbone: nn.Module, embed_dim=512, mlp=False, norm=True):
        super().__init__()
        self.backbone = backbone  # 不调用 reset_classifier，不动它
        self.norm = norm

        # 尝试直接从模型属性拿 logits 维度（num_classes）
        in_dim = getattr(self.backbone, 'num_classes', None)

        # 兜底：跑一次 dummy 推理推断 logits 维度（注意不会反传梯度）
        if in_dim is None or in_dim == 0:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                logits = self.backbone(dummy)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                in_dim = logits.shape[-1]

        if mlp:
            self.head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, embed_dim),
            )
        else:
            self.head = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        # timm 的 forward 默认返回分类 logits（未经 softmax）
        logits = self.backbone(x)                 # (B, num_classes)
        if isinstance(logits, (list, tuple)):     # 保险：有些模型可能返回额外输出
            logits = logits[0]
        z = self.head(logits)                     # (B, embed_dim)
        return l2_normalize(z) if self.norm else z
        
class LogitsAsEmbedding(nn.Module):
    """
    直接用模型最后一层输出（logits）作为 embedding。
    - 不 reset、不移除分类头。
    - 默认做 L2 归一化，便于用余弦相似度。
    """
    def __init__(self, backbone: nn.Module, l2norm: bool = True):
        super().__init__()
        self.backbone = backbone
        self.l2norm = l2norm

    def forward(self, x):
        # timm 模型通常返回 (B, num_classes) 的 logits
        out = self.backbone(x)
        # 兼容某些模型返回 tuple/list 或者空间张量
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim > 2:
            out = out.flatten(1)
        return F.normalize(out, p=2, dim=-1) if self.l2norm else out

def freeze_by_ratio(module: nn.Module, ratio: float):
    """按参数次序冻结前 ratio 比例（0~1）。不依赖具体层名，通用且稳妥。"""
    params = [p for p in module.parameters()]
    cutoff = int(len(params) * ratio)
    for i, p in enumerate(params):
        p.requires_grad = (i >= cutoff)

# ----------------- losses -----------------
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss:
    输入 Z:[B,D], labels:[B]；同 label 皆为正样本，其余为负（自身不计）
    可与双视图拼接一起用：Z=cat(z1,z2), labels=cat(g,g)
    """
    def __init__(self, temperature=0.1, normalize=True):
        super().__init__()
        self.tau = float(temperature)
        self.normalize = normalize

    def forward(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert Z.dim()==2 and labels.dim()==1 and Z.size(0)==labels.size(0)
        if self.normalize:
            Z = F.normalize(Z, dim=1)

        B = Z.size(0)
        sim = (Z @ Z.t()) / self.tau                   # [B,B]
        # 自身不参与对比
        sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=Z.device), float('-inf'))

        # 构造正样本掩码（自身为 False）
        labels = labels.view(-1,1)
        pos_mask = (labels == labels.t()) & (~torch.eye(B, dtype=torch.bool, device=Z.device))
        pos_mask = pos_mask.float()

        # 分母：所有（非自身）样本
        log_denom = torch.logsumexp(sim, dim=1)        # [B]

        # 分子：对所有正样本做 logsumexp（无正样本的行将被忽略）
        # 将非正样本置为 -inf：用 log(pos_mask) 实现（0→-inf, 1→0）
        log_pos = sim + torch.log(pos_mask + 1e-12)
        log_num = torch.logsumexp(log_pos, dim=1)      # [B]

        # 只统计有正样本的行
        valid = (pos_mask.sum(dim=1) > 0).float()      # [B]
        loss = -(log_num - log_denom) * valid
        loss = loss.sum() / (valid.sum() + 1e-12)
        return loss

class BCEPairLossB(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
        self.crit = nn.BCEWithLogitsLoss()
    def forward(self, za, zb, y):
        logit = (za * zb).sum(-1) * self.scale       # [B]
        y = y.float().reshape_as(logit)              # ✅ 对齐形状与类型
        return self.crit(logit, y)
        
import torch.nn.functional as F

class BCEPairLoss(nn.Module):
    def __init__(self, scale=20.0, normalize=True, pos_weight=None):
        super().__init__()
        self.scale = scale
        self.normalize = normalize
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, za, zb, y):
        # 可选：L2 归一化 → 余弦相似度
        if self.normalize:
            za = F.normalize(za, dim=1)
            zb = F.normalize(zb, dim=1)
        s = (za * zb).sum(dim=1)             # [B]
        logit = s * self.scale               # [B]
        y = y.float().reshape_as(logit)      # [B]
        # 若标签为 {-1,+1}，取消注释下一行
        # y = (y > 0).float()
        return self.crit(logit, y)

class BCEAtFixedThresholdLoss(nn.Module):
   """
   把判别阈值 τ 直接编码进 logit：logit = scale * (cosine - τ)。
   这样用 BCEWithLogitsLoss 训练，模型就会直接朝着“s >= τ 判正”的目标优化。
   """
   def __init__(self, tau=0.60, scale=20.0, normalize=True, pos_weight=None):
       super().__init__()
       self.tau = float(tau)
       self.scale = float(scale)
       self.normalize = normalize
       self.crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   def forward(self, za, zb, y):
       if self.normalize:
           za = F.normalize(za, dim=1)
           zb = F.normalize(zb, dim=1)
       s = (za * zb).sum(dim=1)             # [B]  余弦相似度
       logit = (s - self.tau) * self.scale  # 把 τ 移进 logit
       y = y.float().reshape_as(logit)
       return self.crit(logit, y)
# ----------------- TauPullPushLoss -----------------
class TauPullPushLoss(nn.Module):
    """
+    TauPullPushLoss = 判别项(阈值对齐) + 拉近/推远项(极值正则)
+      - 判别项：以固定阈值 τ 为决策面（s=cos≥τ 判正），与部署一致
+      - 拉近/推远：正样本持续往 +1 拉、负样本持续往 −1 推（即便已在 τ 正确一侧）
+    形式：
+      L_pos = softplus(-alpha*(s-τ))  + λ_pull*(1-s)^2
+      L_neg = softplus( alpha*(s-τ))  + λ_push*(s+1)^2
+      L = w_tau * [ y*L_pos + (1-y)*L_neg ] （可叠加 pos_weight）
+      其中 w_tau = exp(- (s-τ)^2 / (2*sigma^2)) 为 τ-聚焦（可关）
+    """
    def __init__(self,
                tau: float = 0.60,
                alpha: float = 10.0,
                lambda_pull: float = 0.2,
                lambda_push: float = 0.2,
                normalize: bool = True,
                pos_weight: Optional[float] = None,
                use_tau_focus: bool = True,
                sigma_tau: float = 0.15):
       super().__init__()
       self.tau = float(tau)
       self.alpha = float(alpha)
       self.lambda_pull = float(lambda_pull)
       self.lambda_push = float(lambda_push)
       self.normalize = normalize
       self.pos_weight = pos_weight
       self.use_tau_focus = use_tau_focus
       self.sigma_tau = float(sigma_tau)

    def forward(self, za: torch.Tensor, zb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       if self.normalize:
           za = F.normalize(za, dim=1)
           zb = F.normalize(zb, dim=1)
       s = (za * zb).sum(dim=1)                 # 余弦相似度 ∈ [-1,1]
       y = y.float().view_as(s)
       # 判别项：以 τ 为中心的对称 softplus（数值稳定、在 τ 附近有梯度）
       cls_pos = F.softplus(-self.alpha * (s - self.tau))   # y=1
       cls_neg = F.softplus( self.alpha * (s - self.tau))   # y=0
       # 拉近/推远项：把正对往 +1 拉、负对往 −1 推
       pull = (1.0 - s).pow(2)                               # y=1
       push = (s + 1.0).pow(2)                               # y=0
       # τ-聚焦：越靠近 τ 权重越大（可关）
       if self.use_tau_focus:
           w_tau = torch.exp(-((s - self.tau) ** 2) / (2 * (self.sigma_tau ** 2)))
       else:
           w_tau = torch.ones_like(s)
       # 类别不均衡权重
       w_pos = float(self.pos_weight) if self.pos_weight is not None else 1.0
       w_neg = 1.0
       loss_pos = w_tau * (cls_pos + self.lambda_pull * pull)
       loss_neg = w_tau * (cls_neg + self.lambda_push * push)
       loss = y * (w_pos * loss_pos) + (1.0 - y) * (w_neg * loss_neg)
       return loss.mean()

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
def eval_group_metrics(model, test_csv, image_size, device, save_confusion_csv=None):
    """
    评测最近邻(1-NN)分类的多指标：accuracy (= micro P/R), macro P/R/F1，和可选混淆矩阵。
    返回一个 dict，含各项指标。
    """
    if not os.path.exists(test_csv):
        return None

    rows = read_items(test_csv)  # [(path, "class/group"), ...]
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # 1) 提取 embedding 与标签
    embs, groups = [], []
    model.eval()
    for path, gid in rows:
        x = tfm(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
        z = model(x).cpu().numpy()[0]
        embs.append(z); groups.append(gid)
    embs = np.stack(embs, 0)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    # 2) 1-NN 预测
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    nn_idx = sims.argmax(1)
    y_true = np.array(groups)
    y_pred = np.array([groups[j] for j in nn_idx])

    # 3) 映射到整数标签，便于统计
    labels = sorted(list(set(groups)))
    lid = {g:i for i,g in enumerate(labels)}
    yt = np.array([lid[g] for g in y_true], dtype=np.int64)
    yp = np.array([lid[g] for g in y_pred], dtype=np.int64)
    C = len(labels)

    # 4) 统计 TP/FP/FN（逐类）
    conf = np.zeros((C, C), dtype=np.int64)  # [true, pred]
    for t, p in zip(yt, yp):
        conf[t, p] += 1

    per_prec, per_rec, per_f1 = [], [], []
    for c in range(C):
        TP = conf[c, c]
        FP = conf[:, c].sum() - TP
        FN = conf[c, :].sum() - TP
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        per_prec.append(prec); per_rec.append(rec); per_f1.append(f1)

    macro_p = float(np.mean(per_prec))
    macro_r = float(np.mean(per_rec))
    macro_f1 = float(np.mean(per_f1))
    accuracy = float(np.trace(conf)) / max(len(yt), 1)  # 也等于 micro-precision/micro-recall

    # 可选：把混淆矩阵存成 CSV，便于排查具体混淆对
    if save_confusion_csv is not None:
        import csv
        os.makedirs(os.path.dirname(save_confusion_csv), exist_ok=True)
        with open(save_confusion_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)  # header: 预测列名
            for i, g in enumerate(labels):
                writer.writerow([g] + conf[i].tolist())

    return {
        "accuracy": accuracy,            # = micro P/R
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "num_classes": C,
        "support": conf.sum(axis=1).tolist(),  # 每类样本数
    }

# ----------------- train -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', type=str, default='dataset/00train_test')
    #ap.add_argument('--method', type=str, choices=['triplet','pairs-bce','pairs-tpp'], default='triplet')
    ap.add_argument('--method', type=str, choices=['triplet','pairs-bce','pairs-tpp','supcon'], default='supcon')
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
    ap.add_argument('--patience', type=int, default=0, help='Eval 指标若连续多少个 epoch 不提升则早停')
    ap.add_argument('--target_recall', type=float, default=None, help='达到该group-NN recall@1阈值（0~1）则提前停止')
    ap.add_argument('--save_best', type=str, default='runs/repvit_embed_512_best.pt', help='最佳模型单独保存路径')
    ap.add_argument('--tau', type=float, default=0.60, help='固定阈值训练/评测所用的 cosine 阈值 τ')
    ap.add_argument('--bce_at_tau', action='store_true', help='pairs 模式下：使用“阈值对齐 BCE”训练（logit=scale*(cos-τ)）')
    # ---- TauPullPushLoss 相关超参 ----
    ap.add_argument('--pos_weight', type=float, default=None, help='正样本权重（类别不均衡时可设为 neg/pos 比）')
    ap.add_argument('--tpp_alpha', type=float, default=10.0, help='TauPullPush 的斜率 alpha（8~12 防饱和）')
    ap.add_argument('--tpp_pull',  type=float, default=0.2,  help='正样本拉近权重 λ_pull')
    ap.add_argument('--tpp_push',  type=float, default=0.2,  help='负样本推远权重 λ_push')
    ap.add_argument('--tpp_sigma', type=float, default=0.15, help='τ-聚焦的带宽 sigma（越小越聚焦边界）')
    ap.add_argument('--no_tau_focus', action='store_true', help='关闭 τ-聚焦权重（默认开启）')
    ap.add_argument('--temperature', type=float, default=0.1, help='SupCon/InfoNCE 温度τ')
    #load ?
    ap.add_argument("--load", type=str, default="", help="path to pretrained checkpoint")
    # ---- 额外：初始化模型在 train.csv 上的阈值评测 ----
    ap.add_argument('--eval_init_train', action='store_true',
                    help='在训练开始前，用初始化模型对 train.csv 扫阈值并打印最佳 F1')
    ap.add_argument('--only_eval_init_train', action='store_true',
                    help='只做初始化评测（train.csv），评测后直接退出，不进入训练')
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    # -------------------- 设备选择（CUDA > DirectML > CPU） --------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        runtime = 'cuda'
    elif '_DML_AVAILABLE' in globals() and _DML_AVAILABLE:
        device = _dml.device()
        runtime = 'dml'
    else:
        device = torch.device('cpu')
        runtime = 'cpu'
    print(f"Device: {device} (runtime={runtime})")

    # 仅在 CUDA 上启用 AMP/GradScaler；DML/CPU 关闭
    use_amp = bool(args.fp16 and isinstance(device, torch.device) and device.type == 'cuda')
    amp_autocast = torch.cuda.amp.autocast if use_amp else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # DataLoader 的 pin_memory 只在 CUDA 有意义
    pin_mem = bool(isinstance(device, torch.device) and device.type == 'cuda')
    print(f"Device: {device}")

    # 1) 复用你指定的 timm 模型
    #backbone = timm.create_model(args.model_name, pretrained=True).eval()
    model_name = "repvit_m1_0"
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load("models/repvit_m1_0_weights_only.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    backbone = model.eval()
    # 移除分类头 + 开池化
    #if hasattr(backbone, 'reset_classifier'):
        #backbone.reset_classifier(num_classes=0, global_pool='avg')

    model = RepViTWithLogitHead(backbone, embed_dim=args.embed_dim, mlp=args.mlp_head).to(device)
    if args.load:
        print(f"Loading from {args.load}")
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"✅ Loaded checkpoint (epoch={ckpt.get('epoch', '?')})")
    #model = LogitsAsEmbedding(backbone, l2norm=True).to(device)

    # 2) 冻结前面一部分参数
    freeze_by_ratio(model.backbone, args.freeze_ratio)
    # head 全部训练
    #for p in model.head.parameters():
    #    p.requires_grad = True
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    # === 初始化模型在 train.csv 上的阈值评测（可选） ===
    if args.eval_init_train or args.only_eval_init_train:
        train_csv_path = os.path.join(args.csv_dir, 'test.csv')
        if os.path.exists(train_csv_path):
            print("\n[Init-Eval] 开始在 train.csv 上评测初始化模型（扫阈值 0.20~0.99） ...")
            init_report = eval_threshold_metrics(
                model, train_csv_path, args.image_size, device,
                thr_list=None,           # 默认 linspace(0.2,0.99,80)
                pos_rule="same_group",
                max_pairs=1_000_000,
                min_index_gap=0
            )
            print(f"[Init-Eval@train.csv] best τ* = {init_report['best']['thr']:.3f}  "
                  f"(F1={init_report['best']['f1']:.3f})")
        else:
            print(f"[Init-Eval] 未找到 {train_csv_path}，跳过初始化评测。")
        if args.only_eval_init_train:
            print("[Init-Eval] 仅评测模式已完成，程序退出。")
            return

    # 3) 数据
    tf_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomResizedCrop(args.image_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    if args.method == 'supcon':
        train_csv = os.path.join(args.csv_dir, 'train.csv')
        ds = SupConCSVDataset(train_csv, tf_train)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_mem, drop_last=True)
        criterion = SupConLoss(temperature=args.temperature, normalize=True)

    elif args.method == 'triplet':
        tri_csv = os.path.join(args.csv_dir, 'triplets.csv')
        ds = TripletDataset(tri_csv, tf_train)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_mem, drop_last=True)
        criterion = nn.TripletMarginLoss(margin=0.2, p=2.0)
    else:
        pos_csv = os.path.join(args.csv_dir, 'pairs_pos.csv')
        neg_csv = os.path.join(args.csv_dir, 'pairs_neg.csv')
        ds = PairDataset(pos_csv, neg_csv, tf_train)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=pin_mem, drop_last=True)
        if args.method == 'pairs-tpp':
           criterion = TauPullPushLoss(
               tau=args.tau,
               alpha=args.tpp_alpha,
               lambda_pull=args.tpp_pull,
               lambda_push=args.tpp_push,
               normalize=True,
               pos_weight=args.pos_weight,
               use_tau_focus=(not args.no_tau_focus),
               sigma_tau=args.tpp_sigma
           )
        elif args.bce_at_tau:
            criterion = BCEAtFixedThresholdLoss(tau=args.tau, scale=20.0, pos_weight=args.pos_weight)
        else:
            criterion = BCEPairLoss(scale=20.0, pos_weight=args.pos_weight)

    # 4) 优化器
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    #scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

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
        
                # 轻量评测（组内最近邻召回）
        test_csv = os.path.join(args.csv_dir, 'test.csv')
        metrics = eval_group_metrics(
            model, test_csv, args.image_size, device,
            save_confusion_csv=os.path.join(os.path.dirname(args.save), "confusion_test.csv")
        )
        if metrics is not None:
            acc = metrics["accuracy"]
            mp  = metrics["macro_precision"]
            mr  = metrics["macro_recall"]
            mf1 = metrics["macro_f1"]
            print(f"[Eval] 1-NN accuracy={acc*100:.2f}% | macro P={mp*100:.2f}% R={mr*100:.2f}% F1={mf1*100:.2f}%")
            # === 阈值式回环评测：给出最佳阈值 τ* ===
            thr_report = eval_threshold_metrics(
                model, test_csv, args.image_size, device,
                thr_list=None,          # 默认 0.2~0.99 扫
                pos_rule="same_group",  # test.csv 中同 class/group 视为回环
                max_pairs=500_000,      # 按需调整
                min_index_gap=0         # 如按时间序列评测可设成 5/10 等
            )
            best_thr = thr_report["best"]["thr"]
            best_f1  = thr_report["best"]["f1"]
            print(f"[ThresholdEval] best τ* = {best_thr:.3f}  (F1={best_f1:.3f})")

            # 若你更关心“召回≥X%下的最高精度”
            sel = choose_threshold_by_target_recall(thr_report["curve"], target_recall=0.90)
            print(f"[ThresholdEval@Recall≥90%] τ = {sel['thr']:.3f} | "
                  f"P={sel['precision']:.3f} R={sel['recall']:.3f} F1={sel['f1']:.3f} ({sel['by']})")
            # 额外：在“固定 τ=args.tau”处，直接打印二分类指标，便于和你的部署阈值一致
            fixed = [r for r in thr_report["curve"] if abs(r["thr"] - args.tau) < 1e-9]
            if not fixed:
                # 若 τ 恰不在扫描网格，做一次单点评测
                fixed_report = eval_threshold_metrics(
                    model, test_csv, args.image_size, device,
                    thr_list=[args.tau], pos_rule="same_group"
                )
                fixed = [fixed_report["curve"][0]]
            fr = fixed[0]
            print(f"[ThresholdEval@τ={args.tau:.3f}] P={fr['precision']:.3f} R={fr['recall']:.3f} F1={fr['f1']:.3f}")

            
            # 用 accuracy 作为“早停/最佳”指标（也可换成 macro_f1）
            if acc > best_rec:
                best_rec = acc
                epochs_no_improve = 0
                torch.save({'epoch': ep, 'model': model.state_dict(), 'args': vars(args)}, args.save_best)
                print(f"[Best Improved] saved best to {args.save_best}")
            else:
                epochs_no_improve += 1

            if args.target_recall is not None and acc >= args.target_recall:
                print(f"[EarlyStop] 达到目标 accuracy {args.target_recall*100:.2f}% ，提前停止。")
                break

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"[EarlyStop] 连续 {args.patience} 个 epoch 指标未提升，提前停止。")
                break

        for it, batch in enumerate(dl, start=1):
            optim.zero_grad(set_to_none=True)
            bsz = batch[0].size(0)  # 当前 batch 大小
            #with torch.cuda.amp.autocast(enabled=args.fp16):
            with amp_autocast():
                if args.method == 'supcon':
                    v1, v2, g = batch
                    v1, v2, g = v1.to(device), v2.to(device), g.to(device)
                    z1, z2 = model(v1), model(v2)                   # [B,D],[B,D]
                    Z  = torch.cat([z1, z2], dim=0)                 # [2B,D]
                    G  = torch.cat([g,  g ], dim=0)                 # [2B]
                    loss = criterion(Z, G)
                elif args.method == 'triplet':
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




        # 常规保存（每个 epoch）
        torch.save({'epoch': ep, 'model': model.state_dict(), 'args': vars(args)}, args.save)
        print(f"[Saved] {args.save}")

    print("Done.")
    if best_rec >= 0:
        print(f"[Summary] best accuracy = {best_rec*100:.2f}% | best_model: {args.save_best}")

# ========= threshold-based binary evaluation for loop detection =========
@torch.no_grad()
def eval_threshold_metrics(model, test_csv, image_size, device,
                           thr_list=None, pos_rule="same_group",
                           max_pairs=1_000_000, min_index_gap=0):
    """
    使用 cosine 相似度 + 阈值进行二分类评测（回环检测）。
    返回：每个阈值的 (precision, recall, f1, tp, fp, fn)，以及最佳阈值。
    - pos_rule: "same_group" 表示同 class/group 视为正例
    - min_index_gap: 过滤过近的帧索引差（如同序列内临近帧），避免 trivial positives
    """
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    if thr_list is None:
        thr_list = np.linspace(0.2, 0.99, 80)

    rows = read_items(test_csv)  # [(path, "class/group"), ...]
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # 1) 提取 embedding 与标签
    embs, groups, paths = [], [], []
    model.eval()
    for path, gid in rows:
        x = tfm(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
        z = model(x).cpu().numpy()[0]
        embs.append(z); groups.append(gid); paths.append(path)
    embs = np.stack(embs, 0)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    
    #print 3 embeddings
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=5, suppress=True)
    sample_ids = random.sample(range(len(embs)), k=min(3, len(embs)))
    print("\n=== Random 3 Embeddings ===")
    for i in sample_ids:
        print(f"[{i}] group={groups[i]}  path={paths[i]}")
        print(embs[i])  # 全量打印 embedding
        print("-" * 80)

    N = len(embs)
    sims = embs @ embs.T
    np.fill_diagonal(sims, -2.0)  # 排除自比对

    # 2) 构建标签矩阵（正例/负例）
    groups = np.array(groups)
    if pos_rule == "same_group":
        y_true = (groups[:, None] == groups[None, :])
    else:
        raise ValueError("unknown pos_rule")
    np.fill_diagonal(y_true, False)

    # 3) 可选：过滤过近的索引对（避免临近帧当正例）
    if min_index_gap > 0:
        idx = np.arange(N)
        too_close = (np.abs(idx[:, None] - idx[None, :]) < min_index_gap)
        y_true = np.where(too_close, False, y_true)

    # 4) 只取上三角以减少重复
    iu = np.triu_indices(N, k=1)
    s = sims[iu]; y = y_true[iu]

    # 5) 采样（大数据集防爆内存）
    if s.shape[0] > max_pairs:
        sel = np.random.choice(s.shape[0], size=max_pairs, replace=False)
        s = s[sel]; y = y[sel]

    # 6) 扫阈值并计算 P/R/F1
    out = []
    best = (-1.0, None)  # (best_f1, best_thr)
    for thr in thr_list:
        y_pred = (s >= thr)
        TP = int(np.sum(y_pred & y))
        FP = int(np.sum(y_pred & ~y))
        FN = int(np.sum((~y_pred) & y))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        out.append({"thr": float(thr), "precision": float(prec),
                    "recall": float(rec), "f1": float(f1),
                    "tp": TP, "fp": FP, "fn": FN})
        if f1 > best[0]:
            best = (f1, thr)

    return {
        "curve": out,
        "best": {"thr": float(best[1]), "f1": float(best[0])}
    }

def choose_threshold_by_target_recall(curve, target_recall=0.90):
    """
    在达到 target_recall 的所有点里，选择 precision 最大（或 f1 最大）。
    """
    cands = []#[r for r in curve if r["recall"] >= target_recall]
    if not cands:
        # 回退到全局 F1 最大
        best = max(curve, key=lambda r: r["f1"])
        return {"thr": best["thr"], "precision": best["precision"],
                "recall": best["recall"], "f1": best["f1"], "by": "max_f1"}
    best = max(cands, key=lambda r: (r["precision"], r["f1"]))
    return {"thr": best["thr"], "precision": best["precision"],
            "recall": best["recall"], "f1": best["f1"], "by": f"target_recall≥{target_recall}"}


if __name__ == "__main__":
    main()
