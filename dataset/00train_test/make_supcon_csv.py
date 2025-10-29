# make_supcon_csv.py
import os, csv, argparse, random
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def scan_leaf_dirs(roots):
    """
    扫描所有叶子目录（直接包含图像文件的目录）。
    假设你的结构是：./<class>/<group>/image.png
    例如：./corridor/000/*.png, ./home/001/*.jpg
    """
    leaf_dirs = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for sub in r.iterdir():
            if sub.is_dir():
                # 认为 sub 是 group 目录（里面放该组的所有图片）
                # 只要里面有图像文件，就纳入
                has_img = any(is_image(x) for x in sub.iterdir() if x.is_file())
                if has_img:
                    leaf_dirs.append(sub)
    return leaf_dirs

def write_csv(rows, out_csv, base_dir):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','class','group'])
        for p, cls, grp in rows:
            # 写相对路径，基于 CSV 所在目录；与你现有代码解析逻辑相吻合
            rel = os.path.relpath(p, start=out_csv.parent)
            w.writerow([rel, cls, grp])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='*', default=[
        './corridor','./home','./market','./office','./restaurant','./station'
    ], help='顶层类别目录列表，每个目录下的子目录是一个组（group）')
    ap.add_argument('--out_dir', type=str, default='dataset/00train_test', help='CSV 输出目录')
    ap.add_argument('--train_ratio', type=float, default=0.8, help='每个组内按比例划分 train/test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--by_group', action='store_true',
                    help='若设定，则按“组”为单位划分（整个组进 train 或 test），默认在组内按图片划分')
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) 找到所有叶子组目录（./<class>/<group>/）
    leaf_dirs = scan_leaf_dirs(args.roots)
    if not leaf_dirs:
        print("No leaf dirs with images found. Check --roots.")
        return

    train_rows, test_rows = [], []

    # 2) 逐组处理
    for gdir in sorted(leaf_dirs):
        cls = gdir.parent.name    # 顶层目录名作为 class
        grp = gdir.name           # 组目录名作为 group
        imgs = [p for p in sorted(gdir.iterdir()) if is_image(p)]
        if not imgs:
            continue

        if args.by_group:
            # 整个组随机进 train 或 test（组间划分，避免“信息泄漏”最严格）
            bucket = train_rows if random.random() < args.train_ratio else test_rows
            for p in imgs:
                bucket.append((str(p), cls, grp))
        else:
            # 组内按图片划分（更均衡；常用于每个组图片较多时）
            n = len(imgs)
            k = max(1, int(round(n * args.train_ratio)))
            idx = list(range(n))
            random.shuffle(idx)
            train_idx = set(idx[:k])
            for i, p in enumerate(imgs):
                (train_rows if i in train_idx else test_rows).append((str(p), cls, grp))

    # 3) 写 CSV（列：path,class,group）
    out_dir = Path(args.out_dir)
    write_csv(train_rows, out_dir/'train_supcon.csv', out_dir)
    write_csv(test_rows,  out_dir/'test_supcon.csv',  out_dir)

    # 4) 简要统计
    print(f"[Done] train.csv: {len(train_rows)} rows | test.csv: {len(test_rows)} rows | out_dir={out_dir}")

if __name__ == '__main__':
    main()
