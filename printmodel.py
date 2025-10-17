import timm, torch, torch.nn as nn

model_name = "repvit_m1_0"
m = timm.create_model(model_name, pretrained=True)
m.eval()

# 尝试找到各个 stage（timm 的 RepViT 一般有 m.stem / m.stages / m.head）
stages = None
if hasattr(m, "stages"):
    stages = m.stages
else:
    # 兜底：从 children 里抓名字里带 "stage" 的
    stages = [mod for name, mod in m.named_children() if "stage" in name.lower()]

print("== RepViT structure ==")
print(type(m))
print("Has stem:", hasattr(m, "stem"))
print("Num stages:", len(stages))
depths = []
for i, s in enumerate(stages):
    # stage 里通常是若干 RepViT block 的 Sequential
    n_blocks = sum(1 for _ in s.modules() if _.__class__.__name__.lower().endswith("block"))
    if n_blocks == 0 and isinstance(s, nn.Sequential):
        n_blocks = len(s)
    depths.append(n_blocks)
    print(f"  Stage {i}: blocks ≈ {n_blocks}, params = {sum(p.numel() for p in s.parameters())/1e6:.2f} M")
print("Depths by stage:", depths, "| total blocks ≈", sum(depths))
print("Head params (classifier/BN/Pooling等):", 
      sum(p.numel() for p in getattr(m, "head", m).parameters())/1e6, "M")

# 统计当前可训练参数
def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params (initial):", trainable_params(m)/1e6, "M")

# 一个按“stage”为单位冻结的工具函数
def freeze_by_stage(model, freeze_stem=True, freeze_upto_stage=0):
    # 先全部解冻
    for p in model.parameters():
        p.requires_grad = True
    # 冻结 stem
    if freeze_stem and hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = False
    # 冻结前 N 个 stage（含该编号）
    for i in range(min(freeze_upto_stage+1, len(stages))):
        for p in stages[i].parameters():
            p.requires_grad = False
    # 一般保留 head 可训练
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

# 示例：冻结 stem + Stage0（最浅层），训练后面所有层
freeze_by_stage(m, freeze_stem=True, freeze_upto_stage=0)
print("Trainable params (after freeze stem+stage0):", trainable_params(m)/1e6, "M")
