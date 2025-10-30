# export_onnx.py
import torch
import timm
from train_from_existing_repvit import RepViTWithHead  # 你自己的定义
device = "cpu"

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

#model = RepViTWithLogitHead(backbone, embed_dim=args.embed_dim, mlp=args.mlp_head).to(device)
#model = LogitsAsEmbedding(backbone, l2norm=True).to(device);
model = RepViTWithHead(backbone, embed_dim=512, mlp=True).to(device)


ckpt = torch.load("runs/repvit_embed_512_withHead.pt", map_location=device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

dummy = torch.randn(1,3,224,224, device=device)
torch.onnx.export(
    model, dummy, "repvit512.onnx",
    input_names=["input"], output_names=["emb"], opset_version=18,
    do_constant_folding=True, dynamic_axes=None  # 固定尺寸
)
print("saved repvit512.onnx")


