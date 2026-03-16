"""
读取保存的 .pth 模型，打印层结构与超参，便于构建推理器和复现训练。
用法: python read_pth.py <path_to.pth>
"""
import os
import sys
import argparse
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def parse_config_txt(config_path: str) -> dict:
    """解析 config.txt。"""
    config = {}
    if not os.path.isfile(config_path):
        return config
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            idx = line.find(":")
            if idx < 0:
                continue
            key = line[:idx].strip()
            val_str = line[idx + 1 :].strip()
            if not val_str:
                continue
            try:
                config[key] = __import__("ast").literal_eval(val_str)
            except (ValueError, SyntaxError):
                config[key] = val_str
    return config


def infer_architecture(state_dict: dict) -> dict:
    """从 state_dict 推断架构：层数、通道、in_channels。"""
    keys = list(state_dict.keys())
    info = {
        "in_channels": 3,
        "encoder_levels": [],
        "decoder_levels": [],
        "has_down3": "down3" in str(keys),
        "has_down4": "down4" in str(keys),
        "has_enc5": "enc5" in str(keys),
    }
    # 从 conv_in 推断 in_channels 和 c1
    for k in keys:
        if k == "conv_in.0.weight" or (k.startswith("conv_in.") and "weight" in k):
            t = state_dict[k]
            if t.dim() == 4:
                info["in_channels"] = int(t.shape[1])
                info["c1"] = int(t.shape[0])
            break
    # 从各层 weight 推断通道
    ch_map = {}
    for k in keys:
        if ".weight" not in k:
            continue
        t = state_dict[k]
        if t.dim() == 4:  # Conv2d
            out_ch = int(t.shape[0])
            in_ch = int(t.shape[1])
            for prefix in ("conv_in", "down1", "down2", "down3", "down4",
                          "enc1", "enc2", "enc3", "enc4", "enc5",
                          "up1", "up2", "up3", "up4",
                          "fuse1", "fuse2", "fuse3", "fuse4",
                          "conv_out"):
                if k.startswith(prefix + "."):
                    ch_map[prefix] = out_ch
                    break
    info["ch_map"] = ch_map
    info.setdefault("c1", ch_map.get("conv_in", 80))
    # 推断 encoder/decoder 通道序列
    if info["has_enc5"] and ("enc5" in ch_map or "down4" in ch_map):
        info["encoder_levels"] = [
            ch_map.get("enc1", ch_map.get("conv_in", 0)),
            ch_map.get("enc2", 0),
            ch_map.get("enc3", 0),
            ch_map.get("enc4", 0),
            ch_map.get("enc5", 0),
        ]
        info["decoder_levels"] = [
            ch_map.get("enc5", 0),
            ch_map.get("fuse1", 0),
            ch_map.get("fuse2", 0),
            ch_map.get("fuse3", 0),
            ch_map.get("fuse4", 0),
            ch_map.get("dec", ch_map.get("conv_out", 0)),
        ]
    else:
        info["encoder_levels"] = [
            ch_map.get("enc1", ch_map.get("c1", 0)),
            ch_map.get("enc2", 0),
            ch_map.get("enc3", 0),
        ]
        info["decoder_levels"] = [
            ch_map.get("enc3", 0),
            ch_map.get("fuse1", 0),
            ch_map.get("fuse2", 0),
            ch_map.get("dec", ch_map.get("conv_out", 0)),
        ]
    return info


def group_state_dict(state_dict: dict) -> OrderedDict:
    """按模块前缀分组。"""
    groups = OrderedDict()
    for k in sorted(state_dict.keys()):
        prefix = k.split(".")[0] if "." in k else k
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((k, state_dict[k].shape))
    return groups


def count_params(state_dict: dict) -> int:
    """参数量。"""
    return sum(p.numel() for p in state_dict.values())


def main():
    parser = argparse.ArgumentParser(description="读取 .pth 模型，打印层结构")
    parser.add_argument("pth", type=str, help=".pth 文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印每层 shape")
    args = parser.parse_args()

    pth_path = os.path.abspath(args.pth)
    if not os.path.isfile(pth_path):
        print(f"错误: 文件不存在 {pth_path}")
        sys.exit(1)

    try:
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(pth_path, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        print("错误: 无法解析 model_state")
        sys.exit(1)

    ckpt_dir = os.path.dirname(pth_path)
    config_path = os.path.join(ckpt_dir, "config.txt")
    config = parse_config_txt(config_path)

    # 元信息
    print("=" * 60)
    print(f"Checkpoint: {pth_path}")
    print("=" * 60)
    if isinstance(ckpt, dict):
        for k in ("stage", "epoch", "save_dir", "best_val_loss", "best_val_psnr", "val_psnr", "val_ssim"):
            if k in ckpt and ckpt[k] is not None:
                print(f"  {k}: {ckpt[k]}")

    n_params = count_params(state_dict)
    print(f"  参数量: {n_params / 1e6:.2f} M")
    if config:
        print("\n--- config.txt ---")
        for k, v in config.items():
            print(f"  {k}: {v}")

    # 推断架构
    arch = infer_architecture(state_dict)
    print("\n--- 推断的模型架构 ---")
    print(f"  in_channels: {arch['in_channels']}")
    print(f"  Encoder 通道: {arch['encoder_levels']}")
    print(f"  Decoder 通道: {arch['decoder_levels']}")
    if arch.get("ch_map"):
        ch_str = " ".join(f"{k}={v}" for k, v in sorted(arch["ch_map"].items()))
        print(f"  各层通道: {ch_str}")

    # 按模块分组打印
    groups = group_state_dict(state_dict)
    print("\n--- 层结构 (按模块) ---")
    for prefix, items in groups.items():
        n = len(items)
        total = sum(s.numel() for _, s in items)
        print(f"  {prefix}: {n} 个参数张量, 共 {total / 1e3:.1f} K")
        if args.verbose:
            for k, shape in items:
                print(f"      {k}: {tuple(shape)}")

    # 推理/复现建议
    img_size = config.get("img_size", 224)
    base_ch = config.get("model_base_ch", 80)
    print("\n--- 推理器构建建议 ---")
    if arch["has_enc5"]:
        print("  # 5 层结构 (64->128->256->384->512)")
        print("  from CNN.train_all_CNN import CNNImageRegressor  # 或 train_patch_0308 内联版")
        print(f"  model = CNNImageRegressor(img_size={img_size}, base_ch={base_ch}, in_channels={arch['in_channels']})")
    else:
        print("  # 3 层结构 (旧版 80->160->320)")
        print("  from CNN.train_all_CNN import CNNImageRegressor")
        print(f"  model = CNNImageRegressor(img_size={img_size}, base_ch={base_ch})")
    print(f"  ckpt = torch.load({repr(pth_path)}, map_location='cpu')")
    print('  model.load_state_dict(ckpt["model_state"], strict=False)')
    print("  model.eval()")
    print("=" * 60)


if __name__ == "__main__":
    main()
