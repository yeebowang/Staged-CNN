"""
NTIRE 提交脚本：CNN + 三频分解联合推理。
输入整图送 CNN（low），中频直接使用输入 mid，高频使用输入 high；
合成 low_ratio*CNN + mid_ratio*mid + high_ratio*high。
适配 train_all_CNN.py：config.txt、GPU、path_remap。
"""
import os
import re
import sys
import time
import argparse
import zipfile
import ast
import glob
import shutil
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

import cv2
import torch.nn as nn


class ResBlock(nn.Module):
    """残差块：Conv3x3->BN->ReLU->Conv3x3->BN + skip，可选 dilation。"""

    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        p = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=p, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=p, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return torch.nn.functional.relu(out + x, inplace=True)


class CNNImageRegressor(nn.Module):
    """
    Encoder-Decoder + Skip (UNet 风格)。
    Conv80 -> 12Res(80) -[skip]-> Down -> 12Res(160) -[skip]-> Down -> 12Res(320)
      -> Up+skip -> Up+skip -> 8Res(80) -> Conv -> RGB
    """

    def __init__(self, img_size: int = 224, base_ch: int = 80, lab_color: bool = False, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.lab_color = lab_color
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4  # 80, 160, 320
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(*[ResBlock(ch1, dilation=1) for _ in range(12)])
        self.down1 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(*[ResBlock(ch2, dilation=1) for _ in range(12)])
        self.down2 = nn.Sequential(
            nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(*[ResBlock(ch3, dilation=2) for _ in range(12)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch3, ch2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(ch2 * 2, ch2, kernel_size=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(ch1 * 2, ch1, kernel_size=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(*[ResBlock(ch1, dilation=1) for _ in range(8)])
        self.conv_out = nn.Conv2d(ch1, 3, kernel_size=3, padding=1)
        self._enc3_stage = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        e1 = self.enc1(h)
        h = self.down1(e1)
        e2 = self.enc2(h)
        h = self.down2(e2)
        h = self.enc3(h)
        h = self.up1(h)
        h = self.fuse1(torch.cat([h, e2], dim=1))
        h = self.up2(h)
        h = self.fuse2(torch.cat([h, e1], dim=1))
        h = self.dec(h)
        return self.conv_out(h)


def _infer_enc3_block_count(state_dict: dict) -> int:
    """
    从 state_dict 推断 enc3 的 block 数量（train_patch：stage1=12, stage2=16, stage3=20）。
    键格式为 enc3.0.conv1.weight 等，取 enc3 后数字的最大值 +1。
    """
    max_idx = -1
    for key in state_dict:
        if "enc3." not in key:
            continue
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "enc3" and i + 1 < len(parts) and parts[i + 1].isdigit():
                max_idx = max(max_idx, int(parts[i + 1]))
                break
    return max_idx + 1 if max_idx >= 0 else 12


def expand_enc3_for_stage(model: nn.Module, stage_idx: int, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Stage 1: enc3=12（默认）。Stage 2: +4 blocks (dilation 2,2,4,4) → 16。Stage 3: +4 blocks (dilation 8,8,4,2) → 20。
    与 train_patch / CNN.train_all_CNN 的 expand_enc3_for_stage 一致，推理时 optimizer 传 None。
    """
    m = getattr(model, "module", model)
    if not isinstance(m, CNNImageRegressor) or not hasattr(m, "enc3"):
        return
    old_blocks = list(m.enc3)
    ch3 = old_blocks[0].conv1.in_channels
    if stage_idx == 0:
        return
    if stage_idx == 1:
        if len(old_blocks) != 12:
            return
        new_blocks = [ResBlock(ch3, d) for d in (2, 2, 4, 4)]
        m.enc3 = nn.Sequential(*(old_blocks + new_blocks))
        m._enc3_stage = 2
    elif stage_idx == 2:
        if len(old_blocks) != 16:
            return
        new_blocks = [ResBlock(ch3, d) for d in (8, 8, 4, 2)]
        m.enc3 = nn.Sequential(*(old_blocks + new_blocks))
        m._enc3_stage = 3
    else:
        return
    dev = next(m.parameters()).device
    m.enc3 = m.enc3.to(dev)
    if optimizer is not None:
        new_params = [p for b in new_blocks for p in b.parameters()]
        if new_params:
            lr = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": new_params, "lr": lr})


def _expand_enc3_to_match_checkpoint(model: CNNImageRegressor, state_dict: dict, device: torch.device) -> None:
    """根据 checkpoint 的 enc3 块数扩展模型（12→16→20），再 load_state_dict 时才能匹配。"""
    n_blocks = _infer_enc3_block_count(state_dict)
    if n_blocks >= 20:
        expand_enc3_for_stage(model, 1, None)
        expand_enc3_for_stage(model, 2, None)
        model.to(device)
        print(f"  enc3 已扩展至 20 块以匹配 checkpoint")
    elif n_blocks >= 16:
        expand_enc3_for_stage(model, 1, None)
        model.to(device)
        print(f"  enc3 已扩展至 16 块以匹配 checkpoint")


# 三频分解（与 freq_deco 一致）
def _gaussian_blur_float(img: np.ndarray, sigma: float) -> np.ndarray:
    k = int(6 * sigma + 1) | 1
    k = max(3, min(k, 51))
    blurred = cv2.GaussianBlur(img, (k, k), sigma)
    return blurred.astype(np.float32)


try:
    from freq_deco import decompose_freq_log
except ImportError:
    def decompose_freq_log(
        img_log: np.ndarray, sigma_low: float, sigma_mid: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        low_log = _gaussian_blur_float(img_log, sigma_low)
        mid_blur = _gaussian_blur_float(img_log, sigma_mid)
        mid_log = (mid_blur - low_log).astype(np.float32)
        high_log = (img_log.astype(np.float32) - mid_blur).astype(np.float32)
        return low_log, mid_log, high_log

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _img_size_hw(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """将 img_size（int 或 (h,w)）转为 (height, width) 二元组。"""
    return (img_size, img_size) if isinstance(img_size, int) else (img_size[0], img_size[1])


def _parse_wh(val, allow_negative: bool = False) -> Optional[Union[int, Tuple[int, int]]]:
    """
    解析尺寸/步长：'N' -> 整数 N（正方形），'W,H' -> (宽, 高)。
    返回 int 或 (width, height)。allow_negative 时允许 -1（仅 stride）。
    """
    if val is None:
        return None
    s = str(val).strip()
    if "," in s:
        parts = [x.strip() for x in s.split(",", 1) if x.strip()]
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    try:
        n = int(s)
        if not allow_negative and n < 0:
            return None
        return n
    except ValueError:
        return None


def _remap_path(path: str, path_remap: Tuple[str, str] = None) -> str:
    """将路径中的盘符/前缀替换，如 F: -> I:，用于跨盘迁移。"""
    if path_remap and len(path_remap) >= 2 and path_remap[0]:
        return path.replace(path_remap[0], path_remap[1])
    return path


def parse_config_txt(config_path: str) -> dict:
    """解析 train_all_CNN.py 保存的 config.txt。"""
    config = {}
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
                config[key] = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                config[key] = val_str
    return config


# train_patch 非 --uu 模式三 stage 对应 patch 尺寸（与 train_patch.STAGES 一致）
STAGE_PATCH_SIZES = (128, 192, 256)


def _find_model_in_dir(ckpt_dir: str) -> Optional[str]:
    """在 checkpoint 目录中查找模型文件。优先级：best_loss_stage_three > best_loss_epoch_* > best_model > latest。"""
    for name in ("best_loss_stage_three.pth", "best_loss_stage_two.pth", "best_loss_stage_one.pth"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    epoch_files = sorted(glob.glob(os.path.join(ckpt_dir, "best_loss_epoch_*.pth")))
    if epoch_files:
        return max(epoch_files, key=os.path.getmtime)
    for name in ("best_model.pth", "latest.pth"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _find_three_stage_paths(ckpt_dir: str) -> Optional[Dict[int, str]]:
    """
    若目录内同时存在 train_patch 非--uu 三 stage 的 pth，返回 {128: path1, 192: path2, 256: path3}；
    否则返回 None。
    """
    names = ("best_loss_stage_one.pth", "best_loss_stage_two.pth", "best_loss_stage_three.pth")
    sizes = STAGE_PATCH_SIZES
    out = {}
    for sz, name in zip(sizes, names):
        p = os.path.join(ckpt_dir, name)
        if not os.path.isfile(p):
            return None
        out[sz] = p
    return out


def resolve_checkpoint_dir(checkpoint_arg: str, project_root: str) -> Tuple[str, str, Optional[Dict[int, str]]]:
    """
    解析 --cnn_checkpoint：目录、.pth 路径，或 'best'/'latest'。
    返回 (checkpoint_dir, best_model_path, stage_paths)。
    stage_paths: 若目录内同时存在三 stage 的 pth（train_patch 非--uu）则为 {128: path, 192: path, 256: path}，否则 None。
    """
    def _resolve_dir(ckpt_dir: str) -> Tuple[str, str, Optional[Dict[int, str]]]:
        stage_paths = _find_three_stage_paths(ckpt_dir)
        best_path = _find_model_in_dir(ckpt_dir)
        if not best_path:
            return ckpt_dir, "", stage_paths
        return ckpt_dir, best_path, stage_paths

    if checkpoint_arg.lower() in ("best", "latest"):
        ckpt_base = os.path.join(project_root, "checkpoints")
        if not os.path.isdir(ckpt_base):
            raise FileNotFoundError(f"未找到 checkpoints 目录: {ckpt_base}")
        candidates = []
        for d in sorted(os.listdir(ckpt_base)):
            cand_dir = os.path.join(ckpt_base, d)
            if not os.path.isdir(cand_dir):
                continue
            p = _find_model_in_dir(cand_dir)
            if p:
                candidates.append((cand_dir, p))
        if not candidates:
            raise FileNotFoundError(f"在 {ckpt_base} 下未找到任何模型 .pth")
        checkpoint_dir, best_path = max(candidates, key=lambda x: os.path.getmtime(x[1]))
        stage_paths = _find_three_stage_paths(checkpoint_dir)
        return checkpoint_dir, best_path, stage_paths

    if os.path.isfile(checkpoint_arg):
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_arg))
        stage_paths = _find_three_stage_paths(ckpt_dir)
        return ckpt_dir, checkpoint_arg, stage_paths

    if os.path.isdir(checkpoint_arg):
        best_path = _find_model_in_dir(checkpoint_arg)
        if not best_path:
            raise FileNotFoundError(
                f"checkpoint 目录 {checkpoint_arg} 下未找到模型 .pth"
            )
        stage_paths = _find_three_stage_paths(checkpoint_arg)
        return os.path.abspath(checkpoint_arg), best_path, stage_paths

    raise FileNotFoundError(f"无效的 checkpoint 路径: {checkpoint_arg}")


def sigma_from_image_freq(
    h: int,
    w: int,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    按频数均分：在 log 频率上把 [f_min, f_max] 三等分，由边界频率反推 sigma。
    每张图按自身尺寸单独计算。
    """
    f_min = 1.0 / max(h, w)
    f_max = 0.5
    log_f_min = np.log(f_min)
    log_f_max = np.log(f_max)
    delta = log_f_max - log_f_min
    f1 = np.exp(log_f_min + delta / 3.0)
    f2 = np.exp(log_f_min + 2.0 * delta / 3.0)
    sigma_mid = 1.0 / (2.0 * np.pi * f2)
    sigma_low = 1.0 / (2.0 * np.pi * f1)
    if sigma_low <= sigma_mid:
        sigma_low = sigma_mid + 1.0
    if verbose:
        print(f"  {h}x{w} sigma_low={sigma_low:.6f} sigma_mid={sigma_mid:.6f}")
    return float(sigma_low), float(sigma_mid)


def _progress_bar(current: int, total: int, prefix: str = "", width: int = 24) -> str:
    if total <= 0:
        return f"{prefix}[{'?' * width}] 0/0"
    n = int(width * current / total) if current < total else width
    bar = ">" * n + " " * (width - n)
    return f"{prefix}[{bar}] {current}/{total}"


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def is_supported_input_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext == ".npy" or is_image_file(name)


def _gt_id_from_npy_name(name: str) -> str:
    """从 .npy 文件名得到 GT 标识（每个 GT 只保存一张 vis）。如 xxx_R0_C0.npy -> xxx。"""
    if not name.lower().endswith(".npy"):
        return name
    base = os.path.splitext(name)[0]
    return re.sub(r"_R\d+_C\d+$", "", base, flags=re.I) if re.search(r"_R\d+_C\d+$", base, re.I) else base


def load_input_as_rgb(path: str, *, npy_is_bgr: bool = True) -> Tuple[Image.Image, np.ndarray]:
    """
    读取输入为 RGB。
    - 普通图片：按 RGB 读取
    - .npy：允许 BGR（默认），支持 [0,255] 或 [0,1]，输出 RGB
    返回 (PIL_RGB, rgb_np_float01 HxWx3)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3 and a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        if a.ndim != 3 or a.shape[-1] != 3:
            raise RuntimeError(f"不支持的 npy 形状: {a.shape}，期望 HxWx3 或 HxW。path={path}")
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        a = np.clip(a, 0.0, 1.0)
        if npy_is_bgr:
            a = a[:, :, ::-1]  # BGR -> RGB
        pil = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
        return pil, a

    pil = Image.open(path).convert("RGB")
    a = np.array(pil).astype(np.float32) / 255.0
    return pil, a


def load_cnn_from_checkpoint(
    checkpoint_dir: str,
    best_model_path: str,
    device: torch.device,
) -> Tuple[CNNImageRegressor, Union[int, Tuple[int, int]]]:
    """从 checkpoint 目录加载 CNN；返回 (model, img_size)。img_size 可为 int 或 (H,W)，优先从 checkpoint 读，其次 config，默认 256。"""
    img_size: Union[int, Tuple[int, int]] = 256
    base_ch = 80
    lab_color = False
    state = torch.load(best_model_path, map_location=device)
    if isinstance(state, dict) and "img_size" in state:
        raw = state["img_size"]
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            img_size = (int(raw[0]), int(raw[1]))
        else:
            img_size = int(raw)
    config_path = os.path.join(checkpoint_dir, "config.txt")
    if os.path.isfile(config_path):
        config = parse_config_txt(config_path)
        if not (isinstance(state, dict) and "img_size" in state):
            cfg = config.get("img_size") or config.get("model_img_size")
            if cfg is not None:
                if isinstance(cfg, (list, tuple)) and len(cfg) >= 2:
                    img_size = (int(cfg[0]), int(cfg[1]))
                else:
                    s = str(cfg).strip().replace(",", " ")
                    parts = s.split()
                    if len(parts) >= 2:
                        img_size = (int(parts[1]), int(parts[0]))  # config 习惯 宽 高 → (H, W)
                    else:
                        img_size = int(parts[0]) if parts else img_size
        base_ch = int(config.get("model_base_ch", 80))
        lab_color = bool(config.get("lab_color", False))

    model = CNNImageRegressor(img_size=img_size, base_ch=base_ch, lab_color=lab_color)
    ckpt = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    if isinstance(ckpt, dict):
        _expand_enc3_to_match_checkpoint(model, ckpt, device)
        if "_gaussian_kernel" in ckpt:
            ckpt = dict(ckpt)
            ckpt["_gaussian_kernel"] = ckpt["_gaussian_kernel"].clone()
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model, img_size


def load_three_stages(
    checkpoint_dir: str,
    stage_paths: Dict[int, str],
    device: torch.device,
) -> Tuple[Dict[int, CNNImageRegressor], int]:
    """
    加载 train_patch 非--uu 三 stage 的 pth，返回 (models_by_patch_size, img_size)。
    每个 stage 的 enc3 块数不同（12/16/20），会根据 checkpoint 自动扩展后再加载。
    """
    models = {}
    for patch_size in STAGE_PATCH_SIZES:
        path = stage_paths[patch_size]
        model, _ = load_cnn_from_checkpoint(checkpoint_dir, path, device)
        models[patch_size] = model
    return models, 256


def _model_for_patch(
    model_or_dict: object,
    ph: int,
    pw: int,
) -> CNNImageRegressor:
    """
    若 model_or_dict 为 Dict[int, CNNImageRegressor]（三 stage），按 patch 尺寸选最近 stage；
    否则返回 model_or_dict 本身。
    """
    if isinstance(model_or_dict, dict):
        s = max(ph, pw)
        best = min(model_or_dict.keys(), key=lambda k: abs(k - s))
        return model_or_dict[best]
    return model_or_dict


def _extract_boundary_morphology(in_np: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    用形态学开/闭操作获取 IN 中的物体边界。
    采用形态学梯度：dilation - erosion，得到物体边缘。
    in_np: (H,W,3) 或 (H,W)，值域 0~1 或 0~255
    """
    if in_np.ndim == 3:
        gray = cv2.cvtColor(
            (np.clip(in_np, 0, 1) * 255).astype(np.uint8) if in_np.max() <= 1.0 else in_np.astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = (np.clip(in_np, 0, 1) * 255).astype(np.uint8) if in_np.max() <= 1.0 else in_np.astype(np.uint8)
    k = kernel_size | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # 形态学梯度 = dilation - erosion，得到边界
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return gradient


def _apply_boundary_to_high(
    high_in: np.ndarray,
    img_np: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    """
    以 boundary 图作为高频过滤器：log 域下 log_high = boundary_weight * log_boundary + log_high，
    再参与最后的高频叠加。img_np 需与 high_in 同尺寸（0~1 float）。
    """
    boundary = _extract_boundary_morphology(img_np)
    boundary_01 = (boundary.astype(np.float32) / 255.0) if boundary.max() > 1.0 else boundary.astype(np.float32)
    boundary_01 = np.clip(boundary_01, 0.0, 1.0)
    log_boundary = np.log1p(boundary_01)
    if high_in.ndim == 3 and log_boundary.ndim == 2:
        log_boundary = np.stack([log_boundary] * 3, axis=-1)
    return boundary_weight * log_boundary + high_in


def _freq_to_vis(low_log: np.ndarray, mid_log: np.ndarray, high_log: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """将三频 log 分量转为可显示的 (arr_uint8, title) 列表。"""
    out = []
    low_linear = np.clip(np.expm1(low_log), 0.0, 1.0)
    out.append(((low_linear * 255).astype(np.uint8), "Low"))
    # mid/high 可正可负，归一化到 [0,1] 显示
    for arr, name in [(mid_log, "Mid"), (high_log, "High")]:
        mn, mx = arr.min(), arr.max()
        r = mx - mn + 1e-8
        norm = ((arr - mn) / r).astype(np.float32)
        if norm.ndim == 2:
            norm = np.stack([norm, norm, norm], axis=-1)
        out.append(((np.clip(norm, 0, 1) * 255).astype(np.uint8), name))
    recon = np.clip(np.expm1(low_log + mid_log + high_log), 0.0, 1.0)
    out.append(((recon * 255).astype(np.uint8), "Recon"))
    return out


def save_vis_components(
    input_pil: Image.Image,
    pred_pil: Image.Image,
    cnn_low: np.ndarray,
    out_path: str,
    patch_merged: Optional[np.ndarray] = None,
    patch_merged_by_size: Optional[List[Tuple[str, np.ndarray]]] = None,
    padded_input: Optional[np.ndarray] = None,
    vis_boundary: bool = False,
    vis_freq_decompose: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    vis_output_no_high: Optional[np.ndarray] = None,
    vis_lock_l: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """
    可视化：Input、Output 两图并排。
    若 patch_merged 非空：显示单块 Patch 子图。
    若 patch_merged_by_size 非空（demo 多尺寸）：每种尺寸一个子图。
    若 padded_input 非空（pad_flip 时）：显示带填充的输入。
    若 vis_boundary：插入 Boundary。
    若 vis_freq_decompose 非空（三频模式）：插入 Low、Mid、High、Recon（分解后合成=原图）。
    若 vis_output_no_high 非空：在 Output 前插入「未加 high」的合成结果。
    若 vis_lock_l 非空（lock_l 模式）：(Input L 灰度图, Pred AB 伪彩, LockL 合成结果) 三张子图。
    """
    if plt is None:
        return
    in_np = np.array(input_pil)
    out_np = np.array(pred_pil)
    arrs = [in_np]
    titles = ["Input"]
    if vis_freq_decompose is not None:
        low_log, mid_log, high_log = vis_freq_decompose
        for arr_u8, title in _freq_to_vis(low_log, mid_log, high_log):
            if arr_u8.shape[:2] != in_np.shape[:2]:
                arr_u8 = np.array(Image.fromarray(arr_u8).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC))
            arrs.append(arr_u8)
            titles.append(title)
    if padded_input is not None and padded_input.size > 0:
        pad_vis = (np.clip(padded_input, 0, 1) * 255).astype(np.uint8) if padded_input.max() <= 1.0 else np.clip(padded_input, 0, 255).astype(np.uint8)
        if pad_vis.shape[:2] != in_np.shape[:2]:
            pad_pil = Image.fromarray(pad_vis).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC)
            pad_vis = np.array(pad_pil)
        arrs.append(pad_vis)
        titles.append("Input+Pad")
    if vis_boundary:
        boundary = _extract_boundary_morphology(in_np.astype(np.float32) / 255.0)
        if boundary.shape[:2] != in_np.shape[:2]:
            boundary = cv2.resize(boundary, (in_np.shape[1], in_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        boundary_vis = np.stack([boundary, boundary, boundary], axis=-1)
        arrs.append(boundary_vis.astype(np.uint8))
        titles.append("Boundary")
    if patch_merged_by_size:
        for label, pm in patch_merged_by_size:
            if pm is None or pm.size == 0:
                continue
            if pm.max() <= 1.0:
                patch_vis = (np.clip(pm, 0, 1) * 255).astype(np.uint8)
            else:
                patch_vis = np.clip(pm, 0, 255).astype(np.uint8)
            if patch_vis.shape[:2] != in_np.shape[:2]:
                patch_pil = Image.fromarray(patch_vis).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC)
                patch_vis = np.array(patch_pil)
            arrs.append(patch_vis)
            titles.append(f"Patch-{label}")
    elif patch_merged is not None and patch_merged.size > 0:
        if patch_merged.max() <= 1.0:
            patch_vis = (np.clip(patch_merged, 0, 1) * 255).astype(np.uint8)
        else:
            patch_vis = np.clip(patch_merged, 0, 255).astype(np.uint8)
        if patch_vis.shape[:2] != in_np.shape[:2]:
            patch_pil = Image.fromarray(patch_vis).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC)
            patch_vis = np.array(patch_pil)
        arrs.append(patch_vis)
        titles.append("Patch")
    if vis_output_no_high is not None and vis_output_no_high.size > 0:
        no_high = vis_output_no_high
        if no_high.max() <= 1.0:
            no_high_u8 = (np.clip(no_high, 0, 1) * 255).astype(np.uint8)
        else:
            no_high_u8 = np.clip(no_high, 0, 255).astype(np.uint8)
        if no_high_u8.shape[:2] != in_np.shape[:2]:
            no_high_u8 = np.array(Image.fromarray(no_high_u8).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC))
        arrs.append(no_high_u8)
        titles.append("NoHigh")
    if vis_lock_l is not None:
        L_vis, AB_vis, lock_l_u8 = vis_lock_l
        for arr_u8, title in [(L_vis, "Input L"), (AB_vis, "Pred AB"), (lock_l_u8, "LockL")]:
            if arr_u8.shape[:2] != in_np.shape[:2]:
                arr_u8 = np.array(Image.fromarray(arr_u8).resize((in_np.shape[1], in_np.shape[0]), Image.BICUBIC))
            arrs.append(arr_u8)
            titles.append(title)
    arrs.append(out_np)
    titles.append("Output")
    n = len(arrs)
    if n == 4:
        nrow, ncol = 2, 2
    else:
        ncol = min(n, 4)
        nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if n == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_2d(axes).flatten()
    for ax, arr, title in zip(axes, arrs, titles):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    for j in range(len(arrs), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _pad_with_flip(img: np.ndarray, pad_t: int, pad_b: int, pad_l: int, pad_r: int) -> np.ndarray:
    """以原图边缘为轴的对称反转填充；四角以四角为中心的中心对称。"""
    return np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="symmetric")


def _pad_with_band(
    img: np.ndarray, pad_t: int, pad_b: int, pad_l: int, pad_r: int, pad_lines: int
) -> np.ndarray:
    """用边缘 pad_lines 行/列的带状像素填充，非单行/列。"""
    h, w = img.shape[:2]
    nh = min(pad_lines, h)
    nw = min(pad_lines, w)
    out = np.zeros((h + pad_t + pad_b, w + pad_l + pad_r, img.shape[2]), dtype=img.dtype)
    out[pad_t : pad_t + h, pad_l : pad_l + w] = img
    if pad_t > 0:
        band = img[0:nh]
        idx = (pad_t - 1 - np.arange(pad_t)) % nh
        out[0:pad_t, pad_l : pad_l + w] = band[idx]
    if pad_b > 0:
        band = img[-nh:]
        idx = np.arange(pad_b) % nh
        out[pad_t + h :, pad_l : pad_l + w] = band[idx]
    if pad_l > 0:
        band = img[:, 0:nw]
        idx = (pad_l - 1 - np.arange(pad_l)) % nw
        out[pad_t : pad_t + h, 0:pad_l] = band[:, idx]
    if pad_r > 0:
        band = img[:, -nw:]
        idx = np.arange(pad_r) % nw
        out[pad_t : pad_t + h, pad_l + w :] = band[:, idx]
    if pad_t > 0 and pad_l > 0:
        band = img[0:nh, 0:nw]
        iy = (pad_t - 1 - np.arange(pad_t)) % nh
        ix = (pad_l - 1 - np.arange(pad_l)) % nw
        out[0:pad_t, 0:pad_l] = band[np.ix_(iy, ix)]
    if pad_t > 0 and pad_r > 0:
        band = img[0:nh, -nw:]
        iy = (pad_t - 1 - np.arange(pad_t)) % nh
        ix = np.arange(pad_r) % nw
        out[0:pad_t, pad_l + w :] = band[np.ix_(iy, ix)]
    if pad_b > 0 and pad_l > 0:
        band = img[-nh:, 0:nw]
        iy = np.arange(pad_b) % nh
        ix = (pad_l - 1 - np.arange(pad_l)) % nw
        out[pad_t + h :, 0:pad_l] = band[np.ix_(iy, ix)]
    if pad_b > 0 and pad_r > 0:
        band = img[-nh:, -nw:]
        iy = np.arange(pad_b) % nh
        ix = np.arange(pad_r) % nw
        out[pad_t + h :, pad_l + w :] = band[np.ix_(iy, ix)]
    return out


def _compute_padding(orig_h: int, orig_w: int, ph: int, pw: int, box: int) -> int:
    """
    计算 padding，不少于 box。
    保证中心 patch 中心严格对齐 IN 图像中心：需 pad_val 使 top_c = cx - ph/2 和 left_c = cy - pw/2 均在 [0, h-ph] / [0, w-pw] 内。
    """
    # top_c = pad_val + orig_h//2 - ph//2 需满足 0 <= top_c <= orig_h + 2*pad_val - ph
    # => pad_val >= ph//2 - orig_h//2 且 pad_val >= (ph - orig_h - 1)//2（由第二式）
    need_h = max(0, (ph - orig_h + 1) // 2, ph // 2 - orig_h // 2)
    need_w = max(0, (pw - orig_w + 1) // 2, pw // 2 - orig_w // 2)
    return max(box, need_h, need_w)


def _sliding_window_grid_centered(
    h: int, w: int, ph: int, pw: int,
    stride_h: int, stride_w: int,
    pad_val: int, orig_h: int, orig_w: int,
) -> list:
    """
    从图像中心构建 patch 网格，向四周拓展。
    中心 patch 对齐原图中心，优先用 stride 步长；若步长导致 patch 越界则减小该方向步长使之刚好在边界内。
    所有 patch 保证在 [0, h-ph] x [0, w-pw] 内。
    """
    if h < ph or w < pw:
        return [(0, 0)]
    cx = pad_val + orig_h // 2
    cy = pad_val + orig_w // 2
    top_c = cx - ph // 2
    left_c = cy - pw // 2
    lo_h, hi_h = 0, h - ph
    lo_w, hi_w = 0, w - pw
    top_c = max(lo_h, min(hi_h, top_c))
    left_c = max(lo_w, min(hi_w, left_c))

    def _expand_from_center(center: int, step: int, lo: int, hi: int) -> List[int]:
        """从中心向两侧展开，优先 stride；接近边界时减小步长使 patch 刚好在 lo..hi 内。"""
        if step <= 0:
            step = 1
        vals = [center]
        v = center - step
        while v >= lo:
            vals.append(v)
            v -= step
        v = center + step
        while v <= hi:
            vals.append(v)
            v += step
        vals = sorted(set(vals))
        if vals and vals[0] != lo:
            vals = [lo] + vals
        if vals and vals[-1] != hi:
            vals = vals + [hi]
        return vals

    tops = _expand_from_center(top_c, stride_h, lo_h, hi_h)
    lefts = _expand_from_center(left_c, stride_w, lo_w, hi_w)
    positions = [(t, l) for t in tops for l in lefts]
    # 按到中心的距离排序，中心优先
    def _dist(tt: int, ll: int) -> float:
        return (tt + ph / 2 - cx) ** 2 + (ll + pw / 2 - cy) ** 2
    positions.sort(key=lambda x: _dist(x[0], x[1]))
    return positions


def _sliding_window_grid(h: int, w: int, patch_h: int, patch_w: int, stride_h: int, stride_w: int) -> list:
    """生成滑动窗口网格 (top, left) 列表，从左上是起点覆盖整图。保留用于 tile_no_overlap 等场景。"""
    if h < patch_h or w < patch_w:
        return [(0, 0)]
    tops = list(range(0, h - patch_h + 1, stride_h))
    lefts = list(range(0, w - patch_w + 1, stride_w))
    if tops[-1] != h - patch_h:
        tops.append(h - patch_h)
    if lefts[-1] != w - patch_w:
        lefts.append(w - patch_w)
    return [(t, l) for t in tops for l in lefts]


def _grid_3x3_patches_centered(orig_h: int, orig_w: int, center_ratio: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    将图像分为 3x3 的 patch，中心 patch 与原图等比例，面积为原图的 center_ratio；其余可非正方形。
    按中心优先顺序返回 (top, left, ph, pw)。
    """
    r = max(0.01, min(1.0, center_ratio))
    h1 = int(round(orig_h * (r ** 0.5)))
    w1 = int(round(orig_w * (r ** 0.5)))
    h1 = max(1, min(h1, orig_h - 2))
    w1 = max(1, min(w1, orig_w - 2))
    h0 = (orig_h - h1) // 2
    h2 = orig_h - h0 - h1
    w0 = (orig_w - w1) // 2
    w2 = orig_w - w0 - w1
    grid = [
        (0, 0, h0, w0),
        (0, w0, h0, w1),
        (0, w0 + w1, h0, w2),
        (h0, 0, h1, w0),
        (h0, w0, h1, w1),
        (h0, w0 + w1, h1, w2),
        (h0 + h1, 0, h2, w0),
        (h0 + h1, w0, h2, w1),
        (h0 + h1, w0 + w1, h2, w2),
    ]
    center_row, center_col = 1, 1

    def _dist_idx(i: int) -> float:
        r, c = i // 3, i % 3
        return (r - center_row) ** 2 + (c - center_col) ** 2

    order = sorted(range(9), key=_dist_idx)
    return [grid[i] for i in order]


def _grid_3x3_uniform_centered(
    orig_h: int, orig_w: int, center_ratio: float = 0.5,
    box_override: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int, int]], int, int, int, int, int]:
    """
    3x3 网格：周边 patch 保持应有的中心位置，形状与中心 patch 一致 (ph x pw)。
    自适应 box 与 padding。返回 (patches, pad_t, pad_b, pad_l, pad_r, box)，
    patches 为 (top_in_padded, left_in_padded, ph, pw)，expand = ph+2*box。
    """
    r = max(0.01, min(1.0, center_ratio))
    ph = int(round(orig_h * (r ** 0.5)))
    pw = int(round(orig_w * (r ** 0.5)))
    ph = max(1, min(ph, orig_h - 2))
    pw = max(1, min(pw, orig_w - 2))
    h0 = (orig_h - ph) // 2
    h2 = orig_h - h0 - ph
    w0 = (orig_w - pw) // 2
    w2 = orig_w - w0 - pw
    centers = [
        (h0 // 2, w0 // 2),
        (h0 // 2, w0 + pw // 2),
        (h0 // 2, w0 + pw + w2 // 2),
        (h0 + ph // 2, w0 // 2),
        (h0 + ph // 2, w0 + pw // 2),
        (h0 + ph // 2, w0 + pw + w2 // 2),
        (h0 + ph + h2 // 2, w0 // 2),
        (h0 + ph + h2 // 2, w0 + pw // 2),
        (h0 + ph + h2 // 2, w0 + pw + w2 // 2),
    ]
    box = box_override if box_override is not None else min(32, ph // 4, pw // 4)
    box = max(0, box)
    expand_h, expand_w = ph + 2 * box, pw + 2 * box
    tops = [cy - ph // 2 - box for cy, _ in centers]
    lefts = [cx - pw // 2 - box for _, cx in centers]
    min_t = min(tops)
    min_l = min(lefts)
    max_b = max(t + expand_h for t in tops)
    max_r = max(l + expand_w for l in lefts)
    pad_t = max(0, -min_t)
    pad_l = max(0, -min_l)
    pad_b = max(0, max_b - orig_h)
    pad_r = max(0, max_r - orig_w)
    patches = [(t + pad_t, l + pad_l, ph, pw) for t, l in zip(tops, lefts)]
    center_row, center_col = 1, 1

    def _dist_idx(i: int) -> float:
        ri, ci = i // 3, i % 3
        return (ri - center_row) ** 2 + (ci - center_col) ** 2

    order = sorted(range(9), key=_dist_idx)
    return [patches[i] for i in order], pad_t, pad_b, pad_l, pad_r, box


def _dist_to_rect_grid(
    ys: np.ndarray, xs: np.ndarray, t: int, l: int, h: int, w: int
) -> np.ndarray:
    """向量化：点 (ys, xs) 到矩形 [t,t+h) x [l,l+w) 的最短距离，内部为 0。"""
    dy = np.zeros_like(ys, dtype=np.float32)
    dy = np.where(ys < t, t - ys, dy)
    dy = np.where(ys >= t + h, ys - (t + h), dy)
    dx = np.zeros_like(xs, dtype=np.float32)
    dx = np.where(xs < l, l - xs, dx)
    dx = np.where(xs >= l + w, xs - (l + w), dx)
    d = np.sqrt(dy ** 2 + dx ** 2)
    inside = (ys >= t) & (ys < t + h) & (xs >= l) & (xs < l + w)
    d = np.where(inside, 0.0, np.where((dy != 0) & (dx != 0), d, np.maximum(np.abs(dy), np.abs(dx))))
    return d


# 3x3 网格顺序（按距中心距离）: 0=中心, 1=上 2=左 3=右 4=下, 5=左上 6=右上 7=左下 8=右下
_EDGE_INDICES = [1, 2, 3, 4]
_CORNER_INDICES = [5, 6, 7, 8]
_EDGE_ADJACENT_CORNERS = {1: [5, 6], 2: [5, 7], 3: [6, 8], 4: [7, 8]}
_CORNER_ADJACENT_EDGES = {5: [1, 2], 6: [1, 3], 7: [2, 4], 8: [3, 4]}


def _uniform_3x3_merge_weights(
    patches: List[Tuple[int, int, int, int]],
    ph: int, pw: int, box: int,
) -> List[np.ndarray]:
    """
    uniform 模式：仅中心 box 与周围、以及四边与四角的重叠区使用线性权重；
    四边之间、四角之间的重叠区不使用线性权重（权重=1，均匀平均）。
    四边优先于四角，故四边-四角重叠：从 box 边界 0 过渡到四角中心 1。
    中心 patch 的非 box 扩展部分仅由中心决定。
    返回每 patch 的 expand_h x expand_w 权重图列表。
    """
    expand_h, expand_w = ph + 2 * box, pw + 2 * box
    top_c, left_c = patches[0][0], patches[0][1]
    core_t, core_l = top_c + box, left_c + box
    rr = np.arange(expand_h, dtype=np.float32) + 0.5
    cc = np.arange(expand_w, dtype=np.float32) + 0.5

    def _in_patch_expanded(ys: np.ndarray, xs: np.ndarray, j: int) -> np.ndarray:
        tj, lj = patches[j][0], patches[j][1]
        return (ys >= tj) & (ys < tj + expand_h) & (xs >= lj) & (xs < lj + expand_w)

    def _patch_core_rect(j: int):
        tj, lj = patches[j][0], patches[j][1]
        return tj + box, lj + box, ph, pw

    weights = []
    for i, (top, left, _, _) in enumerate(patches):
        ys = top + rr[:, None]
        xs = left + cc[None, :]
        d_to_core = _dist_to_rect_grid(ys, xs, core_t, core_l, ph, pw)
        pc_cy = top + box + ph / 2 - 0.5
        pc_cx = left + box + pw / 2 - 0.5
        d_pc_to_core = _dist_to_rect_grid(
            np.array([pc_cy]), np.array([pc_cx]), core_t, core_l, ph, pw
        )[0]
        if d_pc_to_core < 1e-6:
            d_pc_to_core = 1e-6
        if i == 0:
            w = np.where(d_to_core <= 0, 1.0, np.maximum(0.0, 1.0 - d_to_core / (box + 1e-6)))
        else:
            in_center_expanded = _in_patch_expanded(ys, xs, 0)
            # 中心-周围：线性权重
            linear_center = np.where(d_to_core <= 0, 0.0, np.minimum(1.0, d_to_core / d_pc_to_core))

            if i in _EDGE_INDICES:
                # 四边：与中心重叠用线性；与四角重叠用线性（四边优先，1->0 向四角中心）；其余用 1
                in_edge_corner = np.zeros_like(in_center_expanded, dtype=bool)
                for cj in _EDGE_ADJACENT_CORNERS.get(i, []):
                    in_edge_corner = in_edge_corner | _in_patch_expanded(ys, xs, cj)
                in_edge_corner = in_edge_corner & (~in_center_expanded)
                # 四边在 edge-corner 重叠：从 1（靠四边）到 0（靠四角中心），用 d_to_corner_center
                d_to_nearest_corner = np.full_like(ys, 1e9, dtype=np.float32)
                d_edge_to_corner = 0.0
                for cj in _EDGE_ADJACENT_CORNERS.get(i, []):
                    tc, lc = patches[cj][0], patches[cj][1]
                    cc_cy = tc + box + ph / 2 - 0.5
                    cc_cx = lc + box + pw / 2 - 0.5
                    dc = np.sqrt((ys - cc_cy) ** 2 + (xs - cc_cx) ** 2)
                    d_to_nearest_corner = np.minimum(d_to_nearest_corner, dc)
                    d_edge_to_corner = max(d_edge_to_corner, float(np.sqrt((pc_cy - cc_cy) ** 2 + (pc_cx - cc_cx) ** 2)))
                d_edge_to_corner = max(d_edge_to_corner, 1e-6)
                linear_edge = np.minimum(1.0, d_to_nearest_corner / d_edge_to_corner)
                w = np.where(in_center_expanded, linear_center,
                             np.where(in_edge_corner, linear_edge, 1.0))
            elif i in _CORNER_INDICES:
                # 四角：与中心重叠用线性；与四边重叠用线性（0->1 向四角中心）；其余用 1
                in_corner_edge = np.zeros_like(in_center_expanded, dtype=bool)
                for ej in _CORNER_ADJACENT_EDGES.get(i, []):
                    in_corner_edge = in_corner_edge | _in_patch_expanded(ys, xs, ej)
                in_corner_edge = in_corner_edge & (~in_center_expanded)
                # 四角在 edge-corner 重叠：从 0（靠四边边界）到 1（靠四角中心），用 d_to_edge_core
                d_to_nearest_edge = np.full_like(ys, 1e9, dtype=np.float32)
                d_corner_to_edge = 0.0
                for ej in _CORNER_ADJACENT_EDGES.get(i, []):
                    et, el, _, _ = _patch_core_rect(ej)
                    de = _dist_to_rect_grid(ys, xs, et, el, ph, pw)
                    d_to_nearest_edge = np.minimum(d_to_nearest_edge, de)
                    ec_cy = patches[ej][0] + box + ph / 2 - 0.5
                    ec_cx = patches[ej][1] + box + pw / 2 - 0.5
                    d_corner_to_edge = max(d_corner_to_edge, float(np.sqrt((pc_cy - ec_cy) ** 2 + (pc_cx - ec_cx) ** 2)))
                d_corner_to_edge = max(d_corner_to_edge, 1e-6)
                linear_corner = np.minimum(1.0, d_to_nearest_edge / d_corner_to_edge)
                w = np.where(in_center_expanded, linear_center,
                             np.where(in_corner_edge, linear_corner, 1.0))
            else:
                w = np.where(in_center_expanded, linear_center, 1.0)
        weights.append(w.astype(np.float32))
    return weights


def _choose_reverse_scores(pred_roi: np.ndarray, input_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回三个距离：score_b 离纯黑，score_w 离纯白，score_r 离反色。
    依次用 threshold 比较：score_b < thresh -> 黑，否则 score_w < thresh -> 白，否则 score_r < thresh -> 反色。
    """
    pred = np.clip(pred_roi.astype(np.float64), 0.0, 1.0)
    inp = np.clip(input_roi.astype(np.float64), 0.0, 1.0)
    score_b = np.mean(np.abs(pred), axis=2).astype(np.float32)  # 到纯黑
    score_w = np.mean(np.abs(pred - 1.0), axis=2).astype(np.float32)  # 到纯白
    pred_u8 = (pred * 255).astype(np.uint8)
    inp_u8 = (inp * 255).astype(np.uint8)
    lab_out = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2LAB)
    lab_in = cv2.cvtColor(inp_u8, cv2.COLOR_RGB2LAB)
    a_out = lab_out[:, :, 1].astype(np.float64) - 128
    b_out = lab_out[:, :, 2].astype(np.float64) - 128
    a_in = lab_in[:, :, 1].astype(np.float64) - 128
    b_in = lab_in[:, :, 2].astype(np.float64) - 128
    d_ab = np.sqrt((a_out + a_in) ** 2 + (b_out + b_in) ** 2)
    score_r = np.clip(d_ab / (2.0 * 128 * np.sqrt(2)), 0.0, 1.0).astype(np.float32)  # 到反色
    return score_b, score_w, score_r


def _choose_reverse_score(
    pred_roi: np.ndarray,
    input_roi: np.ndarray,
    th_black: float,
    th_white: float,
    th_reverse: float,
) -> np.ndarray:
    """
    choose_reverse 选分：依次比较 score_b < th_black、score_w < th_white、score_r < th_reverse。
    返回用于比较的标量，越小越优。
    """
    score_b, score_w, score_r = _choose_reverse_scores(pred_roi, input_roi)
    score = np.where(
        score_b < th_black,
        score_b,
        np.where(
            score_w < th_white,
            th_black + score_w,
            np.where(score_r < th_reverse, 2.0 + score_r, 3.0),
        ),
    )
    return score.astype(np.float32)


def _apply_set_black_white(
    img: np.ndarray,
    th_black: float,
    th_white: float,
    set_black: float,
    set_white: float,
) -> np.ndarray:
    """
    对满足 choose_black 的像素：若 RGB 方差 < set_black 则与该像素和纯黑做平均；
    对满足 choose_white 的像素：若 RGB 方差 < set_white 则与该像素和纯白做平均。
    choose_black/choose_white 仍用 sb/sw，仅 set_black/set_white 的二次判断改用方差。
    img: [0,1] float (H,W,3)
    """
    if set_black <= 0 and set_white <= 0:
        return img
    pred = np.clip(img.astype(np.float64), 0.0, 1.0)
    sb = np.mean(pred, axis=2).astype(np.float32)
    sw = np.mean(1.0 - pred, axis=2).astype(np.float32)
    var_score = np.var(pred, axis=2).astype(np.float32)  # 每个像素 RGB 三通道方差
    out = pred.astype(np.float32).copy()
    if set_black > 0 and th_black > 0:
        mask = (sb < th_black) & (var_score < set_black)
        out[mask, :] = (out[mask] + 0.0) / 2.0
    if set_white > 0 and th_white > 0:
        mask = (sb >= th_black) & (sw < th_white) & (var_score < set_white)
        out[mask, :] = (out[mask] + 1.0) / 2.0
    return out


def _patch_interior_mask(rh: int, rw: int, edge: int = 32) -> np.ndarray:
    """
    每个 patch 边缘 edge 像素不参与 choose 竞争，减少该 patch 可选项。
    返回 (rh, rw) bool，True= interior = 此 patch 可参与 choose。
    """
    if rh <= 2 * edge or rw <= 2 * edge:
        return np.zeros((rh, rw), dtype=bool)
    r = np.arange(rh, dtype=np.int32)
    c = np.arange(rw, dtype=np.int32)
    return ((r >= edge) & (r < rh - edge))[:, None] & ((c >= edge) & (c < rw - edge))[None, :]


def _overlap_merge_weight(patch_h: int, patch_w: int, use_linear: bool = False) -> np.ndarray:
    """use_linear=False: 均匀权重 1；use_linear=True: 线性衰减权重。"""
    if use_linear:
        xh = np.linspace(0, 1, patch_h)
        xw = np.linspace(0, 1, patch_w)
        wh = np.minimum(xh, 1 - xh) * 2
        ww = np.minimum(xw, 1 - xw) * 2
        return (wh[:, None] * ww[None, :]).astype(np.float32)
    return np.ones((patch_h, patch_w), dtype=np.float32)


def _rgb_to_lab_uint8(rgb_01: np.ndarray) -> np.ndarray:
    """RGB [0,1] float (H,W,3) -> LAB uint8 (H,W,3)，OpenCV 约定 L in [0,255], A/B in [0,255] 128 中性。"""
    rgb_u8 = (np.clip(rgb_01, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)


def _lab_to_rgb_float(lab: np.ndarray) -> np.ndarray:
    """LAB uint8 (H,W,3) -> RGB [0,1] float (H,W,3)。"""
    rgb_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_u8.astype(np.float32) / 255.0


def _lock_l_merge(input_rgb_01: np.ndarray, pred_rgb_01: np.ndarray) -> np.ndarray:
    """用输入的 L 与预测的 AB 合成：output = LAB(L_input, A_pred, B_pred) -> RGB。"""
    lab_in = _rgb_to_lab_uint8(input_rgb_01)
    lab_pred = _rgb_to_lab_uint8(np.clip(pred_rgb_01, 0.0, 1.0))
    L_in = lab_in[:, :, 0]
    A_pred = lab_pred[:, :, 1]
    B_pred = lab_pred[:, :, 2]
    lab_merge = np.stack([L_in, A_pred, B_pred], axis=-1)
    return _lab_to_rgb_float(lab_merge)


def run_inference_sliding_window(
    cnn_model: CNNImageRegressor,
    input_dir: str,
    out_dir: str,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    patch_size: int = 256,
    patch_h: Optional[int] = None,
    patch_w: Optional[int] = None,
    stride: Optional[int] = None,
    stride_h: Optional[int] = None,
    stride_w: Optional[int] = None,
    box: int = 32,
    padding: Optional[int] = None,
    vis_dir: Optional[str] = None,
    vis_max_images: int = 20,
    max_files: Optional[int] = None,
    sigma_low: Optional[float] = None,
    sigma_mid: Optional[float] = None,
    low_ratio: float = 1.0,
    mid_ratio: float = 0.0,
    high_ratio: float = 0.0,
    use_weighted_merge: bool = False,
    tile_no_overlap: bool = False,
    choose_thresholds: Optional[Tuple[float, float, float]] = None,
    set_black: float = 0,
    set_white: float = 0,
    vis_choose_black_var: Optional[float] = None,
    use_full: bool = False,
    multi_sizes: Optional[List[int]] = None,
    multi_weight: Optional[float] = None,
    vis_include_patch: bool = False,
    vis_boundary: bool = False,
    pad_lines: int = 32,
    pad_flip: bool = False,
    use_train_patch_style: bool = False,
    resize_input: Optional[Tuple[int, int]] = None,
    boundary_weight: Optional[float] = None,
    lock_l: bool = False,
) -> float:
    """
    滑动窗口推理。patch_h/patch_w 控制 patch 高/宽，未指定则用 patch_size。
    boundary_weight：在最终高频叠加前，log 域 log_high = boundary_weight*log_boundary + log_high。
    stride 未给定：stride = patch - box（默认 box=32），适配高宽不一致。
    从图像中心构建 patch、向四周拓展，不从 padding 后左上角开始。
    padding：若为 None 则按图像尺寸计算，且不少于 box；可显式指定。
    tile_no_overlap=True：stride=patch 无重叠铺满，仍用原左上角起点网格。
    use_train_patch_style=True：与 train_patch 整图验证完全一致，patch=256 stride=224 左上角网格、线性权重、无 band padding。
    """
    if use_train_patch_style:
        ph = pw = 256
        stride_h = stride_w = 224
        _img_size_train = 256
    else:
        ph = patch_h if patch_h is not None else patch_size
        pw = patch_w if patch_w is not None else patch_size
        _img_size_train = img_size
        # 显式传入的 stride_h/stride_w 优先（如 --stride 512,384）
        if stride_h is not None and stride_w is not None:
            pass
        elif tile_no_overlap:
            stride_h, stride_w = ph, pw
        elif stride == -1:
            stride_h, stride_w = ph, pw
        elif stride is not None:
            stride_h = stride_w = stride
        else:
            stride_h = max(1, ph - box)
            stride_w = max(1, pw - box)

    os.makedirs(out_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    use_freq = mid_ratio != 0.0 or high_ratio != 0.0
    if use_train_patch_style:
        eff_h, eff_w = _img_size_train, _img_size_train
    else:
        eff_h, eff_w = (img_size, img_size) if isinstance(img_size, int) else (img_size[0], img_size[1])
    transform = T.Compose([T.Resize((eff_h, eff_w)), T.ToTensor()])
    to_pil = T.ToPILImage()
    weight_map = _overlap_merge_weight(ph, pw, use_linear=True if use_train_patch_style else use_weighted_merge)

    files = sorted([f for f in os.listdir(input_dir) if is_supported_input_file(f)])
    if len(files) == 0:
        raise RuntimeError(f"输入目录 {input_dir} 中没有找到图像/.npy 文件。")
    if max_files is not None:
        files = files[:max_files]
    total_files = len(files)
    if tile_no_overlap:
        mode_str = "无重叠铺满"
    elif stride == -1:
        mode_str = f"stride=patch({stride_h}x{stride_w}) 无重叠"
    elif stride is None:
        mode_str = f"stride={stride_h}x{stride_w}(patch-{box}) 重叠合并"
    else:
        mode_str = f"stride={stride} 重叠合并"
    patch_str = f"{ph}x{pw}" if (ph != pw) else str(ph)
    print(f"滑动窗口推理：patch={patch_str} {mode_str} 共 {total_files} 张" + ("（纯 CNN）" if not use_freq else "（三频合成）"))

    total_time = 0.0
    vis_saved_gt_ids = set()
    with torch.no_grad():
        for idx, name in enumerate(files):
            in_path = os.path.join(input_dir, name)
            img, img_np = load_input_as_rgb(in_path, npy_is_bgr=True)
            orig_w, orig_h = img.size
            orig_h_save, orig_w_save = orig_h, orig_w
            img_np_for_high = img_np.copy()
            input_is_npy = name.lower().endswith(".npy")
            gt_id = _gt_id_from_npy_name(name) if input_is_npy else ""

            if use_freq and resize_input is not None:
                img_log_orig = np.log1p(np.clip(img_np, 0, None))
                s_orig = (sigma_low, sigma_mid) if (sigma_low is not None and sigma_mid is not None) else sigma_from_image_freq(orig_h, orig_w, verbose=False)
                low_orig, mid_orig, high_in = decompose_freq_log(img_log_orig, s_orig[0], s_orig[1])

            if resize_input is not None:
                resize_w, resize_h = resize_input
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil = img_pil.resize((resize_w, resize_h), Image.BICUBIC)
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                orig_h, orig_w = resize_h, resize_w
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save}) resize(w,h)=({resize_w},{resize_h})")
            else:
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save})")

            if use_freq:
                img_log = np.log1p(np.clip(img_np, 0, None))
                if sigma_low is not None and sigma_mid is not None:
                    s_low, s_mid = sigma_low, sigma_mid
                else:
                    s_low, s_mid = sigma_from_image_freq(orig_h, orig_w, verbose=False)
                low_in, mid_in, high_in_r = decompose_freq_log(img_log, s_low, s_mid)
                if resize_input is None:
                    high_in = high_in_r
            vis_freq_tuple = (low_orig, mid_orig, high_in) if (use_freq and resize_input is not None) else ((low_in, mid_in, high_in) if use_freq else None)

            if use_train_patch_style:
                work_np = img_np.copy()
                h, w = orig_h, orig_w
                pad_val = 0
                if h < ph or w < pw:
                    pad_h, pad_w = max(0, ph - h), max(0, pw - w)
                    work_np = np.pad(work_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                    h, w = h + pad_h, w + pad_w
                positions = _sliding_window_grid(h, w, ph, pw, stride_h, stride_w)
                if not positions:
                    positions = [(0, 0)]
                acc = np.zeros((h, w, 3), dtype=np.float64)
                wacc = np.zeros((h, w), dtype=np.float64)
            elif padding is not None and padding == 0:
                pad_val = 0
                work_np = img_np.copy()
                h, w = orig_h, orig_w
            else:
                pad_val = _compute_padding(orig_h, orig_w, ph, pw, box)
                if multi_sizes:
                    for ms in multi_sizes:
                        pad_val = max(pad_val, _compute_padding(orig_h, orig_w, ms, ms, box))
                if padding is not None and padding > 0:
                    pad_val = max(pad_val, padding // 2)  # 32 → 上下左右各 16，与 train val_fullimage_padding 一致
                pad_val = max(pad_val, box)
                _pad = lambda i, pt, pb, pl, pr: _pad_with_flip(i, pt, pb, pl, pr) if pad_flip else _pad_with_band(i, pt, pb, pl, pr, pad_lines)
                work_np = _pad(img_np, pad_val, pad_val, pad_val, pad_val)
                h, w = orig_h + 2 * pad_val, orig_w + 2 * pad_val
            max_patch_dim = max(ph, pw)
            if multi_sizes:
                max_patch_dim = max(max_patch_dim, max(multi_sizes))
            need_extra = h < max_patch_dim or w < max_patch_dim
            if need_extra and (padding is None or padding != 0) and not use_train_patch_style:
                pad_ht = max(0, max_patch_dim - h)
                pad_wt = max(0, max_patch_dim - w)
                work_np = _pad(work_np, 0, pad_ht, 0, pad_wt)
                h, w = h + pad_ht, w + pad_wt

            if not use_train_patch_style:
                if tile_no_overlap:
                    positions = _sliding_window_grid(h, w, ph, pw, stride_h, stride_w)
                else:
                    positions = _sliding_window_grid_centered(h, w, ph, pw, stride_h, stride_w, pad_val, orig_h, orig_w)
                if not positions:
                    positions = [(0, 0)]

            if not use_train_patch_style:
                if tile_no_overlap:
                    cnn_low = np.zeros((h, w, 3), dtype=np.float32)
                elif choose_thresholds is not None:
                    out_img = np.zeros((h, w, 3), dtype=np.float32)
                    min_score = np.full((h, w), np.inf, dtype=np.float32)
                acc = np.zeros((h, w, 3), dtype=np.float64)
                wacc = np.zeros((h, w), dtype=np.float64)
                acc_fb = np.zeros((h, w, 3), dtype=np.float64)
                wacc_fb = np.zeros((h, w), dtype=np.float64)
                weight_map_linear = _overlap_merge_weight(ph, pw, use_linear=True)
            else:
                acc = np.zeros((h, w, 3), dtype=np.float64)
                wacc = np.zeros((h, w), dtype=np.float64)

            n_patches = len(positions)
            start = time.perf_counter()
            for top, left in positions:
                rh = min(ph, h - top)
                rw = min(pw, w - left)
                patch_np = work_np[top : top + rh, left : left + rw]
                if patch_np.shape[0] < ph or patch_np.shape[1] < pw:
                    patch_pad = np.pad(
                        patch_np,
                        ((0, ph - rh), (0, pw - rw), (0, 0)),
                        mode="edge",
                    )
                else:
                    patch_pad = patch_np
                patch_pil = Image.fromarray((patch_pad * 255).astype(np.uint8))
                x = transform(patch_pil).unsqueeze(0).to(device)
                pred = _model_for_patch(cnn_model, ph, pw)(x)
                pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                pred_np = pred.numpy().transpose(1, 2, 0)
                if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                    pred_pil = to_pil(pred)
                    pred_pil = pred_pil.resize((pw, ph), Image.BICUBIC)
                    pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                if tile_no_overlap:
                    cnn_low[top : top + rh, left : left + rw] = pred_np[:rh, :rw]
                elif choose_thresholds is not None and not use_train_patch_style:
                    score = _choose_reverse_score(pred_np[:rh, :rw], patch_np, *choose_thresholds)
                    roi_min = min_score[top : top + rh, left : left + rw]
                    interior = _patch_interior_mask(rh, rw, edge=32)
                    mask = (score < roi_min) & interior
                    min_score[top : top + rh, left : left + rw] = np.where(mask, score, roi_min)
                    roi_out = out_img[top : top + rh, left : left + rw]
                    out_img[top : top + rh, left : left + rw] = np.where(mask[:, :, None], pred_np[:rh, :rw], roi_out)
                    wpatch = weight_map_linear[:rh, :rw]
                    pass_mask = (score < 3.0)[:, :, None]
                    acc[top : top + rh, left : left + rw] += np.where(pass_mask, pred_np[:rh, :rw] * wpatch[:, :, None], 0.0)
                    wacc[top : top + rh, left : left + rw] += np.where(pass_mask[:, :, 0], wpatch, 0.0)
                    acc_fb[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                    wacc_fb[top : top + rh, left : left + rw] += wpatch
                else:
                    wpatch = weight_map[:rh, :rw]
                    acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                    wacc[top : top + rh, left : left + rw] += wpatch

            if use_full and not tile_no_overlap and not use_train_patch_style:
                n_patches += 1
                full_pil = Image.fromarray((work_np * 255).astype(np.uint8)).resize((eff_w, eff_h))
                x_full = transform(full_pil).unsqueeze(0).to(device)
                pred_full = _model_for_patch(cnn_model, ph, pw)(x_full)
                pred_full = torch.clamp(pred_full.squeeze(0).cpu(), 0.0, 1.0)
                pred_full_np = pred_full.numpy().transpose(1, 2, 0)
                pred_full_np = np.array(
                    Image.fromarray((pred_full_np * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
                ).astype(np.float32) / 255.0
                if choose_thresholds is not None:
                    score_full = _choose_reverse_score(pred_full_np, work_np, *choose_thresholds)
                    mask_full = score_full < min_score
                    min_score[:] = np.where(mask_full, score_full, min_score)
                    out_img[:] = np.where(mask_full[:, :, None], pred_full_np, out_img)
                if multi_weight is not None and multi_weight > 0 and (choose_thresholds is None):
                    w_full = multi_weight * 1.0
                    acc += pred_full_np * w_full
                    wacc += w_full

            if multi_sizes and not tile_no_overlap and not use_train_patch_style:
                use_patch_stride = stride == -1
                mbase_linear = (choose_thresholds is not None) or use_weighted_merge
                for mph in multi_sizes:
                    mpw = mph
                    if use_patch_stride:
                        mstride_h, mstride_w = mph, mpw
                    else:
                        mstride_h = max(1, mph - box)
                        mstride_w = max(1, mpw - box)
                    mpositions = _sliding_window_grid_centered(h, w, mph, mpw, mstride_h, mstride_w, pad_val, orig_h, orig_w)
                    if not mpositions:
                        mpositions = [(0, 0)]
                    n_patches += len(mpositions)
                    mwm = _overlap_merge_weight(mph, mpw, use_linear=mbase_linear)
                    for top, left in mpositions:
                        rh = min(mph, h - top)
                        rw = min(mpw, w - left)
                        patch_np = work_np[top : top + rh, left : left + rw]
                        if patch_np.shape[0] < mph or patch_np.shape[1] < mpw:
                            patch_pad = np.pad(patch_np, ((0, mph - rh), (0, mpw - rw), (0, 0)), mode="edge")
                        else:
                            patch_pad = patch_np
                        patch_pil = Image.fromarray((patch_pad * 255).astype(np.uint8))
                        x = transform(patch_pil).unsqueeze(0).to(device)
                        pred = _model_for_patch(cnn_model, mph, mpw)(x)
                        pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                        pred_np = pred.numpy().transpose(1, 2, 0)
                        if pred_np.shape[0] != mph or pred_np.shape[1] != mpw:
                            pred_pil = to_pil(pred)
                            pred_pil = pred_pil.resize((mpw, mph), Image.BICUBIC)
                            pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                        if choose_thresholds is not None:
                            score = _choose_reverse_score(pred_np[:rh, :rw], patch_np, *choose_thresholds)
                            roi_min = min_score[top : top + rh, left : left + rw]
                            interior = _patch_interior_mask(rh, rw, edge=32)
                            mask = (score < roi_min) & interior
                            min_score[top : top + rh, left : left + rw] = np.where(mask, score, roi_min)
                            roi_out = out_img[top : top + rh, left : left + rw]
                            out_img[top : top + rh, left : left + rw] = np.where(mask[:, :, None], pred_np[:rh, :rw], roi_out)
                        if multi_weight is not None and multi_weight > 0 and (choose_thresholds is None):
                            wpatch = mwm[:rh, :rw] * multi_weight
                            acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                            wacc[top : top + rh, left : left + rw] += wpatch

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

            if not tile_no_overlap:
                if choose_thresholds is not None:
                    satisfied = min_score < 3.0
                    use_choose = satisfied
                    wacc_nz = np.maximum(wacc, 1e-8)
                    merged_pass = (acc / wacc_nz[:, :, None]).astype(np.float32)
                    wacc_fb_nz = np.maximum(wacc_fb, 1e-8)
                    merged_fb = (acc_fb / wacc_fb_nz[:, :, None]).astype(np.float32)
                    merged = np.where(wacc[:, :, None] > 1e-8, merged_pass, merged_fb)
                    cnn_low = np.where(use_choose[:, :, None], out_img, merged)
                else:
                    wacc = np.maximum(wacc, 1e-8)
                    cnn_low = (acc / wacc[:, :, None]).astype(np.float32)

            cnn_low = cnn_low[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]

            vis_lock_l = None
            if lock_l:
                lock_l_result = _lock_l_merge(img_np, cnn_low)
                if vis_dir and (
                    (input_is_npy and gt_id not in vis_saved_gt_ids and len(vis_saved_gt_ids) < vis_max_images)
                    or (not input_is_npy and idx < vis_max_images)
                ):
                    lab_in = _rgb_to_lab_uint8(img_np)
                    L_vis = np.stack([lab_in[:, :, 0], lab_in[:, :, 0], lab_in[:, :, 0]], axis=-1)
                    lab_pred = _rgb_to_lab_uint8(np.clip(cnn_low, 0.0, 1.0))
                    AB_vis = np.stack([lab_pred[:, :, 1], lab_pred[:, :, 2], np.full((orig_h, orig_w), 128, dtype=np.uint8)], axis=-1)
                    vis_lock_l = (L_vis, AB_vis, (np.clip(lock_l_result, 0.0, 1.0) * 255).astype(np.uint8))
                cnn_low = lock_l_result

            if use_freq:
                mid_log = mid_in
                high_log = high_in
                cnn_log = np.log1p(np.clip(cnn_low, 1e-6, None))
                output_low_mid_log = low_ratio * cnn_log + mid_ratio * mid_log
                output_linear = np.clip(np.expm1(output_low_mid_log), 0.0, 1.0)
            else:
                output_linear = np.clip(cnn_low, 0.0, 1.0)
            output_before_set = output_linear.copy() if (choose_thresholds is not None and (set_black > 0 or set_white > 0)) else None
            if choose_thresholds is not None and (set_black > 0 or set_white > 0):
                th_b, th_w, _ = choose_thresholds
                output_linear = _apply_set_black_white(output_linear, th_b, th_w, set_black, set_white)
            if resize_input is not None:
                output_linear = np.array(
                    Image.fromarray((output_linear * 255).astype(np.uint8)).resize((orig_w_save, orig_h_save), Image.BICUBIC)
                ).astype(np.float32) / 255.0
            out_no_high = output_linear.copy() if (use_freq and high_ratio != 0) else None
            if use_freq and high_ratio != 0:
                high_to_add = high_in
                if boundary_weight is not None:
                    high_to_add = _apply_boundary_to_high(high_in, img_np_for_high, boundary_weight)
                output_linear = np.clip(np.expm1(np.log1p(np.clip(output_linear, 1e-6, None)) + high_ratio * high_to_add), 0.0, 1.0)

            pred_img = Image.fromarray((output_linear * 255).astype(np.uint8))

            out_path = os.path.join(out_dir, name)
            if input_is_npy:
                out_bgr = np.clip(output_linear, 0.0, 1.0).astype(np.float32)[:, :, ::-1]
                np.save(out_path, out_bgr)
            else:
                pred_img.save(out_path)
            print("\r" + _progress_bar(idx + 1, total_files, prefix="滑动窗口 ") + f"  {name}: {n_patches} 张 patch", end="", flush=True)

            do_vis = vis_dir and (
                (input_is_npy and gt_id not in vis_saved_gt_ids and len(vis_saved_gt_ids) < vis_max_images)
                or (not input_is_npy and idx < vis_max_images)
            )
            if do_vis:
                if input_is_npy:
                    vis_saved_gt_ids.add(gt_id)
                base, _ = os.path.splitext(name)
                patch_merged_by_size = []
                if vis_include_patch and not tile_no_overlap:
                    def _run_patch_merge(mph: int, mpw: int, mpos) -> np.ndarray:
                        acc_p = np.zeros((h, w, 3), dtype=np.float64)
                        wacc_p = np.zeros((h, w), dtype=np.float64)
                        wm = _overlap_merge_weight(mph, mpw, use_linear=True)
                        for top, left in mpos:
                            rh = min(mph, h - top)
                            rw = min(mpw, w - left)
                            patch_np = work_np[top : top + rh, left : left + rw]
                            if patch_np.shape[0] < mph or patch_np.shape[1] < mpw:
                                patch_pad = np.pad(patch_np, ((0, mph - rh), (0, mpw - rw), (0, 0)), mode="edge")
                            else:
                                patch_pad = patch_np
                            patch_pil = Image.fromarray((patch_pad * 255).astype(np.uint8))
                            x = transform(patch_pil).unsqueeze(0).to(device)
                            pred = _model_for_patch(cnn_model, mph, mpw)(x)
                            pred_t = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                            pred_np = pred_t.numpy().transpose(1, 2, 0)
                            if pred_np.shape[0] != mph or pred_np.shape[1] != mpw:
                                pred_np = np.array(to_pil(pred_t).resize((mpw, mph), Image.BICUBIC)).astype(np.float32) / 255.0
                            wpatch = wm[:rh, :rw]
                            acc_p[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                            wacc_p[top : top + rh, left : left + rw] += wpatch
                        wacc_p = np.maximum(wacc_p, 1e-8)
                        out_p = (acc_p / wacc_p[:, :, None]).astype(np.float32)[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                        return out_p
                    # 主 patch
                    pos_main = _sliding_window_grid_centered(h, w, ph, pw, ph, pw, pad_val, orig_h, orig_w)
                    main_p = _run_patch_merge(ph, pw, pos_main)
                    if use_freq:
                        log_term = low_ratio * np.log1p(np.clip(main_p, 1e-6, None)) + mid_ratio * mid_in
                        if resize_input is None:
                            log_term += high_ratio * high_in
                        main_p = np.clip(np.expm1(log_term), 0.0, 1.0)
                    patch_merged_by_size.append((str(ph), np.clip(main_p, 0.0, 1.0)))
                    # full
                    if use_full:
                        full_pil = Image.fromarray((work_np * 255).astype(np.uint8)).resize((eff_w, eff_h))
                        pred_full = _model_for_patch(cnn_model, ph, pw)(transform(full_pil).unsqueeze(0).to(device))
                        full_p = torch.clamp(pred_full.squeeze(0).cpu(), 0.0, 1.0).numpy().transpose(1, 2, 0)
                        full_p = np.array(Image.fromarray((full_p * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0
                        full_p = full_p[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                        if use_freq:
                            log_term = low_ratio * np.log1p(np.clip(full_p, 1e-6, None)) + mid_ratio * mid_in
                            if resize_input is None:
                                log_term += high_ratio * high_in
                            full_p = np.clip(np.expm1(log_term), 0.0, 1.0)
                        patch_merged_by_size.append(("full", np.clip(full_p, 0.0, 1.0)))
                    # multi_size
                    use_patch_stride = stride == -1
                    for mph in (multi_sizes or []):
                        mpw = mph
                        mstride_h = mph if use_patch_stride else max(1, mph - box)
                        mstride_w = mpw if use_patch_stride else max(1, mpw - box)
                        mpos = _sliding_window_grid_centered(h, w, mph, mpw, mstride_h, mstride_w, pad_val, orig_h, orig_w)
                        mp = _run_patch_merge(mph, mpw, mpos)
                        if use_freq:
                            log_term = low_ratio * np.log1p(np.clip(mp, 1e-6, None)) + mid_ratio * mid_in
                            if resize_input is None:
                                log_term += high_ratio * high_in
                            mp = np.clip(np.expm1(log_term), 0.0, 1.0)
                        patch_merged_by_size.append((str(mph), np.clip(mp, 0.0, 1.0)))
                    if choose_thresholds is not None:
                        th_b, th_w, th_r = choose_thresholds
                        if output_before_set is not None:
                            pred = np.clip(output_before_set.astype(np.float64), 0.0, 1.0)
                            sb = np.mean(pred, axis=2).astype(np.float32)
                            sw = np.mean(1.0 - pred, axis=2).astype(np.float32)
                            var_score = np.var(pred, axis=2).astype(np.float32)
                            var_th_b = vis_choose_black_var if vis_choose_black_var is not None else set_black
                            mask_black = (sb < th_b) & (var_score < var_th_b) if var_th_b > 0 and th_b > 0 else (sb < th_b)
                            mask_white = ((sb >= th_b) & (sw < th_w) & (var_score < set_white)) if set_white > 0 and th_w > 0 else ((sb >= th_b) & (sw < th_w))
                            out_crop_roi = out_img[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                            work_crop = work_np[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                            _, _, sr = _choose_reverse_scores(out_crop_roi, work_crop)
                        else:
                            sb, sw, sr = _choose_reverse_scores(out_img, work_np)
                            mask_black = sb < th_b
                            mask_white = (sb >= th_b) & (sw < th_w)
                            out_crop_roi = out_img[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                        mask_reverse = (sb >= th_b) & (sw >= th_w) & (sr < th_r)
                        out_crop = out_crop_roi
                        for mask, label in [(mask_black, "ChooseBlack"), (mask_white, "ChooseWhite"), (mask_reverse, "ChooseReverse")]:
                            m = mask[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w] if mask.shape != (orig_h, orig_w) else mask
                            vis = np.where(m[:, :, None], out_crop, 0.5)
                            patch_merged_by_size.append((label, np.clip(vis.astype(np.float32), 0.0, 1.0)))
                elif vis_include_patch and tile_no_overlap:
                    pos_patch = _sliding_window_grid(h, w, ph, pw, stride_h, stride_w)
                    cnn_low_p = np.zeros((h, w, 3), dtype=np.float32)
                    for top, left in pos_patch:
                        rh, rw = min(ph, h - top), min(pw, w - left)
                        patch_np = work_np[top : top + rh, left : left + rw]
                        patch_pad = np.pad(patch_np, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge") if (rh < ph or rw < pw) else patch_np
                        pred_t = torch.clamp(_model_for_patch(cnn_model, ph, pw)(transform(Image.fromarray((patch_pad * 255).astype(np.uint8))).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0)
                        pred_np = pred_t.numpy().transpose(1, 2, 0)
                        if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                            pred_np = np.array(to_pil(pred_t).resize((pw, ph), Image.BICUBIC)).astype(np.float32) / 255.0
                        cnn_low_p[top : top + rh, left : left + rw] = pred_np[:rh, :rw]
                    cnn_low_p = cnn_low_p[pad_val : pad_val + orig_h, pad_val : pad_val + orig_w]
                    if use_freq:
                        log_term = low_ratio * np.log1p(np.clip(cnn_low_p, 1e-6, None)) + mid_ratio * mid_in
                        if resize_input is None:
                            log_term += high_ratio * high_in
                        cnn_low_p = np.clip(np.expm1(log_term), 0.0, 1.0)
                    patch_merged_by_size = [(str(ph), np.clip(cnn_low_p, 0.0, 1.0))]
                save_vis_components(
                    img, pred_img, cnn_low,
                    os.path.join(vis_dir, f"sample_{idx:04d}_{base}.png"),
                    patch_merged_by_size=patch_merged_by_size if patch_merged_by_size else None,
                    vis_boundary=vis_boundary,
                    vis_freq_decompose=vis_freq_tuple,
                    vis_output_no_high=out_no_high,
                    vis_lock_l=vis_lock_l,
                )

    if total_files:
        print()
    return total_time / total_files


def run_inference_auto_patch(
    cnn_model: CNNImageRegressor,
    input_dir: str,
    out_dir: str,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    box: Optional[int] = 32,
    center_ratio: float = 0.5,
    stride: Optional[int] = None,
    stride_h: Optional[int] = None,
    stride_w: Optional[int] = None,
    patch_size: int = 256,
    patch_h: Optional[int] = None,
    patch_w: Optional[int] = None,
    use_weighted_merge: bool = True,
    choose_thresholds: Optional[Tuple[float, float, float]] = None,
    set_black: float = 0,
    set_white: float = 0,
    vis_choose_black_var: Optional[float] = None,
    use_full: bool = False,
    multi_sizes: Optional[List[int]] = None,
    multi_weight: Optional[float] = None,
    vis_dir: Optional[str] = None,
    vis_max_images: int = 20,
    max_files: Optional[int] = None,
    sigma_low: Optional[float] = None,
    sigma_mid: Optional[float] = None,
    low_ratio: float = 1.0,
    mid_ratio: float = 0.0,
    high_ratio: float = 0.0,
    vis_include_patch: bool = False,
    vis_boundary: bool = False,
    pad_lines: int = 32,
    pad_flip: bool = False,
    uniform: bool = False,
    resize_input: Optional[Tuple[int, int]] = None,
    boundary_weight: Optional[float] = None,
) -> float:
    """
    auto_patch 推理：若 stride/stride_h/stride_w 给定则按滑动窗口（中心为首个向四周），否则 3x3 分块。
    uniform=True 强行进入 uniform 模式，此时 box 可由参数指定。
    3x3：中心 patch 与原图等比例面积=center_ratio，每 patch 加 box 重叠合并。
    pad_lines：padding 填充使用的边缘行/列带状厚度。
    """
    os.makedirs(out_dir, exist_ok=True)
    use_sliding = stride is not None or stride_h is not None or stride_w is not None
    sh_eff = stride_h if stride_h is not None else stride
    sw_eff = stride_w if stride_w is not None else stride
    if use_sliding and (sh_eff is None or sw_eff is None):
        sh_eff = sh_eff if sh_eff is not None else sw_eff
        sw_eff = sw_eff if sw_eff is not None else sh_eff
    use_uniform_3x3 = not use_sliding and (uniform or box is None)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    use_freq = mid_ratio != 0.0 or high_ratio != 0.0
    eff_h, eff_w = _img_size_hw(img_size)
    transform = T.Compose([T.Resize((eff_h, eff_w)), T.ToTensor()])
    to_pil = T.ToPILImage()

    files = sorted([f for f in os.listdir(input_dir) if is_supported_input_file(f)])
    if len(files) == 0:
        raise RuntimeError(f"输入目录 {input_dir} 中没有找到图像/.npy 文件。")
    if max_files is not None:
        files = files[:max_files]
    total_files = len(files)
    if use_sliding:
        mode_str = "stride=patch 滑动窗口" if (sh_eff == -1 or sw_eff == -1) else f"stride={sh_eff}x{sw_eff} 滑动窗口"
    else:
        mode_str = "3x3 uniform 自适应" if use_uniform_3x3 else "3x3 分块"
    if choose_thresholds is not None:
        mode_str += f" [choose(b/w/r)]"
    print(f"auto_patch {mode_str}：共 {total_files} 张" + ("（纯 CNN）" if not use_freq else "（三频合成）"))

    total_time = 0.0
    vis_saved_gt_ids = set()
    with torch.no_grad():
        for idx, name in enumerate(files):
            in_path = os.path.join(input_dir, name)
            img, img_np = load_input_as_rgb(in_path, npy_is_bgr=True)
            orig_w, orig_h = img.size
            orig_h_save, orig_w_save = orig_h, orig_w
            img_np_for_high = img_np.copy()
            input_is_npy = name.lower().endswith(".npy")
            gt_id = _gt_id_from_npy_name(name) if input_is_npy else ""

            if use_freq and resize_input is not None:
                img_log_orig = np.log1p(np.clip(img_np, 0, None))
                s_orig = (sigma_low, sigma_mid) if (sigma_low is not None and sigma_mid is not None) else sigma_from_image_freq(orig_h, orig_w, verbose=False)
                _, _, high_in = decompose_freq_log(img_log_orig, s_orig[0], s_orig[1])

            if resize_input is not None:
                resize_w, resize_h = resize_input
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil = img_pil.resize((resize_w, resize_h), Image.BICUBIC)
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                orig_h, orig_w = resize_h, resize_w
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save}) resize(w,h)=({resize_w},{resize_h})")
            else:
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save})")

            if use_freq:
                img_log = np.log1p(np.clip(img_np, 0, None))
                if sigma_low is not None and sigma_mid is not None:
                    s_low, s_mid = sigma_low, sigma_mid
                else:
                    s_low, s_mid = sigma_from_image_freq(orig_h, orig_w, verbose=False)
                low_in, mid_in, high_in_r = decompose_freq_log(img_log, s_low, s_mid)
                if resize_input is None:
                    high_in = high_in_r

            box_eff = box if box is not None else 32
            pad_val = box_eff
            crop_t = crop_l = box_eff
            if use_sliding:
                r = max(0.01, min(1.0, center_ratio))
                ph = int(round(orig_h * (r ** 0.5)))
                pw = int(round(orig_w * (r ** 0.5)))
                ph = max(1, min(ph, orig_h - 2))
                pw = max(1, min(pw, orig_w - 2))
                pad_val = max(_compute_padding(orig_h, orig_w, ph, pw, box_eff), box_eff)
                if multi_sizes:
                    for ms in multi_sizes:
                        pad_val = max(pad_val, _compute_padding(orig_h, orig_w, ms, ms, box_eff))
                _pad = lambda i, pt, pb, pl, pr: _pad_with_flip(i, pt, pb, pl, pr) if pad_flip else _pad_with_band(i, pt, pb, pl, pr, pad_lines)
                work_np = _pad(img_np, pad_val, pad_val, pad_val, pad_val)
                h, w = orig_h + 2 * pad_val, orig_w + 2 * pad_val
                max_patch_dim = max(ph, pw)
                if multi_sizes:
                    max_patch_dim = max(max_patch_dim, max(multi_sizes))
                need_extra = h < max_patch_dim or w < max_patch_dim
                if need_extra:
                    pad_ht = max(0, max_patch_dim - h)
                    pad_wt = max(0, max_patch_dim - w)
                    work_np = _pad(work_np, 0, pad_ht, 0, pad_wt)
                    h, w = h + pad_ht, w + pad_wt
                s_h = ph if sh_eff == -1 else sh_eff
                s_w = pw if sw_eff == -1 else sw_eff
                positions = _sliding_window_grid_centered(h, w, ph, pw, s_h, s_w, pad_val, orig_h, orig_w)
                h_canvas, w_canvas = h, w
                crop_t = crop_l = pad_val
                weight_map = _overlap_merge_weight(ph, pw, use_linear=use_weighted_merge)
                if idx == 0:
                    print(f"  中心 patch 大小: {ph} x {pw}")
            else:
                if use_uniform_3x3:
                    box_override = box if box is not None else None
                    patches, pad_t, pad_b, pad_l, pad_r, box_eff = _grid_3x3_uniform_centered(orig_h, orig_w, center_ratio, box_override=box_override)
                    _pad = lambda i, pt, pb, pl, pr: _pad_with_flip(i, pt, pb, pl, pr) if pad_flip else _pad_with_band(i, pt, pb, pl, pr, pad_lines)
                    work_np = _pad(img_np, pad_t, pad_b, pad_l, pad_r)
                    h_canvas = orig_h + pad_t + pad_b
                    w_canvas = orig_w + pad_l + pad_r
                    crop_t, crop_l = pad_t, pad_l
                else:
                    _pad = lambda i, pt, pb, pl, pr: _pad_with_flip(i, pt, pb, pl, pr) if pad_flip else _pad_with_band(i, pt, pb, pl, pr, pad_lines)
                    work_np = _pad(img_np, box_eff, box_eff, box_eff, box_eff)
                    patches = _grid_3x3_patches_centered(orig_h, orig_w, center_ratio)
                    h_canvas = orig_h + 2 * box_eff
                    w_canvas = orig_w + 2 * box_eff
                if idx == 0 and patches:
                    _, _, cph, cpw = patches[0]
                    print(f"  中心 patch 大小: {cph} x {cpw}")
                if multi_sizes:
                    max_ms = max(multi_sizes)
                    if h_canvas < max_ms or w_canvas < max_ms:
                        _pad = lambda i, pt, pb, pl, pr: _pad_with_flip(i, pt, pb, pl, pr) if pad_flip else _pad_with_band(i, pt, pb, pl, pr, pad_lines)
                        pad_ht = max(0, max_ms - h_canvas)
                        pad_wt = max(0, max_ms - w_canvas)
                        work_np = _pad(work_np, 0, pad_ht, 0, pad_wt)
                        h_canvas, w_canvas = h_canvas + pad_ht, w_canvas + pad_wt

            if choose_thresholds is not None:
                out_img = np.zeros((h_canvas, w_canvas, 3), dtype=np.float32)
                min_score = np.full((h_canvas, w_canvas), np.inf, dtype=np.float32)
                acc = np.zeros((h_canvas, w_canvas, 3), dtype=np.float64)
                wacc = np.zeros((h_canvas, w_canvas), dtype=np.float64)
                acc_fb = np.zeros((h_canvas, w_canvas, 3), dtype=np.float64)
                wacc_fb = np.zeros((h_canvas, w_canvas), dtype=np.float64)
                weight_map_linear = _overlap_merge_weight(ph, pw, use_linear=True) if use_sliding else None
            else:
                acc = np.zeros((h_canvas, w_canvas, 3), dtype=np.float64)
                wacc = np.zeros((h_canvas, w_canvas), dtype=np.float64)

            n_patches = len(positions) if use_sliding else sum(1 for (_, _, pph, ppw) in patches if pph > 0 and ppw > 0)
            start = time.perf_counter()
            if use_sliding:
                for top, left in positions:
                    rh = min(ph, h_canvas - top)
                    rw = min(pw, w_canvas - left)
                    patch_np = work_np[top : top + rh, left : left + rw]
                    if patch_np.shape[0] < ph or patch_np.shape[1] < pw:
                        patch_pad = np.pad(patch_np, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge")
                    else:
                        patch_pad = patch_np
                    patch_pil = Image.fromarray((patch_pad * 255).astype(np.uint8))
                    x = transform(patch_pil).unsqueeze(0).to(device)
                    pred = _model_for_patch(cnn_model, ph, pw)(x)
                    pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                    pred_np = pred.numpy().transpose(1, 2, 0)
                    if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                        pred_pil = to_pil(pred)
                        pred_pil = pred_pil.resize((pw, ph), Image.BICUBIC)
                        pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                    if choose_thresholds is not None:
                        score = _choose_reverse_score(pred_np[:rh, :rw], patch_np, *choose_thresholds)
                        roi_min = min_score[top : top + rh, left : left + rw]
                        interior = _patch_interior_mask(rh, rw, edge=32)
                        mask = (score < roi_min) & interior
                        min_score[top : top + rh, left : left + rw] = np.where(mask, score, roi_min)
                        roi_out = out_img[top : top + rh, left : left + rw]
                        out_img[top : top + rh, left : left + rw] = np.where(mask[:, :, None], pred_np[:rh, :rw], roi_out)
                        wpatch = weight_map_linear[:rh, :rw]
                        pass_mask = (score < 3.0)[:, :, None]
                        acc[top : top + rh, left : left + rw] += np.where(pass_mask, pred_np[:rh, :rw] * wpatch[:, :, None], 0.0)
                        wacc[top : top + rh, left : left + rw] += np.where(pass_mask[:, :, 0], wpatch, 0.0)
                        acc_fb[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                        wacc_fb[top : top + rh, left : left + rw] += wpatch
                    else:
                        wpatch = weight_map[:rh, :rw]
                        acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                        wacc[top : top + rh, left : left + rw] += wpatch
            else:
                if use_uniform_3x3:
                    merge_weights = _uniform_3x3_merge_weights(patches, patches[0][2], patches[0][3], box_eff)
                for j, (top, left, pph, ppw) in enumerate(patches):
                    if pph <= 0 or ppw <= 0:
                        continue
                    expand_h, expand_w = pph + 2 * box_eff, ppw + 2 * box_eff
                    t1 = max(0, top)
                    t2 = min(work_np.shape[0], top + expand_h)
                    l1 = max(0, left)
                    l2 = min(work_np.shape[1], left + expand_w)
                    crop = work_np[t1:t2, l1:l2]
                    ch, cw = crop.shape[:2]
                    if ch < expand_h or cw < expand_w:
                        crop = np.pad(crop, ((0, max(0, expand_h - ch)), (0, max(0, expand_w - cw)), (0, 0)), mode="edge")
                    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
                    x = transform(crop_pil).unsqueeze(0).to(device)
                    pred = _model_for_patch(cnn_model, expand_h, expand_w)(x)
                    pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                    pred_np = pred.numpy().transpose(1, 2, 0)
                    if pred_np.shape[0] != expand_h or pred_np.shape[1] != expand_w:
                        pred_pil = to_pil(pred)
                        pred_pil = pred_pil.resize((expand_w, expand_h), Image.BICUBIC)
                        pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                    place_t, place_l = top, left
                    rh = min(expand_h, h_canvas - place_t)
                    rw = min(expand_w, w_canvas - place_l)
                    input_roi_3x3 = crop[:rh, :rw]
                    if choose_thresholds is not None:
                        score = _choose_reverse_score(pred_np[:rh, :rw], input_roi_3x3, *choose_thresholds)
                        roi_min = min_score[place_t : place_t + rh, place_l : place_l + rw]
                        interior = _patch_interior_mask(rh, rw, edge=32)
                        mask = (score < roi_min) & interior
                        min_score[place_t : place_t + rh, place_l : place_l + rw] = np.where(mask, score, roi_min)
                        roi_out = out_img[place_t : place_t + rh, place_l : place_l + rw]
                        out_img[place_t : place_t + rh, place_l : place_l + rw] = np.where(mask[:, :, None], pred_np[:rh, :rw], roi_out)
                        wpatch = (merge_weights[j] if use_uniform_3x3 else _overlap_merge_weight(expand_h, expand_w, use_linear=True))[:rh, :rw]
                        pass_mask = (score < 3.0)[:, :, None]
                        acc[place_t : place_t + rh, place_l : place_l + rw] += np.where(pass_mask, pred_np[:rh, :rw] * wpatch[:, :, None], 0.0)
                        wacc[place_t : place_t + rh, place_l : place_l + rw] += np.where(pass_mask[:, :, 0], wpatch, 0.0)
                        acc_fb[place_t : place_t + rh, place_l : place_l + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                        wacc_fb[place_t : place_t + rh, place_l : place_l + rw] += wpatch
                    else:
                        wpatch = merge_weights[j] if use_uniform_3x3 else _overlap_merge_weight(expand_h, expand_w, use_linear=use_weighted_merge)
                        acc[place_t : place_t + rh, place_l : place_l + rw] += pred_np[:rh, :rw] * wpatch[:rh, :rw, None]
                        wacc[place_t : place_t + rh, place_l : place_l + rw] += wpatch[:rh, :rw]

            if use_full:
                n_patches += 1
                full_pil = Image.fromarray((work_np * 255).astype(np.uint8)).resize((eff_w, eff_h))
                x_full = transform(full_pil).unsqueeze(0).to(device)
                pred_full = _model_for_patch(cnn_model, ph, pw)(x_full)
                pred_full = torch.clamp(pred_full.squeeze(0).cpu(), 0.0, 1.0)
                pred_full_np = pred_full.numpy().transpose(1, 2, 0)
                pred_full_np = np.array(
                    Image.fromarray((pred_full_np * 255).astype(np.uint8)).resize((w_canvas, h_canvas), Image.BICUBIC)
                ).astype(np.float32) / 255.0
                if choose_thresholds is not None:
                    score_full = _choose_reverse_score(pred_full_np, work_np, *choose_thresholds)
                    mask_full = score_full < min_score
                    min_score[:] = np.where(mask_full, score_full, min_score)
                    out_img[:] = np.where(mask_full[:, :, None], pred_full_np, out_img)
                if multi_weight is not None and multi_weight > 0 and (choose_thresholds is None):
                    w_full = multi_weight * 1.0
                    acc += pred_full_np * w_full
                    wacc += w_full

            if multi_sizes:
                use_patch_stride = use_sliding and (sh_eff == -1 or sw_eff == -1)
                pv = pad_val if use_sliding else crop_t
                for mph in multi_sizes:
                    mpw = mph
                    if use_patch_stride:
                        mstride_h, mstride_w = mph, mpw
                    else:
                        mstride_h = max(1, mph - box_eff)
                        mstride_w = max(1, mpw - box_eff)
                    if use_sliding:
                        mpositions = _sliding_window_grid_centered(h_canvas, w_canvas, mph, mpw, mstride_h, mstride_w, pv, orig_h, orig_w)
                    else:
                        mpositions = _sliding_window_grid(h_canvas, w_canvas, mph, mpw, mstride_h, mstride_w)
                    if not mpositions:
                        mpositions = [(0, 0)]
                    n_patches += len(mpositions)
                    mwm = _overlap_merge_weight(mph, mpw, use_linear=(choose_thresholds is not None) or use_weighted_merge)
                    for top, left in mpositions:
                        rh = min(mph, h_canvas - top)
                        rw = min(mpw, w_canvas - left)
                        patch_np = work_np[top : top + rh, left : left + rw]
                        if patch_np.shape[0] < mph or patch_np.shape[1] < mpw:
                            patch_pad = np.pad(patch_np, ((0, mph - rh), (0, mpw - rw), (0, 0)), mode="edge")
                        else:
                            patch_pad = patch_np
                        patch_pil = Image.fromarray((patch_pad * 255).astype(np.uint8))
                        x = transform(patch_pil).unsqueeze(0).to(device)
                        pred = _model_for_patch(cnn_model, mph, mpw)(x)
                        pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                        pred_np = pred.numpy().transpose(1, 2, 0)
                        if pred_np.shape[0] != mph or pred_np.shape[1] != mpw:
                            pred_pil = to_pil(pred)
                            pred_pil = pred_pil.resize((mpw, mph), Image.BICUBIC)
                            pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                        if choose_thresholds is not None:
                            score = _choose_reverse_score(pred_np[:rh, :rw], patch_np, *choose_thresholds)
                            roi_min = min_score[top : top + rh, left : left + rw]
                            interior = _patch_interior_mask(rh, rw, edge=32)
                            mask = (score < roi_min) & interior
                            min_score[top : top + rh, left : left + rw] = np.where(mask, score, roi_min)
                            roi_out = out_img[top : top + rh, left : left + rw]
                            out_img[top : top + rh, left : left + rw] = np.where(mask[:, :, None], pred_np[:rh, :rw], roi_out)
                        if multi_weight is not None and multi_weight > 0 and (choose_thresholds is None):
                            wpatch = mwm[:rh, :rw] * multi_weight
                            acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                            wacc[top : top + rh, left : left + rw] += wpatch

            if choose_thresholds is not None:
                satisfied = min_score < 3.0
                use_choose = satisfied
                wacc_nz = np.maximum(wacc, 1e-8)
                merged_pass = (acc / wacc_nz[:, :, None]).astype(np.float32)
                wacc_fb_nz = np.maximum(wacc_fb, 1e-8)
                merged_fb = (acc_fb / wacc_fb_nz[:, :, None]).astype(np.float32)
                merged = np.where(wacc[:, :, None] > 1e-8, merged_pass, merged_fb)
                cnn_low = np.where(use_choose[:, :, None], out_img, merged)
            else:
                wacc = np.maximum(wacc, 1e-8)
                cnn_low = (acc / wacc[:, :, None]).astype(np.float32)
            cnn_low = cnn_low[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

            if use_freq:
                cnn_log = np.log1p(np.clip(cnn_low, 1e-6, None))
                output_low_mid_log = low_ratio * cnn_log + mid_ratio * mid_in
                output_linear = np.clip(np.expm1(output_low_mid_log), 0.0, 1.0)
            else:
                output_linear = np.clip(cnn_low, 0.0, 1.0)
            output_before_set = output_linear.copy() if (choose_thresholds is not None and (set_black > 0 or set_white > 0)) else None
            if choose_thresholds is not None and (set_black > 0 or set_white > 0):
                th_b, th_w, _ = choose_thresholds
                output_linear = _apply_set_black_white(output_linear, th_b, th_w, set_black, set_white)
            if resize_input is not None:
                output_linear = np.array(
                    Image.fromarray((output_linear * 255).astype(np.uint8)).resize((orig_w_save, orig_h_save), Image.BICUBIC)
                ).astype(np.float32) / 255.0
            out_no_high = output_linear.copy() if (use_freq and high_ratio != 0) else None
            if use_freq and high_ratio != 0:
                high_to_add = high_in
                if boundary_weight is not None:
                    high_to_add = _apply_boundary_to_high(high_in, img_np_for_high, boundary_weight)
                output_linear = np.clip(np.expm1(np.log1p(np.clip(output_linear, 1e-6, None)) + high_ratio * high_to_add), 0.0, 1.0)

            pred_img = Image.fromarray((output_linear * 255).astype(np.uint8))
            out_path = os.path.join(out_dir, name)
            if input_is_npy:
                out_bgr = np.clip(output_linear, 0.0, 1.0).astype(np.float32)[:, :, ::-1]
                np.save(out_path, out_bgr)
            else:
                pred_img.save(out_path)
            print("\r" + _progress_bar(idx + 1, total_files, prefix="auto ") + f"  {name}: {n_patches} 张 patch", end="", flush=True)

            do_vis = vis_dir and (
                (input_is_npy and gt_id not in vis_saved_gt_ids and len(vis_saved_gt_ids) < vis_max_images)
                or (not input_is_npy and idx < vis_max_images)
            )
            if do_vis:
                if input_is_npy:
                    vis_saved_gt_ids.add(gt_id)
                base, _ = os.path.splitext(name)
                patch_merged_by_size = []
                if vis_include_patch:
                    def _freq_merge(arr):
                        if use_freq:
                            log_term = low_ratio * np.log1p(np.clip(arr, 1e-6, None)) + mid_ratio * mid_in
                            if resize_input is None:
                                log_term += high_ratio * high_in
                            return np.clip(np.expm1(log_term), 0.0, 1.0)
                        return np.clip(arr, 0.0, 1.0)
                    cnn_low_p = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
                    if use_sliding:
                        vis_positions = _sliding_window_grid_centered(h_canvas, w_canvas, ph, pw, ph, pw, pad_val, orig_h, orig_w)
                        for top, left in vis_positions:
                            rh, rw = min(ph, h_canvas - top), min(pw, w_canvas - left)
                            patch_np = work_np[top : top + rh, left : left + rw]
                            if patch_np.shape[0] < ph or patch_np.shape[1] < pw:
                                patch_np = np.pad(patch_np, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge")
                            pred_t = torch.clamp(_model_for_patch(cnn_model, ph, pw)(transform(Image.fromarray((patch_np * 255).astype(np.uint8))).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0)
                            pred_np = pred_t.numpy().transpose(1, 2, 0)
                            if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                                pred_np = np.array(to_pil(pred_t).resize((pw, ph), Image.BICUBIC)).astype(np.float32) / 255.0
                            o_top, o_left = max(0, top - pad_val), max(0, left - pad_val)
                            o_bottom = min(orig_h, top - pad_val + ph)
                            o_right = min(orig_w, left - pad_val + pw)
                            pr_t, pr_l = o_top - (top - pad_val), o_left - (left - pad_val)
                            pr_b, pr_r = pr_t + (o_bottom - o_top), pr_l + (o_right - o_left)
                            cnn_low_p[o_top:o_bottom, o_left:o_right] = pred_np[pr_t:pr_b, pr_l:pr_r]
                        patch_merged_by_size.append((str(ph), _freq_merge(cnn_low_p)))
                    elif use_uniform_3x3:
                        order = sorted(range(9), key=lambda i: (i // 3 - 1) ** 2 + (i % 3 - 1) ** 2)
                        pph, ppw = patches[0][2], patches[0][3]
                        cnn_low_p = np.zeros((3 * pph, 3 * ppw, 3), dtype=np.float32)
                        for j, (top, left, _, _) in enumerate(patches):
                            patch_np = work_np[top : top + pph, left : left + ppw]
                            pred_t = torch.clamp(_model_for_patch(cnn_model, pph, ppw)(transform(Image.fromarray((patch_np * 255).astype(np.uint8))).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0)
                            pred_np = pred_t.numpy().transpose(1, 2, 0)
                            if pred_np.shape[0] != pph or pred_np.shape[1] != ppw:
                                pred_np = np.array(to_pil(pred_t).resize((ppw, pph), Image.BICUBIC)).astype(np.float32) / 255.0
                            r, c = order[j] // 3, order[j] % 3
                            cnn_low_p[r * pph : (r + 1) * pph, c * ppw : (c + 1) * ppw] = pred_np[:pph, :ppw]
                        patch_merged_by_size.append(("3x3", _freq_merge(cnn_low_p)))
                    else:
                        for top, left, pph, ppw in patches:
                            if pph <= 0 or ppw <= 0:
                                continue
                            patch_np = img_np[top : top + pph, left : left + ppw]
                            pred_t = torch.clamp(_model_for_patch(cnn_model, pph, ppw)(transform(Image.fromarray((patch_np * 255).astype(np.uint8))).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0)
                            pred_np = pred_t.numpy().transpose(1, 2, 0)
                            if pred_np.shape[0] != pph or pred_np.shape[1] != ppw:
                                pred_np = np.array(to_pil(pred_t).resize((ppw, pph), Image.BICUBIC)).astype(np.float32) / 255.0
                            cnn_low_p[top : top + pph, left : left + ppw] = pred_np[:pph, :ppw]
                        patch_merged_by_size.append(("3x3", _freq_merge(cnn_low_p)))
                    if use_full:
                        full_pil = Image.fromarray((work_np * 255).astype(np.uint8)).resize((eff_w, eff_h))
                        full_p = torch.clamp(_model_for_patch(cnn_model, eff_h, eff_w)(transform(full_pil).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0).numpy().transpose(1, 2, 0)
                        full_p = np.array(Image.fromarray((full_p * 255).astype(np.uint8)).resize((w_canvas, h_canvas), Image.BICUBIC)).astype(np.float32) / 255.0
                        full_p = full_p[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]
                        patch_merged_by_size.append(("full", _freq_merge(full_p)))
                    if multi_sizes:
                        use_patch_stride = use_sliding and (sh_eff == -1 or sw_eff == -1)
                        pv = pad_val if use_sliding else crop_t
                        for mph in multi_sizes:
                            mpw = mph
                            mstride_h = mph if use_patch_stride else max(1, mph - box_eff)
                            mstride_w = mpw if use_patch_stride else max(1, mpw - box_eff)
                            mpos = _sliding_window_grid_centered(h_canvas, w_canvas, mph, mpw, mstride_h, mstride_w, pv, orig_h, orig_w) if use_sliding else _sliding_window_grid(h_canvas, w_canvas, mph, mpw, mstride_h, mstride_w)
                            acc_p = np.zeros((h_canvas, w_canvas, 3), dtype=np.float64)
                            wacc_p = np.zeros((h_canvas, w_canvas), dtype=np.float64)
                            wm = _overlap_merge_weight(mph, mpw, use_linear=True)
                            for top, left in mpos:
                                rh, rw = min(mph, h_canvas - top), min(mpw, w_canvas - left)
                                patch_np = work_np[top : top + rh, left : left + rw]
                                patch_pad = np.pad(patch_np, ((0, mph - rh), (0, mpw - rw), (0, 0)), mode="edge") if (rh < mph or rw < mpw) else patch_np
                                pred_t = torch.clamp(_model_for_patch(cnn_model, mph, mpw)(transform(Image.fromarray((patch_pad * 255).astype(np.uint8))).unsqueeze(0).to(device)).squeeze(0).cpu(), 0.0, 1.0)
                                pred_np = pred_t.numpy().transpose(1, 2, 0)
                                if pred_np.shape[0] != mph or pred_np.shape[1] != mpw:
                                    pred_np = np.array(to_pil(pred_t).resize((mpw, mph), Image.BICUBIC)).astype(np.float32) / 255.0
                                wpatch = wm[:rh, :rw]
                                acc_p[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wpatch[:, :, None]
                                wacc_p[top : top + rh, left : left + rw] += wpatch
                            wacc_p = np.maximum(wacc_p, 1e-8)
                            mp = (acc_p / wacc_p[:, :, None]).astype(np.float32)[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]
                            patch_merged_by_size.append((str(mph), _freq_merge(mp)))
                    if choose_thresholds is not None:
                        th_b, th_w, th_r = choose_thresholds
                        if output_before_set is not None:
                            pred = np.clip(output_before_set.astype(np.float64), 0.0, 1.0)
                            sb = np.mean(pred, axis=2).astype(np.float32)
                            sw = np.mean(1.0 - pred, axis=2).astype(np.float32)
                            var_score = np.var(pred, axis=2).astype(np.float32)
                            var_th_b = vis_choose_black_var if vis_choose_black_var is not None else set_black
                            mask_black = (sb < th_b) & (var_score < var_th_b) if var_th_b > 0 and th_b > 0 else (sb < th_b)
                            mask_white = ((sb >= th_b) & (sw < th_w) & (var_score < set_white)) if set_white > 0 and th_w > 0 else ((sb >= th_b) & (sw < th_w))
                            out_crop_roi = out_img[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]
                            work_crop = work_np[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]
                            _, _, sr = _choose_reverse_scores(out_crop_roi, work_crop)
                        else:
                            sb, sw, sr = _choose_reverse_scores(out_img, work_np)
                            mask_black = sb < th_b
                            mask_white = (sb >= th_b) & (sw < th_w)
                            out_crop_roi = out_img[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w]
                        mask_reverse = (sb >= th_b) & (sw >= th_w) & (sr < th_r)
                        out_crop = out_crop_roi
                        for mask, label in [(mask_black, "ChooseReverse-黑"), (mask_white, "ChooseReverse-白"), (mask_reverse, "ChooseReverse-反色")]:
                            m = mask[crop_t : crop_t + orig_h, crop_l : crop_l + orig_w] if mask.shape != (orig_h, orig_w) else mask
                            vis = np.where(m[:, :, None], out_crop, 0.5)
                            patch_merged_by_size.append((label, np.clip(vis.astype(np.float32), 0.0, 1.0)))
                save_vis_components(
                    img, pred_img, cnn_low,
                    os.path.join(vis_dir, f"sample_{idx:04d}_{base}.png"),
                    patch_merged_by_size=patch_merged_by_size if patch_merged_by_size else None,
                    vis_boundary=vis_boundary,
                    vis_freq_decompose=(low_in, mid_in, high_in) if use_freq else None,
                    vis_output_no_high=out_no_high,
                )

    if total_files:
        print()
    return total_time / total_files


def run_inference_and_save_images(
    cnn_model: CNNImageRegressor,
    input_dir: str,
    out_dir: str,
    img_size: Union[int, Tuple[int, int]],
    device: torch.device,
    vis_dir: Optional[str] = None,
    vis_max_images: int = 20,
    max_files: Optional[int] = None,
    vis_boundary: bool = False,
    sigma_low: Optional[float] = None,
    sigma_mid: Optional[float] = None,
    low_ratio: float = 1.0,
    mid_ratio: float = 1.0,
    high_ratio: float = 1.0,
    resize_input: Optional[Tuple[int, int]] = None,
    boundary_weight: Optional[float] = None,
) -> float:
    """
    三频分解推理：输入整图送 CNN（low），中频直出 mid，高频用输入 high。
    合成 low_ratio*CNN + mid_ratio*mid + high_ratio*high。
    resize_input 时：先 resize 再推理，结果 resize 回原图后加入高频。
    """
    os.makedirs(out_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    eff_h, eff_w = _img_size_hw(img_size)
    transform = T.Compose([T.Resize((eff_h, eff_w)), T.ToTensor()])
    to_pil = T.ToPILImage()

    files = sorted([f for f in os.listdir(input_dir) if is_supported_input_file(f)])
    if len(files) == 0:
        raise RuntimeError(f"输入目录 {input_dir} 中没有找到图像/.npy 文件。")
    if max_files is not None:
        files = files[:max_files]
    total_files = len(files)
    print(f"共 {total_files} 张图待处理")

    total_time = 0.0
    vis_saved_gt_ids = set()
    with torch.no_grad():
        for idx, name in enumerate(files):
            in_path = os.path.join(input_dir, name)
            img, img_np = load_input_as_rgb(in_path, npy_is_bgr=True)
            orig_w, orig_h = img.size
            orig_h_save, orig_w_save = orig_h, orig_w
            img_np_for_high = img_np.copy()
            input_is_npy = name.lower().endswith(".npy")
            gt_id = _gt_id_from_npy_name(name) if input_is_npy else ""

            start = time.perf_counter()
            if sigma_low is not None and sigma_mid is not None:
                s_low, s_mid = sigma_low, sigma_mid
            else:
                s_low, s_mid = sigma_from_image_freq(orig_h, orig_w, verbose=False)
                print(f"  [{idx+1}/{total_files}] {name} {orig_h_save}x{orig_w_save} sigma_low={s_low:.6f} sigma_mid={s_mid:.6f}")

            if resize_input is not None:
                img_log_orig = np.log1p(np.clip(img_np, 0, None))
                _, _, high_in = decompose_freq_log(img_log_orig, s_low, s_mid)
                resize_w, resize_h = resize_input
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil = img_pil.resize((resize_w, resize_h), Image.BICUBIC)
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                orig_h, orig_w = resize_h, resize_w
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save}) resize(w,h)=({resize_w},{resize_h})")
            else:
                print(f"  {name} 原图(w,h)=({orig_w_save},{orig_h_save})")

            img_log = np.log1p(np.clip(img_np, 0, None))
            low_in, mid_in, high_in_r = decompose_freq_log(img_log, s_low, s_mid)
            if resize_input is None:
                high_in = high_in_r

            img_for_cnn = Image.fromarray((img_np * 255).astype(np.uint8))
            x = transform(img_for_cnn).unsqueeze(0).to(device)
            pred = _model_for_patch(cnn_model, eff_h, eff_w)(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

            pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
            pred_np = pred.numpy().transpose(1, 2, 0)
            if pred_np.shape[0] != orig_h or pred_np.shape[1] != orig_w:
                pred_pil = to_pil(pred)
                pred_pil = pred_pil.resize((orig_w, orig_h), Image.BICUBIC)
                pred_np = np.array(pred_pil).astype(np.float32) / 255.0
            cnn_log = np.log1p(np.clip(pred_np, 1e-6, None))

            mid_log = mid_in

            output_log = low_ratio * cnn_log + mid_ratio * mid_log
            output_linear = np.clip(np.expm1(output_log), 0.0, 1.0)
            if resize_input is not None:
                output_linear = np.array(
                    Image.fromarray((output_linear * 255).astype(np.uint8)).resize((orig_w_save, orig_h_save), Image.BICUBIC)
                ).astype(np.float32) / 255.0
            if high_ratio != 0:
                high_to_add = high_in
                if boundary_weight is not None:
                    high_to_add = _apply_boundary_to_high(high_in, img_np_for_high, boundary_weight)
                output_linear = np.clip(np.expm1(np.log1p(np.clip(output_linear, 1e-6, None)) + high_ratio * high_to_add), 0.0, 1.0)
            pred_img = Image.fromarray((output_linear * 255).astype(np.uint8))

            out_path = os.path.join(out_dir, name)
            if input_is_npy:
                out_bgr = np.clip(output_linear, 0.0, 1.0).astype(np.float32)[:, :, ::-1]
                np.save(out_path, out_bgr)
            else:
                pred_img.save(out_path)

            print("\r" + _progress_bar(idx + 1, total_files, prefix="推理 ") + f"  {name}: 1 张 patch", end="", flush=True)

            do_vis = vis_dir and (
                (input_is_npy and gt_id not in vis_saved_gt_ids and len(vis_saved_gt_ids) < vis_max_images)
                or (not input_is_npy and idx < vis_max_images)
            )
            if do_vis:
                if input_is_npy:
                    vis_saved_gt_ids.add(gt_id)
                base, _ = os.path.splitext(name)
                out_no_high = np.clip(np.expm1(low_ratio * cnn_log + mid_ratio * mid_log), 0.0, 1.0) if high_ratio != 0 else None
                save_vis_components(
                    img, pred_img, pred_np,
                    os.path.join(vis_dir, f"sample_{idx:04d}_{base}.png"),
                    vis_boundary=vis_boundary,
                    vis_freq_decompose=(low_in, mid_in, high_in),
                    vis_output_no_high=out_no_high,
                )

    if total_files:
        print()
    return total_time / total_files


def _merge_multi_img_size_outputs(
    tmp_base: str,
    distinct_sizes: List[int],
    out_dir: str,
    run_fn,
    vis_dir: Optional[str] = None,
    vis_max_images: int = 0,
    input_dir: Optional[str] = None,
    stride_info: Optional[str] = None,
) -> float:
    """
    按多种 img_size 分别推理到临时目录，再平均合并到 out_dir。
    run_fn(sz, sub_dir) 接收 patch_size 和输出目录，返回平均每张图耗时。
    若 vis_dir 非空：在删除临时目录前生成可视化，子图标题与推理参数对应（patch=img=sz, stride 等）。
    """
    total_time = 0.0
    n_runs = len(distinct_sizes)
    for i, sz in enumerate(distinct_sizes):
        sub_dir = os.path.join(tmp_base, f"sz_{sz}")
        os.makedirs(sub_dir, exist_ok=True)
        print(f"  [{i+1}/{n_runs}] img_size=patch={sz}")
        t = run_fn(sz, sub_dir)
        total_time += t
    os.makedirs(out_dir, exist_ok=True)
    ref_dir = os.path.join(tmp_base, f"sz_{distinct_sizes[0]}")
    file_list = sorted(
        [n for n in os.listdir(ref_dir)
         if os.path.isfile(os.path.join(ref_dir, n)) and os.path.splitext(n)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]]
    )
    for idx, name in enumerate(file_list):
        path = os.path.join(ref_dir, name)
        ext = os.path.splitext(name)[1].lower()
        arrs = []
        for sz in distinct_sizes:
            p = os.path.join(tmp_base, f"sz_{sz}", name)
            if not os.path.isfile(p):
                continue
            if ext == ".npy":
                a = np.load(p)
            else:
                img = Image.open(p).convert("RGB")
                a = np.array(img).astype(np.float32) / 255.0
            arrs.append(a)
        if not arrs:
            continue
        merged = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)
        out_path = os.path.join(out_dir, name)
        if ext == ".npy":
            np.save(out_path, merged)
        else:
            Image.fromarray((np.clip(merged, 0, 1) * 255).astype(np.uint8)).save(out_path)

        if vis_dir and vis_max_images > 0 and idx < vis_max_images and input_dir and plt is not None:
            base, _ = os.path.splitext(name)
            in_path = os.path.join(input_dir, name)
            if os.path.isfile(in_path):
                input_pil, _ = load_input_as_rgb(in_path, npy_is_bgr=True)
                merged_vis = merged[:, :, ::-1] if ext == ".npy" else merged
                pred_pil = Image.fromarray((np.clip(merged_vis, 0, 1) * 255).astype(np.uint8))
                patch_merged_by_size = []
                for sz in distinct_sizes:
                    p = os.path.join(tmp_base, f"sz_{sz}", name)
                    if os.path.isfile(p) and ext != ".npy":
                        a = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
                        label = f"{sz}" if stride_info is None else f"{sz}(stride={stride_info})"
                        patch_merged_by_size.append((label, a))
                patch_merged_by_size.append(("merged", np.clip(merged_vis, 0, 1)))
                os.makedirs(vis_dir, exist_ok=True)
                save_vis_components(
                    input_pil, pred_pil, merged_vis,
                    os.path.join(vis_dir, f"sample_{idx:04d}_{base}.png"),
                    patch_merged_by_size=patch_merged_by_size,
                )

    try:
        shutil.rmtree(tmp_base)
    except OSError:
        pass
    return total_time / n_runs


def write_readme(
    readme_path: str,
    runtime_per_image: float,
    device: torch.device,
    extra_data: int = 0,
    description: str = "CNN+freq: input_low->CNN, mid/high direct, output=low+mid+high. https://github.com/yeebowang/Staged-CNN",
) -> None:
    cpu_flag = 1 if device.type == "cpu" else 0
    with open(readme_path, "w") as f:
        f.write(f"runtime per image [s] : {runtime_per_image:.4f}\n")
        f.write(f"CPU[1] / GPU[0] : {cpu_flag}\n")
        f.write(f"Extra Data [1] / No Extra Data [0] : {extra_data}\n")
        f.write(f"Other description : {description}\n")


def make_submission_zip(out_dir: str, zip_path: str) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(out_dir)):
            path = os.path.join(out_dir, name)
            if not os.path.isfile(path):
                continue
            zf.write(path, arcname=os.path.basename(path))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="NTIRE CNN 三频提交（中频直出），从 config.txt 读超参，GPU 推理"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="F:/Code/Datasets/NTIRE2026/color_track/test_step1",
        help="测试图像目录",
    )
    parser.add_argument(
        "--cnn_checkpoint",
        type=str,
        default="best",
        help="CNN checkpoint 目录、.pth 路径，或 'best'/'latest'",
    )
    parser.add_argument(
        "--img_size",
        type=str,
        default=None,
        help="手动指定 CNN 输入尺寸（整数或 宽,高 如 544,416），优先于 config.txt",
    )
    parser.add_argument("--output_zip", type=str, default=None)
    parser.add_argument("--tmp_out_dir", type=str, default="./submission_outputs")
    parser.add_argument("--vis_dir", type=str, default="./vis_submission")
    parser.add_argument("--vis_max_images", type=int, default=20)
    parser.add_argument("--sigma_low", type=float, default=None)
    parser.add_argument("--sigma_mid", type=float, default=None)
    parser.add_argument("--low_ratio", type=float, default=1.0)
    parser.add_argument("--mid_ratio", type=float, default=0.0, help="默认 0：纯 CNN，与 train_patch 整图验证一致；>0 启用三频")
    parser.add_argument("--high_ratio", type=float, default=0.0)
    parser.add_argument(
        "--path_remap",
        type=str,
        nargs=2,
        default=["F:", "I:"],
        metavar=("OLD", "NEW"),
        help="路径替换 F: -> I:，默认 ['F:', 'I:']",
    )
    parser.add_argument("--same_as_train_patch", action="store_true",
        help="强制使用 train_patch 整图验证相同方法：patch=256 stride=224 左上角网格、线性权重、无 padding")
    parser.add_argument("--sliding_window", action="store_true", help="使用滑动窗口推理")
    parser.add_argument("--tile_no_overlap", action="store_true", help="无重叠铺满：从 patch 中挑选覆盖原图的子集，直接铺满，stride=patch_size")
    parser.add_argument("--no_weighted_merge", action="store_true", help="禁用线性权重，重叠区用均匀平均；默认使用线性衰减权重")
    parser.add_argument("--patch_size", type=str, default="256", help="patch 尺寸：整数或 宽,高 如 544,416")
    parser.add_argument("--patch_h", type=int, default=None, help="patch 高，未指定则用 patch_size")
    parser.add_argument("--patch_w", type=int, default=None, help="patch 宽，未指定则用 patch_size")
    parser.add_argument("--stride", type=str, default=None, help="步长：整数或 宽,高 如 512,384；-1 表示 stride=patch 无重叠")
    parser.add_argument("--stride_h", type=int, default=None, help="高方向 stride，与 --stride 同理；-1 表示用 patch 高")
    parser.add_argument("--stride_w", type=int, default=None, help="宽方向 stride，与 --stride 同理；-1 表示用 patch 宽")
    parser.add_argument("--box", type=int, default=None, help="重叠 context；auto_patch 3x3 未给定时用自适应 uniform 模式")
    parser.add_argument("--pad_lines", type=int, default=32, help="padding 边缘带状厚度，默认 32；有 --pad_flip 时失效")
    parser.add_argument("--pad_flip", action="store_true", help="padding 以原图边缘为轴对称反转，四角以角为中心的中心对称；启用时 pad_lines 失效")
    parser.add_argument("--padding", type=int, default=None, help="图像四周边界 padding：值为每维总宽度，上下左右各 pad N//2；如 32 表示上下左右各 16。0=不 padding；默认=box")
    parser.add_argument("--demo", nargs="?", type=int, const=3, default=None, metavar="N",
        help="Demo 模式：仅推理前 N 张，不输出 zip，可视化到 vis_sub_demo；默认 --demo 为 3 张，如 --demo 5")
    parser.add_argument("--auto_patch", nargs="?", type=float, const=0.25, default=None, metavar="RATIO",
        help="3x3/滑动分块：中心为首个，中心 patch 等比例且面积为原图 RATIO；未给定值时按 0.25，如 --auto_patch 0.3")
    parser.add_argument("--uniform", action="store_true", help="强行进入 uniform 模式；此时可用 --box 指定 box 值")
    parser.add_argument("--choose_black", type=str, default="0", metavar="TH[,VAR]",
        help="choose 模式：TH 为距离阈值；可加 TH,VAR 如 0.3,0.03 表示 ChooseBlack 画图用方差 < 0.03；与纯黑平均的方差由 --set_black 决定")
    parser.add_argument("--choose_white", type=float, default=0, metavar="TH",
        help="choose 模式：白分数阈值，score_w < TH 则取白；默认 0 禁用；三档相互独立")
    parser.add_argument("--choose_reverse", type=float, default=0, metavar="TH",
        help="choose 模式：反色分数阈值，score_r < TH 则取反色；默认 0 禁用；三档参数相互独立")
    parser.add_argument("--set_black", type=float, default=0, metavar="TH",
        help="choose 模式：满足距离且 RGB 方差 < TH 的像素与纯黑做平均；0 禁用；与 ChooseBlack 画图用的方差（来自 --choose_black TH,VAR 的 VAR）独立")
    parser.add_argument("--set_white", type=float, default=0, metavar="TH",
        help="choose 模式：满足 choose_white 且 RGB 方差 < TH 的像素与纯白做平均；0 禁用")
    parser.add_argument("--full", action="store_true", default=False,
        help="将原图整体作为一个 patch 参与合并，可用于 --choose_reverse 或线性权重；默认 false")
    parser.add_argument("--multi_size", type=str, default=None, metavar="SIZES",
        help="多种 patch 尺寸，逗号分隔，如 480,128；在原有 patch_size 和 full 之外再推理这些尺寸的 patch，参与 choose_reverse；线性加权需 --multi_weight")
    parser.add_argument("--multi_weight", type=float, default=None, metavar="W",
        help="辅助 patch（full、multi_size）参与线性加权时的权重系数，需乘以其应有的线性权重；未指定则不参与线性加权")
    parser.add_argument("--auto_img_size", action="store_true",
        help="img_size 随 patch 改变（=patch 大小）；多种 patch 时按多种 img_size 分别推理，再平均合并")
    parser.add_argument("--resize_input", type=str, default=None, metavar="W,H",
        help="将输入 resize 到宽 W 高 H 后推理，推理完成后再 resize 回原图；高频加入在 resize 回原图之后。如 --resize_input 512,384")
    parser.add_argument("--boundary", type=float, default=None, metavar="W",
        help="以 boundary 图过滤高频：log 域 log_high = W*log_boundary+log_high，再参与最后高频叠加；不设则不过滤。如 --boundary 1.0")
    parser.add_argument("--lock_l", action="store_true",
        help="记录输入图像的 L 通道，推理后用「输入 L + 预测 AB」合成最终图像，并可视化 Input L / Pred AB / LockL 合成过程")

    args = parser.parse_args()

    # 解析 宽,高 格式：img_size / stride / patch_size
    args._img_size = None  # int 或 (height, width) 供 T.Resize
    if args.img_size is not None:
        wh = _parse_wh(args.img_size, allow_negative=False)
        if wh is not None:
            if isinstance(wh, tuple):
                args._img_size = (wh[1], wh[0])  # 用户 宽,高 -> (H, W) 供 Resize
            else:
                args._img_size = wh

    args._stride_wh = None  # (stride_w, stride_h) 或 None，与 args.stride 二选一
    if getattr(args, "stride", None) is not None:
        swh = _parse_wh(args.stride, allow_negative=True)
        if isinstance(swh, tuple):
            args._stride_wh = swh
            args.stride = None
            args.stride_w, args.stride_h = swh[0], swh[1]
        elif swh is not None:
            args.stride = int(swh)
            args._stride_wh = None

    # patch_size 字符串 -> patch_w, patch_h 或单个 patch_size
    _ps = _parse_wh(args.patch_size, allow_negative=False)
    if _ps is not None:
        if isinstance(_ps, tuple):
            args.patch_w, args.patch_h = _ps[0], _ps[1]
            args.patch_size = _ps[0]  # distinct_patch_sizes 用第一个数
        else:
            args.patch_size = _ps
            if args.patch_h is None:
                args.patch_h = args.patch_size
            if args.patch_w is None:
                args.patch_w = args.patch_size

    args._multi_sizes = None
    if args.multi_size:
        parsed = [int(x.strip()) for x in args.multi_size.split(",") if x.strip()]
        if parsed:
            args._multi_sizes = parsed

    args._resize_input = None  # (width, height)
    if args.resize_input:
        parts = [x.strip() for x in args.resize_input.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError("--resize_input 需要 W,H 格式，如 512,384")
        args._resize_input = (int(parts[0]), int(parts[1]))
        print(f"resize_input: 推理时先 resize 到 {args._resize_input[0]}x{args._resize_input[1]}，再 resize 回原图；高频在 resize 之后加入")

    if args.demo is not None:
        args.vis_dir = "./vis_sub_demo"
        args._demo_max_files = args.demo
        args._demo_skip_zip = True
    else:
        args._demo_max_files = None
        args._demo_skip_zip = False

    if args.sigma_low is not None and args.sigma_mid is not None and args.sigma_mid >= args.sigma_low:
        raise ValueError("请设置 sigma_mid < sigma_low")

    # 解析 --choose_black：支持 "TH" 或 "TH,VAR"，VAR 为 ChooseBlack 可视化的方差阈值；set_black 为与纯黑做平均的方差阈值
    _cb_parts = [x.strip() for x in str(args.choose_black).split(",") if x.strip()]
    args._choose_black_th = float(_cb_parts[0]) if _cb_parts else 0.0
    args._choose_black_var = float(_cb_parts[1]) if len(_cb_parts) >= 2 else args.set_black  # 仅用于 ChooseBlack 画图

    args._choose_thresholds = None
    if args._choose_black_th > 0 or args.choose_white > 0 or args.choose_reverse > 0:
        # 三者相互独立，各自使用指定值；0 表示禁用该档（score ∈ [0,1]，th=0 时永不匹配）
        args._choose_thresholds = (args._choose_black_th, args.choose_white, args.choose_reverse)

    path_remap = tuple(args.path_remap) if args.path_remap and args.path_remap[0] else None
    if path_remap:
        print(f"路径替换: {path_remap[0]!r} -> {path_remap[1]!r}")

    args.input_dir = _remap_path(args.input_dir, path_remap)
    args.cnn_checkpoint = _remap_path(args.cnn_checkpoint, path_remap)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 GPU，推理需要 CUDA。请确认 PyTorch GPU 版与驱动正常。"
        )
    device = torch.device("cuda")
    print(f"使用 GPU 推理: {torch.cuda.get_device_name(0)}")

    project_root = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir, best_cnn_path, stage_paths = resolve_checkpoint_dir(args.cnn_checkpoint, project_root)
    if stage_paths:
        print(f"检测到 train_patch 三 stage，加载: {list(stage_paths.keys())} 对应 patch 尺寸")
        cnn_model, img_size = load_three_stages(checkpoint_dir, stage_paths, device)
        print(f"CNN checkpoint: 三 stage 128/192/256 -> {list(stage_paths.values())}")
    else:
        print(f"CNN checkpoint: {best_cnn_path}")
        cnn_model, img_size = load_cnn_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            best_model_path=best_cnn_path,
            device=device,
        )
    if args._img_size is not None:
        img_size = args._img_size
        print(f"使用手动指定 img_size={img_size}")
    else:
        print(f"img_size={img_size} (来自 checkpoint/config.txt 或默认 256)")

    auto_img_size = getattr(args, "auto_img_size", False)
    distinct_patch_sizes = sorted(set([args.patch_size] + (args._multi_sizes or [])))
    if auto_img_size:
        if len(distinct_patch_sizes) == 1:
            img_size = distinct_patch_sizes[0]
            args._multi_sizes = []
            print(f"auto_img_size: 单一 patch 尺寸 {img_size}，img_size={img_size}")
        else:
            print(f"auto_img_size: 多种 patch 尺寸 {distinct_patch_sizes}，将按各 img_size 分别推理并平均合并")
    use_freq = args.mid_ratio != 0 or args.high_ratio != 0
    print(f"合成系数 low={args.low_ratio} mid={args.mid_ratio} high={args.high_ratio}" + ("（纯 CNN，与 train_patch 整图验证一致）" if not use_freq else "（三频合成）"))
    if args._choose_thresholds is not None:
        tb, tw, tr = args._choose_thresholds
        print(f"choose 阈值: 黑={tb} 白={tw} 反色={tr}（优先黑>白>AB反色，不满足的像素用线性权重）")
    if args.full:
        print("full 模式: 原图整体作为一个 patch 参与合并")
    if args._multi_sizes:
        print(f"multi_size 模式: 额外 patch 尺寸 {args._multi_sizes}")
    if args.multi_weight is not None and args.multi_weight > 0:
        print(f"multi_weight={args.multi_weight}: 辅助 patch 参与线性加权")
    if args.auto_patch is not None:
        use_uni = args.uniform or args.box is None
        box_info = f"box={args.box}" if (use_uni and args.box is not None) else ("uniform 自适应" if use_uni else f"box={args.box}")
        wm_str = "失效" if args._choose_thresholds is not None else str(not args.no_weighted_merge)
        print(f"3x3 分块模式: 中心为首个，中心 patch 等比例面积={args.auto_patch}，{box_info}，weighted_merge={wm_str}")
    elif args.sliding_window:
        stride_info = "patch-box" if args.stride is None else ("patch(无重叠)" if args.stride == -1 else f"stride={args.stride}")
        mode = "tile_no_overlap" if args.tile_no_overlap else (f"choose(b/w/r)" if args._choose_thresholds is not None else f"{stride_info} weighted_merge={not args.no_weighted_merge}")
        ph = args.patch_h if args.patch_h is not None else args.patch_size
        pw = args.patch_w if args.patch_w is not None else args.patch_size
        patch_str = f"{ph}x{pw}" if (ph != pw) else str(args.patch_size)
        print(f"滑动窗口: patch={patch_str} {mode}")
    if args.demo:
        print(f"Demo 模式: 仅前 {args.demo} 张，vis=vis_sub_demo，不打包 zip")
    print("开始推理...")
    # 仅 --auto_patch 时走 auto_patch；--uniform 单独使用（无 --auto_patch）不进入
    use_auto_patch = args.auto_patch is not None
    multi_img_size_mode = auto_img_size and len(distinct_patch_sizes) > 1

    if multi_img_size_mode:
        import tempfile
        tmp_base = os.path.join(tempfile.gettempdir(), f"ntire_auto_imgsz_{os.getpid()}")
        if use_auto_patch:
            def _run_ap_sz(sz: int, sub_dir: str):
                return run_inference_auto_patch(
                    cnn_model=cnn_model,
                    input_dir=args.input_dir,
                    out_dir=sub_dir,
                    img_size=sz,
                    device=device,
                    box=args.box,
                    center_ratio=args.auto_patch if args.auto_patch is not None else 0.25,
                    stride=args.stride,
                    stride_h=args.stride_h,
                    stride_w=args.stride_w,
                    patch_size=sz,
                    patch_h=sz,
                    patch_w=sz,
                    multi_sizes=[],
                    use_weighted_merge=False if args._choose_thresholds is not None else (not args.no_weighted_merge),
                    choose_thresholds=args._choose_thresholds,
                    set_black=args.set_black,
                    set_white=args.set_white,
                    vis_choose_black_var=args._choose_black_var,
                    use_full=args.full,
                    multi_weight=args.multi_weight,
                    vis_dir=None,
                    vis_max_images=0,
                    max_files=args._demo_max_files,
                    sigma_low=args.sigma_low,
                    sigma_mid=args.sigma_mid,
                    low_ratio=args.low_ratio,
                    mid_ratio=args.mid_ratio,
                    high_ratio=args.high_ratio,
                    vis_include_patch=False,
                    vis_boundary=False,
                    pad_lines=args.pad_lines,
                    pad_flip=args.pad_flip,
                    uniform=args.uniform,
                    resize_input=args._resize_input,
                    boundary_weight=args.boundary,
                )
            stride_str = None
            if args.stride is not None:
                stride_str = str(args.stride) if args.stride != -1 else "patch"
            avg_time = _merge_multi_img_size_outputs(
                tmp_base, distinct_patch_sizes, args.tmp_out_dir,
                run_fn=_run_ap_sz,
                vis_dir=args.vis_dir if args.demo else None,
                vis_max_images=args.vis_max_images if args.demo else 0,
                input_dir=args.input_dir,
                stride_info=stride_str,
            )
        else:
            sw_box = args.box if args.box is not None else 32
            sw_pad_flip = args.pad_flip
            if args.uniform and args.patch_size == 224 and sw_box == 224 and args.stride == 224 and args.padding == 224 and sw_pad_flip:
                sw_box, sw_pad_flip = 0, True
            def _run_sw_sz(sz: int, sub_dir: str):
                return run_inference_sliding_window(
                    cnn_model=cnn_model,
                    input_dir=args.input_dir,
                    out_dir=sub_dir,
                    img_size=sz,
                    device=device,
                    patch_size=sz,
                    patch_h=sz,
                    patch_w=sz,
                    stride=args.stride if args.stride != -1 else -1,
                    box=sw_box,
                    padding=args.padding,
                    use_weighted_merge=False if args._choose_thresholds is not None else (not args.no_weighted_merge),
                    tile_no_overlap=args.tile_no_overlap,
                    choose_thresholds=args._choose_thresholds,
                    set_black=args.set_black,
                    set_white=args.set_white,
                    vis_choose_black_var=args._choose_black_var,
                    use_full=args.full,
                    multi_sizes=[],
                    multi_weight=args.multi_weight,
                    vis_include_patch=False,
                    vis_boundary=False,
                    vis_dir=None,
                    vis_max_images=0,
                    max_files=args._demo_max_files,
                    pad_lines=args.pad_lines,
                    pad_flip=sw_pad_flip,
                    sigma_low=args.sigma_low,
                    sigma_mid=args.sigma_mid,
                    low_ratio=args.low_ratio,
                    mid_ratio=args.mid_ratio,
                    high_ratio=args.high_ratio,
                    resize_input=args._resize_input,
                    use_train_patch_style=args.same_as_train_patch,
                    boundary_weight=args.boundary,
                    lock_l=args.lock_l,
                )
            stride_str = None
            if args.stride is not None and args.stride != -1:
                stride_str = str(args.stride)
            elif args.stride == -1:
                stride_str = "patch"
            else:
                stride_str = "patch-box"
            avg_time = _merge_multi_img_size_outputs(
                tmp_base, distinct_patch_sizes, args.tmp_out_dir,
                run_fn=_run_sw_sz,
                vis_dir=args.vis_dir if args.demo else None,
                vis_max_images=args.vis_max_images if args.demo else 0,
                input_dir=args.input_dir,
                stride_info=stride_str,
            )
    elif use_auto_patch:
        center_ratio = args.auto_patch if args.auto_patch is not None else 0.25
        avg_time = run_inference_auto_patch(
            cnn_model=cnn_model,
            input_dir=args.input_dir,
            out_dir=args.tmp_out_dir,
            img_size=img_size,
            device=device,
            box=args.box,
            center_ratio=center_ratio,
            stride=args.stride,
            stride_h=args.stride_h,
            stride_w=args.stride_w,
            patch_size=args.patch_size,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            use_weighted_merge=False if args._choose_thresholds is not None else (not args.no_weighted_merge),
            choose_thresholds=args._choose_thresholds,
            set_black=args.set_black,
            set_white=args.set_white,
            vis_choose_black_var=args._choose_black_var,
            use_full=args.full,
            multi_sizes=args._multi_sizes,
            multi_weight=args.multi_weight,
            vis_dir=args.vis_dir,
            vis_max_images=args.vis_max_images,
            max_files=args._demo_max_files,
            sigma_low=args.sigma_low,
            sigma_mid=args.sigma_mid,
            low_ratio=args.low_ratio,
            mid_ratio=args.mid_ratio,
            high_ratio=args.high_ratio,
            vis_include_patch=args.demo,
            vis_boundary=args.demo,
            pad_lines=args.pad_lines,
            pad_flip=args.pad_flip,
            uniform=args.uniform,
            resize_input=args._resize_input,
            boundary_weight=args.boundary,
        )
    elif args.sliding_window:
        sw_patch = args.patch_size
        sw_patch_h = args.patch_h
        sw_patch_w = args.patch_w
        sw_stride = args.stride
        sw_box = args.box if args.box is not None else 32
        sw_padding = args.padding
        sw_pad_flip = args.pad_flip
        # --sliding_window --uniform 特殊模式：patch_size+box=224, stride=224, padding=224, pad_flip
        if args.uniform and sw_patch == 224 and sw_box == 224 and sw_stride == 224 and sw_padding == 224 and sw_pad_flip:
            sw_patch = 224
            sw_box = 0
            sw_stride = 224
            sw_padding = 224
            sw_pad_flip = True
            print("sliding_window+uniform 特殊模式: patch=224, box=0, stride=224, padding=224, pad_flip")
        avg_time = run_inference_sliding_window(
            cnn_model=cnn_model,
            input_dir=args.input_dir,
            out_dir=args.tmp_out_dir,
            img_size=img_size,
            device=device,
            patch_size=sw_patch,
            patch_h=sw_patch_h,
            patch_w=sw_patch_w,
            stride=sw_stride,
            stride_h=args.stride_h,
            stride_w=args.stride_w,
            box=sw_box,
            padding=sw_padding,
            use_weighted_merge=False if args._choose_thresholds is not None else (not args.no_weighted_merge),
            tile_no_overlap=args.tile_no_overlap,
            choose_thresholds=args._choose_thresholds,
            set_black=args.set_black,
            set_white=args.set_white,
            vis_choose_black_var=args._choose_black_var,
            use_full=args.full,
            multi_sizes=args._multi_sizes,
            multi_weight=args.multi_weight,
            vis_include_patch=args.demo,
            vis_boundary=args.demo,
            vis_dir=args.vis_dir,
            vis_max_images=args.vis_max_images,
            max_files=args._demo_max_files,
            pad_lines=args.pad_lines,
            pad_flip=sw_pad_flip,
            sigma_low=args.sigma_low,
            sigma_mid=args.sigma_mid,
            low_ratio=args.low_ratio,
            mid_ratio=args.mid_ratio,
            high_ratio=args.high_ratio,
            resize_input=args._resize_input,
            use_train_patch_style=args.same_as_train_patch,
            boundary_weight=args.boundary,
            lock_l=args.lock_l,
        )
    else:
        avg_time = run_inference_and_save_images(
            cnn_model=cnn_model,
            input_dir=args.input_dir,
            out_dir=args.tmp_out_dir,
            img_size=img_size,
            device=device,
            vis_dir=args.vis_dir,
            vis_max_images=args.vis_max_images,
            max_files=args._demo_max_files,
            vis_boundary=args.demo,
            sigma_low=args.sigma_low,
            sigma_mid=args.sigma_mid,
            low_ratio=args.low_ratio,
            mid_ratio=args.mid_ratio,
            high_ratio=args.high_ratio,
            resize_input=args._resize_input,
            boundary_weight=args.boundary,
        )
    print(f"平均推理时间: {avg_time:.4f} s/图")

    readme_path = os.path.join(args.tmp_out_dir, "readme.txt")
    write_readme(readme_path, avg_time, device)

    if not args._demo_skip_zip:
        if args.output_zip is None:
            args.output_zip = f"submission_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
        n_packed = make_submission_zip(args.tmp_out_dir, args.output_zip)
        print(f"打包到: {args.output_zip}，共 {n_packed} 个文件，完成。")
    else:
        print(f"Demo 完成，可视化: {args.vis_dir}")


if __name__ == "__main__":
    main()
