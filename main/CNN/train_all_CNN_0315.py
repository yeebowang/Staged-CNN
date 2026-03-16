"""
CNN 版本训练脚本，参考 NTIRE/train.py。
使用简单 encoder-decoder CNN 做图像回归（CR -> GT），MSE Loss。
"""
import os
import sys
import time
import shutil
import argparse
from datetime import datetime
from typing import List, Tuple, Optional, Union

# 允许从上级目录引用同一数据集逻辑（可选）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision.utils import save_image

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "需要安装 matplotlib：pip install matplotlib"
    ) from e

try:
    from torchvision import models
except ImportError:
    models = None

# 三频分解（用于 --low_freq_only）
try:
    from freq_deco import decompose_freq_log
except ImportError:
    def _gaussian_blur_float(img: np.ndarray, sigma: float) -> np.ndarray:
        k = int(6 * sigma + 1) | 1
        k = max(3, min(k, 51))
        blurred = cv2.GaussianBlur(img, (k, k), sigma)
        return blurred.astype(np.float32)

    def decompose_freq_log(img_log, sigma_low, sigma_mid):
        low_log = _gaussian_blur_float(img_log, sigma_low)
        mid_blur = _gaussian_blur_float(img_log, sigma_mid)
        mid_log = (mid_blur - low_log).astype(np.float32)
        high_log = (img_log.astype(np.float32) - mid_blur).astype(np.float32)
        return low_log, mid_log, high_log


# sRGB -> XYZ -> LAB 矩阵 (D65)，在 GPU 上做色彩空间转换
_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_XYZ2RGB = torch.tensor([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=torch.float32)

_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA ** 3
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """[已弃用，保留兼容] 线性 RGB [0,1] (H,W,3) -> LAB，归一化到 [0,1]。"""
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lab_float = lab.astype(np.float32) / 255.0
    return lab_float


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """[已弃用，保留兼容] LAB [0,1] (H,W,3) -> RGB [0,1]。"""
    lab_u8 = (np.clip(lab, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.0


def rgb_to_lab_tensor(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    sRGB [0,1] (B,3,H,W) 或 (3,H,W) -> LAB 归一化，在 GPU 上执行。
    L: [-1,1] (L/50-1)，ab: [-1,1] (a/128, b/128)。
    """
    squeeze = rgb.dim() == 3
    if squeeze:
        rgb = rgb.unsqueeze(0)  # (3,H,W) -> (1,3,H,W)
    rgb = rgb.clamp(0.0, 1.0)
    # sRGB 线性化
    rgb_lin = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    # (B,3,H,W) -> (B,H,W,3) 做矩阵乘法
    B, C, H, W = rgb_lin.shape
    rgb_flat = rgb_lin.permute(0, 2, 3, 1)  # (B,H,W,3)
    m = _RGB2XYZ.to(device=device, dtype=rgb.dtype)
    xyz = torch.matmul(rgb_flat, m.T)  # (B,H,W,3)
    # XYZ -> LAB
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    x = x / _XN
    y = y / _YN
    z = z / _ZN

    def _f(t):
        return torch.where(t > _DELTA3, t.pow(1.0 / 3.0), t / (3 * _DELTA * _DELTA) + 4.0 / 29.0)

    l_ = 116.0 * _f(y) - 16.0
    a_ = 500.0 * (_f(x) - _f(y))
    b_ = 200.0 * (_f(y) - _f(z))
    # L: [-1,1] (L/50-1), ab: [-1,1] (a/128, b/128)
    l_norm = (l_.clamp(0, 100) / 50.0 - 1.0).clamp(-1.0, 1.0)
    a_norm = (a_.clamp(-128, 127) / 128.0).clamp(-1.0, 1.0)
    b_norm = (b_.clamp(-128, 127) / 128.0).clamp(-1.0, 1.0)
    lab = torch.stack([l_norm, a_norm, b_norm], dim=-1)  # (B,H,W,3)
    lab = lab.permute(0, 3, 1, 2)  # (B,3,H,W)
    if squeeze:
        lab = lab.squeeze(0)
    return lab


def lab_to_rgb_tensor(lab: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    LAB 归一化 -> sRGB [0,1]，在 GPU 上执行。
    输入：L [-1,1] (L/50-1), a [-1,1], b [-1,1]。
    """
    squeeze = lab.dim() == 3
    if squeeze:
        lab = lab.unsqueeze(0)
    # L: [-1,1] -> [0,100], ab: [-1,1] -> [-128,128]
    l_ = (lab[:, 0:1].clamp(-1.0, 1.0) + 1.0) * 50.0
    a_ = lab[:, 1:2].clamp(-1.0, 1.0) * 128.0
    b_ = lab[:, 2:3].clamp(-1.0, 1.0) * 128.0
    lab_real = torch.cat([l_, a_, b_], dim=1)  # (B,3,H,W)
    lab_flat = lab_real.permute(0, 2, 3, 1)  # (B,H,W,3)
    y = (lab_flat[..., 0] + 16.0) / 116.0
    x = lab_flat[..., 1] / 500.0 + y
    z = y - lab_flat[..., 2] / 200.0

    def _finv(t):
        return torch.where(t > _DELTA, t.pow(3.0), 3.0 * _DELTA * _DELTA * (t - 4.0 / 29.0))

    xn = torch.full_like(x, _XN, device=device, dtype=lab.dtype)
    yn = torch.full_like(y, _YN, device=device, dtype=lab.dtype)
    zn = torch.full_like(z, _ZN, device=device, dtype=lab.dtype)
    xyz = torch.stack([_finv(x) * xn, _finv(y) * yn, _finv(z) * zn], dim=-1)
    m = _XYZ2RGB.to(device=device, dtype=lab.dtype)
    rgb_lin = torch.matmul(xyz, m.T).clamp(0.0, None)
    # 线性 -> sRGB
    rgb = torch.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        1.055 * rgb_lin.pow(1.0 / 2.4) - 0.055,
    )
    rgb = rgb.clamp(0.0, 1.0).permute(0, 3, 1, 2)  # (B,3,H,W)
    if squeeze:
        rgb = rgb.squeeze(0)
    return rgb


def _gradient_loss(pred_L: torch.Tensor, gt_L: torch.Tensor) -> torch.Tensor:
    """L 通道梯度 L1：L1(grad_x) + L1(grad_y)，pred_L/gt_L 为 (B,1,H,W)。"""
    gx_p = pred_L[:, :, :, 1:] - pred_L[:, :, :, :-1]
    gy_p = pred_L[:, :, 1:, :] - pred_L[:, :, :-1, :]
    gx_t = gt_L[:, :, :, 1:] - gt_L[:, :, :, :-1]
    gy_t = gt_L[:, :, 1:, :] - gt_L[:, :, :-1, :]
    return torch.nn.functional.l1_loss(gx_p, gx_t) + torch.nn.functional.l1_loss(gy_p, gy_t)


def _psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """pred/gt [0,1] (B,C,H,W)，返回平均 PSNR (dB)，按图像平均。"""
    import math
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    mse_per_img = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
    psnr_per_img = 10.0 * torch.log10(max_val ** 2 / (mse_per_img + 1e-10))
    return float(psnr_per_img.mean().item())


def _ssim_value(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """SSIM 值 [0,1]，越高越好。pred/target (B,C,H,W) [0,1]。"""
    return 1.0 - 2.0 * _ssim_loss(pred, target).item()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM loss = (1 - SSIM) / 2，归一化到 [0,1]，输入 (B,C,H,W)，支持 1 或 3 通道。"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channel = pred.size(1)
    from math import exp
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype)
    gauss = torch.exp(-(coords - window_size // 2) ** 2 / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    win = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # (w,w)
    win = win.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(pred, win, padding=pad, groups=channel)
    mu2 = torch.nn.functional.conv2d(target, win, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(pred * pred, win, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, win, padding=pad, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, win, padding=pad, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return (1.0 - ssim_map.mean()) / 2.0  # 归一化到 [0,1]


class PerceptualLoss(nn.Module):
    """VGG16 特征 L1 loss，归一化到约 [0,1]，输入 RGB [0,1] (B,3,H,W)。"""

    def __init__(self, device: torch.device, norm_scale: float = 10.0):
        super().__init__()
        if models is None:
            raise ImportError("需要 torchvision：pip install torchvision")
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:16])  # relu3_3
        for p in self.slice1.parameters():
            p.requires_grad = False
        self.slice1.to(device)
        self.slice1.eval()
        self.norm_scale = norm_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_n = (pred - mean) / std
        target_n = (target - mean) / std
        f_pred = self.slice1(pred_n)
        f_target = self.slice1(target_n)
        return torch.nn.functional.l1_loss(f_pred, f_target) / self.norm_scale  # 归一化


def _sigma_from_image_freq(h: int, w: int) -> Tuple[float, float]:
    """按频数均分，根据图像尺寸计算 sigma_low, sigma_mid。"""
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
    return float(sigma_low), float(sigma_mid)


def _path_to_npy(path: str) -> str:
    """列表中的路径（多为 .png）转为同名的 .npy 路径。"""
    base, _ = os.path.splitext(path)
    return base + ".npy"


def _remap_path(path: str, path_remap: Tuple[str, str] = None) -> str:
    """将路径中的盘符/前缀替换，如 F: -> I:，用于跨盘迁移。"""
    if path_remap and len(path_remap) >= 2:
        return path.replace(path_remap[0], path_remap[1])
    return path


class NTIREColorTrackDatasetNpy(Dataset):
    """
    从 .npy 文件加载 CR/GT，列表文件格式与 PNG 版相同（每行 cr_path  mid  gt_path），
    实际读取时把路径扩展名改为 .npy。
    若指定 gt_dir，则 GT 统一从该目录按 CR 文件名取同名 .npy，忽略列表第三列（实验用，如 GT_MID_COM_npy）。
    NPY 可为 (H,W) 灰度或 (H,W,3)，会 resize 到 img_size 并转为 3 通道 tensor。
    path_remap: 如 ("F:", "I:")，将 txt 中所有路径的 F: 替换为 I:，用于不修改 txt 内容时切换盘符。
    use_low_freq_only: 若为 True，对 CR 和 GT 做三频分解（sigma 由首图尺寸决定），仅取低频用于训练。
    use_lab_color: 若为 True，线性 RGB 转为 LAB 色彩空间训练（可与 use_low_freq_only 组合或单独使用）。
    """
    def __init__(
        self,
        list_files,
        img_size: int = 224,
        gt_dir: str = None,
        path_remap: Tuple[str, str] = None,
        use_low_freq_only: bool = False,
        use_lab_color: bool = False,
    ):
        self.samples: List[Tuple[str, str]] = []
        self.img_size = img_size
        self.path_remap = path_remap
        self.use_low_freq_only = use_low_freq_only
        self.use_lab_color = use_lab_color
        self._sigma_low = None
        self._sigma_mid = None
        if isinstance(list_files, str):
            list_files = [list_files]
        for list_file in list_files:
            list_file = _remap_path(list_file, path_remap)
            with open(list_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    cr_path = _remap_path(parts[0], path_remap)
                    gt_path = _remap_path(parts[2], path_remap)
                    cr_npy = _path_to_npy(cr_path)
                    if gt_dir:
                        gt_npy = os.path.join(gt_dir, os.path.basename(cr_npy))
                    else:
                        gt_npy = _path_to_npy(gt_path)
                    if os.path.exists(cr_npy) and os.path.exists(gt_npy):
                        self.samples.append((cr_npy, gt_npy))
        if len(self.samples) == 0:
            raise RuntimeError("提供的列表文件中没有找到对应的 .npy 文件。")
        self._resize = (img_size, img_size)

        if use_low_freq_only:
            arr = np.load(self.samples[0][0]).astype(np.float32)
            h, w = arr.shape[0], arr.shape[1]
            self._sigma_low, self._sigma_mid = _sigma_from_image_freq(h, w)
            lab_str = " + LAB" if use_lab_color else ""
            print(f"  三频低频训练{lab_str}：首图 {h}x{w} sigma_low={self._sigma_low:.4f} sigma_mid={self._sigma_mid:.4f}")
        elif use_lab_color:
            print("  LAB 色彩空间训练")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npy_to_tensor(self, path: str, resize: Tuple[int, int]) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        if self.use_low_freq_only and self._sigma_low is not None and self._sigma_mid is not None:
            img_log = np.log1p(np.clip(arr, 0, None))
            low_log, _, _ = decompose_freq_log(img_log, self._sigma_low, self._sigma_mid)
            arr = np.clip(np.expm1(low_log), 0.0, 1.0).astype(np.float32)
        # use_lab_color 时保持 RGB，在 GPU 上做 RGB->LAB 转换以加速

        arr = cv2.resize(arr, resize, interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
        return t

    def __getitem__(self, idx: int):
        cr_npy, gt_npy = self.samples[idx]
        cr = self._load_npy_to_tensor(cr_npy, self._resize)
        gt = self._load_npy_to_tensor(gt_npy, self._resize)
        return cr, gt


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


def _slide_overlap_loss(
    predA: torch.Tensor, predB: torch.Tensor,
    t1: int, l1: int, t2: int, l2: int, ps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """同图两 patch 重叠区域 L1 与 MSE，返回 (l1_loss, mse_loss)。"""
    r0 = max(t1, t2)
    r1 = min(t1 + ps, t2 + ps)
    c0 = max(l1, l2)
    c1 = min(l1 + ps, l2 + ps)
    if r0 >= r1 or c0 >= c1:
        z = torch.tensor(0.0, device=predA.device, dtype=predA.dtype)
        return z, z
    a_r0, a_r1 = r0 - t1, r1 - t1
    a_c0, a_c1 = c0 - l1, c1 - l1
    b_r0, b_r1 = r0 - t2, r1 - t2
    b_c0, b_c1 = c0 - l2, c1 - l2
    oa = predA[:, :, a_r0:a_r1, a_c0:a_c1]
    ob = predB[:, :, b_r0:b_r1, b_c0:b_c1]
    l1_loss = torch.nn.functional.l1_loss(oa, ob)
    mse_loss = torch.nn.functional.mse_loss(oa, ob)
    return l1_loss, mse_loss


class CNNImageRegressor(nn.Module):
    """
    Encoder-Decoder + Skip (UNet 风格)。
    Conv80 -> 12Res(80) -[skip]-> Down -> 12Res(160) -[skip]-> Down -> 12Res(320)
      -> Up+skip -> Up+skip -> 8Res(80) -> Conv -> RGB
    """

    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224, base_ch: int = 80, lab_color: bool = False, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size  # int 或 (H, W)，仅用于记录，forward 为全卷积
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
        self._enc3_stage = 1  # 1/2/3，对应 enc3 块数 12/16/20

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        e1 = self.enc1(h)
        h = self.down1(e1)
        e2 = self.enc2(h)
        h = self.down2(e2)
        h = self.enc3(h)
        h = self.up1(h)
        if h.shape[2:] != e2.shape[2:]:
            e2 = torch.nn.functional.interpolate(e2, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.fuse1(torch.cat([h, e2], dim=1))
        h = self.up2(h)
        if h.shape[2:] != e1.shape[2:]:
            e1 = torch.nn.functional.interpolate(e1, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = self.fuse2(torch.cat([h, e1], dim=1))
        h = self.dec(h)
        return self.conv_out(h)


def expand_enc3_for_stage(model: nn.Module, stage_idx: int, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Stage 1: enc3=12 (默认)。Stage 2: +4 blocks (dilation 2,2,4,4) → 16。Stage 3: +4 blocks (dilation 8,8,4,2) → 20。
    新增参数会加入 optimizer（若提供）。
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
    # 新 block 移到 model 所在 device，并加入 optimizer
    dev = next(m.parameters()).device
    m.enc3 = m.enc3.to(dev)
    if optimizer is not None:
        new_params = [p for b in new_blocks for p in b.parameters()]
        if new_params:
            lr = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": new_params, "lr": lr})


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = None,
    max_epochs: int = None,
    perceptual_loss_fn: Optional[nn.Module] = None,
    use_rgb_lab_loss: bool = False,
    loss_l1: float = 1.0,
    loss_mse: float = 0.0,
    loss_grad: float = 0.0,
    loss_ssim: float = 0.0,
    loss_percep: float = 0.0,
    loss_ab_l1: float = 0.0,
    loss_ab_mse: float = 0.0,
    loss_consist: float = 0.0,
    loss_slide: float = 0.0,
    loss_consist_l1: float = 0.5,
    loss_consist_mse: float = 0.5,
    loss_slide_l1: float = 0.5,
    loss_slide_mse: float = 0.5,
) -> Tuple[float, Optional[float], Optional[float]]:
    """重建项由权重控制：loss_l1*L1(RGB)+loss_mse*MSE(RGB)。use_rgb_lab_loss 时再加 loss_grad*Grad(L)+loss_ssim*(1-SSIM(L))+loss_percep*Percep(RGB)。
    loss_consist>0 时需 loader 返回 (crA, crB, gt)，额外加 loss_consist * (loss_consist_l1*L1(predA,predB) + loss_consist_mse*MSE(predA,predB))，默认 0.5/0.5。
    loss_slide>0 时需 loader 返回 (crA, crB, gtA, gtB, t1, l1, t2, l2)，额外加 loss_slide * (loss_slide_l1*L1_overlap + loss_slide_mse*MSE_overlap)，默认 0.5/0.5。"""
    model.train()
    l1_fn = nn.L1Loss()
    criterion = nn.MSELoss()
    running_loss = 0.0
    running_loss_rgb = 0.0
    running_loss_lab_raw = 0.0
    total_images = len(loader.dataset)
    total_batches = len(loader)
    epoch_start = time.perf_counter()
    images_seen = 0

    for batch_idx, batch in enumerate(loader):
        use_consist = loss_consist > 0 and len(batch) == 3
        use_slide = loss_slide > 0 and len(batch) == 8
        if use_consist:
            crA, crB, gt = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad()
            predA = model(crA)
            predB = model(crB)
            pred_rgb_a = predA.clamp(0.0, 1.0)
            pred_rgb_b = predB.clamp(0.0, 1.0)
            gt_rgb = gt.clamp(0.0, 1.0)
            if pred_rgb_a.shape[2:] != gt_rgb.shape[2:]:
                pred_rgb_a = F.interpolate(pred_rgb_a, size=gt_rgb.shape[2:], mode="bilinear", align_corners=False)
            if pred_rgb_b.shape[2:] != gt_rgb.shape[2:]:
                pred_rgb_b = F.interpolate(pred_rgb_b, size=gt_rgb.shape[2:], mode="bilinear", align_corners=False)
            # main loss: (L(predA,gt) + L(predB,gt)) / 2
            if use_rgb_lab_loss:
                pred_lab_a = rgb_to_lab_tensor(pred_rgb_a, device)
                pred_lab_b = rgb_to_lab_tensor(pred_rgb_b, device)
                gt_lab = rgb_to_lab_tensor(gt_rgb, device)
                pred_L_a, pred_L_b = pred_lab_a[:, 0:1], pred_lab_b[:, 0:1]
                gt_L = gt_lab[:, 0:1]
                l1_a = l1_fn(pred_rgb_a, gt_rgb)
                l1_b = l1_fn(pred_rgb_b, gt_rgb)
                mse_a = criterion(pred_rgb_a, gt_rgb)
                mse_b = criterion(pred_rgb_b, gt_rgb)
                loss = loss_l1 * 0.5 * (l1_a + l1_b) + loss_mse * 0.5 * (mse_a + mse_b)
                n_b = crA.size(0)
                running_loss_rgb += (0.5 * (l1_a.item() + l1_b.item())) * n_b
                aux = 0.0
                if loss_grad > 0:
                    g_a = _gradient_loss(pred_L_a, gt_L)
                    g_b = _gradient_loss(pred_L_b, gt_L)
                    loss = loss + loss_grad * 0.5 * (g_a + g_b)
                    aux += 0.5 * (g_a.item() + g_b.item()) * loss_grad
                if loss_ssim > 0:
                    pred_L_01_a = (pred_L_a.clamp(-1.0, 1.0) + 1.0) / 2.0
                    pred_L_01_b = (pred_L_b.clamp(-1.0, 1.0) + 1.0) / 2.0
                    gt_L_01 = (gt_L.clamp(-1.0, 1.0) + 1.0) / 2.0
                    s_a = 2.0 * _ssim_loss(pred_L_01_a, gt_L_01)
                    s_b = 2.0 * _ssim_loss(pred_L_01_b, gt_L_01)
                    loss = loss + loss_ssim * 0.5 * (s_a + s_b)
                    aux += 0.5 * (s_a.item() + s_b.item()) * loss_ssim
                if loss_ab_l1 > 0 or loss_ab_mse > 0:
                    pred_ab_a = pred_lab_a[:, 1:3]
                    pred_ab_b = pred_lab_b[:, 1:3]
                    gt_ab = gt_lab[:, 1:3]
                    loss = loss + loss_ab_l1 * 0.5 * (l1_fn(pred_ab_a, gt_ab) + l1_fn(pred_ab_b, gt_ab)) + loss_ab_mse * 0.5 * (criterion(pred_ab_a, gt_ab) + criterion(pred_ab_b, gt_ab))
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    p_a = perceptual_loss_fn(pred_rgb_a, gt_rgb)
                    p_b = perceptual_loss_fn(pred_rgb_b, gt_rgb)
                    loss = loss + loss_percep * 0.5 * (p_a + p_b)
                    aux += 0.5 * (p_a.item() + p_b.item()) * loss_percep
                running_loss_lab_raw += aux * n_b
            else:
                # no LAB：L1(RGB)+MSE(RGB) 由权重控制 + 可选 percep(RGB)
                l1_a = l1_fn(pred_rgb_a, gt_rgb)
                l1_b = l1_fn(pred_rgb_b, gt_rgb)
                mse_a = criterion(pred_rgb_a, gt_rgb)
                mse_b = criterion(pred_rgb_b, gt_rgb)
                loss = loss_l1 * 0.5 * (l1_a + l1_b) + loss_mse * 0.5 * (mse_a + mse_b)
                n_b = crA.size(0)
                running_loss_rgb += (0.5 * (l1_a.item() + l1_b.item())) * n_b
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    p_a = perceptual_loss_fn(pred_rgb_a, gt_rgb)
                    p_b = perceptual_loss_fn(pred_rgb_b, gt_rgb)
                    loss = loss + loss_percep * 0.5 * (p_a + p_b)
            # consistency: loss_consist * (loss_consist_l1*L1 + loss_consist_mse*MSE)(predA, predB)
            if pred_rgb_a.shape[2:] != pred_rgb_b.shape[2:]:
                pred_rgb_b = F.interpolate(pred_rgb_b, size=pred_rgb_a.shape[2:], mode="bilinear", align_corners=False)
            consist_l1 = l1_fn(pred_rgb_a, pred_rgb_b)
            consist_mse = criterion(pred_rgb_a, pred_rgb_b)
            loss = loss + loss_consist * (loss_consist_l1 * consist_l1 + loss_consist_mse * consist_mse)
        elif use_slide:
            crA, crB, gtA, gtB = [b.to(device, non_blocking=True) for b in batch[:4]]
            t1, l1_t, t2, l2_t = batch[4], batch[5], batch[6], batch[7]
            optimizer.zero_grad()
            predA = model(crA)
            predB = model(crB)
            pred_rgb_a = predA.clamp(0.0, 1.0)
            pred_rgb_b = predB.clamp(0.0, 1.0)
            gt_rgb_a = gtA.clamp(0.0, 1.0)
            gt_rgb_b = gtB.clamp(0.0, 1.0)
            if pred_rgb_a.shape[2:] != gt_rgb_a.shape[2:]:
                pred_rgb_a = F.interpolate(pred_rgb_a, size=gt_rgb_a.shape[2:], mode="bilinear", align_corners=False)
            if pred_rgb_b.shape[2:] != gt_rgb_b.shape[2:]:
                pred_rgb_b = F.interpolate(pred_rgb_b, size=gt_rgb_b.shape[2:], mode="bilinear", align_corners=False)
            ps = gtA.shape[2]
            if use_rgb_lab_loss:
                pred_lab_a = rgb_to_lab_tensor(pred_rgb_a, device)
                pred_lab_b = rgb_to_lab_tensor(pred_rgb_b, device)
                gt_lab_a = rgb_to_lab_tensor(gt_rgb_a, device)
                gt_lab_b = rgb_to_lab_tensor(gt_rgb_b, device)
                pred_L_a, pred_L_b = pred_lab_a[:, 0:1], pred_lab_b[:, 0:1]
                gt_L_a, gt_L_b = gt_lab_a[:, 0:1], gt_lab_b[:, 0:1]
                l1_a = l1_fn(pred_rgb_a, gt_rgb_a)
                l1_b = l1_fn(pred_rgb_b, gt_rgb_b)
                mse_a = criterion(pred_rgb_a, gt_rgb_a)
                mse_b = criterion(pred_rgb_b, gt_rgb_b)
                loss = loss_l1 * 0.5 * (l1_a + l1_b) + loss_mse * 0.5 * (mse_a + mse_b)
                n_b = crA.size(0)
                running_loss_rgb += 0.5 * (l1_a.item() + l1_b.item()) * n_b
                aux = 0.0
                if loss_grad > 0:
                    g_a = _gradient_loss(pred_L_a, gt_L_a)
                    g_b = _gradient_loss(pred_L_b, gt_L_b)
                    loss = loss + loss_grad * 0.5 * (g_a + g_b)
                    aux += 0.5 * (g_a.item() + g_b.item()) * loss_grad
                if loss_ssim > 0:
                    pred_L_01_a = (pred_L_a.clamp(-1.0, 1.0) + 1.0) / 2.0
                    pred_L_01_b = (pred_L_b.clamp(-1.0, 1.0) + 1.0) / 2.0
                    gt_L_01_a = (gt_L_a.clamp(-1.0, 1.0) + 1.0) / 2.0
                    gt_L_01_b = (gt_L_b.clamp(-1.0, 1.0) + 1.0) / 2.0
                    s_a = 2.0 * _ssim_loss(pred_L_01_a, gt_L_01_a)
                    s_b = 2.0 * _ssim_loss(pred_L_01_b, gt_L_01_b)
                    loss = loss + loss_ssim * 0.5 * (s_a + s_b)
                    aux += 0.5 * (s_a.item() + s_b.item()) * loss_ssim
                if loss_ab_l1 > 0 or loss_ab_mse > 0:
                    pred_ab_a = pred_lab_a[:, 1:3]
                    pred_ab_b = pred_lab_b[:, 1:3]
                    gt_ab_a = gt_lab_a[:, 1:3]
                    gt_ab_b = gt_lab_b[:, 1:3]
                    loss = loss + loss_ab_l1 * 0.5 * (l1_fn(pred_ab_a, gt_ab_a) + l1_fn(pred_ab_b, gt_ab_b)) + loss_ab_mse * 0.5 * (criterion(pred_ab_a, gt_ab_a) + criterion(pred_ab_b, gt_ab_b))
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    p_a = perceptual_loss_fn(pred_rgb_a, gt_rgb_a)
                    p_b = perceptual_loss_fn(pred_rgb_b, gt_rgb_b)
                    loss = loss + loss_percep * 0.5 * (p_a + p_b)
                    aux += 0.5 * (p_a.item() + p_b.item()) * loss_percep
                running_loss_lab_raw += aux * n_b
            else:
                # no LAB：L1(RGB)+MSE(RGB) 由权重控制 + 可选 percep(RGB)
                l1_a = l1_fn(pred_rgb_a, gt_rgb_a)
                l1_b = l1_fn(pred_rgb_b, gt_rgb_b)
                mse_a = criterion(pred_rgb_a, gt_rgb_a)
                mse_b = criterion(pred_rgb_b, gt_rgb_b)
                loss = loss_l1 * 0.5 * (l1_a + l1_b) + loss_mse * 0.5 * (mse_a + mse_b)
                n_b = crA.size(0)
                running_loss_rgb += 0.5 * (l1_a.item() + l1_b.item()) * n_b
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    p_a = perceptual_loss_fn(pred_rgb_a, gt_rgb_a)
                    p_b = perceptual_loss_fn(pred_rgb_b, gt_rgb_b)
                    loss = loss + loss_percep * 0.5 * (p_a + p_b)
            if loss_slide > 0:
                slide_l1_list, slide_mse_list = [], []
                n_b = crA.size(0)
                for i in range(n_b):
                    def _int_at(t, ii):
                        if isinstance(t, (int, float)):
                            return int(t)
                        return int((t[ii] if t.dim() > 0 else t).item())
                    tt1 = _int_at(t1, i)
                    ll1 = _int_at(l1_t, i)
                    tt2 = _int_at(t2, i)
                    ll2 = _int_at(l2_t, i)
                    sl1, smse = _slide_overlap_loss(pred_rgb_a[i:i+1], pred_rgb_b[i:i+1], tt1, ll1, tt2, ll2, ps)
                    slide_l1_list.append(sl1)
                    slide_mse_list.append(smse)
                slide_l1_mean = torch.stack(slide_l1_list).mean()
                slide_mse_mean = torch.stack(slide_mse_list).mean()
                loss = loss + loss_slide * (loss_slide_l1 * slide_l1_mean + loss_slide_mse * slide_mse_mean)
            n = crA.size(0)
        else:
            cr, gt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(cr)
            if use_rgb_lab_loss:
                pred_rgb = pred.clamp(0.0, 1.0)
                gt_rgb = gt.clamp(0.0, 1.0)
                if pred_rgb.shape[2:] != gt_rgb.shape[2:]:
                    pred_rgb = F.interpolate(pred_rgb, size=gt_rgb.shape[2:], mode="bilinear", align_corners=False)
                pred_lab = rgb_to_lab_tensor(pred_rgb, device)
                gt_lab = rgb_to_lab_tensor(gt_rgb, device)
                pred_L = pred_lab[:, 0:1]
                gt_L = gt_lab[:, 0:1]
                l1_rgb = l1_fn(pred_rgb, gt_rgb)
                mse_rgb = criterion(pred_rgb, gt_rgb)
                loss = loss_l1 * l1_rgb + loss_mse * mse_rgb
                n_b = cr.size(0)
                running_loss_rgb += l1_rgb.item() * n_b
                aux = 0.0
                if loss_grad > 0:
                    grad_l = _gradient_loss(pred_L, gt_L)
                    loss = loss + loss_grad * grad_l
                    aux += grad_l.item() * loss_grad
                if loss_ssim > 0:
                    pred_L_01 = (pred_L.clamp(-1.0, 1.0) + 1.0) / 2.0
                    gt_L_01 = (gt_L.clamp(-1.0, 1.0) + 1.0) / 2.0
                    ssim_l = 2.0 * _ssim_loss(pred_L_01, gt_L_01)
                    loss = loss + loss_ssim * ssim_l
                    aux += ssim_l.item() * loss_ssim
                if loss_ab_l1 > 0 or loss_ab_mse > 0:
                    pred_ab = pred_lab[:, 1:3]
                    gt_ab = gt_lab[:, 1:3]
                    loss = loss + loss_ab_l1 * l1_fn(pred_ab, gt_ab) + loss_ab_mse * criterion(pred_ab, gt_ab)
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    perc_l = perceptual_loss_fn(pred_rgb, gt_rgb)
                    loss = loss + loss_percep * perc_l
                    aux += perc_l.item() * loss_percep
                running_loss_lab_raw += aux * n_b
            else:
                # no LAB：L1(RGB)+MSE(RGB) 由权重控制 + 可选 percep(RGB)
                pred_rgb = pred.clamp(0.0, 1.0)
                gt_rgb = gt.clamp(0.0, 1.0)
                if pred_rgb.shape[2:] != gt_rgb.shape[2:]:
                    pred_rgb = F.interpolate(pred_rgb, size=gt_rgb.shape[2:], mode="bilinear", align_corners=False)
                l1_rgb = l1_fn(pred_rgb, gt_rgb)
                mse_rgb = criterion(pred_rgb, gt_rgb)
                loss = loss_l1 * l1_rgb + loss_mse * mse_rgb
                n_b = cr.size(0)
                running_loss_rgb += l1_rgb.item() * n_b
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    perc_l = perceptual_loss_fn(pred_rgb, gt_rgb)
                    loss = loss + loss_percep * perc_l
        loss.backward()
        optimizer.step()
        n = crA.size(0) if (use_consist or use_slide) else cr.size(0)
        running_loss += loss.item() * n
        images_seen += n

        elapsed = time.perf_counter() - epoch_start
        batches_done = batch_idx + 1
        eta = (elapsed / batches_done * (total_batches - batches_done)) if batches_done > 0 else 0.0
        pct = 100.0 * batches_done / total_batches if total_batches else 0
        epoch_info = f"Epoch {epoch}/{max_epochs} " if (epoch is not None and max_epochs is not None) else ""
        print(
            f"\r  {epoch_info}[ {images_seen}/{total_images} patches ] "
            f"{pct:.1f}%  已耗时 {elapsed:.1f}s  预计剩余 {eta:.1f}s",
            end="",
            flush=True,
        )
    print()
    avg_loss = running_loss / total_images
    if use_rgb_lab_loss:
        avg_L = running_loss_rgb / total_images
        avg_ab = running_loss_lab_raw / total_images
    else:
        avg_L = avg_loss
        avg_ab = 0.0
    return avg_loss, avg_L, avg_ab


def compute_val_patch_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    device: torch.device,
    perceptual_loss_fn: Optional[nn.Module] = None,
    use_rgb_lab_loss: bool = False,
    loss_l1: float = 1.0,
    loss_mse: float = 0.0,
    loss_grad: float = 0.0,
    loss_ssim: float = 0.0,
    loss_percep: float = 0.0,
    loss_ab_l1: float = 0.0,
    loss_ab_mse: float = 0.0,
) -> torch.Tensor:
    """单 patch (pred, gt) 的 val loss，与 eval_one_epoch 中逻辑一致。pred/gt: (B,3,H,W)。LAB 的 AB 已归一化 [-1,1]。"""
    l1_fn = nn.L1Loss()
    criterion = nn.MSELoss()
    pred_rgb = pred.clamp(0.0, 1.0)
    gt_rgb = gt.clamp(0.0, 1.0)
    if use_rgb_lab_loss:
        pred_lab = rgb_to_lab_tensor(pred_rgb, device)
        gt_lab = rgb_to_lab_tensor(gt_rgb, device)
        pred_L = pred_lab[:, 0:1]
        gt_L = gt_lab[:, 0:1]
        loss = loss_l1 * l1_fn(pred_rgb, gt_rgb) + loss_mse * criterion(pred_rgb, gt_rgb)
        if loss_ab_l1 > 0 or loss_ab_mse > 0:
            pred_ab = pred_lab[:, 1:3]
            gt_ab = gt_lab[:, 1:3]
            loss = loss + loss_ab_l1 * l1_fn(pred_ab, gt_ab) + loss_ab_mse * criterion(pred_ab, gt_ab)
        if loss_grad > 0:
            loss = loss + loss_grad * _gradient_loss(pred_L, gt_L)
        if loss_ssim > 0:
            pred_L_01 = (pred_L.clamp(-1.0, 1.0) + 1.0) / 2.0
            gt_L_01 = (gt_L.clamp(-1.0, 1.0) + 1.0) / 2.0
            loss = loss + loss_ssim * 2.0 * _ssim_loss(pred_L_01, gt_L_01)
        if loss_percep > 0 and perceptual_loss_fn is not None:
            loss = loss + loss_percep * perceptual_loss_fn(pred_rgb, gt_rgb)
    else:
        # no LAB：L1(RGB)+MSE(RGB) 由权重控制 + 可选 percep(RGB)
        loss = loss_l1 * l1_fn(pred_rgb, gt_rgb) + loss_mse * criterion(pred_rgb, gt_rgb)
        if loss_percep > 0 and perceptual_loss_fn is not None:
            loss = loss + loss_percep * perceptual_loss_fn(pred_rgb, gt_rgb)
    return loss


def _lock_l_merge_single(input_rgb_01: np.ndarray, pred_rgb_01: np.ndarray) -> np.ndarray:
    """单张图：用输入 L + 预测 AB 合成，RGB [0,1] (H,W,3)。"""
    rgb_u8_in = (np.clip(input_rgb_01, 0.0, 1.0) * 255).astype(np.uint8)
    lab_in = cv2.cvtColor(rgb_u8_in, cv2.COLOR_RGB2LAB)
    rgb_u8_pred = (np.clip(pred_rgb_01, 0.0, 1.0) * 255).astype(np.uint8)
    lab_pred = cv2.cvtColor(rgb_u8_pred, cv2.COLOR_RGB2LAB)
    lab_merge = np.stack([lab_in[:, :, 0], lab_pred[:, :, 1], lab_pred[:, :, 2]], axis=-1)
    rgb_u8 = cv2.cvtColor(lab_merge.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_u8.astype(np.float32) / 255.0


def _lock_l_merge_batch(cr: torch.Tensor, pred_rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(B,3,H,W) [0,1] RGB：用 cr 的 L + pred_rgb 的 AB 合成，返回 (B,3,H,W)。"""
    cr_np = cr.cpu().numpy().transpose(0, 2, 3, 1)
    pred_np = pred_rgb.cpu().numpy().transpose(0, 2, 3, 1)
    merged = np.zeros_like(cr_np, dtype=np.float32)
    for i in range(cr_np.shape[0]):
        merged[i] = _lock_l_merge_single(cr_np[i], pred_np[i])
    return torch.from_numpy(merged.transpose(0, 3, 1, 2)).float().to(device)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    perceptual_loss_fn: Optional[nn.Module] = None,
    use_rgb_lab_loss: bool = False,
    loss_l1: float = 1.0,
    loss_mse: float = 0.0,
    loss_grad: float = 0.0,
    loss_ssim: float = 0.0,
    loss_percep: float = 0.0,
    loss_ab_l1: float = 0.0,
    loss_ab_mse: float = 0.0,
    lock_l: bool = False,
) -> Tuple[float, Optional[float], Optional[float], float, float]:
    """在验证集上计算 loss、PSNR、SSIM。返回 (total_loss, loss_L, loss_ab, psnr, ssim)。lock_l 为 True 时用输入 L + 预测 AB 合成后再算指标。"""
    model.eval()
    l1_fn = nn.L1Loss()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_loss_rgb = 0.0
    total_loss_lab_raw = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0
    total_patches = len(loader.dataset)
    total_batches = len(loader)
    val_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (cr, gt) in enumerate(loader):
            cr, gt = cr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            pred = model(cr)
            pred_rgb = pred.clamp(0.0, 1.0)
            if lock_l:
                pred_rgb = _lock_l_merge_batch(cr, pred_rgb, device)
            gt_rgb = gt.clamp(0.0, 1.0)
            n = cr.size(0)
            total_psnr += _psnr(pred_rgb, gt_rgb) * n
            total_ssim += _ssim_value(pred_rgb, gt_rgb) * n
            if use_rgb_lab_loss:
                pred_lab = rgb_to_lab_tensor(pred_rgb, device)
                gt_lab = rgb_to_lab_tensor(gt_rgb, device)
                pred_L = pred_lab[:, 0:1]
                gt_L = gt_lab[:, 0:1]
                l1_rgb = l1_fn(pred_rgb, gt_rgb)
                mse_rgb = criterion(pred_rgb, gt_rgb)
                loss = loss_l1 * l1_rgb + loss_mse * mse_rgb
                n = cr.size(0)
                total_loss_rgb += l1_rgb.item() * n
                aux = 0.0
                if loss_ab_l1 > 0 or loss_ab_mse > 0:
                    pred_ab = pred_lab[:, 1:3]
                    gt_ab = gt_lab[:, 1:3]
                    loss = loss + loss_ab_l1 * l1_fn(pred_ab, gt_ab) + loss_ab_mse * criterion(pred_ab, gt_ab)
                if loss_grad > 0:
                    grad_l = _gradient_loss(pred_L, gt_L)
                    loss = loss + loss_grad * grad_l
                    aux += grad_l.item() * loss_grad
                if loss_ssim > 0:
                    pred_L_01 = (pred_L.clamp(-1.0, 1.0) + 1.0) / 2.0
                    gt_L_01 = (gt_L.clamp(-1.0, 1.0) + 1.0) / 2.0
                    ssim_l = 2.0 * _ssim_loss(pred_L_01, gt_L_01)  # 1-SSIM
                    loss = loss + loss_ssim * ssim_l
                    aux += ssim_l.item() * loss_ssim
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    perc_l = perceptual_loss_fn(pred_rgb, gt_rgb)
                    loss = loss + loss_percep * perc_l
                    aux += perc_l.item() * loss_percep
                total_loss_lab_raw += aux * n
            else:
                # no LAB：L1(RGB)+MSE(RGB) 由权重控制 + 可选 percep(RGB)
                l1_rgb = l1_fn(pred_rgb, gt_rgb)
                mse_rgb = criterion(pred_rgb, gt_rgb)
                loss = loss_l1 * l1_rgb + loss_mse * mse_rgb
                total_loss_rgb += l1_rgb.item() * n
                if loss_percep > 0 and perceptual_loss_fn is not None:
                    perc_l = perceptual_loss_fn(pred_rgb, gt_rgb)
                    loss = loss + loss_percep * perc_l
            n = cr.size(0)
            total_loss += loss.item() * n
            total_images += n
            elapsed = time.perf_counter() - val_start
            batches_done = batch_idx + 1
            eta = (elapsed / batches_done * (total_batches - batches_done)) if batches_done > 0 else 0.0
            pct = 100.0 * batches_done / total_batches if total_batches else 0
            print(
                f"\r  Val [ {total_images}/{total_patches} patches ] "
                f"{pct:.1f}%  已耗时 {elapsed:.1f}s  预计剩余 {eta:.1f}s",
                end="",
                flush=True,
            )
    print()
    model.train()
    avg = total_loss / total_images if total_images else 0.0
    if use_rgb_lab_loss and total_images:
        avg_L = total_loss_rgb / total_images
        avg_ab = total_loss_lab_raw / total_images
    else:
        avg_L = avg
        avg_ab = 0.0
    psnr = total_psnr / total_images if total_images else 0.0
    ssim = total_ssim / total_images if total_images else 0.0
    return avg, avg_L, avg_ab, psnr, ssim


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_batches: int = 1,
    use_lab_color: bool = False,
    show_L: bool = True,
    prefix: str = "sample",
) -> None:
    """prefix 用于文件名。show_L 时除 RGB 外还展示 Pred_L 与 GT_L。"""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idx = 0
    with torch.no_grad():
        for batch_idx, (cr, gt) in enumerate(loader):
            cr, gt = cr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            if use_lab_color:
                cr = rgb_to_lab_tensor(cr, device)
                gt = rgb_to_lab_tensor(gt, device)
            pred = model(cr)
            if use_lab_color:
                pred_lab = torch.cat([
                    pred[:, 0:1].clamp(-1.0, 1.0),
                    pred[:, 1:3].clamp(-1.0, 1.0),
                ], dim=1)
                gt_L = gt[:, 0:1].clone()
                pred = lab_to_rgb_tensor(pred_lab, device)
                cr = lab_to_rgb_tensor(cr, device)
                gt = lab_to_rgb_tensor(gt, device)
                pred_L = pred_lab[:, 0:1]
            else:
                pred = torch.clamp(pred, 0.0, 1.0)
                pred_lab = rgb_to_lab_tensor(pred, device)
                gt_lab = rgb_to_lab_tensor(gt, device)
                pred_L = pred_lab[:, 0:1]
                gt_L = gt_lab[:, 0:1]
            for i in range(cr.size(0)):
                c = cr[i].cpu().numpy().transpose(1, 2, 0)
                p = pred[i].cpu().numpy().transpose(1, 2, 0)
                g = gt[i].cpu().numpy().transpose(1, 2, 0)
                imgs = [
                    torch.from_numpy(c).permute(2, 0, 1),
                    torch.from_numpy(p).permute(2, 0, 1),
                    torch.from_numpy(g).permute(2, 0, 1),
                ]
                if show_L:
                    # L: [-1,1] -> [0,1]，复制为 3 通道以显示灰度
                    def _L_to_vis(L_1ch):
                        v = (L_1ch.clamp(-1.0, 1.0) + 1.0) / 2.0
                        return v.expand(3, -1, -1)
                    pred_L_vis = _L_to_vis(pred_L[i])
                    gt_L_vis = _L_to_vis(gt_L[i])
                    imgs.extend([pred_L_vis.cpu(), gt_L_vis.cpu()])
                grid = torch.stack(imgs, dim=0)
                nrow = 5 if show_L else 3
                save_image(grid, os.path.join(out_dir, f"{prefix}_{idx:04d}.png"), nrow=nrow)
                idx += 1
            if batch_idx + 1 >= max_batches:
                break


def main():
    _default_base = "F:/Code/Datasets/NTIRE2026/color_track/Train/metadata"
    parser = argparse.ArgumentParser(description="NTIRE Color Track CNN：split2+3 训练，split1 验证")
    parser.add_argument(
        "--train_list",
        type=str,
        nargs="+",
        default=[f"{_default_base}/split2.txt", f"{_default_base}/split3.txt"],
        help="训练集列表文件（默认 split2, split3）",
    )
    parser.add_argument(
        "--val_list",
        type=str,
        nargs="+",
        default=[f"{_default_base}/split1.txt"],
        help="验证集列表文件（默认 split1）",
    )
    parser.add_argument(
        "--path_remap",
        type=str,
        nargs=2,
        default=["F:", "I:"],
        metavar=("OLD", "NEW"),
        help="路径替换：将 txt 中的 OLD 换为 NEW，如 F: -> I:。默认 ['F:', 'I:']，无需替换可传 --path_remap '' ''",
    )
    parser.add_argument("--img_size", type=int, default=384, help="输入分辨率")
    parser.add_argument("--base_ch", type=int, default=80, help="主干通道数 width（默认 80）")
    parser.add_argument("--batch_size", type=int, default=7, help="batch 大小")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader 子进程数")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="最多训练轮数，配合 early stopping 使用",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="训练 loss 连续多少 epoch 不下降则提前结束",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="checkpoint 保存目录，默认 <项目根>/checkpoints",
    )
    parser.add_argument("--vis_batches", type=int, default=1, help="val 可视化 batch 数（仅 PSNR 最优时保存）")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练：可传 .pth 路径、checkpoint 文件夹（如 checkpoints/20250305-123456）、或 best/latest 自动找最新",
    )
    parser.add_argument(
        "--low_freq_only",
        action="store_true",
        help="仅用低频训练：对 CR/GT 做三频分解（sigma 由首图尺寸决定），取低频送入 CNN",
    )
    parser.add_argument(
        "--loss_rgb_lab",
        action="store_true",
        help="网络只输出 RGB，Loss=L1(RGB_pred,RGB_gt)+λ*L1(LAB_pred,LAB_gt)。覆盖 lab_color",
    )
    parser.add_argument("--loss_l1", type=float, default=1.0, help="L1(RGB) 权重（默认 1.0）")
    parser.add_argument("--loss_mse", type=float, default=0.0, help="MSE(RGB) 权重（默认 0），与 loss_l1 共同控制重建项")
    parser.add_argument("--loss_grad", type=float, default=0.1, help="loss_rgb_lab 时 Gradient(L) 权重（默认 0.1）")
    parser.add_argument("--loss_ssim", type=float, default=0.1, help="loss_rgb_lab 时 (1-SSIM(L)) 权重（默认 0.1）")
    parser.add_argument("--loss_percep", type=float, default=0.01, help="loss_rgb_lab 时 Perceptual(RGB) 权重（默认 0.01）")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # NTIRE 项目根目录
    if args.save_dir is None:
        args.save_dir = os.path.join(project_root, "checkpoints")

    # 解析 --resume best/latest：在 <项目根>/checkpoints 下查找
    if args.resume and args.resume.lower() in ("best", "latest"):
        fname = "latest.pth" if args.resume.lower() == "latest" else "best_model.pth"
        ckpt_base = args.save_dir
        candidates = []
        if os.path.isdir(ckpt_base):
            for d in os.listdir(ckpt_base):
                p = os.path.join(ckpt_base, d, fname)
                if os.path.isfile(p):
                    candidates.append(p)
                elif fname == "latest.pth":
                    # latest 不存在时 fallback 到 best_model.pth
                    pb = os.path.join(ckpt_base, d, "best_model.pth")
                    if os.path.isfile(pb):
                        candidates.append(pb)
        if candidates:
            args.resume = max(candidates, key=os.path.getmtime)
        else:
            cand = os.path.join(project_root, fname)
            if not os.path.isfile(cand) and fname == "latest.pth":
                cand = os.path.join(project_root, "best_model.pth")
            if os.path.isfile(cand):
                args.resume = cand
        if not args.resume or not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f"未找到 checkpoint：请将 best_model.pth 与 latest.pth 放在 {ckpt_base}/<run>/ 下，或使用 --resume <path>"
            )
    elif args.resume and os.path.isdir(args.resume):
        # 指定文件夹时，优先 latest.pth，否则 best_model.pth
        for fname in ("latest.pth", "best_model.pth"):
            p = os.path.join(args.resume, fname)
            if os.path.isfile(p):
                args.resume = p
                break
        else:
            raise FileNotFoundError(f"文件夹 {args.resume} 下未找到 latest.pth 或 best_model.pth")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 GPU，请确认：1) 已安装 PyTorch GPU 版（pip install torch --index-url https://download.pytorch.org/whl/cu124）；"
            "2) NVIDIA 驱动正常（nvidia-smi 可用）；3) CUDA 版本匹配。"
        )
    device = torch.device("cuda")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    start_epoch = 1
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    best_val_loss = float("inf")
    best_val_psnr = -float("inf")
    no_improve_count = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"未找到 checkpoint 文件: {args.resume}")
        print(f"从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # 使用 checkpoint 里记录的 save_dir，便于在同一轮次目录下继续保存
        save_dir = ckpt.get("save_dir", os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
        run_name = os.path.basename(save_dir)
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        val_psnrs = ckpt.get("val_psnrs", [])
        val_ssims = ckpt.get("val_ssims", [])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("train_loss", float("inf")))
        best_val_psnr = ckpt.get("best_val_psnr", -float("inf"))
        no_improve_count = ckpt.get("no_improve_count", 0)
        print(f"  将从未完成的 epoch {start_epoch} 继续，best_val_loss={best_val_loss:.6f}, best_val_psnr={best_val_psnr:.2f}, no_improve={no_improve_count}")
    else:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(args.save_dir, run_name)

    best_model_path = os.path.join(save_dir, "best_model.pth")
    latest_model_path = os.path.join(save_dir, "latest.pth")

    path_remap = tuple(args.path_remap) if args.path_remap and args.path_remap[0] else None
    _use_lab_dataset = False  # 固定 RGB 输入
    train_dataset = NTIREColorTrackDatasetNpy(
        args.train_list,
        img_size=args.img_size,
        path_remap=path_remap,
        use_low_freq_only=args.low_freq_only,
        use_lab_color=_use_lab_dataset,
    )
    val_dataset = NTIREColorTrackDatasetNpy(
        args.val_list,
        img_size=args.img_size,
        path_remap=path_remap,
        use_low_freq_only=args.low_freq_only,
        use_lab_color=_use_lab_dataset,
    )
    _dl_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        _dl_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **_dl_kw)
    model = CNNImageRegressor(img_size=args.img_size, base_ch=args.base_ch, lab_color=False)
    if args.resume:
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("  已加载模型权重")
    model.to(device)

    perceptual_loss_fn = None
    if args.loss_percep > 0:
        perceptual_loss_fn = PerceptualLoss(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config_lines = [
        f"=== 模型参数量 ===",
        f"总参数量: {n_params:,}",
        f"可训练: {n_trainable:,}",
        f"约 {n_params/1e6:.2f}M",
        "",
        f"=== 超参数与配置 ===",
        f"run_name: {run_name}",
        f"train_list: {args.train_list}",
        f"val_list: {args.val_list}",
        f"path_remap: {args.path_remap}",
        f"img_size: {args.img_size}",
        f"batch_size: {args.batch_size}",
        f"num_workers: {args.num_workers}",
        f"max_epochs: {args.max_epochs}",
        f"patience: {args.patience}",
        f"lr: {args.lr}",
        f"save_dir: {args.save_dir}",
        f"vis_batches: {args.vis_batches}",
        f"resume: {args.resume}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"model_base_ch: {args.base_ch}",
        f"model_params_M: {n_params/1e6:.2f}",
        f"low_freq_only: {args.low_freq_only}",
        f"loss_rgb_lab: {args.loss_rgb_lab}",
        f"loss_grad: {args.loss_grad}",
        f"loss_ssim: {args.loss_ssim}",
        f"loss_percep: {args.loss_percep}",
    ]
    if args.low_freq_only and train_dataset._sigma_low is not None:
        config_lines.append(f"sigma_low: {train_dataset._sigma_low}")
        config_lines.append(f"sigma_mid: {train_dataset._sigma_mid}")
    config_str = "\n".join(config_lines)
    _skip = ("总参数量:", "可训练:", "train_list:", "val_list:", "path_remap:", "=== 模型参数量 ===", "=== 超参数与配置 ===")
    print_lines = [l for l in config_lines if l and not any(l.startswith(s) for s in _skip) and "约 " not in l]
    print("\n--- 超参数与配置 ---")
    print("\n".join(print_lines))
    print("---\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.resume and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("  已加载优化器状态")
        except Exception as e:
            print(f"  优化器状态未恢复（将使用当前 lr 重新开始）: {e}")
    # 每 10 个 epoch 保存该段内最优：记录当前段内最优 loss 及对应 state_dict、epoch
    best_in_window_loss = float("inf")
    best_in_window_state = None
    best_in_window_epoch = None
    # 最近一轮 val_loss，用于 early stopping
    prev_val_loss = None

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss, train_L, train_ab = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, max_epochs=args.max_epochs,
            perceptual_loss_fn=perceptual_loss_fn,
            use_rgb_lab_loss=args.loss_rgb_lab,
            loss_l1=args.loss_l1,
            loss_mse=getattr(args, "loss_mse", 0.0),
            loss_grad=args.loss_grad,
            loss_ssim=args.loss_ssim,
            loss_percep=args.loss_percep,
        )
        val_loss, val_L, val_ab, val_psnr, val_ssim = eval_one_epoch(
            model, val_loader, device,
            perceptual_loss_fn=perceptual_loss_fn,
            use_rgb_lab_loss=args.loss_rgb_lab,
            loss_l1=args.loss_l1,
            loss_mse=getattr(args, "loss_mse", 0.0),
            loss_grad=args.loss_grad,
            loss_ssim=args.loss_ssim,
            loss_percep=args.loss_percep,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)
        print(f"Epoch [{epoch}/{args.max_epochs}] Train: {train_loss:.6f}  Val: {val_loss:.6f}  PSNR: {val_psnr:.2f}  SSIM: {val_ssim:.4f}")

        # 训练完一个 epoch 后再创建 checkpoint 目录并保存 config
        log_txt = os.path.join(save_dir, "log.txt")
        if epoch == start_epoch:
            os.makedirs(save_dir, exist_ok=True)
            config_txt = os.path.join(save_dir, "config.txt")
            with open(config_txt, "w", encoding="utf-8") as f:
                f.write(config_str)
            print(f"  已保存到 {config_txt}")
            write_header = not os.path.isfile(log_txt)
        else:
            write_header = False
        with open(log_txt, "a" if not write_header else "w", encoding="utf-8") as f:
            if write_header:
                f.write("epoch\ttrain_loss\tval_loss\tval_PSNR\tval_SSIM\n")
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{val_psnr:.2f}\t{val_ssim:.4f}\n")

        # 按 val_loss 保存 best 模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_psnrs": val_psnrs,
                "val_ssims": val_ssims,
                "best_val_loss": best_val_loss,
                "best_val_psnr": max(best_val_psnr, val_psnr),
                "no_improve_count": no_improve_count,
                "save_dir": save_dir,
            }, best_model_path)
            shutil.copy(best_model_path, latest_model_path)
            print(f"  保存当前最优 loss (val_loss={val_loss:.6f}) 到 {best_model_path}")

        # 按 val_psnr 保存 best_PSNR 模型，并保存 val 可视化
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            vis_dir = os.path.join(save_dir, "vis_val_sample")
            os.makedirs(vis_dir, exist_ok=True)
            visualize_predictions(
                model, val_loader, device,
                out_dir=vis_dir,
                max_batches=args.vis_batches,
                use_lab_color=_use_lab_dataset,
                prefix="val_sample",
            )
            print(f"  可视化已保存到 {vis_dir}/")
            best_psnr_path = os.path.join(save_dir, f"best_PSNR_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_val_psnr": best_val_psnr,
                "save_dir": save_dir,
            }, best_psnr_path)
            print(f"  保存当前最优 PSNR ({val_psnr:.2f}) 到 {best_psnr_path}")

        # early stopping：只看“是否比上一轮下降”
        if prev_val_loss is not None and val_loss >= prev_val_loss - 1e-8:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"  Loss 连续 {args.patience} 轮未下降（相对上一轮），提前结束训练。")
                break
        else:
            no_improve_count = 0
        prev_val_loss = val_loss

        # 当前段内最优：用于每 10 epoch 保存（按 val_loss）
        if val_loss < best_in_window_loss:
            best_in_window_loss = val_loss
            best_in_window_epoch = epoch
            best_in_window_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 每 10 个 epoch 保存该段内最优，文件名含 epoch
        if epoch % 10 == 0 and best_in_window_state is not None:
            segment_path = os.path.join(save_dir, f"best_epoch_{best_in_window_epoch}.pth")
            torch.save({
                "epoch": best_in_window_epoch,
                "model_state": best_in_window_state,
                "train_loss": best_in_window_loss,
                "save_dir": save_dir,
            }, segment_path)
            print(f"  保存本段(1~{epoch})最优 (epoch={best_in_window_epoch}, loss={best_in_window_loss:.6f}) 到 {segment_path}")
            best_in_window_loss = float("inf")
            best_in_window_state = None
            best_in_window_epoch = None

    # 提前结束时，若当前段未满 10 个 epoch 也保存该段最优
    if best_in_window_state is not None and (no_improve_count >= args.patience or epoch >= args.max_epochs):
        segment_path = os.path.join(save_dir, f"best_epoch_{best_in_window_epoch}.pth")
        torch.save({
            "epoch": best_in_window_epoch,
            "model_state": best_in_window_state,
            "train_loss": best_in_window_loss,
            "save_dir": save_dir,
        }, segment_path)
        print(f"  保存本段最优 (epoch={best_in_window_epoch}, loss={best_in_window_loss:.6f}) 到 {segment_path}")

    print(f"已保存最优模型到 {best_model_path} (best val loss: {best_val_loss:.6f})")
    shutil.copy(best_model_path, latest_model_path)
    print(f"checkpoint 目录 {save_dir} 下已包含 best_model.pth 与 latest.pth")

    x = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(x, train_losses, label="Train Loss")
    ax1.plot(x, val_losses, label="Val Loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    if val_psnrs and len(val_psnrs) == len(x):
        ax2.plot(x, val_psnrs, label="Val PSNR", color="C2")
        ax2.legend(loc="upper left")
    if val_ssims and len(val_ssims) == len(x):
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, val_ssims, label="Val SSIM", color="C3")
        ax2_twin.set_ylabel("SSIM")
        ax2_twin.legend(loc="upper right")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    plt.suptitle(f"CNN Train Curves ({run_name})")
    plt.tight_layout()
    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线已保存到: {loss_curve_path}")


if __name__ == "__main__":
    main()
