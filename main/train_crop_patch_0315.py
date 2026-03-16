"""
Patch 训练脚本：按 GT 目录决定数据，train GT=75，val GT=10。
默认 crop_resize：GT_crop_resize（*_GT_crop_resize_R*_C*.npy）、IN_CR_COM_crop_resize（*_IN_crop_resize_R*_C*.npy）。
Fallback：GT（*_GT.npy）、IN_CR_COM（*_IN.npy）或 36-crop（*_GT_crop_r*_c*.npy）。
"""
import os
import re
import sys
import glob
import shutil
import argparse
import time
from datetime import datetime
from typing import List, Tuple, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from CNN.train_all_CNN_0315 import (
    CNNImageRegressor,
    train_one_epoch,
    eval_one_epoch,
    compute_val_patch_loss,
    visualize_predictions,
    PerceptualLoss,
    expand_enc3_for_stage,
    _sigma_from_image_freq,
    _psnr,
    _ssim_value,
)

try:
    import lpips
    _lpips_fn = lpips.LPIPS(net="alex")
except Exception:
    _lpips_fn = None

# log.txt / fullimage_metrics.txt 列对齐格式
LOG_EPOCH_W = 14
LOG_TRAIN_W = 12
LOG_VAL_W = 12
LOG_PSNR_W = 8
LOG_SSIM_W = 8
FULLIMAGE_TAG_W = 20
FULLIMAGE_METRIC_W = 10


def _fmt_log_line(epoch_str: str, train_loss: float, val_loss: float, val_psnr: float, val_ssim: float) -> str:
    return f"{epoch_str:<{LOG_EPOCH_W}}\t{train_loss:>{LOG_TRAIN_W}.6f}\t{val_loss:>{LOG_VAL_W}.6f}\t{val_psnr:>{LOG_PSNR_W}.2f}\t{val_ssim:>{LOG_SSIM_W}.4f}"


def _fmt_log_header() -> str:
    return f"{'epoch':<{LOG_EPOCH_W}}\t{'train_loss':>{LOG_TRAIN_W}}\t{'val_loss':>{LOG_VAL_W}}\t{'val_PSNR':>{LOG_PSNR_W}}\t{'val_SSIM':>{LOG_SSIM_W}}"


def _fmt_fullimage_line(tag: str, psnr: float, ssim: float, lpips_val: float) -> str:
    return f"{tag:<{FULLIMAGE_TAG_W}}\t{psnr:>{FULLIMAGE_METRIC_W}.4f}\t{ssim:>{FULLIMAGE_METRIC_W}.4f}\t{lpips_val:>{FULLIMAGE_METRIC_W}.4f}"


def _fmt_fullimage_header() -> str:
    return f"{'tag':<{FULLIMAGE_TAG_W}}\t{'PSNR':>{FULLIMAGE_METRIC_W}}\t{'SSIM':>{FULLIMAGE_METRIC_W}}\t{'LPIPS':>{FULLIMAGE_METRIC_W}}"


def _to_size(x, default=(256, 256), wh_order: bool = False) -> Optional[Tuple[int, int]]:
    """将 int 或 [int,int]/(int,int) 规范为 (H, W)。单值表示正方形。None 保持 None。
    wh_order=True 时，两值按 (宽, 高) 解析并转为 (H, W)。"""
    if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
        return None
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        a, b = int(x[0]), int(x[1])
        return (b, a) if wh_order else (a, b)
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return (int(x[0]), int(x[0]))
    return default


def _patch_hw(patch_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """从 patch_size（int 或 (H,W)）得到 (ph, pw)。"""
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    return (int(patch_size[0]), int(patch_size[1]))


def _fmt_size_wh(hw: Union[int, Tuple[int, int]]) -> str:
    """将内部 (H, W) 格式化为习惯的 宽 高 字符串。"""
    if isinstance(hw, int):
        return str(hw)
    return f"{hw[1]} {hw[0]}"


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


def _save_rgb_lab_vis(img_bgr: np.ndarray, out_path: str) -> None:
    """
    按 原图 | R-G-B / 原图 | L-a-b 两行四列可视化，检查 LAB 空间是否正确。
    img_bgr: BGR uint8 或 float [0,1]
    """
    if plt is None:
        return
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(r, cmap="Reds")
    axes[0, 1].set_title("R")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(g, cmap="Greens")
    axes[0, 2].set_title("G")
    axes[0, 2].axis("off")
    axes[0, 3].imshow(b, cmap="Blues")
    axes[0, 3].set_title("B")
    axes[0, 3].axis("off")
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(l_ch, cmap="gray")
    axes[1, 1].set_title("L")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(a_ch, cmap="RdYlGn_r")
    axes[1, 2].set_title("a")
    axes[1, 2].axis("off")
    axes[1, 3].imshow(b_ch, cmap="RdYlBu_r")
    axes[1, 3].set_title("b")
    axes[1, 3].axis("off")
    plt.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  已保存 RGB/LAB 可视化: {out_path}")


def _unwrap_model(m: nn.Module) -> nn.Module:
    """DataParallel 时返回 module，否则返回自身。用于 state_dict / getattr。"""
    return m.module if hasattr(m, "module") else m


def _save_loss_curve(
    save_dir: str,
    train_losses: List[float],
    val_losses: List[float],
    val_psnrs: List[float],
    val_ssims: List[float],
    stage_starts: Optional[List[int]] = None,
    val_psnrs_fullimage: Optional[List[float]] = None,
    val_ssims_fullimage: Optional[List[float]] = None,
) -> None:
    """保存 loss 曲线到 loss_curve.png。
    stage_starts: 各 stage 起始的 global epoch（1-based）。整图验证 PSNR/SSIM 仅在运行过 run_val_sliding_window_vis 的 epoch 有值，其余为 nan。
    """
    if plt is None or not train_losses:
        return
    n = len(train_losses)
    tl = np.array(train_losses, dtype=float)
    vl = np.array((val_losses + [0.0] * n)[:n], dtype=float)
    vp = np.array((val_psnrs + [0.0] * n)[:n], dtype=float)
    vp_full = np.array((val_psnrs_fullimage or []) + [np.nan] * n)[:n]
    # 占位 0 视为缺失，用 nan 不连线
    first_real = next((i for i in range(n) if tl[i] > 1e-8), n)
    if first_real > 0:
        tl[:first_real] = np.nan
        vl[:first_real] = np.nan
        vp[:first_real] = np.nan

    x = np.arange(1, n + 1, dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    def _plot_segments(ax, y, color, label, marker_at: Optional[List[int]] = None, linestyle: str = "-", markers_only: bool = False):
        if stage_starts and len(stage_starts) >= 1:
            for i, s in enumerate(stage_starts):
                start_idx = s - 1
                end_idx = (stage_starts[i + 1] - 1) if i + 1 < len(stage_starts) else n
                if start_idx >= n:
                    continue
                end_idx = min(end_idx, n)
                xs = x[start_idx:end_idx].astype(float)
                ys = y[start_idx:end_idx].copy()
                mask = ~np.isnan(ys)
                if not np.any(mask):
                    continue
                if markers_only:
                    ax.plot(xs[mask], ys[mask], color=color, linestyle="", marker="o", markersize=5, label=label if i == 0 else "")
                else:
                    ax.plot(xs[mask], ys[mask], color=color, linestyle=linestyle, label=label if i == 0 else "")
                    if marker_at and start_idx < n and not np.isnan(y[start_idx]):
                        lab = f"stage_{i+1}"
                        ax.plot(xs[0], ys[0], "o", color=color, markersize=8)
                        ax.annotate(lab, (xs[0], ys[0]), textcoords="offset points", xytext=(4, 4), fontsize=9)
                    if end_idx > start_idx:
                        last_idx = end_idx - 1 - start_idx
                        if last_idx >= 0 and not np.isnan(ys[last_idx]):
                            ax.plot(xs[last_idx], ys[last_idx], "s", color=color, markersize=6)
        else:
            mask = ~np.isnan(y)
            if np.any(mask):
                if markers_only:
                    ax.plot(x[mask], y[mask], color=color, linestyle="", marker="o", markersize=5, label=label)
                else:
                    ax.plot(x[mask], y[mask], color=color, linestyle=linestyle, label=label)

    def _plot_fullimage_psnr(ax, x_vals, y_vals, color: str, label: str, stage_starts_opt):
        """整图 PSNR：有数据的连续段用实线，段与段之间用虚线连接。"""
        n_ = len(y_vals)
        valid = ~np.isnan(y_vals)
        if not np.any(valid):
            return
        runs = []
        i = 0
        while i < n_:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n_ and valid[j]:
                j += 1
            runs.append((i, j))
            i = j
        for idx, (start, end) in enumerate(runs):
            ax.plot(x_vals[start:end], y_vals[start:end], color=color, linestyle="-", label=label if idx == 0 else "")
        for k in range(len(runs) - 1):
            end0 = runs[k][1] - 1
            start1 = runs[k + 1][0]
            ax.plot([x_vals[end0], x_vals[start1]], [y_vals[end0], y_vals[start1]], color=color, linestyle="--")
        if stage_starts_opt:
            for i, s in enumerate(stage_starts_opt):
                start_idx = s - 1
                if start_idx < n_ and not np.isnan(y_vals[start_idx]):
                    ax.plot(x_vals[start_idx], y_vals[start_idx], "o", color=color, markersize=8)
                    ax.annotate(f"stage_{i+1}", (x_vals[start_idx], y_vals[start_idx]), textcoords="offset points", xytext=(4, 4), fontsize=9)

    _plot_segments(ax1, tl, "C0", "Train Loss", marker_at=stage_starts)
    _plot_segments(ax1, vl, "C1", "Val Loss", marker_at=stage_starts)
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best")
    ax1.grid(True)

    if np.any(~np.isnan(vp)):
        _plot_segments(ax2, vp, "C2", "Val PSNR (patch)", marker_at=stage_starts)
    # 整图验证：绘制方式与 patch PSNR 一致，缺失 epoch 处用 nan 断开
    if np.any(~np.isnan(vp_full)):
        _plot_segments(ax2, vp_full, "C4", "Val PSNR (full image)", marker_at=stage_starts)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    from matplotlib.ticker import MultipleLocator
    ax1.xaxis.set_major_locator(MultipleLocator(3))
    lines1, labels1 = ax2.get_legend_handles_labels()
    if lines1:
        ax2.legend(lines1, labels1, loc="best")
    plt.suptitle(f"Patch Train Curves ({os.path.basename(save_dir)})")
    plt.tight_layout()
    curve_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()


class PatchDatasetNpy(Dataset):
    """
    Progressive + Overlap：patch_per_image 对每个 (IN, GT) 对。
    每个 GT 对应约 40 个 IN，样本 = 所有 (in_path, gt_path) 对。
    GT: {base}/GT/{id}_GT.npy
    输入: {base}/IN_CR_COM/{id}_*_IN.npy
    use_consist=True 时返回 (inputA, inputB, GT)，同一 GT 下随机取 2 个不同 IN。
    use_slide=True 时返回 (crA, crB, gtA, gtB, t1, l1, t2, l2)，同一 IN 下两重叠 patch。
    """

    def __init__(
        self,
        base_dir: str,
        sample_indices: List[int],
        patch_size: int = 224,
        patch_per_image: int = 1,
        use_low_freq_only: bool = False,
        random_crop: bool = True,
        use_consist: bool = False,
        use_slide: bool = False,
        slide_stride: int = 112,
        gt_subdir: str = "GT_crop_resize",
        in_subdir: str = "IN_CR_COM_crop_resize",
        all_crops_per_id: bool = False,
        aug_flip: bool = True,
        pad_to_patch_size: bool = False,
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.all_crops_per_id = all_crops_per_id
        self.aug_flip = aug_flip
        self.pad_to_patch_size = pad_to_patch_size
        self.gt_dir = os.path.join(self.base_dir, gt_subdir)
        self.in_dir = os.path.join(self.base_dir, in_subdir)
        self.patch_size = patch_size
        self.patch_per_image = patch_per_image
        self.use_low_freq_only = use_low_freq_only
        self.random_crop = random_crop
        self.use_consist = use_consist
        self.use_slide = use_slide
        self.slide_stride = slide_stride
        self._sigma_low = None
        self._sigma_mid = None

        # pairs: (in_path, gt_path, gt_idx)，每个 (IN, GT) 对一条
        self.pairs: List[Tuple[str, str, int]] = []
        # gt_idx -> [pair_idx, ...]，同 GT 的 IN 的 pair 索引
        self.gt_to_pair_indices: List[List[int]] = []
        def _crop_id_key(s):
            p = s.split("_")[0]
            return (int(p), s) if p.isdigit() else (0, s)
        gt_crop_resize_files = sorted(glob.glob(os.path.join(self.gt_dir, "*_GT_crop_resize_R*_C*.npy")))
        if gt_crop_resize_files:
            # 4-crop_resize 模式：*_crop_resize_R{R}_C{C}.npy，R,C in 0,1；每 4 轮遍历完 4 个位置，单 epoch 内各 GT 位置随机不同
            _resize_re = re.compile(r"^(.+)_GT_crop_resize_R(\d+)_C(\d+)\.npy$")
            id_to_crops = {}
            for gt_path in gt_crop_resize_files:
                name = os.path.basename(gt_path)
                m = _resize_re.match(name)
                if not m:
                    continue
                sid, R, C = m.group(1), int(m.group(2)), int(m.group(3))
                if R <= 1 and C <= 1:
                    id_to_crops.setdefault(sid, []).append((R, C, gt_path))
            if id_to_crops and all(len(v) == 4 for v in id_to_crops.values()):
                unique_ids = sorted(id_to_crops.keys(), key=_crop_id_key)
                id_to_in_crops = {}
                for sid in unique_ids:
                    in_prefix = sid.split("_")[0] if sid.split("_")[0].isdigit() else sid
                    for R, C, _ in id_to_crops[sid]:
                        pattern = os.path.join(self.in_dir, f"{in_prefix}_*_IN_crop_resize_R{R}_C{C}.npy")
                        id_to_in_crops[(sid, R, C)] = sorted(glob.glob(pattern))
                self._crop_mode = True
                self._crop_resize_4 = True
                self._id_to_crops = id_to_crops
                self._id_to_in_crops = id_to_in_crops
                self._unique_ids = unique_ids
                self._sample_indices = [i for i in sample_indices if i < len(unique_ids)]
                self._build_crop_epoch_pairs(0)
            else:
                gt_crop_resize_files = []
        if not gt_crop_resize_files or not getattr(self, "_crop_mode", False):
            gt_crop_files = sorted(glob.glob(os.path.join(self.gt_dir, "*_GT_crop_r*_c*.npy")))
            if gt_crop_files:
                self._crop_resize_4 = False
                _crop_re = re.compile(r"^(.+)_GT_crop_r(\d+)_c(\d+)\.npy$")
                id_to_crops = {}
                for gt_path in gt_crop_files:
                    name = os.path.basename(gt_path)
                    m = _crop_re.match(name)
                    if not m:
                        continue
                    sid, r, c = m.group(1), int(m.group(2)), int(m.group(3))
                    id_to_crops.setdefault(sid, []).append((r, c, gt_path))
                unique_ids = sorted(id_to_crops.keys(), key=_crop_id_key)
                id_to_in_crops = {}
                for sid in unique_ids:
                    in_prefix = sid.split("_")[0] if sid.split("_")[0].isdigit() else sid
                    for r, c, _ in id_to_crops[sid]:
                        pattern = os.path.join(self.in_dir, f"{in_prefix}_*_IN_crop_r{r}_c{c}.npy")
                        matches = sorted(glob.glob(pattern))
                        id_to_in_crops[(sid, r, c)] = matches
                self._crop_mode = True
                self._id_to_crops = id_to_crops
                self._id_to_in_crops = id_to_in_crops
                self._unique_ids = unique_ids
                self._sample_indices = [i for i in sample_indices if i < len(unique_ids)]
                self._build_crop_epoch_pairs(0)
            else:
                gt_files = sorted(glob.glob(os.path.join(self.gt_dir, "*_GT.npy")))
                for idx in sample_indices:
                    if idx >= len(gt_files):
                        break
                    gt_path = gt_files[idx]
                    name = os.path.basename(gt_path)
                    if not name.endswith("_GT.npy"):
                        continue
                    sid = name[:-7]
                    pattern = os.path.join(self.in_dir, f"{sid}_*_IN.npy")
                    matches = sorted(glob.glob(pattern))
                    if not matches:
                        continue
                    gt_idx = len(self.gt_to_pair_indices)
                    self.gt_to_pair_indices.append([])
                    for in_path in matches:
                        pair_idx = len(self.pairs)
                        self.pairs.append((in_path, gt_path, gt_idx))
                        self.gt_to_pair_indices[gt_idx].append(pair_idx)
                self._crop_mode = False

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"未找到有效样本。GT 目录: {self.gt_dir}，输入目录: {self.in_dir}；"
                f"支持 *_GT_crop_resize_R*_C*.npy、*_GT_crop_r*_c*.npy 或 *_GT.npy 与对应 IN"
            )

        if use_low_freq_only and self.pairs:
            arr = np.load(self.pairs[0][0]).astype(np.float32)
            h, w = arr.shape[0], arr.shape[1]
            self._sigma_low, self._sigma_mid = _sigma_from_image_freq(h, w)

    def _build_crop_epoch_pairs(self, epoch: int):
        """crop 模式：重建 self.pairs。36-crop 每轮随机一位置；4-crop_resize 每 4 轮遍历完 4 位置，同 epoch 内各 GT 位置随机不同。"""
        self.pairs = []
        self.gt_to_pair_indices = []
        if getattr(self, "_crop_resize_4", False):
            n_pos = 4
            for i, idx in enumerate(self._sample_indices):
                sid = self._unique_ids[idx]
                choices = sorted(self._id_to_crops[sid])
                if len(choices) != n_pos:
                    continue
                # val 时 all_crops_per_id=True：每个 id 加入 4 个 crop，得到 4 个 GT sample；训练时每 epoch 只取 1 个位置
                positions_to_add = list(range(n_pos)) if self.all_crops_per_id else [(i + epoch) % n_pos]
                for pos_idx in positions_to_add:
                    R, C, gt_path = choices[pos_idx]
                    in_paths = self._id_to_in_crops.get((sid, R, C), [])
                    if not in_paths:
                        continue
                    gt_idx = len(self.gt_to_pair_indices)
                    self.gt_to_pair_indices.append([])
                    for in_path in in_paths:
                        pair_idx = len(self.pairs)
                        self.pairs.append((in_path, gt_path, gt_idx))
                        self.gt_to_pair_indices[gt_idx].append(pair_idx)
        else:
            rng = np.random.default_rng(epoch)
            for idx in self._sample_indices:
                sid = self._unique_ids[idx]
                choices = self._id_to_crops[sid]
                r, c, gt_path = choices[rng.integers(0, len(choices))]
                in_paths = self._id_to_in_crops.get((sid, r, c), [])
                if not in_paths:
                    continue
                gt_idx = len(self.gt_to_pair_indices)
                self.gt_to_pair_indices.append([])
                for in_path in in_paths:
                    pair_idx = len(self.pairs)
                    self.pairs.append((in_path, gt_path, gt_idx))
                    self.gt_to_pair_indices[gt_idx].append(pair_idx)

    def set_epoch(self, epoch: int):
        """crop 模式：每轮调用，使本 epoch 内每张 GT 固定一个 (r,c)，4-crop 时每 4 轮遍历完且同轮内各 GT 位置随机不同。"""
        if getattr(self, "_crop_mode", False):
            self._build_crop_epoch_pairs(epoch)

    def __len__(self) -> int:
        return len(self.pairs) * self.patch_per_image

    def _load_npy(self, path: str) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.use_low_freq_only and self._sigma_low is not None:
            img_log = np.log1p(np.clip(arr, 0, None))
            low_log, _, _ = decompose_freq_log(img_log, self._sigma_low, self._sigma_mid)
            arr = np.clip(np.expm1(low_log), 0.0, 1.0).astype(np.float32)
        return arr

    def _crop_patch(self, arr: np.ndarray, top: int, left: int) -> torch.Tensor:
        ph, pw = _patch_hw(self.patch_size)
        patch = arr[top : top + ph, left : left + pw]
        return torch.from_numpy(patch).permute(2, 0, 1)

    def _to_train_size(self, t: torch.Tensor) -> torch.Tensor:
        """若设置了 train_img_size，将 (3,H,W) 缩放到 (3, H, W)。"""
        sz = getattr(self, "train_img_size", None)
        if sz is None:
            return t
        h, w = (sz[0], sz[1]) if isinstance(sz, (tuple, list)) else (sz, sz)
        if t.shape[1] == h and t.shape[2] == w:
            return t
        t = t.unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t.squeeze(0)

    def _apply_same_flip(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """对多个 (C,H,W) 做相同的随机水平/垂直翻转，仅当 random_crop 且 aug_flip 时在外部调用。"""
        if not (getattr(self, "random_crop", False) and getattr(self, "aug_flip", False)):
            return tensors
        flip_h = np.random.rand() < 0.5
        flip_w = np.random.rand() < 0.5
        out = []
        for t in tensors:
            if flip_h:
                t = torch.flip(t, [1])
            if flip_w:
                t = torch.flip(t, [2])
            out.append(t)
        return tuple(out)

    def __getitem__(self, idx: int):
        pair_idx = idx // self.patch_per_image
        in_path, gt_path, gt_idx = self.pairs[pair_idx]
        gt_arr = self._load_npy(gt_path)
        h, w = gt_arr.shape[0], gt_arr.shape[1]
        ph, pw = _patch_hw(self.patch_size)
        if h < ph or w < pw:
            if getattr(self, "pad_to_patch_size", False):
                # 上下左右 pad 到 (ph, pw)，均分：如 768->800 则上下各 16
                pad_top = (ph - h) // 2
                pad_bottom = ph - h - pad_top
                pad_left = (pw - w) // 2
                pad_right = pw - w - pad_left
                gt_arr = np.pad(gt_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
                h, w = ph, pw
            else:
                scale = max(ph / h, pw / w)
                new_h, new_w = int(h * scale), int(w * scale)
                gt_arr = cv2.resize(gt_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w = new_h, new_w
        val_pad = getattr(self, "_val_padding", 0)
        if not self.random_crop and getattr(self, "_val_cover_positions", None) is not None and val_pad > 0:
            p2 = val_pad // 2  # 上下左右平分：上16下16左16右16
            gt_arr = np.pad(gt_arr, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
        if self.random_crop:
            top = np.random.randint(0, max(1, h - ph + 1))
            left = np.random.randint(0, max(1, w - pw + 1))
        else:
            sub_idx = idx % self.patch_per_image
            cover_positions = getattr(self, "_val_cover_positions", None)
            if cover_positions is not None:
                top, left = cover_positions[min(sub_idx, len(cover_positions) - 1)]
            else:
                # 固定位置：0=中心, 1=左上, 2=右上, 3=左下, 4=右下
                positions = [
                    ((h - ph) // 2, (w - pw) // 2),
                    (0, 0),
                    (0, max(0, w - pw)),
                    (max(0, h - ph), 0),
                    (max(0, h - ph), max(0, w - pw)),
                ]
                top, left = positions[min(sub_idx, 4)]
        gt = self._crop_patch(gt_arr, top, left)

        if self.use_consist:
            same_gt = self.gt_to_pair_indices[gt_idx]
            other = [i for i in same_gt if i != pair_idx]
            if other:
                other_pair_idx = int(np.random.choice(other))
                other_in_path = self.pairs[other_pair_idx][0]
            else:
                other_in_path = in_path
            crA_arr = self._load_npy(in_path)
            crB_arr = self._load_npy(other_in_path)
            if crA_arr.shape[0] != h or crA_arr.shape[1] != w:
                crA_arr = cv2.resize(crA_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            if crB_arr.shape[0] != h or crB_arr.shape[1] != w:
                crB_arr = cv2.resize(crB_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            if val_pad > 0:
                p2 = val_pad // 2
                crA_arr = np.pad(crA_arr, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
                crB_arr = np.pad(crB_arr, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
            crA = self._crop_patch(crA_arr, top, left)
            crB = self._crop_patch(crB_arr, top, left)
            crA, crB, gt = self._to_train_size(crA), self._to_train_size(crB), self._to_train_size(gt)
            crA, crB, gt = self._apply_same_flip(crA, crB, gt)
            return crA, crB, gt
        elif self.use_slide:
            cr_arr = self._load_npy(in_path)
            if cr_arr.shape[0] != h or cr_arr.shape[1] != w:
                cr_arr = cv2.resize(cr_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            stride = self.slide_stride if self.slide_stride > 0 else (min(ph, pw) // 2)
            positions = _sliding_window_grid(h, w, ph, pw, stride)
            if len(positions) < 2:
                t1, l1 = 0, 0
                t2, l2 = 0, 0
            else:
                # ppi 次采样尽量覆盖全图：网格按索引分 n_slot 段，sub_idx 决定从哪两段取 patch（两段相距约半图）
                sub_idx = idx % self.patch_per_image
                n_slot = min(self.patch_per_image, len(positions), 64)
                if n_slot < 2:
                    i1, i2 = np.random.choice(len(positions), 2, replace=False)
                else:
                    seg_size = (len(positions) + n_slot - 1) // n_slot
                    seg0 = min((sub_idx % n_slot) * seg_size, len(positions) - 1)
                    seg1_idx = (sub_idx + n_slot // 2) % n_slot
                    seg1 = min(seg1_idx * seg_size, len(positions) - 1)
                    end0 = min(seg0 + seg_size, len(positions))
                    end1 = min(seg1 + seg_size, len(positions))
                    pool0 = list(range(seg0, end0))
                    pool1 = list(range(seg1, end1))
                    if not pool0:
                        pool0 = [seg0]
                    if not pool1:
                        pool1 = [seg1]
                    if seg0 == seg1 or (seg0 < end1 and seg1 < end0):
                        i1, i2 = np.random.choice(len(positions), 2, replace=False)
                    else:
                        i1 = int(np.random.choice(pool0))
                        i2 = int(np.random.choice(pool1))
                        if i1 == i2 and len(positions) > 1:
                            i2 = (i2 + 1) % len(positions)
                (t1, l1), (t2, l2) = positions[int(i1)], positions[int(i2)]
            crA = self._crop_patch(cr_arr, t1, l1)
            crB = self._crop_patch(cr_arr, t2, l2)
            gtA = self._crop_patch(gt_arr, t1, l1)
            gtB = self._crop_patch(gt_arr, t2, l2)
            crA = self._to_train_size(crA)
            crB = self._to_train_size(crB)
            gtA = self._to_train_size(gtA)
            gtB = self._to_train_size(gtB)
            crA, crB, gtA, gtB = self._apply_same_flip(crA, crB, gtA, gtB)
            return crA, crB, gtA, gtB, t1, l1, t2, l2
        else:
            cr_arr = self._load_npy(in_path)
            if cr_arr.shape[0] != h or cr_arr.shape[1] != w:
                if getattr(self, "pad_to_patch_size", False) and (cr_arr.shape[0] < h or cr_arr.shape[1] < w):
                    pad_top = (h - cr_arr.shape[0]) // 2
                    pad_bottom = h - cr_arr.shape[0] - pad_top
                    pad_left = (w - cr_arr.shape[1]) // 2
                    pad_right = w - cr_arr.shape[1] - pad_left
                    cr_arr = np.pad(cr_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
                else:
                    cr_arr = cv2.resize(cr_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            if val_pad > 0:
                p2 = val_pad // 2
                cr_arr = np.pad(cr_arr, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
            cr = self._crop_patch(cr_arr, top, left)
            cr, gt = self._to_train_size(cr), self._to_train_size(gt)
            cr, gt = self._apply_same_flip(cr, gt)
            return cr, gt


def _sliding_window_grid(h: int, w: int, patch_h: int, patch_w: int, stride: int) -> list:
    """滑动窗口网格 (top, left)。"""
    if h < patch_h or w < patch_w:
        return [(0, 0)]
    tops = list(range(0, h - patch_h + 1, stride))
    lefts = list(range(0, w - patch_w + 1, stride))
    if tops[-1] != h - patch_h:
        tops.append(h - patch_h)
    if lefts[-1] != w - patch_w:
        lefts.append(w - patch_w)
    return [(t, l) for t in tops for l in lefts]


def _cover_grid_positions(img_h: int, img_w: int, patch_h: int, patch_w: int, max_stride: int = 224) -> list:
    """
    覆盖整图，stride 不超过 max_stride（默认 224），最后一列/行贴边。
    返回 [(top, left), ...]。patch_h/patch_w 可为矩形。
    """
    if img_h < patch_h or img_w < patch_w:
        return [(0, 0)]
    n_cols = 1 + max(0, (img_w - patch_w + max_stride - 1) // max_stride)
    n_rows = 1 + max(0, (img_h - patch_h + max_stride - 1) // max_stride)
    n_cols = max(1, n_cols)
    n_rows = max(1, n_rows)
    stride_w = (img_w - patch_w) // (n_cols - 1) if n_cols > 1 else 0
    stride_h = (img_h - patch_h) // (n_rows - 1) if n_rows > 1 else 0
    positions = []
    for row in range(n_rows):
        top = (img_h - patch_h) if (n_rows > 1 and row == n_rows - 1) else (row * stride_h)
        for col in range(n_cols):
            left = (img_w - patch_w) if (n_cols > 1 and col == n_cols - 1) else (col * stride_w)
            positions.append((top, left))
    return positions


def _set_val_cover(val_dataset, patch_size: Union[int, Tuple[int, int]], max_stride: int = 224, val_padding: int = 0):
    """val_ppi=-1 时：用首张 val GT 的尺寸计算覆盖网格。patch_size 可为 int 或 (H,W)。"""
    if not val_dataset.pairs:
        return
    ph, pw = _patch_hw(patch_size)
    _, gt_path, _ = val_dataset.pairs[0]
    arr = np.load(gt_path)
    h, w = arr.shape[0], arr.shape[1]
    if h < ph or w < pw:
        scale = max(ph / h, pw / w)
        h = int(h * scale)
        w = int(w * scale)
    if val_padding > 0:
        nh, nw = h + val_padding, w + val_padding
        positions = _cover_grid_positions(nh, nw, ph, pw, max_stride=max_stride)
        val_dataset._val_padding = val_padding
    else:
        positions = _cover_grid_positions(h, w, ph, pw, max_stride=max_stride)
        if getattr(val_dataset, "_val_padding", None) is not None:
            del val_dataset._val_padding
    val_dataset.patch_per_image = len(positions)
    val_dataset._val_cover_positions = positions


def _overlap_merge_weight(patch_h: int, patch_w: Optional[int] = None) -> np.ndarray:
    """线性衰减权重，overlap 平滑融合。patch_w 缺省时与 patch_h 相同（正方形）。"""
    if patch_w is None:
        patch_w = patch_h
    xh = np.linspace(0, 1, patch_h)
    xw = np.linspace(0, 1, patch_w)
    wh = np.minimum(xh, 1 - xh) * 2
    ww = np.minimum(xw, 1 - xw) * 2
    return (wh[:, None] * ww[None, :]).astype(np.float32)


def _rgb_to_lab_uint8(rgb_01: np.ndarray) -> np.ndarray:
    """RGB [0,1] float (H,W,3) -> LAB uint8 (H,W,3)，OpenCV 约定。"""
    rgb_u8 = (np.clip(rgb_01, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)


def _lab_to_rgb_float(lab: np.ndarray) -> np.ndarray:
    """LAB uint8 (H,W,3) -> RGB [0,1] float (H,W,3)。"""
    rgb_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_u8.astype(np.float32) / 255.0


def _lock_l_merge(input_rgb_01: np.ndarray, pred_rgb_01: np.ndarray) -> np.ndarray:
    """验证时锁定 L：用输入的 L 与预测的 AB 合成。input/pred 均为 RGB [0,1] (H,W,3)。"""
    lab_in = _rgb_to_lab_uint8(input_rgb_01)
    lab_pred = _rgb_to_lab_uint8(np.clip(pred_rgb_01, 0.0, 1.0))
    lab_merge = np.stack([lab_in[:, :, 0], lab_pred[:, :, 1], lab_pred[:, :, 2]], axis=-1)
    return _lab_to_rgb_float(lab_merge)


def _load_npy_float_for_vis(path: str) -> np.ndarray:
    """加载 npy 为 [0,1] float RGB (H,W,3)。"""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return arr


def run_val_full_coverage_unified(
    model: nn.Module,
    val_dataset: PatchDatasetNpy,
    device: torch.device,
    save_dir: str,
    stage_idx: int,
    epoch: int,
    suffix: str,
    patch_size: Union[int, Tuple[int, int]],
    stride: int,
    img_size: Union[int, Tuple[int, int]],
    val_padding: int,
    perceptual_loss_fn: Optional[nn.Module],
    use_rgb_lab_loss: bool,
    loss_l1: float,
    loss_mse: float,
    loss_grad: float,
    loss_ssim: float,
    loss_percep: float,
    loss_ab_l1: float = 0.0,
    loss_ab_mse: float = 0.0,
    lock_l: bool = False,
) -> Tuple[float, float, float]:
    """
    全覆盖 val 与整图验证合并：遍历每个 GT 的**全部** IN，一次前向得到 val_loss（逐 patch 平均）+ 整图 PSNR + 保存 IN-OUT-GT 图。
    patch_size / img_size 可为 int（正方形）或 (H, W)。lock_l 为 True 时用输入 L + 预测 AB 合成后再算 loss/PSNR 与整图。
    """
    model.eval()
    ph, pw = _patch_hw(patch_size)
    transform = T.ToTensor()  # 不 resize，按 patch 直接前向
    weight_w = _overlap_merge_weight(ph, pw)

    # 每个 GT 文件（每个 crop）只用 1 个 IN；4-crop 时每个 GT id 有 4 个 gt_idx，故每 id 共 4 个 val 样本
    val_samples: List[Tuple[str, str, int]] = []
    for gt_idx in range(len(val_dataset.gt_to_pair_indices)):
        pair_indices = val_dataset.gt_to_pair_indices[gt_idx]
        if not pair_indices:
            continue
        pair_idx = pair_indices[0]
        in_path, gt_path, _ = val_dataset.pairs[pair_idx]
        val_samples.append((in_path, gt_path, 0))

    vis_dir = os.path.join(save_dir, f"vis_fullimage_stage{stage_idx + 1}{suffix}")
    os.makedirs(vis_dir, exist_ok=True)

    total_loss = 0.0
    total_patches = 0
    total_psnr_patch = 0.0
    total_fullimage_psnr = 0.0
    n_samples = 0
    n_total_samples = len(val_samples)
    _val_start = time.perf_counter()

    with torch.no_grad():
        for idx, (in_path, gt_path, in_sub_idx) in enumerate(val_samples):
            in_np = _load_npy_float_for_vis(in_path)
            gt_np = _load_npy_float_for_vis(gt_path)
            h, w = in_np.shape[0], in_np.shape[1]
            if gt_np.shape[0] != h or gt_np.shape[1] != w:
                gt_np = cv2.resize(gt_np, (w, h), interpolation=cv2.INTER_LINEAR)
            need_pad = h < ph or w < pw
            if need_pad:
                pad_h, pad_w = max(0, ph - h), max(0, pw - w)
                work_np = np.pad(in_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                work_gt = np.pad(gt_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                nh, nw = h + pad_h, w + pad_w
            else:
                work_np = in_np
                work_gt = gt_np
                nh, nw = h, w
            if val_padding > 0:
                p2 = val_padding // 2
                work_np = np.pad(work_np, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
                work_gt = np.pad(work_gt, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
                nh, nw = h + val_padding, w + val_padding
                positions = _cover_grid_positions(nh, nw, ph, pw, max_stride=stride)
            else:
                positions = _sliding_window_grid(nh, nw, ph, pw, stride)
            if not positions:
                positions = [(0, 0)]

            acc = np.zeros((nh, nw, 3), dtype=np.float64)
            wacc = np.zeros((nh, nw), dtype=np.float64)

            for top, left in positions:
                rh = min(ph, nh - top)
                rw = min(pw, nw - left)
                patch_in = work_np[top : top + rh, left : left + rw]
                patch_gt = work_gt[top : top + rh, left : left + rw]
                if patch_in.shape[0] < ph or patch_in.shape[1] < pw:
                    patch_in = np.pad(patch_in, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge")
                    patch_gt = np.pad(patch_gt, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge")
                x_in = transform(Image.fromarray((patch_in * 255).astype(np.uint8))).unsqueeze(0).to(device)
                x_gt = transform(Image.fromarray((patch_gt * 255).astype(np.uint8))).unsqueeze(0).to(device)
                pred = model(x_in)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                    pred_np = np.array(T.ToPILImage()(pred.squeeze(0).cpu()).resize((pw, ph), Image.BICUBIC)).astype(np.float32) / 255.0
                if lock_l:
                    pred_np = _lock_l_merge(patch_in, pred_np)
                    pred_for_loss = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                else:
                    pred_for_loss = pred
                loss = compute_val_patch_loss(
                    pred_for_loss, x_gt, device,
                    perceptual_loss_fn=perceptual_loss_fn,
                    use_rgb_lab_loss=use_rgb_lab_loss,
                    loss_l1=loss_l1, loss_mse=loss_mse, loss_grad=loss_grad, loss_ssim=loss_ssim, loss_percep=loss_percep,
                    loss_ab_l1=loss_ab_l1, loss_ab_mse=loss_ab_mse,
                )
                total_loss += loss.item()
                total_patches += 1
                total_psnr_patch += float(_psnr(pred_for_loss, x_gt))

                wp = weight_w[:rh, :rw]
                acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wp[:, :, None]
                wacc[top : top + rh, left : left + rw] += wp

            wacc = np.maximum(wacc, 1e-8)
            out_np = (acc / wacc[:, :, None]).astype(np.float32)
            if need_pad:
                out_np = out_np[:h, :w]
            elif val_padding > 0:
                p2 = val_padding // 2
                out_np = out_np[p2 : p2 + h, p2 : p2 + w]
            if lock_l:
                out_np = _lock_l_merge(in_np, out_np)

            vis = np.concatenate([in_np, out_np, gt_np], axis=1)
            vis = (np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8)
            sid = os.path.basename(gt_path).replace("_GT.npy", "").replace(".npy", "")
            out_path = os.path.join(vis_dir, f"{sid}_IN{in_sub_idx}_OUT_GT.png")
            Image.fromarray(vis).save(out_path)

            pred_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            total_fullimage_psnr += float(_psnr(pred_t, gt_t))
            n_samples += 1
            elapsed = time.perf_counter() - _val_start
            pct = 100.0 * (idx + 1) / n_total_samples if n_total_samples else 0.0
            eta = (elapsed / (idx + 1) * (n_total_samples - idx - 1)) if (idx + 1) > 0 else 0.0
            print(
                f"\r  全覆盖 val+整图 [ {idx + 1}/{n_total_samples} samples, {total_patches} patches ] "
                f"{pct:.1f}%  已耗时 {elapsed:.1f}s  预计剩余 {eta:.1f}s",
                end="",
                flush=True,
            )

    print()
    val_loss = total_loss / total_patches if total_patches else 0.0
    val_psnr_patch = total_psnr_patch / total_patches if total_patches else 0.0
    fullimage_psnr = total_fullimage_psnr / n_samples if n_samples else 0.0
    print(f"  全覆盖 val+整图 {vis_dir}/  val_loss: {val_loss:.6f}  patch_PSNR: {val_psnr_patch:.2f}  整图PSNR: {fullimage_psnr:.2f}")
    model.train()
    return val_loss, val_psnr_patch, fullimage_psnr


def run_val_sliding_window_vis(
    model: nn.Module,
    val_dataset: PatchDatasetNpy,
    device: torch.device,
    save_dir: str,
    stage_idx: int,
    patch_size: Union[int, Tuple[int, int]] = 256,
    stride: int = 224,
    img_size: Union[int, Tuple[int, int]] = 256,
    suffix: str = "",
    val_padding: int = 0,
    lock_l: bool = False,
) -> Tuple[float, float, float]:
    """
    整图验证：滑动窗口推理，输出原始分辨率 IN-OUT-GT 对比可视化。
    patch_size / img_size 可为 int 或 (H, W)。lock_l 为 True 时用输入 L + 预测 AB 合成后再算 PSNR/SSIM。
    """
    model.eval()
    ph, pw = _patch_hw(patch_size)
    if _lpips_fn is not None:
        _lpips_fn.to(device)
        _lpips_fn.eval()
    transform = T.ToTensor()  # 不 resize，按 patch 直接前向
    to_pil = T.ToPILImage()
    weight_w = _overlap_merge_weight(ph, pw)

    # 每个 GT 文件（每个 crop）只用 1 个 IN，与 run_val_full_coverage_unified 一致
    val_samples: List[Tuple[str, str, int]] = []  # (in_path, gt_path, in_sub_idx)
    for gt_idx in range(len(val_dataset.gt_to_pair_indices)):
        pair_indices = val_dataset.gt_to_pair_indices[gt_idx]
        if not pair_indices:
            continue
        pair_idx = pair_indices[0]
        in_path, gt_path, _ = val_dataset.pairs[pair_idx]
        val_samples.append((in_path, gt_path, 0))

    vis_dir = os.path.join(save_dir, f"vis_fullimage_stage{stage_idx + 1}{suffix}")
    os.makedirs(vis_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    n_samples = 0
    total_samples = len(val_samples)

    with torch.no_grad():
        for idx, (in_path, gt_path, in_sub_idx) in enumerate(val_samples):
            in_np = _load_npy_float_for_vis(in_path)
            gt_np = _load_npy_float_for_vis(gt_path)
            h, w = in_np.shape[0], in_np.shape[1]
            if gt_np.shape[0] != h or gt_np.shape[1] != w:
                gt_np = cv2.resize(gt_np, (w, h), interpolation=cv2.INTER_LINEAR)
            need_pad = h < ph or w < pw
            if need_pad:
                pad_h, pad_w = max(0, ph - h), max(0, pw - w)
                work_np = np.pad(in_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                nh, nw = h + pad_h, w + pad_w
            else:
                work_np = in_np
                nh, nw = h, w
            if val_padding > 0:
                p2 = val_padding // 2  # 上下左右平分：上16下16左16右16
                work_np = np.pad(work_np, ((p2, p2), (p2, p2), (0, 0)), mode="edge")
                nh, nw = h + val_padding, w + val_padding
                positions = _cover_grid_positions(nh, nw, ph, pw, max_stride=stride)
            else:
                positions = _sliding_window_grid(nh, nw, ph, pw, stride)
            if not positions:
                positions = [(0, 0)]

            acc = np.zeros((nh, nw, 3), dtype=np.float64)
            wacc = np.zeros((nh, nw), dtype=np.float64)

            for top, left in positions:
                rh = min(ph, nh - top)
                rw = min(pw, nw - left)
                patch = work_np[top : top + rh, left : left + rw]
                if patch.shape[0] < ph or patch.shape[1] < pw:
                    patch = np.pad(patch, ((0, ph - rh), (0, pw - rw), (0, 0)), mode="edge")
                patch_pil = Image.fromarray((patch * 255).astype(np.uint8))
                x = transform(patch_pil).unsqueeze(0).to(device)
                pred = model(x)
                pred = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
                pred_np = pred.numpy().transpose(1, 2, 0)
                if pred_np.shape[0] != ph or pred_np.shape[1] != pw:
                    pred_pil = to_pil(pred)
                    pred_pil = pred_pil.resize((pw, ph), Image.BICUBIC)
                    pred_np = np.array(pred_pil).astype(np.float32) / 255.0
                wp = weight_w[:rh, :rw]
                acc[top : top + rh, left : left + rw] += pred_np[:rh, :rw] * wp[:, :, None]
                wacc[top : top + rh, left : left + rw] += wp

            wacc = np.maximum(wacc, 1e-8)
            out_np = (acc / wacc[:, :, None]).astype(np.float32)
            if need_pad:
                out_np = out_np[:h, :w]
            elif val_padding > 0:
                p2 = val_padding // 2
                out_np = out_np[p2 : p2 + h, p2 : p2 + w]
            if lock_l:
                out_np = _lock_l_merge(in_np, out_np)

            # 原始分辨率 IN | OUT | GT 横向拼接
            vis = np.concatenate([in_np, out_np, gt_np], axis=1)
            vis = (np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8)
            sid = os.path.basename(gt_path).replace("_GT.npy", "")
            out_path = os.path.join(vis_dir, f"{sid}_IN{in_sub_idx}_OUT_GT.png")
            Image.fromarray(vis).save(out_path)

            # PSNR, SSIM, LPIPS (OUT vs GT)
            pred_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            total_psnr += _psnr(pred_t, gt_t)
            total_ssim += _ssim_value(pred_t, gt_t)
            if _lpips_fn is not None:
                total_lpips += float(_lpips_fn(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).mean().item())
            n_samples += 1
            print(f"\r  整图验证 [ {idx + 1}/{total_samples} ]", end="", flush=True)

    print()
    psnr_avg = total_psnr / n_samples if n_samples else 0.0
    ssim_avg = total_ssim / n_samples if n_samples else 0.0
    lpips_avg = total_lpips / n_samples if (n_samples and _lpips_fn is not None) else -1.0
    lpips_str = f" LPIPS: {lpips_avg:.4f}" if lpips_avg >= 0 else ""
    print(f"  整图验证 {vis_dir}/  PSNR: {psnr_avg:.2f}  SSIM: {ssim_avg:.4f}{lpips_str}")
    full_log = os.path.join(save_dir, "fullimage_metrics.txt")
    lpips_out = lpips_avg if lpips_avg >= 0 else -1.0
    with open(full_log, "a", encoding="utf-8") as f:
        f.write(_fmt_fullimage_line(f"stage{stage_idx+1}{suffix}", psnr_avg, ssim_avg, lpips_out) + "\n")

    model.train()
    return psnr_avg, ssim_avg, lpips_avg


# Progressive + Overlap: (patch, patch_per_image, batch, epochs, 学习目标 [, patch_range, val_override])
# 三个 stage 默认最大轮数均为 20；stage4 使用 IN_CR_COM_pred，img_size=patch=输入图尺寸
STAGES = [
    (128, 10, 8, 20, "texture, illumination statistics", None),
    (192, 8, 5, 20, "mid-range structure", None),
    # stage3: val 固定 288/256/32 → 4×3 铺满、相邻重叠 32；(val_patch, val_max_stride, val_padding)
    (256, 6, 3, 20, "global consistency", (224, 288), (288, 256, 32)),
    # stage4: 仅用 IN_CR_COM_pred，patch=img_size=输入图尺寸+上下左右各 pad 16，1 patch/图
    (256, 1, 3, 20, "full-image pred refinement", None),
]
IN_SUBDIR_STAGE4 = "IN_CR_COM_pred"
# Stage4 使用「输入图尺寸 + 上下左右各 pad 16」：(H, W) = (768+32, 1024+32) = (800, 1056)
REF_IMAGE_H, REF_IMAGE_W = 768, 1024
STAGE4_PAD = 16
STAGE4_IMG_SIZE = (REF_IMAGE_H + 2 * STAGE4_PAD, REF_IMAGE_W + 2 * STAGE4_PAD)  # (800, 1056) (H, W)

# U-Net U 型：3down->2down->1->2up->3up，(patch, ppi, batch, epochs, name, in_channels)
# 2up/3up 使用 6ch 输入 [rgb, skip_rgb]，skip 来自同名 down 的 best_loss
UU_STAGES = [
    (256, 4, 3, 10, "3down", 3),
    (192, 6, 5, 10, "2down", 3),
    (128, 10, 8, 20, "1", 3),
    (192, 6, 5, 10, "2up", 6),   # skip from 2down
    (256, 4, 3, 10, "3up", 6),   # skip from 3down
]


class ModelWithSkip(nn.Module):
    """方案 B：将 skip_model 输出拼接到输入，main_model 接收 6ch。"""
    def __init__(self, main_model: nn.Module, skip_model: nn.Module, device: torch.device):
        super().__init__()
        self.main_model = main_model
        self.skip_model = skip_model
        self.skip_model.eval()
        for p in self.skip_model.parameters():
            p.requires_grad = False
        self._device = device
        self.img_size = getattr(main_model, "img_size", 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            skip = self.skip_model(x)
        x6 = torch.cat([x, skip], dim=1)
        return self.main_model(x6)


def run_stage(
    stage_idx: int,
    patch_size: int,
    patch_per_image: int,
    batch_size: int,
    epochs: int,
    stage_desc: str,
    train_dataset: PatchDatasetNpy,
    val_dataset: PatchDatasetNpy,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_dir: str,
    perceptual_loss_fn: Optional[nn.Module],
    args,
    train_losses: List[float],
    val_losses: List[float],
    val_psnrs: List[float],
    val_ssims: List[float],
    best_val_loss: float,
    best_val_psnr: float,
    start_epoch: int = 1,
    stage_starts: Optional[List[int]] = None,
    patience_stage: Optional[int] = None,
    patience_neighbor: Optional[int] = None,
    patch_size_range: Optional[Tuple[int, int]] = None,
    val_override: Optional[Tuple[int, int, int]] = None,
    val_psnrs_fullimage: Optional[List[float]] = None,
    val_ssims_fullimage: Optional[List[float]] = None,
    skip_train_resize: bool = False,
) -> Tuple[float, float]:
    """运行一个 stage，返回更新后的 best_val_loss, best_val_psnr。整图验证仅在 1/3/5/10 轮及 stage3 每轮运行，结果写入 val_psnrs_fullimage/val_ssims_fullimage 供曲线绘制。"""
    ps = patience_stage if patience_stage is not None else args.patience_stage
    pn = patience_neighbor if patience_neighbor is not None else args.patience_neighbor
    if stage_starts is not None:
        stage_starts.append(len(train_losses) + 1)
    train_dataset.patch_size = patch_size
    train_dataset.patch_per_image = patch_per_image
    train_dataset.random_crop = getattr(train_dataset, "random_crop", True)  # Stage4 已设为 False，不覆盖
    # Stage4 skip_train_resize=True 时不 resize，直接 800×1056 送入模型
    train_dataset.train_img_size = None if skip_train_resize else (getattr(train_dataset, "train_img_size", None) or getattr(args, "train_img_size", None))
    val_patch = (val_override[0] if val_override else patch_size)
    val_dataset.patch_size = val_patch
    val_ppi = getattr(args, "val_ppi", -1)
    if val_ppi == -1:
        val_dataset.random_crop = False
        if val_override is not None:
            _set_val_cover(val_dataset, val_override[0], val_override[1], val_override[2])
        else:
            # 整图推理：stride=img_size-32，上下左右各 pad 16
            ph, pw = _patch_hw(patch_size)
            _set_val_cover(val_dataset, patch_size, min(ph, pw) - 32, 32)
    else:
        val_dataset.patch_per_image = val_ppi
        val_dataset.random_crop = False
        if getattr(val_dataset, "_val_cover_positions", None) is not None:
            del val_dataset._val_cover_positions

    num_gpus = getattr(args, "num_gpus", 1)
    effective_batch = batch_size * num_gpus
    _dl_kw = {
        "batch_size": effective_batch,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        _dl_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **_dl_kw)

    no_improve_count = 0
    no_improve_neighbor = 0
    stage_best_loss = float("inf")   # 本 stage 内最优，用于 patience_stage 判断
    stage_best_psnr = float("-inf")
    prev_val_loss, prev_val_psnr = None, None

    for epoch in range(start_epoch, epochs + 1):
        if getattr(train_dataset, "set_epoch", None) is not None:
            train_dataset.set_epoch(epoch)
        if patch_size_range is not None:
            lo, hi = patch_size_range[0], patch_size_range[1]
            # 支持矩形：( (lo_h,lo_w), (hi_h,hi_w) ) 或 正方形 (lo, hi)
            if isinstance(lo, (tuple, list)) and isinstance(hi, (tuple, list)):
                lo_h, lo_w, hi_h, hi_w = lo[0], lo[1], hi[0], hi[1]
                ps_train = (int(np.random.randint(lo_h, hi_h + 1)), int(np.random.randint(lo_w, hi_w + 1)))
            else:
                span = hi - lo + 1
                margin = max(1, span // 4)
                small_hi = lo + margin - 1
                large_lo = hi - margin + 1
                mid_ratio = getattr(args, "patch_mid_ratio", 0.6)
                large_ratio = getattr(args, "patch_large_ratio", 0.15)
                small_ratio = 1.0 - mid_ratio - large_ratio
                if large_lo > small_hi + 1 and small_ratio >= 0 and large_ratio >= 0:
                    r = np.random.rand()
                    if r < small_ratio:
                        ps_train = int(np.random.randint(lo, small_hi + 1))
                    elif r < small_ratio + mid_ratio:
                        ps_train = int(np.random.randint(small_hi + 1, large_lo))
                    else:
                        ps_train = int(np.random.randint(large_lo, hi + 1))
                else:
                    ps_train = int(np.random.randint(lo, hi + 1))
            train_dataset.patch_size = ps_train
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, max_epochs=epochs,
            perceptual_loss_fn=perceptual_loss_fn,
            use_rgb_lab_loss=args.loss_rgb_lab,
            loss_l1=getattr(args, "loss_l1", 1.0),
            loss_mse=getattr(args, "loss_mse", 0.0),
            loss_grad=args.loss_grad,
            loss_ssim=args.loss_ssim,
            loss_percep=args.loss_percep,
            loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
            loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
            loss_consist=args.loss_consist,
            loss_slide=args.loss_slide,
            loss_consist_l1=args.loss_consist_l1,
            loss_consist_mse=args.loss_consist_mse,
            loss_slide_l1=args.loss_slide_l1,
            loss_slide_mse=args.loss_slide_mse,
        )
        val_ppi = getattr(args, "val_ppi", -1)
        if val_ppi == -1 and getattr(val_dataset, "_val_cover_positions", None) is not None:
            # 全覆盖：val 与整图验证合并，一次前向得到 val_loss + 整图 PSNR + vis，遍历每个 GT 全部 IN
            if val_override:
                _vps, _vst, _vpad = val_override[0], val_override[1], val_override[2]
            else:
                _vps = patch_size
                ph, pw = _patch_hw(patch_size)
                _vst = min(ph, pw) - 32
                _vpad = 32  # 上下左右各 pad 16
            val_loss, val_psnr, fullimage_psnr = run_val_full_coverage_unified(
                model=model,
                val_dataset=val_dataset,
                device=device,
                save_dir=save_dir,
                stage_idx=stage_idx,
                epoch=epoch,
                suffix=f"_epoch{epoch}",
                patch_size=_vps,
                stride=_vst,
                img_size=(getattr(args, "val_img_size", None) or getattr(_unwrap_model(model), "img_size", 256)),
                val_padding=_vpad,
                perceptual_loss_fn=perceptual_loss_fn,
                use_rgb_lab_loss=args.loss_rgb_lab,
                loss_l1=getattr(args, "loss_l1", 1.0),
                loss_mse=getattr(args, "loss_mse", 0.0),
                loss_grad=args.loss_grad,
                loss_ssim=args.loss_ssim,
                loss_percep=args.loss_percep,
                loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
                loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
                lock_l=getattr(args, "lock_l", False),
            )
            val_ssim = 0.0
            if val_psnrs_fullimage is not None and val_ssims_fullimage is not None:
                global_epoch = len(train_losses) + 1
                while len(val_psnrs_fullimage) < global_epoch:
                    val_psnrs_fullimage.append(float("nan"))
                while len(val_ssims_fullimage) < global_epoch:
                    val_ssims_fullimage.append(float("nan"))
                val_psnrs_fullimage[global_epoch - 1] = fullimage_psnr
                val_ssims_fullimage[global_epoch - 1] = 0.0
            # 全覆盖时也写入 fullimage_metrics.txt（PSNR 有值，SSIM/LPIPS 未算填 0/-1）
            full_log = os.path.join(save_dir, "fullimage_metrics.txt")
            with open(full_log, "a", encoding="utf-8") as f:
                f.write(_fmt_fullimage_line(f"stage{stage_idx+1}_epoch{epoch}", fullimage_psnr, 0.0, -1.0) + "\n")
        else:
            val_loss, _, _, val_psnr, val_ssim = eval_one_epoch(
                model, val_loader, device,
                perceptual_loss_fn=perceptual_loss_fn,
                use_rgb_lab_loss=args.loss_rgb_lab,
                loss_l1=getattr(args, "loss_l1", 1.0),
                loss_mse=getattr(args, "loss_mse", 0.0),
                loss_grad=args.loss_grad,
                loss_ssim=args.loss_ssim,
                loss_percep=args.loss_percep,
                loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
                loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
                lock_l=getattr(args, "lock_l", False),
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        global_epoch = len(train_losses)
        stage_names = ("one", "two", "three", "four")
        stage_name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage{stage_idx+1}"
        print(f"[stage {stage_idx+1}] Epoch [{epoch}/{epochs}] (global {global_epoch}) "
              f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  PSNR: {val_psnr:.2f}  SSIM: {val_ssim:.4f}")

        # 第 1、3、5、10 轮保存 pth；stage3/4 每轮跑整图验证
        if epoch in (1, 3, 5, 10) or stage_idx >= 2:
            ck = {
                "stage": stage_idx,
                "epoch": epoch,
                "model_state": _unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_psnrs": val_psnrs,
                "val_ssims": val_ssims,
                "val_psnrs_fullimage": list(val_psnrs_fullimage or []),
                "val_ssims_fullimage": list(val_ssims_fullimage or []),
                "best_val_loss": best_val_loss,
                "best_val_psnr": max(best_val_psnr, val_psnr),
                "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(model), "img_size", 256),
            }
            if stage_starts is not None:
                ck["stage_starts"] = list(stage_starts)
            torch.save(ck, os.path.join(save_dir, f"epoch[{epoch}]_stage_{stage_name}.pth"))
            if val_ppi != -1:
                if val_override:
                    _vps, _vst, _vpad = val_override[0], val_override[1], val_override[2]
                else:
                    _vps = patch_size
                    ph, pw = _patch_hw(patch_size)
                    _vst = min(ph, pw) - 32
                    _vpad = 32  # 上下左右各 pad 16
                psnr_full, ssim_full, _ = run_val_sliding_window_vis(
                    model=model,
                    val_dataset=val_dataset,
                    device=device,
                    save_dir=save_dir,
                    stage_idx=stage_idx,
                    patch_size=_vps,
                    stride=_vst,
                    img_size=(getattr(args, "val_img_size", None) or getattr(_unwrap_model(model), "img_size", 256)),
                    suffix=f"_epoch{epoch}",
                    val_padding=_vpad,
                    lock_l=getattr(args, "lock_l", False),
                )
                if val_psnrs_fullimage is not None and val_ssims_fullimage is not None:
                    while len(val_psnrs_fullimage) < global_epoch:
                        val_psnrs_fullimage.append(float("nan"))
                    while len(val_ssims_fullimage) < global_epoch:
                        val_ssims_fullimage.append(float("nan"))
                    val_psnrs_fullimage[global_epoch - 1] = psnr_full
                    val_ssims_fullimage[global_epoch - 1] = ssim_full

        log_txt = os.path.join(save_dir, "log.txt")
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(_fmt_log_line(f"stage{stage_idx+1}_{epoch}", train_loss, val_loss, val_psnr, val_ssim) + "\n")

        improved_loss = val_loss < best_val_loss
        improved_psnr = val_psnr > best_val_psnr
        improved_this_stage = val_loss < stage_best_loss or val_psnr > stage_best_psnr
        if improved_this_stage:
            stage_best_loss = min(stage_best_loss, val_loss)
            stage_best_psnr = max(stage_best_psnr, val_psnr)

        if improved_loss:
            best_val_loss = val_loss
            loss_path = os.path.join(save_dir, f"best_loss_epoch_{global_epoch}.pth")
            for old in glob.glob(os.path.join(save_dir, "best_loss_epoch_*.pth")):
                if old != loss_path:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            ck = {
                "stage": stage_idx,
                "epoch": epoch,
                "model_state": _unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_psnrs": val_psnrs,
                "val_ssims": val_ssims,
                "val_psnrs_fullimage": list(val_psnrs_fullimage or []),
                "val_ssims_fullimage": list(val_ssims_fullimage or []),
                "best_val_loss": best_val_loss,
                "best_val_psnr": max(best_val_psnr, val_psnr),
                "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(model), "img_size", 256),
            }
            if stage_starts is not None:
                ck["stage_starts"] = list(stage_starts)
            torch.save(ck, loss_path)
            shutil.copy(loss_path, os.path.join(save_dir, f"best_loss_stage_{stage_name}.pth"))
            print(f"  保存最优 loss 到 best_loss_epoch_{global_epoch}.pth")
        elif improved_this_stage:
            # 本 stage 有提升但未破全局最优时，也保存 best_loss_stage_X（避免 early stop 后缺失）
            ck = {
                "stage": stage_idx,
                "epoch": epoch,
                "model_state": _unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_psnrs": val_psnrs,
                "val_ssims": val_ssims,
                "val_psnrs_fullimage": list(val_psnrs_fullimage or []),
                "val_ssims_fullimage": list(val_ssims_fullimage or []),
                "best_val_loss": best_val_loss,
                "best_val_psnr": max(best_val_psnr, val_psnr),
                "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(model), "img_size", 256),
            }
            if stage_starts is not None:
                ck["stage_starts"] = list(stage_starts)
            stage_path = os.path.join(save_dir, f"best_loss_stage_{stage_name}.pth")
            torch.save(ck, stage_path)
            print(f"  保存本 stage 最优到 best_loss_stage_{stage_name}.pth")

        if improved_psnr:
            best_val_psnr = val_psnr
            vis_dir = os.path.join(save_dir, "vis_val_sample")
            os.makedirs(vis_dir, exist_ok=True)
            visualize_predictions(
                model, val_loader, device,
                out_dir=vis_dir,
                max_batches=args.vis_batches,
                use_lab_color=False,
                prefix="val_sample",
            )
            psnr_path = os.path.join(save_dir, f"best_PSNR_epoch_{global_epoch}.pth")
            for old in glob.glob(os.path.join(save_dir, "best_PSNR_epoch_*.pth")):
                if old != psnr_path:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            torch.save({
                "stage": stage_idx,
                "epoch": epoch,
                "model_state": _unwrap_model(model).state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_val_psnr": best_val_psnr,
                "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(model), "img_size", 256),
            }, psnr_path)
            print(f"  保存最优 PSNR ({val_psnr:.2f}) 到 best_PSNR_epoch_{global_epoch}.pth")

        if not improved_this_stage:
            no_improve_count += 1
        else:
            no_improve_count = 0

        improved_vs_prev = (prev_val_loss is None) or (val_loss < prev_val_loss or val_psnr > prev_val_psnr)
        if not improved_vs_prev:
            no_improve_neighbor += 1
        else:
            no_improve_neighbor = 0
        prev_val_loss, prev_val_psnr = val_loss, val_psnr

        if no_improve_count >= ps:
            print(f"  Early stop: 本 stage 连续 {ps} 轮无 best_loss 且无 best_psnr")
        elif no_improve_neighbor >= pn:
            print(f"  Early stop: 连续 {pn} 轮没比上一轮好")
        else:
            # 未 early stop：正常进入下一轮
            pass
        if no_improve_count >= ps or no_improve_neighbor >= pn:
            # early stop 时也要保存 latest 并画入最后一轮曲线
            ck = {
                "stage": stage_idx,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_psnrs": val_psnrs,
                "val_ssims": val_ssims,
                "val_psnrs_fullimage": list(val_psnrs_fullimage or []),
                "val_ssims_fullimage": list(val_ssims_fullimage or []),
                "best_val_loss": best_val_loss,
                "best_val_psnr": max(best_val_psnr, val_psnr),
                "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(model), "img_size", 256),
            }
            if stage_starts is not None:
                ck["stage_starts"] = list(stage_starts)
            torch.save(ck, os.path.join(save_dir, "latest.pth"))
            _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=stage_starts,
                             val_psnrs_fullimage=val_psnrs_fullimage, val_ssims_fullimage=val_ssims_fullimage)
            break

        # 每轮保存 latest.pth（含 stage、epoch，便于 --resume latest 恢复）
        ck = {
            "stage": stage_idx,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_psnrs": val_psnrs,
            "val_ssims": val_ssims,
            "val_psnrs_fullimage": list(val_psnrs_fullimage or []),
            "val_ssims_fullimage": list(val_ssims_fullimage or []),
            "best_val_loss": best_val_loss,
            "best_val_psnr": max(best_val_psnr, val_psnr),
            "save_dir": save_dir,
            "img_size": getattr(_unwrap_model(model), "img_size", 256),
        }
        if stage_starts is not None:
            ck["stage_starts"] = list(stage_starts)
        torch.save(ck, os.path.join(save_dir, "latest.pth"))
        _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=stage_starts,
                         val_psnrs_fullimage=val_psnrs_fullimage, val_ssims_fullimage=val_ssims_fullimage)

    # Stage 结束：若本 stage 有提升则已在循环内保存 best_loss_stage_X；若无提升则从 best_loss_epoch 复制（避免覆盖已保存的 stage best）
    stage_names = ("one", "two", "three", "four")
    stage_name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage{stage_idx+1}"
    stage_path = os.path.join(save_dir, f"best_loss_stage_{stage_name}.pth")
    if not os.path.isfile(stage_path):
        best_epoch_files = sorted(glob.glob(os.path.join(save_dir, "best_loss_epoch_*.pth")))
        if best_epoch_files:
            latest = max(best_epoch_files, key=os.path.getmtime)
            shutil.copy(latest, stage_path)
            print(f"  保存 best_loss_stage_{stage_name}.pth（本 stage 无提升，从 best_loss_epoch 复制）")

    # 非全覆盖时：stage 结束再跑一次整图验证并写 fullimage 曲线；全覆盖时每轮已由 unified 完成，不再重复
    if getattr(args, "val_ppi", -1) != -1:
        if val_override:
            _vps, _vst, _vpad = val_override[0], val_override[1], val_override[2]
        else:
            _vps = patch_size
            ph, pw = _patch_hw(patch_size)
            _vst = min(ph, pw) - 32
            _vpad = 32  # 上下左右各 pad 16
        psnr_full, ssim_full, _ = run_val_sliding_window_vis(
            model=model,
            val_dataset=val_dataset,
            device=device,
            save_dir=save_dir,
            stage_idx=stage_idx,
            patch_size=_vps,
            stride=_vst,
            img_size=(getattr(args, "val_img_size", None) or getattr(_unwrap_model(model), "img_size", 256)),
            val_padding=_vpad,
            lock_l=getattr(args, "lock_l", False),
        )
        if val_psnrs_fullimage is not None and val_ssims_fullimage is not None:
            n_epochs = len(train_losses)
            while len(val_psnrs_fullimage) < n_epochs:
                val_psnrs_fullimage.append(float("nan"))
            while len(val_ssims_fullimage) < n_epochs:
                val_ssims_fullimage.append(float("nan"))
            val_psnrs_fullimage[n_epochs - 1] = psnr_full
            val_ssims_fullimage[n_epochs - 1] = ssim_full

    return best_val_loss, best_val_psnr


def run_uu_stage(
    stage_idx: int,
    patch_size: int,
    patch_per_image: int,
    batch_size: int,
    epochs: int,
    stage_name: str,
    in_channels: int,
    train_dataset: PatchDatasetNpy,
    val_dataset: PatchDatasetNpy,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_dir: str,
    perceptual_loss_fn: Optional[nn.Module],
    args,
    train_losses: List[float],
    val_losses: List[float],
    val_psnrs: List[float],
    val_ssims: List[float],
    best_val_loss: float,
    best_val_psnr: float,
    skip_model: Optional[nn.Module] = None,
    start_epoch: int = 1,
) -> Tuple[float, float]:
    """UU 模式：运行一个 stage。in_channels=6 时用 skip_model 构造 ModelWithSkip。"""
    train_dataset.patch_size = patch_size
    train_dataset.patch_per_image = patch_per_image
    train_dataset.random_crop = True
    val_dataset.patch_size = patch_size
    val_ppi = getattr(args, "val_ppi", -1)
    if val_ppi == -1:
        val_dataset.random_crop = False
        _set_val_cover(val_dataset, patch_size, getattr(args, "val_max_stride", 224))
    else:
        val_dataset.patch_per_image = val_ppi
        val_dataset.random_crop = False
        if getattr(val_dataset, "_val_cover_positions", None) is not None:
            del val_dataset._val_cover_positions

    train_model = model
    if in_channels == 6 and skip_model is not None:
        skip_model = skip_model.to(device)
        train_model = ModelWithSkip(model, skip_model, device)
        train_model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.01))

    num_gpus = getattr(args, "num_gpus", 1)
    effective_batch = batch_size * num_gpus
    _dl_kw = {
        "batch_size": effective_batch,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        _dl_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **_dl_kw)

    no_improve_count = 0
    no_improve_neighbor = 0
    stage_best_loss = float("inf")
    stage_best_psnr = float("-inf")
    prev_val_loss, prev_val_psnr = None, None

    for epoch in range(start_epoch, epochs + 1):
        if getattr(train_dataset, "set_epoch", None) is not None:
            train_dataset.set_epoch(epoch)
        train_loss, _, _ = train_one_epoch(
            train_model, train_loader, optimizer, device,
            epoch=epoch, max_epochs=epochs,
            perceptual_loss_fn=perceptual_loss_fn,
            use_rgb_lab_loss=args.loss_rgb_lab,
            loss_l1=getattr(args, "loss_l1", 1.0),
            loss_mse=getattr(args, "loss_mse", 0.0),
            loss_grad=args.loss_grad,
            loss_ssim=args.loss_ssim,
            loss_percep=args.loss_percep,
            loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
            loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
            loss_consist=0.0,  # UU 模式暂不用 consist
            loss_slide=0.0,    # UU 模式暂不用 slide
            loss_consist_l1=args.loss_consist_l1,
            loss_consist_mse=args.loss_consist_mse,
            loss_slide_l1=args.loss_slide_l1,
            loss_slide_mse=args.loss_slide_mse,
        )
        val_loss, _, _, val_psnr, val_ssim = eval_one_epoch(
            train_model, val_loader, device,
            perceptual_loss_fn=perceptual_loss_fn,
            use_rgb_lab_loss=args.loss_rgb_lab,
            loss_l1=getattr(args, "loss_l1", 1.0),
            loss_mse=getattr(args, "loss_mse", 0.0),
            loss_grad=args.loss_grad,
            loss_ssim=args.loss_ssim,
            loss_percep=args.loss_percep,
            loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
            loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
            lock_l=getattr(args, "lock_l", False),
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        global_epoch = len(train_losses)
        print(f"[UU {stage_name}] Epoch [{epoch}/{epochs}] (global {global_epoch}) "
              f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  PSNR: {val_psnr:.2f}  SSIM: {val_ssim:.4f}")

        improved_loss = val_loss < best_val_loss
        improved_psnr = val_psnr > best_val_psnr
        improved_this_stage = val_loss < stage_best_loss or val_psnr > stage_best_psnr
        if improved_this_stage:
            stage_best_loss = min(stage_best_loss, val_loss)
            stage_best_psnr = max(stage_best_psnr, val_psnr)

        if improved_loss:
            best_val_loss = val_loss
            ckpt = {"model_state": _unwrap_model(model).state_dict(), "optimizer_state": optimizer.state_dict(),
                    "train_losses": train_losses, "val_losses": val_losses,
                    "val_psnrs": val_psnrs, "val_ssims": val_ssims,
                    "best_val_loss": best_val_loss, "best_val_psnr": max(best_val_psnr, val_psnr),
                    "stage": stage_idx, "epoch": epoch, "save_dir": save_dir,
                    "img_size": getattr(_unwrap_model(train_model), "img_size", 256)}
            loss_path = os.path.join(save_dir, f"best_loss_uu_{stage_name}.pth")
            torch.save(ckpt, loss_path)
            print(f"  保存 best_loss_uu_{stage_name}.pth")
        if improved_psnr:
            best_val_psnr = val_psnr

        if not improved_this_stage:
            no_improve_count += 1
        else:
            no_improve_count = 0

        improved_vs_prev = (prev_val_loss is None) or (val_loss < prev_val_loss or val_psnr > prev_val_psnr)
        if not improved_vs_prev:
            no_improve_neighbor += 1
        else:
            no_improve_neighbor = 0
        prev_val_loss, prev_val_psnr = val_loss, val_psnr

        if no_improve_count >= args.patience_stage:
            print(f"  Early stop: 本 stage 连续 {args.patience_stage} 轮无提升")
        elif no_improve_neighbor >= args.patience_neighbor:
            print(f"  Early stop: 连续 {args.patience_neighbor} 轮没比上一轮好")

        ckpt = {"model_state": _unwrap_model(model).state_dict(), "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses, "val_losses": val_losses,
                "val_psnrs": val_psnrs, "val_ssims": val_ssims,
                "best_val_loss": best_val_loss, "best_val_psnr": max(best_val_psnr, val_psnr),
                "stage": stage_idx, "epoch": epoch, "save_dir": save_dir,
                "img_size": getattr(_unwrap_model(train_model), "img_size", 256)}
        torch.save(ckpt, os.path.join(save_dir, "latest_uu.pth"))
        _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims)

        if no_improve_count >= args.patience_stage or no_improve_neighbor >= args.patience_neighbor:
            break

        if epoch in (1, 3, 5, 10):
            run_val_sliding_window_vis(
                model=train_model,
                val_dataset=val_dataset,
                device=device,
                save_dir=save_dir,
                stage_idx=stage_idx,
                patch_size=256,
                stride=224,
                img_size=(getattr(args, "val_img_size", None) or getattr(_unwrap_model(train_model), "img_size", 256)),
                suffix=f"_uu_{stage_name}_epoch{epoch}",
                lock_l=getattr(args, "lock_l", False),
            )

        log_txt = os.path.join(save_dir, "log.txt")
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(_fmt_log_line(f"uu_{stage_name}_{epoch}", train_loss, val_loss, val_psnr, val_ssim) + "\n")

    run_val_sliding_window_vis(
        model=train_model,
        val_dataset=val_dataset,
        device=device,
        save_dir=save_dir,
        stage_idx=stage_idx,
        patch_size=256,
        stride=224,
        img_size=(getattr(args, "val_img_size", None) or getattr(_unwrap_model(train_model), "img_size", 256)),
        suffix=f"_uu_{stage_name}",
        lock_l=getattr(args, "lock_l", False),
    )
    return best_val_loss, best_val_psnr


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    default_train_base = os.path.normpath(os.path.join(project_root, "..", "Train"))

    parser = argparse.ArgumentParser(
        description="Patch 训练：3-stage 课程学习 (128→192→256) 或单阶段"
    )
    parser.add_argument(
        "--train_base",
        type=str,
        default=default_train_base,
        help=f"数据根目录（默认 {default_train_base}）",
    )
    parser.add_argument("--gt_subdir", type=str, default="GT_crop_resize", help="GT 子目录（默认 GT_crop_resize）")
    parser.add_argument("--in_subdir", type=str, default="IN_CR_COM_crop_resize", help="输入子目录（默认 IN_CR_COM_crop_resize）")
    _default_val = [1, 2, 9, 12, 16, 21, 25, 31, 41, 45, 49, 57, 59, 66, 70, 73, 77, 80]
    parser.add_argument(
        "--val_indices",
        type=int,
        nargs="*",
        default=None,
        help=f"验证集 GT 索引，对应文件名 id 如 1_GT.npy 的 1（默认 {_default_val}）",
    )
    _default_disable = [4, 6, 8, 15, 20, 24, 28, 30, 34, 36, 38, 40, 44, 48, 52, 54, 56, 65, 69, 83, 85]
    parser.add_argument(
        "--disable_gt",
        type=int,
        nargs="*",
        default=None,
        help=f"禁用的 GT 索引，对应文件名 id（默认 {_default_disable}），不参与训练和验证",
    )
    parser.add_argument(
        "--select_gt",
        type=str,
        default=None,
        metavar="IDS",
        help="仅用这些 GT id 做 train 和 val，逗号分隔，如 66,67,68,69,73,74,75,80,81,82,83,84,85",
    )
    parser.add_argument("--no_staged", action="store_true", help="禁用 3-stage，使用单阶段 img_size/batch/epochs")
    parser.add_argument("--img_size", type=str, nargs="*", default=[256], help="单阶段时 patch/输入尺寸：单值或 宽 高（习惯顺序），如 544 408 或 544,408")
    parser.add_argument("--base_ch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8, help="每 GPU batch 大小（总 batch = batch_size * num_gpus）")
    parser.add_argument("--batch_per_gpu", type=int, default=None, help="与 --batch_size 同义（指定后覆盖 batch_size）")
    parser.add_argument("--num_gpus", type=int, default=2, help="使用 GPU 数量（默认 2）")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=120, help="单阶段时总 epoch（3-stage 时为 60+40+20）")
    parser.add_argument("--patience_stage", type=int, default=4, help="连续 N 轮无 best_loss/best_psnr 则停止当前 stage")
    parser.add_argument("--patience_neighbor", type=int, default=2, help="连续 N 轮没比上一轮好（loss 或 PSNR）则停止")
    parser.add_argument("--val_ppi", type=int, default=-1, help="验证集每图 patch 数：-1=覆盖整图（默认）；>0 则固定点数如 5（中心+四角）")
    parser.add_argument("--val_max_stride", type=int, default=224, help="val 覆盖整图时 stride 上限（默认 224），第三阶段 patch=256 时重叠区在 val 上按 patch 平均 loss，可视化用权重融合")
    parser.add_argument("--train_img_size", type=str, nargs="*", default=None, help="训练时固定输入分辨率：单值或 宽 高，未设则用 patch_size 不 resize")
    parser.add_argument("--val_img_size", type=str, nargs="*", default=None, help="val 时固定输入分辨率：单值或 宽 高，未设则用 model.img_size")
    parser.add_argument("--lock_l", action="store_true", help="验证时锁定 LAB 的 L：用输入 L + 预测 AB 合成后再算 val loss/PSNR/SSIM 与整图指标")
    parser.add_argument("--lr", type=float, default=1e-4, help="base 学习率；stage2=lr/2，stage3=lr/4")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 权重衰减（L2 正则，默认 0.01）；设 0 或 --no_weight_decay 关闭")
    parser.add_argument("--no_weight_decay", action="store_true", help="关闭 weight decay（等价于 --weight_decay 0）")
    parser.add_argument("--no_aug_flip", action="store_true", help="关闭训练时随机水平/垂直翻转增强")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vis_batches", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径、latest（从最新 run 的 latest.pth 恢复）、或 stage1/2/3（从最新 run 的对应 stage 恢复）")
    parser.add_argument("--start_stage", type=int, default=None, choices=[1, 2, 3, 4], help="从指定 stage 开始：无 --resume 时随机初始化仅跑该 stage；有 --resume 时加载 pth 后从该 stage 继续")
    parser.add_argument("--low_freq_only", action="store_true")
    parser.add_argument("--loss_rgb_lab", action="store_true", default=True, help="启用 L1+grad+ssim+percep loss（默认开启）")
    parser.add_argument("--no_loss_rgb_lab", action="store_true", help="禁用 L1+grad+ssim+percep loss")
    parser.add_argument("--loss_l1", type=float, default=1.0, help="L1(RGB) 权重（默认 1.0）")
    parser.add_argument("--loss_mse", type=float, default=0.0, help="MSE(RGB) 权重（默认 0），与 loss_l1 共同控制重建项")
    parser.add_argument("--loss_ab_l1", type=float, default=0.0, help="L1(LAB 的 AB 通道) 权重，AB 已归一化 [-1,1]（默认 0）")
    parser.add_argument("--loss_ab_mse", type=float, default=0.0, help="MSE(LAB 的 AB 通道) 权重，AB 已归一化 [-1,1]（默认 0）")
    parser.add_argument("--loss_grad", type=float, default=0.1)
    parser.add_argument("--loss_ssim", type=float, default=0.1)
    parser.add_argument("--loss_percep", type=float, default=0.01)
    parser.add_argument("--loss_consist", type=float, default=0.1, help="consistency loss 总权重（默认 0.1，设 0 关闭）")
    parser.add_argument("--loss_consist_l1", type=float, default=None, help="consist 项内 L1 权重；未传则与 loss_l1/loss_mse 同配比")
    parser.add_argument("--loss_consist_mse", type=float, default=None, help="consist 项内 MSE 权重；未传则与 loss_l1/loss_mse 同配比")
    parser.add_argument("--loss_slide", type=float, default=0.1, help="slide loss 总权重（设 0 关闭，与 consist 互斥）")
    parser.add_argument("--loss_slide_l1", type=float, default=None, help="slide 项内 L1 权重；未传则与 loss_l1/loss_mse 同配比")
    parser.add_argument("--loss_slide_mse", type=float, default=None, help="slide 项内 MSE 权重；未传则与 loss_l1/loss_mse 同配比")
    parser.add_argument("--slide_stride", type=int, default=0, help="slide loss 采样 stride，0 表示 patch_size//2（默认 0）")
    parser.add_argument("--patch_mid_ratio", type=float, default=0.6, help="stage3 随机 patch 时中间段比例（默认 0.6）")
    parser.add_argument("--patch_large_ratio", type=float, default=0.15, help="stage3 大 patch 段比例（默认 0.15），大 patch 易 OOM")
    parser.add_argument("--uu", action="store_true", help="U-Net U 型：3down->2down->1->2up->3up，同名 stage 间 skip 融合")
    parser.add_argument("--init_3down_from", type=str, default=None, help="UU 模式：用非 UU 的 checkpoint 初始化 3down（与 3down 同构：patch 256, 3ch）")
    parser.add_argument("--demo", nargs="?", type=int, default=None, const=-1, metavar="N", help="调试：--demo 不跟数字时 train/val 用同批 GT（默认 1 个）；--demo N 时取前 N，train 与 val 分池")
    parser.add_argument("--demo_same_gt", action="store_true", help="与 --demo N 同用：强制 train 与 val 使用同一批 GT")
    parser.add_argument("--train_all", action="store_true", help="训练时也包含 val_ids 中的 GT，验证集仍用于 val；默认仅用非 val 的 GT 训练")
    parser.add_argument("--train_all_crop", action="store_true", help="4-crop 时训练每 epoch 用齐 4 个 crop（4-in-1），默认每 epoch 每 id 仅 1 个 crop")
    parser.add_argument("--final_patch_size", type=str, nargs="*", default=None, help="最终 stage 的 patch 与模型 img_size：单值或 宽 高，如 544 408")
    parser.add_argument("--patch_size", type=str, nargs="*", default=None, help="与 --final_patch_size 同义：单值或 宽 高")
    args = parser.parse_args()
    if getattr(args, "batch_per_gpu", None) is not None:
        args.batch_size = args.batch_per_gpu
    if getattr(args, "no_weight_decay", False):
        args.weight_decay = 0.0

    def _parse_size_list(v):
        """解析 544,408 或 [544, 408] 为 int 列表；命令行习惯 (宽, 高)。"""
        if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
            return None
        if isinstance(v, (list, tuple)) and len(v) == 1:
            s = v[0]
            if isinstance(s, str):
                return [int(x.strip()) for x in s.replace(",", " ").split()]
            return [int(s)]
        return [int(x) for x in v]

    # 规范尺寸：命令行按 (宽, 高) 写，内部存 (H, W)。--img_size 为最高优先级，设后 model/final_patch/val 等均用此尺寸
    _img = _parse_size_list(args.img_size)
    args.img_size = _to_size(_img, wh_order=True) if _img else (256, 256)
    _train_sz = _parse_size_list(getattr(args, "train_img_size", None))
    args.train_img_size = _to_size(_train_sz, wh_order=True) if _train_sz else None
    _val_sz = _parse_size_list(getattr(args, "val_img_size", None))
    args.val_img_size = _to_size(_val_sz, wh_order=True) if _val_sz else None
    _fp = _parse_size_list(args.final_patch_size)
    _pp = _parse_size_list(getattr(args, "patch_size", None))
    args.final_patch_size = _to_size(_fp, wh_order=True) if _fp else None
    if _pp:
        args.final_patch_size = args.final_patch_size or _to_size(_pp, wh_order=True)
    if args.img_size != (256, 256):
        args.final_patch_size = args.img_size
        if args.train_img_size is None:
            args.train_img_size = args.img_size
        if args.val_img_size is None:
            args.val_img_size = args.img_size
    elif args.final_patch_size is None and not args.no_staged:
        args.final_patch_size = args.img_size
    if args.no_loss_rgb_lab:
        args.loss_rgb_lab = False

    # consist/slide 的 L1/MSE 配比：未传则与 loss_l1/loss_mse 同配比
    _tl1 = getattr(args, "loss_l1", 1.0)
    _tmse = getattr(args, "loss_mse", 0.0)
    _total = _tl1 + _tmse
    if _total > 0:
        _def_l1, _def_mse = _tl1 / _total, _tmse / _total
    else:
        _def_l1, _def_mse = 0.5, 0.5
    if getattr(args, "loss_consist_l1", None) is None and getattr(args, "loss_consist_mse", None) is None:
        args.loss_consist_l1, args.loss_consist_mse = _def_l1, _def_mse
    elif getattr(args, "loss_consist_l1", None) is None:
        args.loss_consist_l1 = 1.0 - args.loss_consist_mse
    elif getattr(args, "loss_consist_mse", None) is None:
        args.loss_consist_mse = 1.0 - args.loss_consist_l1
    if getattr(args, "loss_slide_l1", None) is None and getattr(args, "loss_slide_mse", None) is None:
        args.loss_slide_l1, args.loss_slide_mse = _def_l1, _def_mse
    elif getattr(args, "loss_slide_l1", None) is None:
        args.loss_slide_l1 = 1.0 - args.loss_slide_mse
    elif getattr(args, "loss_slide_mse", None) is None:
        args.loss_slide_mse = 1.0 - args.loss_slide_l1

    # 自动选择最新 checkpoint 文件夹
    def _latest_run_dir(base: str) -> Optional[str]:
        if not base or not os.path.isdir(base):
            return None
        pat = re.compile(r"^\d{8}-\d{6}$")
        subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and pat.match(d)]
        if not subdirs:
            return None
        return os.path.join(base, max(subdirs))

    # 解析 --resume：latest / stage1/2/3 / 文件路径
    resume_file = None
    resume_is_latest = False  # 仅加载模型 vs 完整恢复（含 stage/epoch/optimizer 等）
    if args.resume:
        r = args.resume.strip().lower()
        base = args.save_dir or os.path.join(project_root, "checkpoints")
        latest_run = _latest_run_dir(base)
        stage_map = {"stage1": "one", "stage2": "two", "stage3": "three", "stage4": "four"}
        if r == "latest":
            if not latest_run:
                raise FileNotFoundError(f"未找到可恢复的 run 目录（在 {base} 下查找 YYYYMMDD-HHMMSS 格式）")
            p = os.path.join(latest_run, "latest.pth")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"未找到 {p}，请先完成至少一个 epoch 的训练")
            resume_file = p
            resume_is_latest = True
        elif r in ("stage1", "stage2", "stage3", "stage4"):
            if not latest_run:
                raise FileNotFoundError(f"未找到可恢复的 run 目录")
            sname = stage_map[r]
            for fname in (f"best_loss_stage_{sname}.pth", f"epoch[1]_stage_{sname}.pth"):
                p = os.path.join(latest_run, fname)
                if os.path.isfile(p):
                    resume_file = p
                    break
            if not resume_file:
                raise FileNotFoundError(f"在 {latest_run} 下未找到 stage {sname} 的 checkpoint")
        elif os.path.isfile(args.resume):
            resume_file = args.resume
        else:
            for suf in ("/stage1", "/stage2", "/stage3", "/stage4"):
                if r.endswith(suf):
                    run_dir = args.resume[:-len(suf)]
                    sname = stage_map[suf[1:]]
                    for fname in (f"best_loss_stage_{sname}.pth", f"epoch[1]_stage_{sname}.pth"):
                        p = os.path.join(run_dir, fname)
                        if os.path.isfile(p):
                            resume_file = p
                            break
                    if not resume_file:
                        raise FileNotFoundError(f"在 {run_dir} 下未找到 stage {sname} 的 checkpoint")
                    break
            if not resume_file:
                raise FileNotFoundError(f"未找到 checkpoint: {args.resume}")
        args.resume = resume_file
        args._resume_is_latest = resume_is_latest

    if args.start_stage is not None and getattr(args, "_resume_is_latest", False):
        raise ValueError("--start_stage 不能与 --resume latest 同时使用，latest 会自动恢复 stage")

    if args.val_indices is None:
        args.val_indices = [1, 2, 9, 12, 16, 21, 25, 31, 41, 45, 49, 57, 59, 66, 70, 73, 77, 80]
    if args.disable_gt is None:
        args.disable_gt = [4, 6, 8, 15, 20, 24, 28, 30, 34, 36, 38, 40, 44, 48, 52, 54, 56, 65, 69, 83, 85]

    if args.save_dir is None:
        args.save_dir = os.path.join(project_root, "checkpoints")

    gt_dir = os.path.join(args.train_base, args.gt_subdir)
    def _crop_id_key(s):
        p = s.split("_")[0]
        return (int(p), s) if p.isdigit() else (0, s)
    gt_files = []
    gt_crop_resize_glob = sorted(glob.glob(os.path.join(gt_dir, "*_GT_crop_resize_R*_C*.npy")))
    if gt_crop_resize_glob:
        _resize_re = re.compile(r"^(.+)_GT_crop_resize_R(\d+)_C(\d+)\.npy$")
        by_base = {}
        for p in gt_crop_resize_glob:
            m = _resize_re.match(os.path.basename(p))
            if m and int(m.group(2)) <= 1 and int(m.group(3)) <= 1:
                by_base.setdefault(m.group(1), []).append(p)
        if by_base and all(len(v) == 4 for v in by_base.values()):
            unique_ids = sorted(by_base.keys(), key=_crop_id_key)
            gt_files = [by_base[sid][0] for sid in unique_ids]
        else:
            gt_crop_resize_glob = []
    if not gt_files:
        gt_crop_glob = sorted(glob.glob(os.path.join(gt_dir, "*_GT_crop_r*_c*.npy")))
        if gt_crop_glob:
            _crop_re = re.compile(r"^(.+)_GT_crop_r(\d+)_c(\d+)\.npy$")
            unique_ids = sorted(set(m.group(1) for p in gt_crop_glob for m in [_crop_re.match(os.path.basename(p))] if m), key=_crop_id_key)
            for sid in unique_ids:
                one = next((p for p in gt_crop_glob if os.path.basename(p).startswith(sid + "_")), None)
                if one:
                    gt_files.append(one)
        if not gt_files:
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*_GT.npy")))

    # 按 R-G-B / L-a-b 两行三列保存首张 GT、IN 的可视化到 checkpoints
    vis_dir = args.save_dir
    if gt_files and plt is not None:
        gt_path = gt_files[0]
        try:
            gt_arr = np.load(gt_path).astype(np.float32)
            if gt_arr.ndim == 2:
                gt_arr = np.stack([gt_arr, gt_arr, gt_arr], axis=-1)
            if gt_arr.max() > 1.0:
                gt_arr = gt_arr / 255.0
            gt_bgr = (np.clip(gt_arr, 0, 1) * 255).astype(np.uint8)
            _save_rgb_lab_vis(gt_bgr, os.path.join(vis_dir, "GT_vis.png"))
        except Exception as e:
            print(f"  保存 GT_vis.png 失败: {e}")
        in_dir = os.path.join(args.train_base, args.in_subdir)
        name = os.path.basename(gt_path)
        if name.endswith("_GT.npy"):
            sid = name[:-7]
            in_pattern = f"{sid}_*_IN.npy"
        elif "_GT_crop_resize_" in name:
            sid = name.split("_GT_crop_resize")[0]
            in_pattern = f"{sid.split('_')[0]}_*_IN_crop_resize_R*_C*.npy"
        else:
            m = re.match(r"^(.+)_GT_crop_r\d+_c\d+\.npy$", name)
            sid = m.group(1) if m else name.split("_GT_crop")[0]
            in_pattern = f"{sid.split('_')[0]}_*_IN_crop_r*_c*.npy"
        in_matches = sorted(glob.glob(os.path.join(in_dir, in_pattern)))
        if in_matches:
            try:
                in_arr = np.load(in_matches[0]).astype(np.float32)
                if in_arr.ndim == 2:
                    in_arr = np.stack([in_arr, in_arr, in_arr], axis=-1)
                if in_arr.max() > 1.0:
                    in_arr = in_arr / 255.0
                in_bgr = (np.clip(in_arr, 0, 1) * 255).astype(np.uint8)
                _save_rgb_lab_vis(in_bgr, os.path.join(vis_dir, "IN_vis.png"))
            except Exception as e:
                print(f"  保存 IN_vis.png 失败: {e}")
        else:
            print(f"  未找到首张 GT 对应的 IN 文件（{args.in_subdir}），跳过 IN_vis.png")
        # 额外保存 IN_CR_COM_pred 的可视化（Stage4 输入），与 in_pattern 同规则
        in_pred_dir = os.path.join(args.train_base, IN_SUBDIR_STAGE4)
        if in_pred_dir != in_dir:
            pred_matches = sorted(glob.glob(os.path.join(in_pred_dir, in_pattern)))
            if pred_matches:
                try:
                    pred_arr = np.load(pred_matches[0]).astype(np.float32)
                    if pred_arr.ndim == 2:
                        pred_arr = np.stack([pred_arr, pred_arr, pred_arr], axis=-1)
                    if pred_arr.max() > 1.0:
                        pred_arr = pred_arr / 255.0
                    pred_bgr = (np.clip(pred_arr, 0, 1) * 255).astype(np.uint8)
                    _save_rgb_lab_vis(pred_bgr, os.path.join(vis_dir, "IN_vis_pred.png"))
                except Exception as e:
                    print(f"  保存 IN_vis_pred.png 失败: {e}")

    total = len(gt_files)
    # 文件名 id -> gt_files 列表索引，如 1_GT.npy -> id=1；crop 时为 1_GT_crop_r*_c*.npy -> id=1
    id_to_idx = {}
    for idx, path in enumerate(gt_files):
        name = os.path.basename(path)
        if name.endswith("_GT.npy"):
            try:
                fid = int(name[:-7])
                id_to_idx[fid] = idx
            except ValueError:
                pass
        elif "_GT_crop_" in name:
            try:
                sid = name.split("_GT_crop")[0]
                fid = int(sid.split("_")[0])
                id_to_idx[fid] = idx
            except (ValueError, IndexError):
                pass
    disabled = set(id_to_idx[i] for i in args.disable_gt if i in id_to_idx)
    available = [i for i in range(total) if i not in disabled]
    if getattr(args, "select_gt", None):
        # --select_gt 优先级最高：仅用这些 GT，val_ids 也强制为这些
        select_gt_ids = set(int(x.strip()) for x in args.select_gt.split(",") if x.strip())
        selected_indices = sorted(id_to_idx[i] for i in select_gt_ids if i in id_to_idx and id_to_idx[i] not in disabled)
        available = [i for i in available if i in set(selected_indices)]
        val_indices = list(selected_indices)
        omitted = []
        print(f"  [--select_gt] 仅用 GT ids={sorted(select_gt_ids)}，val_ids=这些，共 {len(available)} 用于 train/val")
    else:
        val_indices = [id_to_idx[i] for i in args.val_indices if i in id_to_idx and id_to_idx[i] not in disabled]
        omitted = [i for i in args.val_indices if i not in id_to_idx or id_to_idx[i] in disabled]
    if omitted:
        print(f"  警告: val_indices 中已忽略（文件不存在或禁用）: {omitted}")
    train_indices = available if (getattr(args, "train_all", False) or getattr(args, "select_gt", None)) else [i for i in available if i not in set(val_indices)]
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    if args.demo is not None:
        # --demo 不跟数字时 args.demo=-1（const），表示 train/val 同批 GT、默认 1 个；--demo N 则 N 个且分池
        demo_same_gt = (args.demo == -1) or getattr(args, "demo_same_gt", False)
        n_demo = 1 if args.demo == -1 else max(1, int(args.demo))
        if demo_same_gt:
            if len(available) < n_demo:
                raise RuntimeError(f"--demo 同批 GT 需要至少 {n_demo} 个可用 GT，当前 {len(available)}")
            train_indices = list(available[:n_demo])
            val_indices = list(train_indices)
            print(f"  [--demo 同批 GT] train=val: {len(train_indices)} (ids={', '.join(str(idx_to_id[i]) for i in train_indices)})")
        else:
            if not train_indices:
                raise RuntimeError("--demo 时至少需要 1 个 train GT（默认 val 外的可用 GT）")
            if not val_indices:
                raise RuntimeError("--demo 时至少需要 1 个 val GT（默认 val_indices）")
            train_indices = list(train_indices[:n_demo])
            val_indices = list(val_indices[:n_demo])
            print(f"  [--demo {n_demo}] train={len(train_indices)} (ids={', '.join(str(idx_to_id[i]) for i in train_indices)}), val={len(val_indices)} (ids={', '.join(str(idx_to_id[i]) for i in val_indices)})")
    if disabled:
        print(f"  禁用 GT（对应文件名 id）: {sorted(idx_to_id[i] for i in disabled)}")
    if getattr(args, "train_all", False):
        print(f"  [--train_all] val_ids 中的 GT 也参与训练")
    if not train_indices:
        raise RuntimeError("训练集为空，请调整 --val_indices")
    if not val_indices:
        raise RuntimeError("验证集为空，请调整 --val_indices")

    init_patch = UU_STAGES[0][0] if args.uu else (STAGES[0][0] if not args.no_staged else args.img_size)
    init_ppi = UU_STAGES[0][1] if args.uu else (STAGES[0][1] if not args.no_staged else 1)
    use_slide = args.loss_slide > 0 and not args.uu
    use_consist = args.loss_consist > 0 and not args.uu and not use_slide
    train_dataset = PatchDatasetNpy(
        args.train_base,
        train_indices,
        patch_size=init_patch,
        patch_per_image=init_ppi,
        use_low_freq_only=args.low_freq_only,
        random_crop=True,
        use_consist=use_consist,
        use_slide=use_slide,
        slide_stride=args.slide_stride,
        gt_subdir=args.gt_subdir,
        in_subdir=args.in_subdir,
        aug_flip=not getattr(args, "no_aug_flip", False),
        all_crops_per_id=getattr(args, "train_all_crop", False),
    )
    val_ppi = getattr(args, "val_ppi", -1)
    val_dataset = PatchDatasetNpy(
        args.train_base,
        val_indices,
        patch_size=init_patch,
        patch_per_image=val_ppi if val_ppi > 0 else 1,
        use_low_freq_only=args.low_freq_only,
        gt_subdir=args.gt_subdir,
        in_subdir=args.in_subdir,
        random_crop=False,
        all_crops_per_id=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    num_gpus = min(getattr(args, "num_gpus", 1), torch.cuda.device_count())
    args.num_gpus = num_gpus
    gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(num_gpus))
    print(f"使用 {num_gpus} 张 GPU: {gpu_names}")
    print(f"数据: {args.train_base}/{args.gt_subdir} + {args.in_subdir}  train_pairs={len(train_dataset.pairs)} val_pairs={len(val_dataset.pairs)}")

    final_img_size = getattr(args, "final_patch_size", None) or (
        STAGES[-1][0] if not args.no_staged else args.img_size
    )
    model = CNNImageRegressor(
        img_size=final_img_size,
        base_ch=args.base_ch,
        lab_color=False,
    )
    perceptual_loss_fn = PerceptualLoss(device) if args.loss_percep > 0 else None

    start_stage = 0
    start_epoch = 1
    resume_ckpt = None
    enc3_expanded_before_load = False
    if args.resume and os.path.isfile(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device)
        sd = resume_ckpt.get("model_state", resume_ckpt)
        if isinstance(sd, dict):
            def _enc3_block_idx(key):
                if "enc3." not in key:
                    return None
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "enc3" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        return int(parts[i + 1])
                return None
            enc3_indices = [idx for k in sd for idx in (_enc3_block_idx(k),) if idx is not None]
            enc3_max = max(enc3_indices, default=-1)
            if enc3_max >= 16 and not args.no_staged:
                for si in range(1, 3):
                    expand_enc3_for_stage(model, si, None)
                    m = getattr(model, "module", model)
                    if hasattr(m, "enc3") and len(m.enc3) > enc3_max:
                        break
                enc3_expanded_before_load = True
                print(f"  已扩展 enc3 以匹配 checkpoint（enc3 含 {enc3_max + 1} 块）")
            elif enc3_max >= 12 and enc3_max < 16 and not args.no_staged:
                expand_enc3_for_stage(model, 1, None)
                enc3_expanded_before_load = True
                print(f"  已扩展 enc3 以匹配 checkpoint（enc3 含 16 块）")
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        print(f"  已从 {args.resume} 加载模型")
        if getattr(args, "_resume_is_latest", False) and not args.no_staged:
            start_stage = resume_ckpt.get("stage", 0)
            epoch_done = resume_ckpt.get("epoch", 0)
            max_epochs_this_stage = STAGES[start_stage][3]
            if epoch_done >= max_epochs_this_stage:
                start_stage += 1
                start_epoch = 1
            else:
                start_epoch = epoch_done + 1
            print(f"  恢复: stage {start_stage+1}, 从 epoch {start_epoch} 继续")
        elif args.start_stage is not None and not args.no_staged:
            # --resume <pth> --start_stage N：加载指定 pth，从 stage N 开始
            start_stage = args.start_stage - 1
            start_epoch = 1
            print(f"  从 stage {args.start_stage} 开始（跳过 stage 1..{args.start_stage-1}）")
    elif args.start_stage is not None and not args.no_staged:
        # --start_stage N 且无 --resume：从 stage N 从头训（随机初始化，不加载 stage1/2 权重）
        start_stage = args.start_stage - 1
        start_epoch = 1
        print(f"  从 stage {args.start_stage} 开始（无预训练，随机初始化）")
    model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.01))
    resume_from_latest = getattr(args, "_resume_is_latest", False) and resume_ckpt is not None and not args.no_staged
    if resume_from_latest:
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            print(f"  已恢复 optimizer 状态")
    elif args.start_stage is not None and resume_ckpt is not None and "optimizer_state" in resume_ckpt and not enc3_expanded_before_load:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            print(f"  已恢复 optimizer 状态")
        except (ValueError, KeyError) as e:
            print(f"  跳过 optimizer 恢复（param groups 不匹配）: {e}")
    elif args.start_stage is not None and resume_ckpt is not None and enc3_expanded_before_load:
        print(f"  跳过 optimizer 恢复（enc3 已扩展，param groups 结构不同）")
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.uu:
        # U-Net U 型训练：3down->2down->1->2up->3up
        # --resume 仅支持 stage 3down
        uu_resume_ckpt = None
        if args.resume:
            base = args.save_dir or os.path.join(project_root, "checkpoints")
            r = args.resume.strip().lower()
            if r in ("latest", "best"):
                latest_run = _latest_run_dir(base)
                if not latest_run:
                    raise FileNotFoundError(f"未找到可恢复的 run 目录（在 {base} 下）")
                for name in ("latest_uu.pth", "best_loss_uu_3down.pth"):
                    p = os.path.join(latest_run, name)
                    if os.path.isfile(p):
                        args.resume = p
                        break
                if not os.path.isfile(args.resume):
                    raise FileNotFoundError(f"在 {latest_run} 下未找到 UU checkpoint（需 latest_uu.pth 或 best_loss_uu_3down.pth）")
            elif not os.path.isfile(args.resume):
                raise FileNotFoundError(f"未找到 checkpoint: {args.resume}")
            uu_resume_ckpt = torch.load(args.resume, map_location=device)
            ckpt_stage = uu_resume_ckpt.get("stage", -1)
            if ckpt_stage != 0:
                raise ValueError(f"--uu 模式下 --resume 仅支持 stage 3down，当前 checkpoint 为 stage {ckpt_stage + 1}")
            _sd = uu_resume_ckpt.get("save_dir", "")
            if _sd and os.path.isdir(_sd):
                save_dir = _sd
            else:
                save_dir = _latest_run_dir(base) if r in ("latest", "best") else os.path.dirname(os.path.abspath(args.resume))
            print(f"  已从 {args.resume} 恢复 UU stage 3down")
        else:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_dir = os.path.join(args.save_dir, run_name)
            os.makedirs(save_dir, exist_ok=True)

        if uu_resume_ckpt is None:
            config_lines = [
            f"mode: uu",
            f"train_base: {args.train_base}",
            f"gt_subdir: {args.gt_subdir}",
            f"in_subdir: {args.in_subdir}",
            f"val_ids: {sorted(idx_to_id[i] for i in val_indices)}",
            f"train_all: {getattr(args, 'train_all', False)}",
            f"val_ppi: {getattr(args, 'val_ppi', -1)} (-1=覆盖整图, >0=固定点数)",
            f"disable_ids: {sorted(idx_to_id[i] for i in disabled)}",
            f"uu_stages: " + "; ".join(f"{s[4]} patch={s[0]} epochs={s[3]} ch={s[5]}" for s in UU_STAGES),
        ]
        with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(config_lines))
        with open(os.path.join(save_dir, "log.txt"), "w", encoding="utf-8") as f:
            f.write(_fmt_log_header() + "\n")
        with open(os.path.join(save_dir, "fullimage_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(_fmt_fullimage_header() + "\n")

        perceptual_loss_fn = PerceptualLoss(device) if args.loss_percep > 0 else None
        if uu_resume_ckpt is not None:
            train_losses = list(uu_resume_ckpt.get("train_losses", []))
            val_losses = list(uu_resume_ckpt.get("val_losses", []))
            val_psnrs = list(uu_resume_ckpt.get("val_psnrs", []))
            val_ssims = list(uu_resume_ckpt.get("val_ssims", []))
            best_val_loss = float(uu_resume_ckpt.get("best_val_loss", float("inf")))
            best_val_psnr = float(uu_resume_ckpt.get("best_val_psnr", -float("inf")))
            uu_start_epoch = int(uu_resume_ckpt.get("epoch", 0)) + 1
            if train_losses:
                log_path = os.path.join(save_dir, "log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(_fmt_log_header() + "\n")
                    for j in range(len(train_losses)):
                        tl = train_losses[j] if j < len(train_losses) else 0.0
                        vl = val_losses[j] if j < len(val_losses) else 0.0
                        vp = val_psnrs[j] if j < len(val_psnrs) else 0.0
                        vs = val_ssims[j] if j < len(val_ssims) else 0.0
                        f.write(_fmt_log_line(str(j + 1), tl, vl, vp, vs) + "\n")
                _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims)
                print(f"  已恢复 log 与 loss 曲线（共 {len(train_losses)} 轮）")
        else:
            train_losses = []
            val_losses = []
            val_psnrs = []
            val_ssims = []
            best_val_loss = float("inf")
            best_val_psnr = float("-inf")
            uu_start_epoch = 1

        for i, (patch_size, ppi, batch_size, epochs, stage_name, in_ch) in enumerate(UU_STAGES):
            uu_model = CNNImageRegressor(
                img_size=patch_size,
                base_ch=args.base_ch,
                lab_color=False,
                in_channels=in_ch,
            )
            uu_model.to(device)
            se = uu_start_epoch if (i == 0 and uu_resume_ckpt is not None) else 1
            if i == 0 and uu_resume_ckpt is not None:
                uu_model.load_state_dict(uu_resume_ckpt["model_state"], strict=True)
            elif i == 0 and args.init_3down_from and os.path.isfile(args.init_3down_from):
                init_ck = torch.load(args.init_3down_from, map_location=device)
                init_sd = init_ck["model_state"] if isinstance(init_ck, dict) and "model_state" in init_ck else init_ck
                uu_model.load_state_dict(init_sd, strict=True)
                print(f"  3down 已从 {args.init_3down_from} 初始化（非 UU 模型与 3down 同构）")
            if num_gpus > 1:
                uu_model = nn.DataParallel(uu_model, device_ids=list(range(num_gpus)))
            uu_optimizer = torch.optim.AdamW(uu_model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.01))
            if i == 0 and uu_resume_ckpt is not None and "optimizer_state" in uu_resume_ckpt:
                uu_optimizer.load_state_dict(uu_resume_ckpt["optimizer_state"])

            skip_model = None
            if in_ch == 6:
                if stage_name == "2up":
                    p = os.path.join(save_dir, "best_loss_uu_2down.pth")
                else:
                    p = os.path.join(save_dir, "best_loss_uu_3down.pth")
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"UU skip 需要 {p}，请先完成前一 stage")
                ck = torch.load(p, map_location=device)
                skip_model = CNNImageRegressor(
                    img_size=256 if stage_name == "3up" else 192,
                    base_ch=args.base_ch,
                    lab_color=False,
                    in_channels=3,
                )
                skip_model.load_state_dict(ck["model_state"], strict=True)
                print(f"  已加载 skip 模型: {p}")

            print(f"\n=== UU Stage {i+1}/5: {stage_name} patch={patch_size} ppi={ppi} batch={batch_size} epochs={epochs} in_ch={in_ch} ===")
            best_val_loss, best_val_psnr = run_uu_stage(
                i, patch_size, ppi, batch_size, epochs, stage_name, in_ch,
                train_dataset, val_dataset,
                uu_model, uu_optimizer, device, save_dir,
                perceptual_loss_fn, args,
                train_losses, val_losses, val_psnrs, val_ssims,
                best_val_loss, best_val_psnr,
                skip_model=skip_model,
                start_epoch=se,
            )
        print(f"\nUU 训练结束，best_val_loss={best_val_loss:.6f} best_val_psnr={best_val_psnr:.2f}")
        print(f"最终模型: best_loss_uu_3up.pth（推理时需同时加载 3down 与 3up）")
        return

    n_params = sum(p.numel() for p in model.parameters())
    model_img_size = getattr(args, "final_patch_size", None) or (
        STAGES[-1][0] if not args.no_staged else args.img_size
    )
    enc3_info = "stage1=12 stage2=16 stage3=20" if not args.no_staged else "12"
    config_path = os.path.join(save_dir, "config.txt")
    resume_config_src = None
    if (resume_ckpt is not None) and (resume_from_latest or args.start_stage is not None):
        src_dir = resume_ckpt.get("save_dir", "") or os.path.dirname(os.path.abspath(args.resume))
        src_config = os.path.join(src_dir, "config.txt") if src_dir else ""
        if src_config and os.path.isfile(src_config):
            resume_config_src = src_config
    if resume_config_src:
        src_dir = os.path.dirname(resume_config_src)
        for name in ("log.txt", "fullimage_metrics.txt"):
            src_f = os.path.join(src_dir, name)
            if os.path.isfile(src_f):
                shutil.copy(src_f, os.path.join(save_dir, name))
        print(f"  已从 {src_dir} 拷贝 log/fullimage（若有）到新 run 目录")
    config_lines = [
        f"train_base: {args.train_base}",
        f"gt_subdir: {args.gt_subdir}",
        f"in_subdir: {args.in_subdir}",
        f"in_subdir_stage4: {IN_SUBDIR_STAGE4}",
        f"val_ids: {sorted(idx_to_id[i] for i in val_indices)}",
        f"train_all: {getattr(args, 'train_all', False)}",
        *([f"select_gt: {args.select_gt}"] if getattr(args, "select_gt", None) else []),
        f"disable_ids: {sorted(idx_to_id[i] for i in disabled)}",
        f"train_indices_n: {len(train_indices)}",
        f"val_indices_n: {len(val_indices)}",
        f"staged: {not args.no_staged}",
        f"train_pairs: {len(train_dataset.pairs)} (IN,GT)",
        f"val_pairs: {len(val_dataset.pairs)} (IN,GT)",
        f"val_ppi: {val_ppi} (-1=覆盖整图, >0=固定点数)",
        f"train_patches_per_epoch: {len(train_dataset)}",
        f"val_patches_per_epoch: {len(val_dataset)}",
        f"img_size: {_fmt_size_wh(args.img_size)} (宽 高，最高优先级)",
        f"model_img_size: {_fmt_size_wh(model_img_size)} (宽 高，同 img_size 或 final_patch_size)",
        f"model_base_ch: {args.base_ch}",
        f"model_params_M: {n_params/1e6:.2f}",
        f"enc3_blocks: {enc3_info}",
        f"lr: {args.lr} (stage i 时 lr = base_lr * 0.5^i)",
        f"weight_decay: {getattr(args, 'weight_decay', 0.01)}",
        f"aug_flip: {not getattr(args, 'no_aug_flip', False)}",
        f"train_all_crop: {getattr(args, 'train_all_crop', False)}",
        f"final_patch_size: {_fmt_size_wh(args.final_patch_size) if getattr(args, 'final_patch_size', None) else 'default(256)'} (宽 高，同 img_size 时由 img_size 决定)",
    ]
    # 整图 val：stride=img_size-32，上下左右 pad 16，输入不 resize
    _fp = getattr(args, "final_patch_size", None)
    if not args.no_staged and _fp is not None:
        val_pad = 32
        fph, fpw = _patch_hw(_fp)
        max_st = min(fph, fpw) - 32
        config_lines.extend([
            f"val_fullimage_patch: {_fmt_size_wh(_fp) if isinstance(_fp, (tuple, list)) else _fp}",
            f"val_fullimage_stride: {max_st} (img_size-32)",
            f"val_fullimage_padding: {val_pad} (上下左右各 {val_pad//2})",
        ])
    config_lines.extend([
        f"patience: stage1=4/2, stage2=6/3, stage3/4=off",
        f"num_workers: {args.num_workers}",
        f"loss_rgb_lab: {args.loss_rgb_lab}",
        f"loss_l1: {getattr(args, 'loss_l1', 1.0)}",
        f"loss_mse: {getattr(args, 'loss_mse', 0.0)}",
        f"loss_ab_l1: {getattr(args, 'loss_ab_l1', 0.0)}",
        f"loss_ab_mse: {getattr(args, 'loss_ab_mse', 0.0)}",
        f"loss_grad: {args.loss_grad}",
        f"loss_ssim: {args.loss_ssim}",
        f"loss_percep: {args.loss_percep}",
        f"loss_consist: {args.loss_consist} (l1={args.loss_consist_l1} mse={args.loss_consist_mse})",
        f"loss_slide: {args.loss_slide} (l1={args.loss_slide_l1} mse={args.loss_slide_mse})",
        f"slide_stride: {args.slide_stride if args.slide_stride > 0 else '0 (auto patch//2)'}",
        f"batch_size_per_gpu: {args.batch_size}",
        f"num_gpus: {args.num_gpus}",
        f"max_epochs: {args.max_epochs}",
    ])
    if not args.no_staged:
        _eff_batch = args.batch_size * num_gpus
        config_lines.append("stages: " + "; ".join(
            f"patch={_fmt_size_wh(s[0])} ppi={_eff_batch} batch_per_gpu={args.batch_size} epochs={s[3]} ({s[4]})" for s in STAGES
        ))
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(config_lines))
    if not resume_config_src:
        with open(os.path.join(save_dir, "log.txt"), "w", encoding="utf-8") as f:
            f.write(_fmt_log_header() + "\n")
        with open(os.path.join(save_dir, "fullimage_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(_fmt_fullimage_header() + "\n")

    stage_starts: List[int]
    if resume_from_latest and resume_ckpt is not None:
        stage_starts = list(resume_ckpt.get("stage_starts", []))
        train_losses = list(resume_ckpt.get("train_losses", []))
        val_losses = list(resume_ckpt.get("val_losses", []))
        val_psnrs = list(resume_ckpt.get("val_psnrs", []))
        val_ssims = list(resume_ckpt.get("val_ssims", []))
        val_psnrs_fullimage = list(resume_ckpt.get("val_psnrs_fullimage", []))
        val_ssims_fullimage = list(resume_ckpt.get("val_ssims_fullimage", []))
        best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf")))
        best_val_psnr = float(resume_ckpt.get("best_val_psnr", -float("inf")))
        # 从 checkpoint 重建 log.txt 并保存 loss 曲线，保留 resume 前的历史
        if train_losses:
            log_path = os.path.join(save_dir, "log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(_fmt_log_header() + "\n")
                for i in range(len(train_losses)):
                    tl = train_losses[i] if i < len(train_losses) else 0.0
                    vl = val_losses[i] if i < len(val_losses) else 0.0
                    vp = val_psnrs[i] if i < len(val_psnrs) else 0.0
                    vs = val_ssims[i] if i < len(val_ssims) else 0.0
                    f.write(_fmt_log_line(str(i + 1), tl, vl, vp, vs) + "\n")
            _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=stage_starts if stage_starts else None,
                             val_psnrs_fullimage=val_psnrs_fullimage, val_ssims_fullimage=val_ssims_fullimage)
            print(f"  已从 checkpoint 恢复 log.txt 和 loss_curve.png（共 {len(train_losses)} 轮历史）")
    elif args.start_stage is not None and args.start_stage > 1 and not args.no_staged and resume_from_latest:
        # --start_stage N 且 --resume latest：从 checkpoint 恢复完整历史，图中 epoch 连续（如 41-60）
        stage_starts = list(resume_ckpt.get("stage_starts", [])) if resume_ckpt else []
        epochs_before = sum(STAGES[j][3] for j in range(args.start_stage - 1))
        ckpt_tl = list(resume_ckpt.get("train_losses", [])) if resume_ckpt else []
        ckpt_vl = list(resume_ckpt.get("val_losses", [])) if resume_ckpt else []
        ckpt_vp = list(resume_ckpt.get("val_psnrs", [])) if resume_ckpt else []
        ckpt_vs = list(resume_ckpt.get("val_ssims", [])) if resume_ckpt else []
        if not stage_starts and len(ckpt_tl) >= 1:
            stage_starts = [1]
        if len(ckpt_tl) >= 1:
            n = len(ckpt_tl)
            train_losses = ckpt_tl
            val_losses = (ckpt_vl + [0.0] * n)[:n]
            val_psnrs = (ckpt_vp + [0.0] * n)[:n]
            val_ssims = (ckpt_vs + [0.0] * n)[:n]
            val_psnrs_fullimage = list(resume_ckpt.get("val_psnrs_fullimage", []))
            val_ssims_fullimage = list(resume_ckpt.get("val_ssims_fullimage", []))
            if train_losses:
                log_path = os.path.join(save_dir, "log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(_fmt_log_header() + "\n")
                    for i in range(len(train_losses)):
                        tl = train_losses[i] if i < len(train_losses) else 0.0
                        vl = val_losses[i] if i < len(val_losses) else 0.0
                        vp = val_psnrs[i] if i < len(val_psnrs) else 0.0
                        vs = val_ssims[i] if i < len(val_ssims) else 0.0
                        f.write(_fmt_log_line(str(i + 1), tl, vl, vp, vs) + "\n")
                _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=stage_starts if stage_starts else None,
                                 val_psnrs_fullimage=val_psnrs_fullimage, val_ssims_fullimage=val_ssims_fullimage)
            print(f"  已从 checkpoint 恢复 loss/PSNR 历史（共 {len(train_losses)} 轮）")
        else:
            train_losses = [0.0] * epochs_before
            val_losses = [0.0] * epochs_before
            val_psnrs = [0.0] * epochs_before
            val_ssims = [0.0] * epochs_before
            val_psnrs_fullimage = []
            val_ssims_fullimage = []
        best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf"))) if resume_ckpt else float("inf")
        best_val_psnr = float(resume_ckpt.get("best_val_psnr", -float("inf"))) if resume_ckpt else -float("inf")
    else:
        stage_starts = []
        train_losses = []
        val_losses = []
        val_psnrs = []
        val_ssims = []
        val_psnrs_fullimage = []
        val_ssims_fullimage = []
        best_val_loss = float("inf")
        best_val_psnr = -float("inf")

    if not args.no_staged:
        stage_names = ("one", "two", "three", "four")
        # 打印各阶段耐性
        print("\n各阶段 patience（early stop 条件）：")
        for si, st in enumerate(STAGES):
            ep = st[3]
            _ps = ep + 1000 if si >= 2 else (6 if si == 1 else args.patience_stage + si)
            _pn = ep + 1000 if si >= 2 else (3 if si == 1 else args.patience_neighbor + si)
            pa = "off（跑满）" if si >= 2 else f"stage={_ps} neighbor={_pn}"
            print(f"  Stage {si+1}: {pa}")
        for i, stage in enumerate(STAGES):
            patch_size = stage[0]
            batch_size = args.batch_size  # 统一用 --batch_per_gpu/--batch_size，effective_batch = num_gpus * batch_size
            patch_per_image = batch_size * num_gpus  # ppi 默认 = effective_batch
            epochs = stage[3]
            desc = stage[4]
            patch_size_range = stage[5] if len(stage) > 5 else None
            val_override = stage[6] if len(stage) > 6 else None
            # stage4: img_size=patch=--img_size+上下左右各 pad 16，仅用 IN_CR_COM_pred；ppi/effective_batch 与其它 stage 一致
            if i == 3:
                stage4_img_size = (args.img_size[0] + 2 * STAGE4_PAD, args.img_size[1] + 2 * STAGE4_PAD)
                patch_size = stage4_img_size
                patch_size_range = None
                train_ds = PatchDatasetNpy(
                    args.train_base,
                    train_indices,
                    patch_size=patch_size,
                    patch_per_image=patch_per_image,
                    use_low_freq_only=args.low_freq_only,
                    random_crop=False,
                    use_consist=False,
                    use_slide=False,
                    gt_subdir=args.gt_subdir,
                    in_subdir=IN_SUBDIR_STAGE4,
                    aug_flip=not getattr(args, "no_aug_flip", False),
                    all_crops_per_id=getattr(args, "train_all_crop", False),
                    pad_to_patch_size=True,
                )
                val_ds = PatchDatasetNpy(
                    args.train_base,
                    val_indices,
                    patch_size=patch_size,
                    patch_per_image=val_ppi if val_ppi > 0 else patch_per_image,
                    use_low_freq_only=args.low_freq_only,
                    gt_subdir=args.gt_subdir,
                    in_subdir=IN_SUBDIR_STAGE4,
                    random_crop=False,
                    all_crops_per_id=True,
                    pad_to_patch_size=True,
                )
                if val_ppi == -1:
                    ph4, pw4 = _patch_hw(patch_size)
                    _set_val_cover(val_ds, patch_size, min(ph4, pw4) - 32, 32)  # stride=img_size-32，上下左右 pad 16
                # Stage4 不 resize，直接送入模型（train_img_size 不设）
                print(f"  Stage 4 使用输入目录: {IN_SUBDIR_STAGE4}, patch=img_size={_fmt_size_wh(patch_size)} (输入图+pad16，不 resize), train_pairs={len(train_ds.pairs)} val_pairs={len(val_ds.pairs)}")
            else:
                train_ds = train_dataset
                val_ds = val_dataset
            if getattr(args, "final_patch_size", None) is not None and i == 2:
                patch_size = args.final_patch_size
                fp = args.final_patch_size
                # 给定 img_size 时不再 random patch，固定用 fp；否则 stage3 用 (fp±64) 随机
                if args.img_size != (256, 256):
                    patch_size_range = None
                else:
                    if isinstance(fp, (tuple, list)):
                        patch_size_range = ((fp[0] - 64, fp[1] - 64), (fp[0] + 64, fp[1] + 64))
                    else:
                        patch_size_range = (fp - 64, fp + 64)
                # stride=img_size-32，上下左右各 pad 16
                val_pad = 32
                fph, fpw = _patch_hw(fp)
                max_st = min(fph, fpw) - 32
                val_override = (fp, max_st, val_pad)
            if i < start_stage:
                continue
            se = start_epoch if i == start_stage else 1
            # 进入下一 stage 时加载「最近可用的」上一 stage 的 best_loss
            if i > 0:
                if i == 3:
                    # Stage 4：enc3 与 stage3 相同（20 块），直接加载 stage3 best_loss
                    _stage3_dirs = [save_dir]
                    if resume_ckpt:
                        rd = resume_ckpt.get("save_dir") or (os.path.dirname(args.resume) if args.resume and os.path.isfile(args.resume) else None)
                        if rd and rd not in _stage3_dirs:
                            _stage3_dirs.append(rd)
                    prev_ckpt = None
                    for d in _stage3_dirs:
                        p = os.path.join(d, "best_loss_stage_three.pth")
                        if os.path.isfile(p):
                            prev_ckpt = p
                            break
                    if prev_ckpt:
                        ck = torch.load(prev_ckpt, map_location=device)
                        _unwrap_model(model).load_state_dict(ck["model_state"], strict=True)
                        if "optimizer_state" in ck:
                            optimizer.load_state_dict(ck["optimizer_state"])
                        print(f"  已加载 stage 3 best_loss 到 Stage4（enc3=20，同 stage3）: {prev_ckpt}")
                    else:
                        print(f"  未找到 best_loss_stage_three.pth，使用当前模型权重")
                else:
                    loaded = False
                    for j in range(i - 1, -1, -1):
                        prev_name = stage_names[j]
                        prev_ckpt = os.path.join(save_dir, f"best_loss_stage_{prev_name}.pth")
                        if os.path.isfile(prev_ckpt):
                            ck = torch.load(prev_ckpt, map_location=device)
                            _unwrap_model(model).load_state_dict(ck["model_state"], strict=True)
                            if "optimizer_state" in ck:
                                optimizer.load_state_dict(ck["optimizer_state"])
                            print(f"  已加载 stage {j+1} best_loss: {prev_ckpt}")
                            loaded = True
                            break
                    if not loaded:
                        print(f"  未找到 stage 1..{i} 的 best_loss 文件，使用当前模型权重")
            # Stage 2/3 渐进扩展 enc3：按当前 enc3 长度顺序扩展到目标（12→16→20），stage4 不扩展
            if 1 <= i <= 2:
                target_blocks = 16 if i == 1 else 20
                m = _unwrap_model(model)
                enc3_list = list(getattr(m, "enc3", []))
                current_len = len(enc3_list)
                while current_len < target_blocks:
                    # 下一步：12→16 用 stage_idx=1，16→20 用 stage_idx=2
                    next_stage_idx = 1 if current_len == 12 else 2
                    expand_enc3_for_stage(model, next_stage_idx, optimizer)
                    new_len = len(list(_unwrap_model(model).enc3))
                    if new_len == current_len:
                        break
                    current_len = new_len
                enc3_len = len(list(_unwrap_model(model).enc3))
                print(f"  enc3 已扩展至 {enc3_len} blocks")
            # lr 按当前 stage 索引计算：stage1→base, stage2→base/2, stage3→base/4, stage4→base（直接用 --lr）
            stage_lr = args.lr if i == 3 else args.lr * (0.5 ** i)
            for pg in optimizer.param_groups:
                pg["lr"] = stage_lr
            # stage2 默认 stage=6 neighbor=3；stage3 默认取消 patience（跑满 epochs）
            ps = epochs + 1000 if i >= 2 else (6 if i == 1 else args.patience_stage + i)
            pn = epochs + 1000 if i >= 2 else (3 if i == 1 else args.patience_neighbor + i)
            patch_info = f"{_fmt_size_wh(patch_size_range[0])}-{_fmt_size_wh(patch_size_range[1])} (random)" if patch_size_range else _fmt_size_wh(patch_size)
            patience_info = "off" if i == 2 else f"stage={ps} neighbor={pn}"
            print(f"\n=== Stage {i+1}/{len(STAGES)}: patch={patch_info} ppi={patch_per_image} batch={batch_size} epochs={epochs} lr={stage_lr:.2e} ({desc}) patience={patience_info} ===")
            # 追加本 stage 的超参数到 config.txt，便于追溯每 stage 的设定
            config_path = os.path.join(save_dir, "config.txt")
            patch_str = f"{_fmt_size_wh(patch_size_range[0])}-{_fmt_size_wh(patch_size_range[1])} (random), val={_fmt_size_wh(patch_size)}" if patch_size_range else _fmt_size_wh(patch_size)
            if val_override:
                patch_str += f" (val_override: patch={_fmt_size_wh(val_override[0])} stride={val_override[1]} pad={val_override[2]})"
            # ppi 与 effective_batch 均为 num_gpus*batch_size_per_gpu，与真实使用一致
            _eff = args.batch_size * num_gpus
            stage_block = (
                f"\n--- stage {i+1} ---\n"
                f"lr: {stage_lr:.2e}\n"
                f"patch: {patch_str}\n"
                f"ppi: {patch_per_image}\n"
                f"batch_size_per_gpu: {args.batch_size}\n"
                f"effective_batch: {batch_size * num_gpus}\n"
                f"patience_stage: {'off' if i >= 2 else ps}\n"
                f"patience_neighbor: {'off' if i >= 2 else pn}\n"
                f"epochs: {epochs}\n"
                f"desc: {desc}\n"
            )
            with open(config_path, "a", encoding="utf-8") as f:
                f.write(stage_block)
            best_val_loss, best_val_psnr = run_stage(
                i, patch_size, patch_per_image, batch_size, epochs, desc,
                train_ds, val_ds,
                model, optimizer, device, save_dir,
                perceptual_loss_fn, args,
                train_losses, val_losses, val_psnrs, val_ssims,
                best_val_loss, best_val_psnr,
                start_epoch=se,
                stage_starts=stage_starts,
                patience_stage=ps,
                patience_neighbor=pn,
                patch_size_range=patch_size_range,
                val_override=val_override,
                val_psnrs_fullimage=val_psnrs_fullimage,
                val_ssims_fullimage=val_ssims_fullimage,
                skip_train_resize=(i == 3),
            )
    else:
        train_dataset.patch_size = args.img_size
        train_dataset.patch_per_image = 1
        train_dataset.train_img_size = getattr(args, "train_img_size", None)
        val_dataset.patch_size = args.img_size
        val_ppi = getattr(args, "val_ppi", -1)
        if val_ppi == -1:
            val_dataset.random_crop = False
            _set_val_cover(val_dataset, args.img_size, getattr(args, "val_max_stride", 224))
        else:
            val_dataset.patch_per_image = val_ppi
            val_dataset.random_crop = False
            if getattr(val_dataset, "_val_cover_positions", None) is not None:
                del val_dataset._val_cover_positions
        effective_batch = args.batch_size * num_gpus
        _dl_kw = {
            "batch_size": effective_batch,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            _dl_kw["prefetch_factor"] = 4
        train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kw)
        val_loader = DataLoader(val_dataset, shuffle=False, **_dl_kw)
        no_improve = 0
        no_improve_neighbor = 0
        prev_val_loss, prev_val_psnr = None, None
        for epoch in range(1, args.max_epochs + 1):
            if getattr(train_dataset, "set_epoch", None) is not None:
                train_dataset.set_epoch(epoch)
            tl, _, _ = train_one_epoch(
                model, train_loader, optimizer, device,
                epoch=epoch, max_epochs=args.max_epochs,
                perceptual_loss_fn=perceptual_loss_fn,
                use_rgb_lab_loss=args.loss_rgb_lab,
                loss_l1=getattr(args, "loss_l1", 1.0),
                loss_mse=getattr(args, "loss_mse", 0.0),
                loss_grad=args.loss_grad, loss_ssim=args.loss_ssim, loss_percep=args.loss_percep,
                loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
                loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
                loss_consist=args.loss_consist,
                loss_slide=args.loss_slide,
                loss_consist_l1=args.loss_consist_l1,
                loss_consist_mse=args.loss_consist_mse,
                loss_slide_l1=args.loss_slide_l1,
                loss_slide_mse=args.loss_slide_mse,
            )
            vl, _, _, vp, vs = eval_one_epoch(
                model, val_loader, device,
                perceptual_loss_fn=perceptual_loss_fn,
                use_rgb_lab_loss=args.loss_rgb_lab,
                loss_l1=getattr(args, "loss_l1", 1.0),
                loss_mse=getattr(args, "loss_mse", 0.0),
                loss_grad=args.loss_grad, loss_ssim=args.loss_ssim, loss_percep=args.loss_percep,
                loss_ab_l1=getattr(args, "loss_ab_l1", 0.0),
                loss_ab_mse=getattr(args, "loss_ab_mse", 0.0),
                lock_l=getattr(args, "lock_l", False),
            )
            train_losses.append(tl)
            val_losses.append(vl)
            val_psnrs.append(vp)
            val_ssims.append(vs)
            print(f"[no_staged] Epoch [{epoch}/{args.max_epochs}] Train: {tl:.6f}  Val: {vl:.6f}  PSNR: {vp:.2f}  SSIM: {vs:.4f}")
            with open(os.path.join(save_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(_fmt_log_line(str(epoch), tl, vl, vp, vs) + "\n")
            _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=None)
            imp_loss = vl < best_val_loss
            imp_psnr = vp > best_val_psnr
            if imp_loss:
                best_val_loss = vl
                loss_path = os.path.join(save_dir, f"best_loss_epoch_{epoch}.pth")
                for old in glob.glob(os.path.join(save_dir, "best_loss_epoch_*.pth")):
                    if old != loss_path:
                        try:
                            os.remove(old)
                        except OSError:
                            pass
                torch.save({
                    "epoch": epoch, "model_state": _unwrap_model(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_losses": train_losses, "val_losses": val_losses,
                    "val_psnrs": val_psnrs, "val_ssims": val_ssims,
                    "best_val_loss": best_val_loss, "best_val_psnr": max(best_val_psnr, vp),
                    "save_dir": save_dir,
                    "img_size": getattr(_unwrap_model(model), "img_size", 256),
                }, loss_path)
            if imp_psnr:
                best_val_psnr = vp
                psnr_path = os.path.join(save_dir, f"best_PSNR_epoch_{epoch}.pth")
                for old in glob.glob(os.path.join(save_dir, "best_PSNR_epoch_*.pth")):
                    if old != psnr_path:
                        try:
                            os.remove(old)
                        except OSError:
                            pass
                torch.save({
                    "epoch": epoch, "model_state": _unwrap_model(model).state_dict(),
                    "val_psnr": vp, "val_ssim": vs, "best_val_psnr": best_val_psnr, "save_dir": save_dir,
                    "img_size": getattr(_unwrap_model(model), "img_size", 256),
                }, psnr_path)
            if not imp_loss and not imp_psnr:
                no_improve += 1
            else:
                no_improve = 0

            improved_vs_prev = (prev_val_loss is None) or (vl < prev_val_loss or vp > prev_val_psnr)
            if not improved_vs_prev:
                no_improve_neighbor += 1
            else:
                no_improve_neighbor = 0
            prev_val_loss, prev_val_psnr = vl, vp

            if no_improve >= args.patience_stage:
                print(f"  Early stop: 连续 {args.patience_stage} 轮无 best_loss/best_psnr")
                break
            if no_improve_neighbor >= args.patience_neighbor:
                print(f"  Early stop: 连续 {args.patience_neighbor} 轮没比上一轮好")
                break

    print(f"\n训练结束，best_val_loss={best_val_loss:.6f} best_val_psnr={best_val_psnr:.2f}")
    _save_loss_curve(save_dir, train_losses, val_losses, val_psnrs, val_ssims, stage_starts=stage_starts if stage_starts else None)
    if train_losses and plt:
        print(f"Loss 曲线: {os.path.join(save_dir, 'loss_curve.png')}")


if __name__ == "__main__":
    main()
