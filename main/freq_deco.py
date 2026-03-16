"""
对 demo_1/2/3 涉及的 CR、SH 图像做频域分解（log 空间 + 高斯模糊核），
得到 low / mid / high 三频段，满足 原图(log) = low + mid + high；仅保存中频 mid 到指定目录，不保存低频与高频。
"""
import os
import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
import cv2


def _gaussian_blur_float(img: np.ndarray, sigma: float) -> np.ndarray:
    """img (H,W,3) float，返回同 shape 的高斯模糊结果。"""
    k = int(6 * sigma + 1) | 1
    k = max(3, min(k, 51))
    blurred = cv2.GaussianBlur(img, (k, k), sigma)
    return blurred.astype(np.float32)


def decompose_freq_log(
    img_log: np.ndarray,
    sigma_low: float,
    sigma_mid: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 log 空间将图像分解为 low / mid / high 三频，满足 img_log = low + mid + high。
    - low_log  = blur(img_log, sigma_low)
    - mid_log  = blur(img_log, sigma_mid) - low_log
    - high_log = img_log - blur(img_log, sigma_mid)
    要求 0 < sigma_mid < sigma_low。
    """
    low_log = _gaussian_blur_float(img_log, sigma_low)
    mid_blur = _gaussian_blur_float(img_log, sigma_mid)
    mid_log = (mid_blur - low_log).astype(np.float32)
    high_log = (img_log.astype(np.float32) - mid_blur).astype(np.float32)
    return low_log, mid_log, high_log


def _variance_per_image(arr: np.ndarray) -> float:
    """(H,W,C) 整体方差。"""
    return float(np.var(arr))


def estimate_sigma_equal_freq(
    sample_paths: list,
    load_image_fn,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    按频数均分：在 log 频率上把 [f_min, f_max] 三等分，由边界频率反推 sigma。
    f_min = 1/max(H,W)，f_max = 0.5（奈奎斯特）。高斯 sigma 与频率关系取 sigma = 1/(2*pi*f)。
    返回 (sigma_low, sigma_mid)。
    """
    if not sample_paths:
        return 10.0, 3.0
    h, w = None, None
    for p in sample_paths:
        try:
            img = load_image_fn(p)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                break
        except Exception:
            continue
    if h is None or w is None:
        return 10.0, 3.0
    f_min = 1.0 / max(h, w)
    f_max = 0.5
    # log 频率三等分：log(f1)=log(f_min)+delta/3, log(f2)=log(f_min)+2*delta/3
    log_f_min = np.log(f_min)
    log_f_max = np.log(f_max)
    delta = log_f_max - log_f_min
    f1 = np.exp(log_f_min + delta / 3.0)   # low/mid 边界
    f2 = np.exp(log_f_min + 2.0 * delta / 3.0)  # mid/high 边界
    # sigma = 1/(2*pi*f)，f 越大 sigma 越小，故 sigma_mid=1/(2*pi*f2), sigma_low=1/(2*pi*f1)
    sigma_mid = 1.0 / (2.0 * np.pi * f2)
    sigma_low = 1.0 / (2.0 * np.pi * f1)
    if sigma_low <= sigma_mid:
        sigma_low = sigma_mid + 1.0
    if verbose:
        print(f"    频数均分：图像 {h}x{w}，f_min={f_min:.6f} f_max={f_max:.4f}")
        print(f"    边界 f1={f1:.6f} f2={f2:.6f} -> sigma_low={sigma_low:.4f} sigma_mid={sigma_mid:.4f}")
    return float(sigma_low), float(sigma_mid)


def estimate_sigma_equal_third(
    sample_paths: list,
    load_image_fn,
    max_samples: int = 85,
    sigma_mid_range: Tuple[float, float] = (0.8, 40.0),
    sigma_low_range: Tuple[float, float] = (2.0, 60.0),
    initial_sigma_mid: Optional[float] = None,
    initial_sigma_low: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    根据样本图像统计频域分量，二分搜索 sigma_mid、sigma_low 使 low/mid/high 三档方差均分（各约 1/3）。
    若提供 initial_sigma_mid / initial_sigma_low，则以之为中心缩窄二分区间。
    返回 (sigma_low, sigma_mid)。
    """
    paths = sample_paths[:max_samples]
    if not paths:
        return 10.0, 3.0
    total_vars = []
    for p in paths:
        try:
            img = load_image_fn(p)
            img_log = np.log1p(np.clip(img, 0, None))
            total_vars.append(_variance_per_image(img_log))
        except Exception:
            continue
    if not total_vars:
        return 10.0, 3.0
    target_var = np.mean(total_vars) / 3.0
    if verbose:
        print(f"    目标方差(1/3): {target_var:.6f}  (样本数 {len(paths)})")

    # 若给定初始 sigma，则以之为中心缩窄二分区间
    mid_lo, mid_hi = sigma_mid_range[0], sigma_mid_range[1]
    if initial_sigma_mid is not None:
        mid_lo = max(sigma_mid_range[0], initial_sigma_mid * 0.25)
        mid_hi = min(sigma_mid_range[1], initial_sigma_mid * 4.0)
        if verbose:
            print(f"    以给定 sigma_mid={initial_sigma_mid} 为初始，区间 [{mid_lo:.4f}, {mid_hi:.4f}]")
    low_lo, low_hi = sigma_low_range[0], sigma_low_range[1]
    if initial_sigma_low is not None:
        low_lo = max(sigma_low_range[0], initial_sigma_low * 0.25)
        low_hi = min(sigma_low_range[1], initial_sigma_low * 4.0)
        if verbose:
            print(f"    以给定 sigma_low={initial_sigma_low} 为初始，区间 [{low_lo:.4f}, {low_hi:.4f}]")

    def mean_high_var(sigma_mid: float) -> float:
        v = []
        for p in paths:
            try:
                img = load_image_fn(p)
                img_log = np.log1p(np.clip(img, 0, None))
                high = img_log - _gaussian_blur_float(img_log, sigma_mid)
                v.append(_variance_per_image(high))
            except Exception:
                continue
        return np.mean(v) if v else 0.0

    def mean_low_var(sigma_low: float) -> float:
        v = []
        for p in paths:
            try:
                img = load_image_fn(p)
                img_log = np.log1p(np.clip(img, 0, None))
                low = _gaussian_blur_float(img_log, sigma_low)
                v.append(_variance_per_image(low))
            except Exception:
                continue
        return np.mean(v) if v else 0.0

    if verbose:
        print("    二分 sigma_mid (high 方差 -> target):")
    lo, hi = mid_lo, mid_hi
    for step in range(32):
        mid = (lo + hi) * 0.5
        v = mean_high_var(mid)
        if verbose:
            print(f"      step {step + 1:2d}: sigma_mid={mid:.4f}  mean_var(high)={v:.6f}  target={target_var:.6f}  [{lo:.4f}, {hi:.4f}]")
        if v >= target_var:
            lo = mid
        else:
            hi = mid
    sigma_mid = (lo + hi) * 0.5
    if sigma_mid < 0.5:
        sigma_mid = 0.5

    if verbose:
        print("    二分 sigma_low (low 方差 -> target):")
    lo_low, hi_low = max(sigma_mid + 0.5, low_lo), low_hi
    for step in range(32):
        mid = (lo_low + hi_low) * 0.5
        v = mean_low_var(mid)
        if verbose:
            print(f"      step {step + 1:2d}: sigma_low={mid:.4f}  mean_var(low)={v:.6f}  target={target_var:.6f}  [{lo_low:.4f}, {hi_low:.4f}]")
        if v <= target_var:
            hi_low = mid
        else:
            lo_low = mid
    sigma_low = (lo_low + hi_low) * 0.5
    if sigma_low <= sigma_mid:
        sigma_low = sigma_mid + 1.0
    return float(sigma_low), float(sigma_mid)


def _format_sec(s: float) -> str:
    """秒数转为 1m23s 或 0m05s 格式。"""
    if s < 0 or not np.isfinite(s):
        return "?"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}m{sec:02d}s"


def _progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    width: int = 24,
    elapsed_sec: Optional[float] = None,
    remaining_sec: Optional[float] = None,
) -> str:
    """生成 [>>>>>>>>>>] 45/100 已用 1m23s 剩余 ~2m00s 样式的进度条。"""
    if total <= 0:
        return f"{prefix}[{'?' * width}] 0/0"
    n = int(width * current / total) if current < total else width
    bar = ">" * n + " " * (width - n)
    out = f"{prefix}[{bar}] {current}/{total}"
    if elapsed_sec is not None:
        out += f"  已用 {_format_sec(elapsed_sec)}"
    if remaining_sec is not None and remaining_sec >= 0:
        out += f"  剩余 ~{_format_sec(remaining_sec)}"
    return out


def collect_npy_from_dirs(base_dir: str):
    """从 base_dir 下 IN_CR_COM / IN_SH_COM / GT 扫描所有 .npy，返回 (cr_paths, sh_paths, gt_paths)。"""
    cr_dir = os.path.join(base_dir, "IN_CR_COM")
    sh_dir = os.path.join(base_dir, "IN_SH_COM")
    gt_dir = os.path.join(base_dir, "GT")
    cr_paths = sorted(glob.glob(os.path.join(cr_dir, "*.npy")))
    sh_paths = sorted(glob.glob(os.path.join(sh_dir, "*.npy")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    return cr_paths, sh_paths, gt_paths


def parse_list_files(list_files):
    """解析列表文件，每行: CR_path SH_path GT_path，返回 (cr_paths, sh_paths, gt_paths)。"""
    if isinstance(list_files, str):
        list_files = [list_files]
    cr_paths, sh_paths, gt_paths = [], [], []
    for list_file in list_files:
        if not os.path.exists(list_file):
            continue
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                cr_path, sh_path, gt_path = parts[0], parts[1], parts[2]
                if os.path.exists(cr_path):
                    cr_paths.append(cr_path)
                if os.path.exists(sh_path):
                    sh_paths.append(sh_path)
                if os.path.exists(gt_path):
                    gt_paths.append(gt_path)
    return cr_paths, sh_paths, gt_paths


def load_image(path: str) -> np.ndarray:
    """加载图像为 (H,W,3) float32 [0,1]。支持 .npy 与常见图像格式。"""
    path_lower = path.lower()
    if path_lower.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=np.float32)
        arr = np.atleast_3d(arr)
        if arr.ndim != 3:
            raise ValueError(f".npy 需为 (H,W,C)，当前 shape: {arr.shape}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] != 3:
            raise ValueError(f".npy 通道数需为 1 或 3，当前: {arr.shape[2]}")
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0).astype(np.float32)


def save_component(arr: np.ndarray, out_path: str) -> None:
    """保存频段为 .npy（log 空间 float32）。"""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.save(out_path, arr)


def _process_one_worker(
    path: str,
    mid_dir: str,
    sigma_low: float,
    sigma_mid: float,
) -> None:
    """供多进程调用的单图处理（参数需可 pickle），只保存 mid .npy，不保存低频与高频。"""
    img_lin = load_image(path)
    img_log = np.log1p(np.clip(img_lin, 0, None))
    low_log, mid_log, high_log = decompose_freq_log(img_log, sigma_low, sigma_mid)
    name = os.path.splitext(os.path.basename(path))[0]
    save_component(mid_log, os.path.join(mid_dir, f"{name}.npy"))


def main():
    parser = argparse.ArgumentParser(description="CR/SH 频域分解，仅保存中频 mid 到指定目录")
    parser.add_argument(
        "--list_files",
        type=str,
        nargs="*",
        default=[
            "F:/Code/Datasets/NTIRE2026/color_track/Train/metadata/split1.txt",
            "F:/Code/Datasets/NTIRE2026/color_track/Train/metadata/split2.txt",
            "F:/Code/Datasets/NTIRE2026/color_track/Train/metadata/split3.txt",
        ],
        help="列表文件，每行 CR_path SH_path GT_path（可为 .npy）；不传则从 base_dir 下 IN_CR_COM/IN_SH_COM/GT 扫描 .npy",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="F:/Code/Datasets/NTIRE2026/color_track/Train",
        help="输出目录根，下建 IN_CR_MID_COM 及 IN_SH_MID_COM、GT_MID_COM（仅保存中频）",
    )
    parser.add_argument(
        "--sigma_low",
        type=float,
        default=None,
        help="低频频段高斯模糊 sigma；不设则与 sigma_mid 一起由统计均分得到",
    )
    parser.add_argument(
        "--sigma_mid",
        type=float,
        default=None,
        help="中/高分界高斯模糊 sigma，需 < sigma_low；不设则与 sigma_low 一起由统计均分得到",
    )
    parser.add_argument(
        "--auto_samples",
        type=int,
        default=85,
        help="方差均分(--sigma_split variance)时用于估计 sigma 的最大样本图数量，默认 85；频数均分时不使用",
    )
    parser.add_argument(
        "--bisect",
        action="store_true",
        help="进行二分估计得到 sigma；未指定时默认不二分、用默认或给定 sigma",
    )
    parser.add_argument(
        "--sigma_sample",
        type=str,
        choices=["cr", "sh", "gt"],
        default="cr",
        help="估计 sigma 时用哪类图：cr / sh / gt，默认 sh",
    )
    parser.add_argument(
        "--sigma_split",
        type=str,
        choices=["variance", "freq"],
        default="freq",
        help="等分方式：freq=按频数均分（log 频率三等分），variance=按方差均分（二分搜索），默认 freq",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="调试：仅处理 demo_1 第一行（一张 CR + 一张 SH）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数，默认 min(4, CPU核心数-1)；设为 1 则单进程",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已存在输出 .npy 的样本，用于断点续跑",
    )
    args = parser.parse_args()

    base = args.base_dir
    if args.list_files:
        cr_paths, sh_paths, gt_paths = parse_list_files(args.list_files)
    else:
        cr_paths, sh_paths, gt_paths = collect_npy_from_dirs(base)
        print(f"默认从 npy 目录读取: IN_CR_COM / IN_SH_COM / GT")
    cr_paths = sorted(set(cr_paths))
    sh_paths = sorted(set(sh_paths))
    gt_paths = sorted(set(gt_paths))

    # 未给定的 sigma 用默认值
    if args.sigma_low is None:
        args.sigma_low = 50.0
    if args.sigma_mid is None:
        args.sigma_mid = 5.0
    if args.sigma_mid >= args.sigma_low:
        raise ValueError("请设置 sigma_mid < sigma_low")

    if args.bisect:
        path_sets = {"cr": cr_paths, "sh": sh_paths, "gt": gt_paths}
        sample_paths = list(sorted(set(path_sets[args.sigma_sample])))
        if not sample_paths:
            print(f"{args.sigma_sample.upper()} 样本为空，保留当前 sigma_low={args.sigma_low}, sigma_mid={args.sigma_mid}")
        elif args.sigma_split == "freq":
            print(f"正在以 {args.sigma_sample.upper()} 样本图按频数均分计算 sigma...")
            args.sigma_low, args.sigma_mid = estimate_sigma_equal_freq(
                sample_paths, load_image, verbose=True
            )
            print(f"  结果: sigma_low={args.sigma_low:.4f}, sigma_mid={args.sigma_mid:.4f}")
        else:
            print(f"正在以 {args.sigma_sample.upper()} 样本图按方差均分二分估计 sigma（给定值作初始）...")
            init_mid = args.sigma_mid
            init_low = args.sigma_low
            args.sigma_low, args.sigma_mid = estimate_sigma_equal_third(
                sample_paths,
                load_image,
                max_samples=args.auto_samples,
                verbose=True,
                initial_sigma_mid=init_mid,
                initial_sigma_low=init_low,
            )
            print(f"  估计结果: sigma_low={args.sigma_low:.4f}, sigma_mid={args.sigma_mid:.4f}")

    if args.sigma_mid >= args.sigma_low:
        raise ValueError("sigma_mid 必须小于 sigma_low")

    cr_mid_dir = os.path.join(base, "IN_CR_MID_COM_npy")
    sh_mid_dir = os.path.join(base, "IN_SH_MID_COM_npy")
    gt_mid_dir = os.path.join(base, "GT_MID_COM_npy")
    for d in [cr_mid_dir, sh_mid_dir, gt_mid_dir]:
        os.makedirs(d, exist_ok=True)

    if args.demo:
        cr_paths = cr_paths[:1] if cr_paths else []
        sh_paths = sh_paths[:1] if sh_paths else []
        gt_paths = gt_paths[:1] if gt_paths else []
        print(f"[--demo] 仅处理第一个样本: CR {len(cr_paths)}, SH {len(sh_paths)}, GT {len(gt_paths)}")

    if args.skip_existing:
        def _filter_missing(paths, mid_d):
            out = []
            for p in paths:
                name = os.path.splitext(os.path.basename(p))[0]
                npy_path = os.path.join(mid_d, f"{name}.npy")
                if not os.path.isfile(npy_path):
                    out.append(p)
            return out
        cr_orig, sh_orig, gt_orig = len(cr_paths), len(sh_paths), len(gt_paths)
        cr_paths = _filter_missing(cr_paths, cr_mid_dir)
        sh_paths = _filter_missing(sh_paths, sh_mid_dir)
        gt_paths = _filter_missing(gt_paths, gt_mid_dir)
        skipped = (cr_orig - len(cr_paths)) + (sh_orig - len(sh_paths)) + (gt_orig - len(gt_paths))
        if skipped:
            print(f"[--skip_existing] 跳过已存在输出: 共 {skipped} 个（CR {cr_orig - len(cr_paths)}，SH {sh_orig - len(sh_paths)}，GT {gt_orig - len(gt_paths)}）")

    n_workers = args.workers
    if n_workers is None:
        n_workers = min(4, max(1, (os.cpu_count() or 4) - 1))
    print(f"待处理: CR {len(cr_paths)}，SH {len(sh_paths)}，GT {len(gt_paths)}")
    print(f"sigma_low={args.sigma_low}, sigma_mid={args.sigma_mid}，并行进程数: {n_workers}")

    def _make_tasks(paths, mid_d):
        return [
            (p, mid_d, args.sigma_low, args.sigma_mid)
            for p in paths
        ]

    all_tasks = []
    all_tasks.extend(_make_tasks(cr_paths, cr_mid_dir))
    all_tasks.extend(_make_tasks(sh_paths, sh_mid_dir))
    all_tasks.extend(_make_tasks(gt_paths, gt_mid_dir))
    total = len(all_tasks)

    start_time = time.perf_counter()
    if n_workers <= 1:
        for i, t in enumerate(all_tasks):
            _process_one_worker(*t)
            elapsed = time.perf_counter() - start_time
            remaining = (elapsed / (i + 1)) * (total - (i + 1)) if (i + 1) > 0 else None
            print("\r" + _progress_bar(i + 1, total, prefix="", elapsed_sec=elapsed, remaining_sec=remaining), end="", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one_worker, *t): t for t in all_tasks}
            for fut in as_completed(futures):
                fut.result()
                done += 1
                elapsed = time.perf_counter() - start_time
                remaining = (elapsed / done) * (total - done) if done > 0 else None
                print("\r" + _progress_bar(done, total, prefix="", elapsed_sec=elapsed, remaining_sec=remaining), end="", flush=True)
    if total:
        elapsed_total = time.perf_counter() - start_time
        print()
        print(f"  总耗时: {_format_sec(elapsed_total)}")

    print("完成。输出目录（仅中频 mid）:")
    print(f"  CR: {cr_mid_dir}")
    print(f"  SH: {sh_mid_dir}")
    print(f"  GT: {gt_mid_dir}")


if __name__ == "__main__":
    # Windows 多进程会重新执行本脚本，仅主进程运行 main()
    try:
        from multiprocessing import parent_process
    except ImportError:
        parent_process = None
    if parent_process is None or parent_process() is None:
        main()
