"""
数据预处理耗时基准：6000×4000 → resize 1024×768 → 裁剪 256×256 patches。
用于评估训练时数据增强/预处理是否可接受。
支持从 GT_MID_COM_npy 目录读取真实 .npy 做实验。
"""
import os
import glob
import time
import argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def resize_cv2(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """cv2.resize: (H,W,C) -> target (h,w)."""
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def resize_pil(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """PIL resize: (H,W,C) -> target (h,w)."""
    pil = Image.fromarray(img)
    pil = pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(pil)


def crop_to_patches(img: np.ndarray, patch_size: int) -> list:
    """将 (H,W,C) 裁剪成不重叠的 patch_size x patch_size patches，返回 list of patches。"""
    h, w = img.shape[0], img.shape[1]
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if y + patch_size <= h and x + patch_size <= w:
                patches.append(img[y : y + patch_size, x : x + patch_size])
    return patches


def _npy_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """(H,W) 或 (H,W,C) float/uint8 -> (H,W,3) uint8，供 resize/crop。"""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3):
        arr = np.atleast_3d(arr)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def run_benchmark_gt_npy(
    gt_npy_dir: str,
    resize_to: tuple = (768, 1024),
    patch_size: int = 256,
    n_warmup: int = 2,
    n_repeat: int = 20,
    max_files: int = 50,
    use_cv2: bool = True,
):
    """
    从 GT_MID_COM_npy 等目录读取真实 .npy，做 resize → 裁剪 256 的耗时基准。
    """
    if use_cv2 and not HAS_CV2:
        use_cv2 = False
    if not use_cv2 and not HAS_PIL:
        raise RuntimeError("需要 cv2 或 PIL")
    resize_fn = resize_cv2 if use_cv2 else resize_pil

    npy_paths = sorted(glob.glob(os.path.join(gt_npy_dir, "*.npy")))
    if not npy_paths:
        raise FileNotFoundError(f"目录下没有 .npy 文件: {gt_npy_dir}")
    npy_paths = npy_paths[:max_files]
    print(f"实验：从 GT 目录读取 .npy，共 {len(npy_paths)} 个文件")
    print(f"  目录: {gt_npy_dir}")
    print(f"  Resize 到: {resize_to[0]}×{resize_to[1]}，Patch: {patch_size}×{patch_size}")
    print()

    # warmup
    arr0 = np.load(npy_paths[0])
    img0 = _npy_to_uint8_rgb(arr0)
    for _ in range(n_warmup):
        r = resize_fn(img0, resize_to)
        _ = crop_to_patches(r, patch_size)

    t_load_list = []
    t_resize_list = []
    t_crop_list = []
    t_full_list = []

    for path in npy_paths:
        t0 = time.perf_counter()
        arr = np.load(path)
        img = _npy_to_uint8_rgb(arr)
        t_load_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        r = resize_fn(img, resize_to)
        t_resize_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        patches = crop_to_patches(r, patch_size)
        t_crop_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        arr = np.load(path)
        img = _npy_to_uint8_rgb(arr)
        r = resize_fn(img, resize_to)
        patches = crop_to_patches(r, patch_size)
        t_full_list.append(time.perf_counter() - t0)

    n_patches = (resize_to[0] // patch_size) * (resize_to[1] // patch_size)
    t_load_ms = np.median(t_load_list) * 1000
    t_resize_ms = np.median(t_resize_list) * 1000
    t_crop_ms = np.median(t_crop_list) * 1000
    t_full_ms = np.median(t_full_list) * 1000

    print("=" * 60)
    print("数据预处理耗时（真实 GT .npy，单张图）")
    print("=" * 60)
    print(f"  样本数:      {len(npy_paths)} 个 .npy")
    print(f"  Resize 到:   {resize_to[0]} × {resize_to[1]}")
    print(f"  Patch:       {patch_size}×{patch_size}，{n_patches} 个/图")
    print(f"  Resize 实现: {'cv2' if use_cv2 else 'PIL'}")
    print()
    print(f"  仅 Load:          {t_load_ms:7.2f} ms/图")
    print(f"  仅 Resize:        {t_resize_ms:7.2f} ms/图")
    print(f"  仅 裁剪:          {t_crop_ms:7.2f} ms/图")
    print(f"  Load+Resize+裁剪: {t_full_ms:7.2f} ms/图")
    print()
    per_epoch = (1000 * (t_full_ms / 1000)) / 4
    print("  粗算（1000 张/epoch，4 worker）：预处理约 {:.2f} s/epoch".format(per_epoch))
    print("=" * 60)


def run_benchmark(
    orig_size: tuple = (4000, 6000),  # (H, W)
    resize_to: tuple = (768, 1024),   # (h, w)
    patch_size: int = 256,
    n_warmup: int = 3,
    n_repeat: int = 20,
    use_cv2: bool = True,
):
    """
    orig_size: 原始图 (H, W)
    resize_to: resize 后 (h, w)，如 1024×768 则 (768, 1024)
    """
    if use_cv2 and not HAS_CV2:
        use_cv2 = False
        print("未安装 cv2，改用 PIL 做 resize")
    if not use_cv2 and not HAS_PIL:
        raise RuntimeError("需要 cv2 或 PIL")

    resize_fn = resize_cv2 if use_cv2 else resize_pil
    H, W = orig_size
    h, w = resize_to

    # 构造模拟图 (H, W, 3) uint8，避免读盘
    print(f"构造模拟图 {H}×{W}...")
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    # warmup
    for _ in range(n_warmup):
        r = resize_fn(img, resize_to)
        _ = crop_to_patches(r, patch_size)

    # 只测 resize
    t_resize_list = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        r = resize_fn(img, resize_to)
        if use_cv2 and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            pass  # 若用 GPU 可 sync
        t_resize_list.append(time.perf_counter() - t0)
    t_resize = np.median(t_resize_list) * 1000  # ms

    # resize + 裁剪
    t_full_list = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        r = resize_fn(img, resize_to)
        patches = crop_to_patches(r, patch_size)
        t_full_list.append(time.perf_counter() - t0)
    t_full = np.median(t_full_list) * 1000  # ms
    n_patches = (resize_to[0] // patch_size) * (resize_to[1] // patch_size)

    print()
    print("=" * 60)
    print("数据预处理耗时（单张图）")
    print("=" * 60)
    print(f"  原始尺寸:     {H} × {W}")
    print(f"  Resize 到:   {resize_to[0]} × {resize_to[1]} (h×w)")
    print(f"  Patch 尺寸:  {patch_size} × {patch_size}")
    print(f"  Patch 数量:  {n_patches} 个/图")
    print(f"  Resize 实现: {'cv2' if use_cv2 else 'PIL'}")
    print()
    print(f"  仅 Resize:         {t_resize:7.2f} ms/图")
    print(f"  Resize + 裁剪:     {t_full:7.2f} ms/图  (含 {n_patches} 个 patch)")
    print(f"  单 patch 等效:     {t_full / n_patches:7.2f} ms/patch")
    print()

    # 训练场景粗算
    # 假设 DataLoader num_workers=4，batch_size=8，每个 batch 有 8 张图
    # 若 1000 张图/epoch，预处理总时间 ≈ 1000 * (t_full/1000) = t_full 秒（若单线程）
    # 多 worker 时每个 worker 处理 1000/4 张，每张 t_full ms，总时间约 (1000/4)*(t_full/1000)/4 量级
    per_epoch_sec = (1000 * (t_full / 1000)) / 4  # 粗算：1000 张图，4 worker 并行
    print("  粗算（1000 张/epoch，4 worker 并行）：")
    print(f"    预处理约 {per_epoch_sec:.2f} s/epoch（若 CPU 不成为瓶颈）")
    print()
    if t_full < 50:
        print("  结论: 单图 < 50 ms，训练中通常可接受（DataLoader 多 worker 可掩盖）。")
    elif t_full < 200:
        print("  结论: 单图几十～两百 ms，可接受；若卡 IO/预处理可适当加大 num_workers。")
    else:
        print("  结论: 单图耗时较高，建议加大 num_workers 或考虑离线预处理成 256×256 再训练。")
    print("=" * 60)


# 默认实验用 GT 目录（中频 npy）
DEFAULT_GT_NPY_DIR = r"F:/Code/Datasets/NTIRE2026/color_track/Train/GT_MID_COM_npy"


def main():
    parser = argparse.ArgumentParser(description="数据预处理耗时：6000×4000 → 1024×768 → 256×256 patches；或从 GT_MID_COM_npy 读 .npy 做实验")
    parser.add_argument("--gt_npy_dir", type=str, default=None, help=f"用真实 .npy 做实验：从该目录读取（默认: {DEFAULT_GT_NPY_DIR}）；不传则用随机图")
    parser.add_argument("--orig_h", type=int, default=4000, help="原图高（仅随机图模式）")
    parser.add_argument("--orig_w", type=int, default=6000, help="原图宽（仅随机图模式）")
    parser.add_argument("--resize_h", type=int, default=768, help="resize 后高")
    parser.add_argument("--resize_w", type=int, default=1024, help="resize 后宽")
    parser.add_argument("--patch_size", type=int, default=256, help="裁剪 patch 边长")
    parser.add_argument("--repeat", type=int, default=20, help="计时重复次数（取中位数）")
    parser.add_argument("--max_files", type=int, default=50, help="GT 实验时最多用多少个 .npy 文件")
    parser.add_argument("--pil", action="store_true", help="用 PIL 做 resize（默认 cv2）")
    args = parser.parse_args()

    use_gt = args.gt_npy_dir is not None or os.path.isdir(DEFAULT_GT_NPY_DIR)
    if use_gt:
        gt_dir = args.gt_npy_dir or DEFAULT_GT_NPY_DIR
        run_benchmark_gt_npy(
            gt_npy_dir=gt_dir,
            resize_to=(args.resize_h, args.resize_w),
            patch_size=args.patch_size,
            n_repeat=args.repeat,
            max_files=args.max_files,
            use_cv2=not args.pil,
        )
    else:
        run_benchmark(
            orig_size=(args.orig_h, args.orig_w),
            resize_to=(args.resize_h, args.resize_w),
            patch_size=args.patch_size,
            n_repeat=args.repeat,
            use_cv2=not args.pil,
        )


if __name__ == "__main__":
    main()
