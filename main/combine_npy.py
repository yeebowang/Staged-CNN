"""
将 36 个 crop（6x6）按 3x3 合并成 4 个，按 stride 拼成一大块后 resize 到 1024x768。
输入：GT_crop / IN_CR_COM_crop（*_crop_r{r}_c{c}.npy，r,c in 0..5）
输出：GT_crop_resize / IN_CR_COM_crop_resize（*_crop_resize_R{R}_C{C}.npy，R,C in 0,1）
stride 默认 (995, 646)，与 crop_npy 的 6000x4000->1024x768 一致。
"""
import os
import re
import argparse
import numpy as np
import cv2

# 与 crop_npy 一致：单 crop 尺寸与 stride
CROP_W, CROP_H = 1024, 768
STRIDE_W, STRIDE_H = 995, 646


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.max() > 1.0:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    return arr


def merge_3x3_and_resize(
    grid: dict,
    stride_w: int = STRIDE_W,
    stride_h: int = STRIDE_H,
    crop_w: int = CROP_W,
    crop_h: int = CROP_H,
    out_w: int = 1024,
    out_h: int = 768,
) -> np.ndarray:
    """
    grid: (r,c) -> array，9 个 crop（3x3），每个 crop_w x crop_h。
    按 stride 放置：crop(r,c) 放在 (c*stride_w, r*stride_h)。
    合并后 canvas 为 (2*stride_h+crop_h) x (2*stride_w+crop_w)，再 resize 到 out_w x out_h。
    """
    merge_w = 2 * stride_w + crop_w
    merge_h = 2 * stride_h + crop_h
    ch = 3
    canvas = np.zeros((merge_h, merge_w, ch), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            arr = grid.get((r, c))
            if arr is None:
                continue
            y0 = r * stride_h
            x0 = c * stride_w
            canvas[y0 : y0 + crop_h, x0 : x0 + crop_w] = arr
    out = cv2.resize(
        (np.clip(canvas, 0, 1) * 255).astype(np.uint8),
        (out_w, out_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def combine_dir(
    in_dir: str,
    out_dir: str,
    pattern: str = "*_crop_r*_c*.npy",
    suffix_out: str = "_crop_resize",
    stride_w: int = STRIDE_W,
    stride_h: int = STRIDE_H,
) -> int:
    """
    遍历 in_dir 下符合 pattern 的 .npy，按 id 分组，每组 36 个 (r,c) 按 3x3 合并成 4 个 (R,C)，保存到 out_dir。
    命名：{base}_crop_resize_R{R}_C{C}.npy，R,C in 0,1。
    """
    os.makedirs(out_dir, exist_ok=True)
    re_crop = re.compile(r"^(.+)_crop_r(\d+)_c(\d+)\.npy$")
    files = []
    for name in os.listdir(in_dir):
        if not name.lower().endswith(".npy"):
            continue
        m = re_crop.match(name)
        if not m:
            continue
        base, r, c = m.group(1), int(m.group(2)), int(m.group(3))
        if r < 6 and c < 6:
            files.append((base, r, c, os.path.join(in_dir, name)))
    by_base = {}
    for base, r, c, path in files:
        by_base.setdefault(base, {})[(r, c)] = path
    total = 0
    for base, full_grid in by_base.items():
        if len(full_grid) != 36:
            print(f"  跳过 {base}: 需要 36 个 crop，当前 {len(full_grid)}")
            continue
        for R in range(2):
            for C in range(2):
                sub = {}
                for lr in range(3):
                    for lc in range(3):
                        r, c = 3 * R + lr, 3 * C + lc
                        p = full_grid.get((r, c))
                        if p is None:
                            continue
                        sub[(lr, lc)] = _load_npy(p)
                if len(sub) != 9:
                    continue
                out_arr = merge_3x3_and_resize(sub, stride_w=stride_w, stride_h=stride_h)
                out_name = f"{base}{suffix_out}_R{R}_C{C}.npy"
                out_path = os.path.join(out_dir, out_name)
                np.save(out_path, out_arr)
                total += 1
        print(f"  {base}: 36 -> 4 已写入 {out_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="将 6x6=36 个 crop 按 3x3 合并成 2x2=4 个，stride 拼图后 resize 到 1024x768。"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="..",
        help="数据根目录（默认上一级），其下 train/GT_crop、train/IN_CR_COM_crop",
    )
    parser.add_argument(
        "--only_dir",
        type=str,
        default=None,
        help="只处理该目录（否则默认处理 train/GT_crop 和 train/IN_CR_COM_crop）",
    )
    parser.add_argument("--stride_w", type=int, default=STRIDE_W, help="列 stride（默认 995）")
    parser.add_argument("--stride_h", type=int, default=STRIDE_H, help="行 stride（默认 646）")
    args = parser.parse_args()
    root = args.root
    total = 0
    if args.only_dir:
        in_dir = args.only_dir
        out_dir = in_dir.rstrip("/").rstrip("\\") + "_resize"
        print(f"处理: {in_dir} -> {out_dir}")
        total = combine_dir(in_dir, out_dir, stride_w=args.stride_w, stride_h=args.stride_h)
    else:
        gt_in = os.path.join(root, "train", "GT_crop")
        gt_out = os.path.join(root, "train", "GT_crop_resize")
        in_in = os.path.join(root, "train", "IN_CR_COM_crop")
        in_out = os.path.join(root, "train", "IN_CR_COM_crop_resize")
        if os.path.isdir(gt_in):
            n = combine_dir(gt_in, gt_out, stride_w=args.stride_w, stride_h=args.stride_h)
            print(f"[GT_crop] 共 {n} 个 -> {gt_out}")
            total += n
        else:
            print(f"[GT_crop] 目录不存在: {gt_in}")
        if os.path.isdir(in_in):
            n = combine_dir(in_in, in_out, stride_w=args.stride_w, stride_h=args.stride_h)
            print(f"[IN_CR_COM_crop] 共 {n} 个 -> {in_out}")
            total += n
        else:
            print(f"[IN_CR_COM_crop] 目录不存在: {in_in}")
    print(f"合计写入 {total} 个 crop_resize 文件。")


if __name__ == "__main__":
    main()
