"""
从指定文件夹读取 .npy 图像，按固定尺寸裁剪并保存。
支持 6000×4000 等大图，计算覆盖全图且重叠尽量少所需的裁剪网格。

例：原图 6000×4000，裁剪 1024×768（宽×高）
  - 最少需要 6×6 = 36 个 crops 才能覆盖全图且重叠尽量小
  - 列 stride=995（重叠 29px），行 stride=646（重叠 122px）
"""
import os
import argparse
import numpy as np


def compute_grid(img_w: int, img_h: int, crop_w: int, crop_h: int):
    """
    计算覆盖整图、重叠尽量小的裁剪网格。
    最后一列/行贴边，保证全图被覆盖。
    返回 (n_cols, n_rows), (stride_w, stride_h), [(left, top), ...], (overlap_w, overlap_h)
    """
    n_cols = max(1, (img_w + crop_w - 1) // crop_w)
    n_rows = max(1, (img_h + crop_h - 1) // crop_h)
    if n_cols == 1:
        stride_w = 0
    else:
        stride_w = (img_w - crop_w) // (n_cols - 1)
    if n_rows == 1:
        stride_h = 0
    else:
        stride_h = (img_h - crop_h) // (n_rows - 1)
    positions = []
    for row in range(n_rows):
        if n_rows == 1:
            top = 0
        elif row == n_rows - 1:
            top = img_h - crop_h
        else:
            top = row * stride_h
        for col in range(n_cols):
            if n_cols == 1:
                left = 0
            elif col == n_cols - 1:
                left = img_w - crop_w
            else:
                left = col * stride_w
            positions.append((left, top))
    overlap_w = crop_w - stride_w if n_cols > 1 else 0
    overlap_h = crop_h - stride_h if n_rows > 1 else 0
    return (n_cols, n_rows), (stride_w, stride_h), positions, (overlap_w, overlap_h)


def crop_npy_dir(
    in_dir: str,
    out_dir: str,
    crop_w: int = 1024,
    crop_h: int = 768,
    ext: str = ".npy",
    suffix: str = "_crop",
) -> int:
    """
    遍历 in_dir 下所有 .npy，按 crop_w×crop_h 裁剪并保存到 out_dir。
    输出命名：原文件名 + suffix + "_r{R}_c{C}.npy"（行、列从 0 起）。
    """
    os.makedirs(out_dir, exist_ok=True)
    count_files = 0
    total_crops = 0
    for name in sorted(os.listdir(in_dir)):
        if not name.lower().endswith(ext):
            continue
        path = os.path.join(in_dir, name)
        arr = np.load(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.max() > 1.0:
            arr = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
        h, w = arr.shape[0], arr.shape[1]
        if w < crop_w or h < crop_h:
            print(f"  跳过 {name}: 尺寸 {w}x{h} 小于裁剪 {crop_w}x{crop_h}")
            continue
        (n_cols, n_rows), (stride_w, stride_h), positions, (overlap_w, overlap_h) = compute_grid(
            w, h, crop_w, crop_h
        )
        base, _ = os.path.splitext(name)
        n_crops = 0
        for idx, (left, top) in enumerate(positions):
            crop = arr[top : top + crop_h, left : left + crop_w]
            if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
                continue
            r, c = idx // n_cols, idx % n_cols
            out_name = f"{base}{suffix}_r{r}_c{c}.npy"
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, crop.astype(np.float32))
            n_crops += 1
        if n_crops:
            print(f"  {name} ({w}x{h}) -> {n_rows}x{n_cols}={n_crops} crops, stride=({stride_w},{stride_h}), overlap=({overlap_w},{overlap_h})")
            count_files += 1
            total_crops += n_crops
    return total_crops


def main():
    parser = argparse.ArgumentParser(
        description=(
            "将目录下 .npy 图像裁剪为固定尺寸（覆盖全图、重叠尽量小）。\n"
            "默认会同时处理 ../train/GT 和 ../train/IN_CR_COM，输出到 ../train/GT_crop 和 ../train/IN_CR_COM_crop。"
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="..",
        help="数据根目录，默认上一级（期望存在 train/GT 和 train/IN_CR_COM）",
    )
    parser.add_argument(
        "--only_dir",
        type=str,
        default=None,
        help="仅处理指定目录（覆盖默认的 GT/IN_CR_COM 双目录），例如 ../train/GT",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="输出目录（仅当 --only_dir 生效时使用）；默认 input_dir/crops_WxH",
    )
    parser.add_argument("--cw", type=int, default=1024, help="裁剪宽度")
    parser.add_argument("--ch", type=int, default=768, help="裁剪高度")
    parser.add_argument("--suffix", type=str, default="_crop", help="输出文件名中缀")
    args = parser.parse_args()
    total = 0

    if args.only_dir is not None:
        in_dir = args.only_dir
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = os.path.join(in_dir, f"crops_{args.cw}x{args.ch}")
        print(f"处理单个目录: {in_dir}")
        total = crop_npy_dir(in_dir, out_dir, crop_w=args.cw, crop_h=args.ch, suffix=args.suffix)
        print(f"共生成 {total} 个 crop，保存到 {out_dir}")
    else:
        # 默认：处理 ../train/GT 和 ../train/IN_CR_COM
        root = args.root
        gt_dir = os.path.join(root, "train", "GT")
        in_dir = os.path.join(root, "train", "IN_CR_COM")
        gt_out = os.path.join(root, "train", "GT_crop")
        in_out = os.path.join(root, "train", "IN_CR_COM_crop")

        print(f"默认模式：处理 {gt_dir} -> {gt_out} 和 {in_dir} -> {in_out}")
        if os.path.isdir(gt_dir):
            t_gt = crop_npy_dir(gt_dir, gt_out, crop_w=args.cw, crop_h=args.ch, suffix=args.suffix)
            print(f"[GT] 共生成 {t_gt} 个 crop，保存到 {gt_out}")
            total += t_gt
        else:
            print(f"[GT] 目录不存在，跳过: {gt_dir}")

        if os.path.isdir(in_dir):
            t_in = crop_npy_dir(in_dir, in_out, crop_w=args.cw, crop_h=args.ch, suffix=args.suffix)
            print(f"[IN_CR_COM] 共生成 {t_in} 个 crop，保存到 {in_out}")
            total += t_in
        else:
            print(f"[IN_CR_COM] 目录不存在，跳过: {in_dir}")

        print(f"两类目录共生成 {total} 个 crop")


if __name__ == "__main__":
    main()
