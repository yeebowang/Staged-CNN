"""
可视化目录下所有图片：.npy 及常见图像格式。按 空格/右键 下一张，左键 上一张，q/ESC 退出。
"""
import os
import sys
import glob
import argparse
import random
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Train", "IN_CR_COM_pred"))


def load_image(path: str) -> np.ndarray:
    """加载 .npy 或图像，返回 BGR uint8（cv2.imshow 用）。.npy 存的是 BGR。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.max() > 1.0:
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255).astype(np.uint8)  # .npy 存 BGR，直接用于 imshow
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法加载: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(description="显示目录下所有图片")
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help=f"目录（默认 {DEFAULT_DIR}）",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="起始索引（0-based）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"目录不存在: {args.dir}")
        sys.exit(1)

    exts = ["*.npy", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.dir, ext)))
    files = sorted(set(files))
    random.shuffle(files)
    if not files:
        print(f"未找到图片: {args.dir}")
        sys.exit(1)

    idx = max(0, min(args.idx, len(files) - 1))
    print(f"共 {len(files)} 张，空格/d 下一张，a 上一张，q/ESC 退出")

    while True:
        path = files[idx]
        try:
            img = load_image(path)
        except Exception as e:
            print(f"加载失败 [{idx}] {path}: {e}")
            idx = (idx + 1) % len(files)
            continue
        name = os.path.basename(path)
        cv2.setWindowTitle("npy_vis", f"[{idx + 1}/{len(files)}] {name}")
        cv2.imshow("npy_vis", img)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (32, ord("d"), ord("D")):
            idx = (idx + 1) % len(files)
        elif k in (ord("a"), ord("A")):
            idx = (idx - 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
