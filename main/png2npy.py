"""
将目录下 PNG 图片转为 NumPy 数组并保存为 .npy 文件。
"""
import os
import cv2
import numpy as np

cr_path = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_CR_COM/"
sh_path = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_SH_COM/"
gt_path = "F:/Code/Datasets/NTIRE2026/color_track/Train/GT/"

def png_dir_to_npy(
    png_dir: str,
    out_dir: str = None,
    grayscale: bool = True,
    ext: str = ".png",
) -> int:
    """
    将 png_dir 下所有 PNG 转为 .npy 并保存。
    :param png_dir: 存放 PNG 的目录
    :param out_dir: 保存 .npy 的目录，默认与 png_dir 相同
    :param grayscale: True 用灰度读取，False 用 BGR 读取
    :param ext: 要处理的图片后缀，默认 .png
    :return: 成功转换的文件数量
    """
    if out_dir is None:
        out_dir = png_dir
    os.makedirs(out_dir, exist_ok=True)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    count = 0
    for name in os.listdir(png_dir):
        if not name.lower().endswith(ext):
            continue
        path = os.path.join(png_dir, name)
        img = cv2.imread(path, flag)
        if img is None:
            print(f"  跳过（读取失败）: {path}")
            continue
        base, _ = os.path.splitext(name)
        npy_path = os.path.join(out_dir, base + ".npy")
        np.save(npy_path, img)
        count += 1
        print(f"  {name} -> {base}.npy  shape={img.shape}")
    return count


if __name__ == "__main__":
    # 可选：指定保存目录，否则保存在原目录
    # out_cr = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_CR_COM"
    #out_sh = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_SH_COM"

    #print("CR 目录 PNG -> NPY (彩色):")
    #n_cr = png_dir_to_npy(cr_path, out_dir=None, grayscale=False)
    #print(f"  共 {n_cr} 个文件\n")

    #print("SH 目录 PNG -> NPY (彩色):")
    #n_sh = png_dir_to_npy(sh_path, out_dir=None, grayscale=False)
    #print(f"  共 {n_sh} 个文件")

    print("GT 目录 PNG -> NPY (彩色):")
    n_gt = png_dir_to_npy(gt_path, out_dir=None, grayscale=False)
    print(f"  共 {n_gt} 个文件")