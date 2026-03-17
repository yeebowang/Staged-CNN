# Staged-CNN
# NTIRE 2026 Ambient Lighting Normalization (Color Lighting)

Staged-CNN 流程说明：数据准备 → 训练（Stage 1–4）→ 测试与提交。

---

## 0. Prepare（数据准备）

训练与评测所需数据需先转为 `.npy` 并做裁剪/合并，脚本使用顺序如下。

| 脚本 | 作用 | 典型用法 |
|------|------|----------|
| **png2npy.py** | 将目录下 PNG 转为 NumPy 数组并保存为 `.npy` | 处理 `Train/GT/`、`Train/IN_CR_COM/`、`Train/IN_SH_COM/` 等，使后续脚本统一读 `.npy` |
| **crop_npy.py** | 将大图 `.npy` 按固定尺寸裁剪（覆盖全图、重叠尽量小） | 默认处理 `train/GT`、`train/IN_CR_COM`，输出到 `train/GT_crop`、`train/IN_CR_COM_crop`（默认裁剪 1024×768）；也可 `--only_dir` 只处理一个目录 |
| **combine_npy.py** | 将 6×6=36 个 crop 按 3×3 合并成 2×2=4 个，按 stride 拼成一大块后 resize 到 1024×768 | 输入 `train/GT_crop`、`train/IN_CR_COM_crop`，输出 `train/GT_crop_resize`、`train/IN_CR_COM_crop_resize`（训练默认使用 crop_resize 数据） |
| **split_dataset.py** | 按 id 划分 train/val 列表，写入 `Train/metadata/` | 生成 train/val 文件名列表，供训练脚本按索引划分验证集 |

其他可选脚本：

- **rename_png.py**：按规则重命名/整理 PNG（如将 `IN_CR` 子目录合并到 `IN_CR_COM`）
- **benchmark_preprocess.py**：数据预处理耗时基准（resize + 裁 patch），用于评估数据管线性能

训练数据目录约定（crop_resize 流程）：

- **GT**：`*_GT.npy` 或裁剪后 `*_GT_crop_r*_c*.npy`，或合并后 `*_GT_crop_resize_R*_C*.npy`
- **输入**：`*_IN.npy` 或 `*_IN_crop_resize_R*_C*.npy` 等，与 GT 命名对应
- **Stage 4 专用**：`IN_CR_COM_pred`（由 Stage 1–3 预测得到，整图精修用）

---

## 1. Train（训练）

主脚本：**train_crop_patch_0315.py**，采用 4 个 Stage 的课程学习。

| Stage | patch 尺寸 | 目标 | 说明 |
|-------|------------|------|------|
| **Stage 1** | 128 | 纹理、光照统计 | 每图多 patch，小 patch 学局部 |
| **Stage 2** | 192 | 中尺度结构 | 扩大感受野 |
| **Stage 3** | 256 | 全局一致性 | val 固定 288/256/32 铺满；可选整图验证 |
| **Stage 4** | 整图 (img_size+pad16) | 全图精修 | 仅用 **IN_CR_COM_pred**（Stage 1–3 的预测结果），单图 1 patch，不 random_crop |

常用参数示例：

```bash
# 从 Stage 1 开始完整训练（需先准备好 GT_crop_resize / IN_CR_COM_crop_resize）
python train_crop_patch_0315.py --train_base <path_to_Train> --num_gpus 2 --batch_per_gpu 2

# 从 Stage 4 开始（加载 Stage 3 的 best_loss，只训整图精修）
python train_crop_patch_0315.py --start_stage 4 --resume checkpoints/<run>/best_loss_epoch_3.pth --img_size 544,416 ...
```

- **--start_stage**：从第几 stage 开始（1/2/3/4）
- **--resume**：恢复训练的 checkpoint（如 `best_loss_epoch_3.pth` 或 `latest.pth`）
- **--img_size**：Stage 4 时与输入分辨率一致（如 544,416），内部会加 pad16
- **--lock_l**：验证时锁定 LAB 的 L，用「输入 L + 预测 AB」算 val 指标

输出：每个 run 在 `checkpoints/<timestamp>/` 下，包含 `best_loss_epoch_*.pth`、`best_PSNR_epoch_*.pth`、`latest.pth`、`config.txt`、`loss_curve.png` 等；Stage 4 会使用 `IN_CR_COM_pred` 目录下的预测图作为输入。

---

## 2. Test（测试与提交）

主脚本：**submission_patch.py**，用于推理并打包提交（含 readme、输出图、可选 zip）。

### 输入文件夹含义（test → test_step1 → test_step2）

| 目录 | 含义 |
|------|------|
| **test** | 原始测试图像（若竞赛提供的是未处理图像，可先放这里） |
| **test_step1** | 第一步预处理后的输入目录，**submission_patch.py 默认 `--input_dir` 指向此处**；目录内为待推理的输入文件（.npy 或 .png/.jpg 等） |
| **test_step2** | 若有分阶段评测或第二步流程，可把 step1 的输出再处理后放入 test_step2 作为下一阶段输入；本仓库中 submission 直接读 **test_step1** 并写出结果 |

即：**test（原始）→ test_step1（当前推理脚本的输入）→ 输出到指定目录或打包 zip**；test_step2 仅在你有额外第二步流程时使用。

### 基本用法

```bash
# 使用默认输入目录 test_step1、从 checkpoint 目录自动找 best
python submission_patch.py --cnn_checkpoint checkpoints/<run_dir> --sliding_window

# 指定输入/输出与 checkpoint
python submission_patch.py --input_dir /path/to/test_step1 --tmp_out_dir ./submission_outputs --cnn_checkpoint checkpoints/<run_dir>/best_PSNR_epoch_3.pth --sliding_window

# 生成提交 zip（含 readme、输出图）
python submission_patch.py --input_dir /path/to/test_step1 --cnn_checkpoint checkpoints/<run_dir> --sliding_window --output_zip submission.zip
```

### 主要参数

- **--input_dir**：输入目录，默认 `test_step1`；支持 **.npy** 与 **.png/.jpg/.bmp/.tif** 等
- **--cnn_checkpoint**：checkpoint 目录或单个 `.pth` 路径；可为 `best`/`latest` 自动在目录中查找
- **--sliding_window**：使用滑动窗口推理（否则为整图或其它模式）
- **--tmp_out_dir**：推理结果保存目录（再被打包进 zip）
- **--output_zip**：若指定则打包为 zip 用于提交
- **--img_size**：与训练一致（如 544,416）；不指定则从 checkpoint 的 config.txt 读
- **--lock_l**：推理后用「输入 L + 预测 AB」合成最终图像
- **--path_remap**：路径替换（如 `F:` → `I:`），便于换盘符

推理结果会写入 `--tmp_out_dir`，readme 中会包含 runtime、CPU/GPU、Other description（如仓库链接）等；若指定 `--output_zip` 则生成可直接提交的 zip。
