# For NTIRE 2026 Ambient Lighting Normalization (Color Lighting)

Staged-CNN workflow: **Data preparation → Training (Stage 1–4) → Test & submission**.

---

## 0. Prepare (Data preparation)

Training and evaluation data must be converted to `.npy` and optionally cropped/merged. Use the scripts in this order:

| Script | Purpose | Typical usage |
|--------|---------|---------------|
| **png2npy.py** | Convert PNGs in a directory to NumPy arrays and save as `.npy` | Process `Train/GT/`, `Train/IN_CR_COM/`, `Train/IN_SH_COM/`, etc., so downstream scripts read `.npy` |
| **crop_npy.py** | Crop large `.npy` images to a fixed size (full coverage, minimal overlap) | By default processes `train/GT` and `train/IN_CR_COM` → `train/GT_crop`, `train/IN_CR_COM_crop` (default crop 1024×768); use `--only_dir` to process a single directory |
| **combine_npy.py** | Merge 6×6=36 crops into 2×2=4 by combining 3×3 blocks with stride, then resize to 1024×768 | Input: `train/GT_crop`, `train/IN_CR_COM_crop` → output: `train/GT_crop_resize`, `train/IN_CR_COM_crop_resize` (default training data) |

Optional scripts:

- **rename_png.py**: Rename or reorganize PNGs (e.g. merge `IN_CR` subdirs into `IN_CR_COM`)
- **benchmark_preprocess.py**: Data preprocessing time benchmark (resize + patch crop) for pipeline performance

Data layout (crop_resize pipeline):

- **GT**: `*_GT.npy`, or after crop `*_GT_crop_r*_c*.npy`, or after combine `*_GT_crop_resize_R*_C*.npy`
- **Input**: `*_IN.npy` or `*_IN_crop_resize_R*_C*.npy`, etc., matching GT naming
- **Stage 4 only**: `IN_CR_COM_pred` (predictions from Stage 1–3, used for full-image refinement)

---

## 1. Train

Main script: **train_crop_patch_0315.py**, 4-stage curriculum learning.

| Stage | Patch size | Focus | Notes |
|-------|------------|-------|--------|
| **Stage 1** | 128 | Texture, illumination statistics | Multiple patches per image, small patches for local patterns |
| **Stage 2** | 192 | Mid-scale structure | Larger receptive field |
| **Stage 3** | 256 | Global consistency | Val uses fixed 288/256/32 tiling; optional full-image validation |
| **Stage 4** | Full image (img_size+pad16) | Full-image refinement | Uses only **IN_CR_COM_pred** (Stage 1–3 outputs), 1 patch per image, no random_crop |

Example commands:

```bash
# Full training from Stage 1 (requires GT_crop_resize / IN_CR_COM_crop_resize)
python train_crop_patch_0315.py --train_base <path_to_Train> --num_gpus 2 --batch_per_gpu 2

# Start from Stage 4 (load Stage 3 best_loss, train full-image refinement only)
python train_crop_patch_0315.py --start_stage 4 --resume checkpoints/<run>/best_loss_epoch_3.pth --img_size 544,416 ...
```

- **--start_stage**: Which stage to start from (1/2/3/4)
- **--resume**: Checkpoint to resume (e.g. `best_loss_epoch_3.pth` or `latest.pth`)
- **--img_size**: For Stage 4, match input resolution (e.g. 544,416); pad16 is applied internally
- **--lock_l**: Lock LAB L during validation; compute val metrics with “input L + predicted AB”

Output: each run under `checkpoints/<timestamp>/` with `best_loss_epoch_*.pth`, `best_PSNR_epoch_*.pth`, `latest.pth`, `config.txt`, `loss_curve.png`, etc. Stage 4 reads prediction images from the `IN_CR_COM_pred` directory.

---

## 2. Test (Inference & submission)

Main script: **submission_patch.py** for inference and packaging (readme, output images, optional zip).

### Input folders (test → test_step1 → test_step2)

| Folder | Meaning |
|--------|---------|
| **test** | Raw test images (if the challenge provides unprocessed images, place them here first) |
| **test_step1** | Preprocessed input directory; **submission_patch.py defaults `--input_dir` to this**. Contains input files to run inference on (.npy or .png/.jpg, etc.) |
| **test_step2** | If you have a second-stage evaluation or pipeline, put step1 outputs (after your processing) here as the next stage input. In this repo, submission reads **test_step1** and writes results directly. |

Flow: **test (raw) → test_step1 (input to the inference script) → output to a directory or zip**. Use test_step2 only if you have an extra second step.

### Basic usage

```bash
# Default input dir test_step1, auto-detect best checkpoint in directory
python submission_patch.py --cnn_checkpoint checkpoints/<run_dir> --sliding_window

# Custom input/output and checkpoint
python submission_patch.py --input_dir /path/to/test_step1 --tmp_out_dir ./submission_outputs --cnn_checkpoint checkpoints/<run_dir>/best_PSNR_epoch_3.pth --sliding_window

# Build submission zip (readme + output images)
python submission_patch.py --input_dir /path/to/test_step1 --cnn_checkpoint checkpoints/<run_dir> --sliding_window --output_zip submission.zip
```

### Main arguments

- **--input_dir**: Input directory (default `test_step1`). Supports **.npy** and **.png/.jpg/.bmp/.tif**
- **--cnn_checkpoint**: Checkpoint directory or path to a `.pth`; use `best` or `latest` to auto-select in a directory
- **--sliding_window**: Use sliding-window inference (otherwise full-image or other modes)
- **--tmp_out_dir**: Directory for inference outputs (then packed into zip)
- **--output_zip**: If set, pack results into a zip for submission
- **--img_size**: Match training (e.g. 544,416); if omitted, read from checkpoint’s config.txt
- **--lock_l**: Compose final image with “input L + predicted AB” after inference
- **--path_remap**: Path replacement (e.g. `F:` → `I:`), useful when changing drive letters

Results are written to `--tmp_out_dir`; the generated readme includes runtime, CPU/GPU, and Other description (e.g. repo link). With `--output_zip` you get a zip ready for submission.

### 3. Check (Staged-CNN's Final Result in NTIRE 2026)
```bash
# test->test_step1
python .\submission_patch.py --cnn_checkpoint checkpoints/step1/best_loss_epoch_32.pth --sliding_window --patch_size 256 --stride 224 --resize_input 474,400 --high_ratio 0.33

# test_step1->test_step2
python .\submission_patch.py --cnn_checkpoint checkpoints/step2/best_PSNR_epoch_9.pth --sliding_window --img_size 544,416 --patch_size 544,416 --stride 512,384 --padding 32

# test_step2->submission
python .\submission_patch.py --cnn_checkpoint checkpoints/step3/best_PSNR_epoch_20.pth --sliding_window --img_size 544,416 --patch_size 544,416 --stride 512,384 --padding 32 --lock_l
```
