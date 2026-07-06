# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
﻿import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib
matplotlib.use("Agg")

# =========================
# 参数区（按需修改）
# =========================
# 输入原始 DCE-MRI NIfTI 文件
INPUT_NII = r"<PROJECT_ROOT>\Benign_nii\31\3\3.nii"

# 输入肿瘤 mask 文件（肿瘤区域=1，其他区域=0）
MASK_NII = r"<DCM_ROOT>\benign-DCE-221subs\part1-175subs+part2\31\2-Untitled.nii.gz"

# 输出目录
OUTPUT_DIR = r"<PROJECT_ROOT>\habitat_0403\dce_gray"

# 按 figshow-1.py 的写法使用窗位/窗宽
# min_v = WL - WW/2, max_v = WL + WW/2
WINDOW_LEVEL = 1600
WINDOW_WIDTH = 3000

# 是否额外保存全部切片 PNG
SAVE_ALL_SLICE_PNG = False


def figshow_style_to_uint8(volume: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """
    按 figshow-1.py 写法做灰度映射：
    1) min_v = wl - ww/2, max_v = wl + ww/2
    2) out = volume - min_v
    3) out[out > max_v] = max_v
    4) out = out / (max_v - min_v) * 255
    """
    min_v = wl - ww / 2.0
    max_v = wl + ww / 2.0

    out = np.asarray(volume, dtype=np.float32) - float(min_v)
    out[out > max_v] = float(max_v)
    out = out / float(max_v - min_v) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def main():
    if not os.path.exists(INPUT_NII):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_NII}")
    if not os.path.exists(MASK_NII):
        raise FileNotFoundError(f"mask 文件不存在: {MASK_NII}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = nib.load(INPUT_NII)
    data = img.get_fdata()
    mask_data = nib.load(MASK_NII).get_fdata()

    if data.shape != mask_data.shape:
        raise ValueError(f"图像与 mask 尺寸不一致: image={data.shape}, mask={mask_data.shape}")

    # 按 figshow 风格窗口化
    gray_u8 = figshow_style_to_uint8(data, WINDOW_LEVEL, WINDOW_WIDTH)

    # 保存整卷灰度 NIfTI
    base = os.path.splitext(os.path.basename(INPUT_NII))[0]
    out_nii = os.path.join(OUTPUT_DIR, f"{base}_gray_wl{WINDOW_LEVEL}_ww{WINDOW_WIDTH}.nii")
    out_img = nib.Nifti1Image(gray_u8, img.affine, img.header)
    nib.save(out_img, out_nii)

    print(f"输入 shape: {data.shape}")
    print(f"窗位/窗宽: WL={WINDOW_LEVEL}, WW={WINDOW_WIDTH}")
    print(f"灰度 NIfTI 已保存: {out_nii}")

    # 选取肿瘤体素最多的切片（按 Z 方向）
    tumor_mask = (mask_data > 0)
    slice_tumor_counts = np.sum(tumor_mask, axis=(0, 1))
    max_slice_idx = int(np.argmax(slice_tumor_counts))
    max_slice_count = int(slice_tumor_counts[max_slice_idx])

    if max_slice_count == 0:
        print("警告：mask 中没有肿瘤体素，默认输出第 0 层。")
        max_slice_idx = 0

    # 与 figshow 一致，直接取 data[:, :, z] 对应切片
    best_slice = gray_u8[:, :, max_slice_idx]
    best_slice_png = os.path.join(
        OUTPUT_DIR,
        f"{base}_largest_tumor_slice_{max_slice_idx:03d}_wl{WINDOW_LEVEL}_ww{WINDOW_WIDTH}.png",
    )
    cv2.imwrite(best_slice_png, best_slice, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print(f"肿瘤最多切片索引: {max_slice_idx}, 肿瘤体素数: {max_slice_count}")
    print(f"肿瘤最多切片 PNG 已保存: {best_slice_png}")

    if SAVE_ALL_SLICE_PNG:
        slice_dir = os.path.join(OUTPUT_DIR, f"{base}_slices_wl{WINDOW_LEVEL}_ww{WINDOW_WIDTH}")
        os.makedirs(slice_dir, exist_ok=True)

        num_slices = gray_u8.shape[2]
        for z in range(num_slices):
            slice_img = gray_u8[:, :, z]
            out_png = os.path.join(slice_dir, f"slice_{z:03d}.png")
            cv2.imwrite(out_png, slice_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print(f"切片 PNG 已保存目录: {slice_dir}")
        print(f"切片数量: {num_slices}")


if __name__ == "__main__":
    main()
