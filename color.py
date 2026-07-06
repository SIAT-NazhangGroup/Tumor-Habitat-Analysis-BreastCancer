import os
import gc

import numpy as np
import nibabel as nib
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt


# ===== 读取 NIfTI =====
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata()


# ===== 应用 mask =====
def apply_mask(volume, mask):
    mask_bin = (mask > 0.5).astype(np.float32)
    volume_masked = volume * mask_bin
    return volume_masked, mask_bin


# ===== 计算 2D 邻域均值 =====
def compute_2d_3x3_mean(volume, mask_bin):
    kernel = np.ones((3, 3, 1), dtype=np.float32)
    vol_masked = volume * mask_bin
    num = convolve(vol_masked, kernel, mode="constant", cval=0.0)
    den = convolve(mask_bin, kernel, mode="constant", cval=0.0)
    mean_2d = np.where(den > 0, num / den, 0.0)
    return mean_2d * mask_bin


# ===== 计算 3D 邻域均值 =====
def compute_3d_3x3x3_mean(volume, mask_bin):
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    vol_masked = volume * mask_bin
    num = convolve(vol_masked, kernel, mode="constant", cval=0.0)
    den = convolve(mask_bin, kernel, mode="constant", cval=0.0)
    mean_3d = np.where(den > 0, num / den, 0.0)
    return mean_3d * mask_bin


# ===== jet + alpha 透明图 =====
def make_jet_rgba(s_data_2d):
    normalized = cv2.normalize(s_data_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    jet = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    rgba = cv2.cvtColor(jet, cv2.COLOR_BGR2BGRA)
    rgba[s_data_2d == 0, 3] = 0  # mask 外透明
    return rgba


# ===== 颜色条 =====
def make_colorbar_image(s_data_2d, height, save_path):
    vmin, vmax = float(s_data_2d.min()), float(s_data_2d.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(1, 5))
    fig.subplots_adjust(left=0.5, right=0.6, top=0.95, bottom=0.05)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        cax=ax,
    )
    cbar.set_label("Value")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    colorbar = cv2.imread(save_path)
    h, w = colorbar.shape[:2]
    new_w = int(w * height / h)
    colorbar = cv2.resize(colorbar, (new_w, height))
    return colorbar


# ===== 保存透明 jet 图 =====
def save_transparent_jet(s_data_2d, out_dir, prefix="", name_suffix="out"):
    os.makedirs(out_dir, exist_ok=True)

    rgba = make_jet_rgba(s_data_2d)
    rgba_path = os.path.join(out_dir, f"{prefix}{name_suffix}_rgba.png")
    cv2.imwrite(rgba_path, rgba)

    colorbar_path = os.path.join(out_dir, f"{prefix}{name_suffix}_colorbar.png")
    colorbar = make_colorbar_image(s_data_2d, rgba.shape[0], colorbar_path)

    rgba_bgr = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
    combined = np.hstack((rgba_bgr, colorbar))
    combined_path = os.path.join(out_dir, f"{prefix}{name_suffix}_combined.png")
    cv2.imwrite(combined_path, combined)

    del rgba, colorbar, combined
    gc.collect()
    print("✅ Saved:", combined_path)

# ===== 找出肿瘤像素最多的层 =====
def find_max_mask_slice(mask_bin):
    # 计算每层的 mask 像素数
    pixel_counts = [np.sum(mask_bin[:, :, z]) for z in range(mask_bin.shape[2])]
    z_max = int(np.argmax(pixel_counts))
    print(f"🧠 肿瘤像素最多的层: z={z_max} (mask体素数={pixel_counts[z_max]:.0f})")
    return z_max


# ===== 主流程 =====
if __name__ == "__main__":
    image_path = r"E:\liuzhou_breastcancer\figure-res-1\washOut\MalignantP005_NOR.nii"  # 原始 MRI
    mask_path = r"E:\DCESummary_2019-202004\malignant-DCE-451subs\part1-288subs+part2\P005_NOR\8.nii"    # 对应 mask
    out_root = "./res_pic"
    volume = load_nifti(image_path)
    mask = load_nifti(mask_path)
    volume_masked, mask_bin = apply_mask(volume, mask)

     # 自动找出肿瘤像素最多的那层
    z = find_max_mask_slice(mask_bin)

    mean_2d = compute_2d_3x3_mean(volume_masked, mask_bin)
    mean_3d = compute_3d_3x3x3_mean(volume_masked, mask_bin)

    out_dir = os.path.join(out_root, f"slice_{z}")

    # 原始值 jet 透明图
    save_transparent_jet(volume_masked[:, :, z], out_dir, name_suffix="ori_out")
    # 2D 均值 jet 透明图
    save_transparent_jet(mean_2d[:, :, z], out_dir, name_suffix="2dmean_out")
    # 3D 均值 jet 透明图
    save_transparent_jet(mean_3d[:, :, z], out_dir, name_suffix="3dmean_out")

# ======================
# 6. 主流程示例
# ======================
# if __name__ == "__main__":
#     # 1）修改成你自己的文件路径
#     image_path = r"E:\liuzhou_breastcancer\figure-res-1\washIn\MalignantP005_NOR.nii"  # 原始 MRI
#     mask_path = r"E:\DCESummary_2019-202004\malignant-DCE-451subs\part1-288subs+part2\P005_NOR\8.nii"    # 对应 mask

#     # 2）读取影像和 mask
#     volume = load_nifti(image_path)  # shape: (X, Y, Z)
#     mask = load_nifti(mask_path)     # shape 必须和 volume 一致

#     # 3）应用 mask
#     volume_masked, mask_bin = apply_mask(volume, mask)

#     # 4）绘制原始（被 mask 限制后的）伪彩图
#     plot_middle_slice(volume_masked, "Original volume (masked) - middle slice")

#     # 5）计算每个像素在“水平所在层” 3×3 邻域的平均值（2D）
#     mean_2d = compute_2d_3x3_mean(volume_masked, mask_bin)
#     plot_middle_slice(mean_2d, "2D 3x3 neighborhood mean (within slice)")

#     # 6）计算每个像素在“三维 3×3×3 邻域”的平均值（3D）
#     mean_3d = compute_3d_3x3x3_mean(volume_masked, mask_bin)
#     plot_middle_slice(mean_3d, "3D 3x3x3 neighborhood mean (across slices)")
