import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# This script saves 3 separate images for subregion 1/2/3.
# 1 -> green, 2 -> blue, 3 -> red.
def normalize_to_uint8(slice_data, mask):
    """Normalize masked values in one slice to 0-255 for texture display."""
    out = np.zeros(slice_data.shape, dtype=np.uint8)
    if not np.any(mask):
        return out

    values = slice_data[mask].astype(np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if vmax > vmin:
        scaled = (values - vmin) / (vmax - vmin)
    else:
        scaled = np.ones_like(values, dtype=np.float32)

    out[mask] = np.clip(np.round(scaled * 255), 0, 255).astype(np.uint8)
    return out


def build_textured_rgba(texture_slice, region_mask, color_rgb):
    """Keep region color while making in-region texture much more visible."""
    gray = normalize_to_uint8(texture_slice, region_mask)
    rgba = np.zeros((texture_slice.shape[0], texture_slice.shape[1], 4), dtype=np.uint8)

    gray_float = gray.astype(np.float32) / 255.0
    # Stretch contrast inside the region so subtle local texture is easier to see.
    if np.any(region_mask):
        region_values = gray_float[region_mask]
        low = float(np.percentile(region_values, 5))
        high = float(np.percentile(region_values, 95))
        if high > low:
            gray_float = np.clip((gray_float - low) / (high - low), 0.0, 1.0)

    gray_enhanced = np.power(gray_float, 0.65)
    texture_factor = 0.45 + 0.85 * gray_enhanced

    for channel_idx, color_value in enumerate(color_rgb):
        channel = np.clip(
            np.round(float(color_value) * texture_factor),
            0,
            255,
        ).astype(np.uint8)
        rgba[..., channel_idx][region_mask] = channel[region_mask]

    rgba[..., 3][region_mask] = 255

    return rgba


def save_rgba_image(rgba, out_file):
    """Save one RGBA image directly, preserving transparent background."""
    plt.imsave(out_file, rgba)


def read_png_rgba(png_path):
    """Read a PNG file as RGBA uint8."""
    png = plt.imread(png_path)
    if png.ndim != 3:
        raise ValueError(f"PNG must be HxWxC, got shape={png.shape}")

    if png.shape[2] == 3:
        alpha = np.ones((png.shape[0], png.shape[1], 1), dtype=png.dtype)
        png = np.concatenate([png, alpha], axis=2)
    elif png.shape[2] != 4:
        raise ValueError(f"PNG must have 3 or 4 channels, got shape={png.shape}")

    if np.issubdtype(png.dtype, np.floating):
        png = np.clip(np.round(png * 255), 0, 255).astype(np.uint8)
    else:
        png = png.astype(np.uint8)

    return png


def split_png_by_subregion(nii_path, in_png_path, out_png_path, output_dir=None):
    """Split existing in/out PNG overlays into 3 transparent PNGs using habitat subregions."""
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.int32)

    if data.ndim != 3:
        raise ValueError(f"Input NIfTI must be 3D, got shape={data.shape}")

    valid_mask = np.isin(data, [1, 2, 3])
    if not np.any(valid_mask):
        raise ValueError("No valid voxels with value 1/2/3 were found.")

    slice_counts = np.sum(valid_mask, axis=(0, 1))
    max_slice_idx = int(np.argmax(slice_counts))
    slice_data = data[:, :, max_slice_idx]

    in_png = read_png_rgba(in_png_path)
    out_png = read_png_rgba(out_png_path)

    expected_shape = slice_data.shape
    if in_png.shape[:2] != expected_shape:
        raise ValueError(f"in PNG shape mismatch: {in_png.shape[:2]} vs slice {expected_shape}")
    if out_png.shape[:2] != expected_shape:
        raise ValueError(f"out PNG shape mismatch: {out_png.shape[:2]} vs slice {expected_shape}")

    if output_dir is None:
        output_dir = os.path.dirname(in_png_path)
    os.makedirs(output_dir, exist_ok=True)

    base_in = os.path.splitext(os.path.basename(in_png_path))[0]
    base_out = os.path.splitext(os.path.basename(out_png_path))[0]

    output_files = {"in": {}, "out": {}}
    for label in [1, 2, 3]:
        region_mask = slice_data == label

        in_region = np.zeros_like(in_png)
        in_region[region_mask] = in_png[region_mask]
        in_file = os.path.join(output_dir, f"{base_in}_subregion_{label}.png")
        save_rgba_image(in_region, in_file)
        output_files["in"][label] = in_file
        print(f"Saved in PNG for subregion {label}: {in_file}")

        out_region = np.zeros_like(out_png)
        out_region[region_mask] = out_png[region_mask]
        out_file = os.path.join(output_dir, f"{base_out}_subregion_{label}.png")
        save_rgba_image(out_region, out_file)
        output_files["out"][label] = out_file
        print(f"Saved out PNG for subregion {label}: {out_file}")

    print(f"Slice index with the most valid voxels: {max_slice_idx}")
    print(f"Valid voxel count in this slice: {int(slice_counts[max_slice_idx])}")

    return max_slice_idx, output_files


def visualize_largest_valid_slice(nii_path, washin_path, washout_path, out_png=None):
    """Save textured wash-in and wash-out overlays for each subregion on the largest valid slice."""
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.int32)
    washin_data = nib.load(washin_path).get_fdata()
    washout_data = nib.load(washout_path).get_fdata()

    if data.ndim != 3:
        raise ValueError(f"Input NIfTI must be 3D, got shape={data.shape}")
    if washin_data.shape != data.shape:
        raise ValueError(f"wash-in shape mismatch: {washin_data.shape} vs habitat {data.shape}")
    if washout_data.shape != data.shape:
        raise ValueError(f"wash-out shape mismatch: {washout_data.shape} vs habitat {data.shape}")

    valid_mask = np.isin(data, [1, 2, 3])
    if not np.any(valid_mask):
        raise ValueError("No valid voxels with value 1/2/3 were found.")

    # Count valid voxels in each axial slice and keep the largest one.
    slice_counts = np.sum(valid_mask, axis=(0, 1))
    max_slice_idx = int(np.argmax(slice_counts))
    slice_data = data[:, :, max_slice_idx]
    washin_slice = washin_data[:, :, max_slice_idx]
    washout_slice = washout_data[:, :, max_slice_idx]

    if out_png is None:
        base_name = os.path.splitext(os.path.basename(nii_path))[0]
        if base_name.endswith(".nii"):
            base_name = os.path.splitext(base_name)[0]
        out_png = os.path.join(os.path.dirname(nii_path), f"{base_name}_largest_slice.png")

    out_dir = os.path.dirname(out_png)
    out_name = os.path.splitext(os.path.basename(out_png))[0]
    os.makedirs(out_dir, exist_ok=True)

    color_map = {
        1: [0, 255, 0],     # green
        2: [0, 0, 255],     # blue
        3: [255, 0, 0],     # red
    }

    output_files = {"washin": {}, "washout": {}}
    for label in [1, 2, 3]:
        region_mask = slice_data == label

        washin_rgba = build_textured_rgba(washin_slice, region_mask, color_map[label])
        washin_out_png = os.path.join(out_dir, f"{out_name}_subregion_{label}_washin_texture.png")
        save_rgba_image(washin_rgba, washin_out_png)
        output_files["washin"][label] = washin_out_png
        print(f"Saved wash-in figure for subregion {label}: {washin_out_png}")

        washout_rgba = build_textured_rgba(washout_slice, region_mask, color_map[label])
        washout_out_png = os.path.join(out_dir, f"{out_name}_subregion_{label}_washout_texture.png")
        save_rgba_image(washout_rgba, washout_out_png)
        output_files["washout"][label] = washout_out_png
        print(f"Saved wash-out figure for subregion {label}: {washout_out_png}")

    print(f"Slice index with the most valid voxels: {max_slice_idx}")
    print(f"Valid voxel count in this slice: {int(slice_counts[max_slice_idx])}")

    return max_slice_idx, output_files


if __name__ == "__main__":
    nii_path = r"E:\liuzhou_breastcancer\habitat_0403\nifti\MalignantP005_NOR_8_new_region_by_score.nii"
    in_png_path = r"E:\liuzhou_breastcancer\res_pic\result_fig\MalignantP005_NOR\8_in.png"
    out_png_path = r"E:\liuzhou_breastcancer\res_pic\result_fig\MalignantP005_NOR\8_out.png"
    output_dir = r"E:\liuzhou_breastcancer\res_pic\result_fig\MalignantP005_NOR"

    split_png_by_subregion(nii_path, in_png_path, out_png_path, output_dir)
