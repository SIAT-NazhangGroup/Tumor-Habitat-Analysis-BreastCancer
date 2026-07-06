# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
﻿import os
import re
import traceback
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. 全局路径配置
# =========================
# 样本清单
meta_csv = r"<PROJECT_ROOT>\datas_clear_final.csv"

# 原始分区图目录（每个样本一个分区图）
# 分区图命名规则：{Benign|Malignant}{patient_name}_{label_name去后缀}.nii
habitat_root = r"<PROJECT_ROOT>\habitat_test"

# wash-in / wash-out 参数图根目录
wash_root_benign = r"<PROJECT_ROOT>\figure-res-0"
wash_root_malignant = r"<PROJECT_ROOT>\figure-res-1"

# 总输出目录（按你的要求固定为 habitat_0403）
out_root = r"<PROJECT_ROOT>\habitat_0403"
out_csv_dir = os.path.join(out_root, "csv")
out_curve_dir = os.path.join(out_root, "curve")
out_nifti_dir = os.path.join(out_root, "nifti")

os.makedirs(out_csv_dir, exist_ok=True)
os.makedirs(out_curve_dir, exist_ok=True)
os.makedirs(out_nifti_dir, exist_ok=True)


# =========================
# 2. 通用工具函数
# =========================
def strip_nii_suffix(name: str) -> str:
    """去掉 .nii 或 .nii.gz 后缀，得到基础名。"""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def safe_name(s: str) -> str:
    """把文件名中的非法字符替换为下划线，避免保存失败。"""
    return re.sub(r'[\\/:*?"<>|]', '_', str(s))


def resolve_case_paths(row: pd.Series):
    """根据一行元数据，拼出该样本所有输入/输出路径。"""
    patient_name = str(row["patient_name"]).strip()
    label_name = str(row["label_name"]).strip()
    grade = int(row["grade"])
    nii_data = str(row["nii_data"]).strip()

    # grade=0 为良性，其他为恶性
    is_benign = (grade == 0)
    prefix = "Benign" if is_benign else "Malignant"
    wash_root = wash_root_benign if is_benign else wash_root_malignant

    label_base = strip_nii_suffix(label_name)
    case_tag = f"{prefix}{patient_name}_{label_base}"
    case_tag_safe = safe_name(case_tag)

    # 输入路径
    habitat_file = os.path.join(habitat_root, f"{case_tag}.nii")
    washin_file = os.path.join(wash_root, "washIn", f"{prefix}{patient_name}.nii")
    washout_file = os.path.join(wash_root, "washOut", f"{prefix}{patient_name}.nii")
    dce_files = [os.path.join(nii_data, str(i), f"{i}.nii") for i in range(1, 10)]

    # 输出路径
    out_tic_csv = os.path.join(out_csv_dir, f"{case_tag_safe}_tic.csv")
    out_map_csv = os.path.join(out_csv_dir, f"{case_tag_safe}_mapping.csv")
    out_tic_png = os.path.join(out_curve_dir, f"{case_tag_safe}_tic.png")
    # 当前流程中的“最大切片新分区图”也存到 curve 目录
    out_slice_png = os.path.join(out_curve_dir, f"{case_tag_safe}_new_region_largest_slice.png")
    out_new_nifti = os.path.join(out_nifti_dir, f"{case_tag_safe}_new_region_by_score.nii")

    return {
        "patient_name": patient_name,
        "label_name": label_name,
        "grade": grade,
        "prefix": prefix,
        "case_tag": case_tag,
        "case_tag_safe": case_tag_safe,
        "habitat_file": habitat_file,
        "washin_file": washin_file,
        "washout_file": washout_file,
        "dce_files": dce_files,
        "out_tic_csv": out_tic_csv,
        "out_map_csv": out_map_csv,
        "out_tic_png": out_tic_png,
        "out_slice_png": out_slice_png,
        "out_new_nifti": out_new_nifti,
    }


def check_inputs(paths: dict):
    """检查输入文件是否齐全，返回缺失项列表。"""
    missing = []
    for k in ["habitat_file", "washin_file", "washout_file"]:
        if not os.path.exists(paths[k]):
            missing.append(k)
    for i, f in enumerate(paths["dce_files"], start=1):
        if not os.path.exists(f):
            missing.append(f"dce_phase_{i}")
    return missing


# =========================
# 3. 单病例完整流程
# =========================
def process_one_case(paths: dict):
    """对单个肿瘤完成：新分区计算、新分区最大切片可视化、按新分区画 TIC。"""
    regions = [1, 2, 3]

    # 读取输入图像
    habitat_img = nib.load(paths["habitat_file"])
    habitat_data = habitat_img.get_fdata().astype(np.int32)
    washin_data = nib.load(paths["washin_file"]).get_fdata()
    washout_data = nib.load(paths["washout_file"]).get_fdata()

    # 尺寸一致性检查
    if washin_data.shape != habitat_data.shape:
        raise ValueError(
            f"wash-in 尺寸不一致: {washin_data.shape} vs habitat {habitat_data.shape}"
        )
    if washout_data.shape != habitat_data.shape:
        raise ValueError(
            f"wash-out 尺寸不一致: {washout_data.shape} vs habitat {habitat_data.shape}"
        )

    # 计算原分区统计与 score
    # score = 0.45 * washin - 0.55 * washout
    region_stats = []
    for r in regions:
        mask = (habitat_data == r)
        voxel_num = int(np.sum(mask))

        if voxel_num == 0:
            mean_washin = np.nan
            mean_washout = np.nan
            score = np.nan
        else:
            mean_washin = float(np.mean(washin_data[mask]))
            mean_washout = float(np.mean(washout_data[mask]))
            score = float(0.45 * mean_washin - 0.55 * mean_washout)

        region_stats.append({
            "original_region": r,
            "voxel_count": voxel_num,
            "mean_washin": mean_washin,
            "mean_washout": mean_washout,
            "score": score,
        })

    # 按 score 升序映射到新分区 1/2/3
    valid_stats = [x for x in region_stats if not np.isnan(x["score"])]
    valid_stats_sorted = sorted(valid_stats, key=lambda x: x["score"])

    mapping = {}
    for new_label, item in enumerate(valid_stats_sorted, start=1):
        mapping[item["original_region"]] = new_label

    # 缺失分区兜底
    for r in regions:
        if r not in mapping:
            mapping[r] = r

    # 生成新分区体素图
    new_region_data = np.array(habitat_data, copy=True)
    for old_label, new_label in mapping.items():
        if old_label != new_label:
            new_region_data[habitat_data == old_label] = new_label

    # 判断是否发生变化
    valid_mask = np.isin(habitat_data, regions)
    changed_mask = valid_mask & (new_region_data != habitat_data)
    changed_voxels = int(np.sum(changed_mask))
    all_valid_voxels = int(np.sum(valid_mask))
    changed_ratio = (changed_voxels / all_valid_voxels * 100.0) if all_valid_voxels > 0 else 0.0
    has_change = changed_voxels > 0

    # 仅在有变化时保存新 nifti
    if has_change:
        new_region_img = nib.Nifti1Image(
            new_region_data.astype(np.int16),
            habitat_img.affine,
            habitat_img.header,
        )
        nib.save(new_region_img, paths["out_new_nifti"])

    # 保存映射与统计 CSV
    map_rows = []
    for r in regions:
        row_stat = next((x for x in region_stats if x["original_region"] == r), None)
        map_rows.append({
            "case_tag": paths["case_tag"],
            "original_region": r,
            "new_region": mapping[r],
            "is_changed": int(mapping[r] != r),
            "voxel_count": 0 if row_stat is None else row_stat["voxel_count"],
            "mean_washin": np.nan if row_stat is None else row_stat["mean_washin"],
            "mean_washout": np.nan if row_stat is None else row_stat["mean_washout"],
            "score": np.nan if row_stat is None else row_stat["score"],
        })
    pd.DataFrame(map_rows).to_csv(paths["out_map_csv"], index=False, encoding="utf-8-sig")

    # 新分区最大切片可视化（1绿2蓝3红，其余透明）
    new_valid_mask = np.isin(new_region_data, regions)
    slice_voxel_counts = np.sum(new_valid_mask, axis=(0, 1))
    max_slice_idx = int(np.argmax(slice_voxel_counts))
    slice_data = new_region_data[:, :, max_slice_idx]

    rgba = np.zeros((slice_data.shape[0], slice_data.shape[1], 4), dtype=np.uint8)

    mask1 = (slice_data == 1)
    rgba[mask1, 1] = 255
    rgba[mask1, 3] = 255

    mask2 = (slice_data == 2)
    rgba[mask2, 2] = 255
    rgba[mask2, 3] = 255

    mask3 = (slice_data == 3)
    rgba[mask3, 0] = 255
    rgba[mask3, 3] = 255

    plt.figure(figsize=(6, 6))
    plt.imshow(rgba)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(paths["out_slice_png"], dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 计算 9 期 DCE 在“新分区”中的均值并画折线图
    tic_rows = []
    for phase_idx, dce_file in enumerate(paths["dce_files"], start=1):
        dce_data = nib.load(dce_file).get_fdata()
        if dce_data.shape != habitat_data.shape:
            raise ValueError(
                f"{paths['case_tag']} 第{phase_idx}期 DCE 尺寸不一致: {dce_data.shape} vs {habitat_data.shape}"
            )

        row = {"Phase": phase_idx}
        for r in regions:
            mask = (new_region_data == r)
            row[f"Region_{r}"] = float(np.mean(dce_data[mask])) if np.any(mask) else np.nan
        tic_rows.append(row)

    tic_df = pd.DataFrame(tic_rows)

    # 保留你原先的修正规则：若第2期小于第1期，则按差值修正
    for r in regions:
        col = f"Region_{r}"
        if tic_df.loc[1, col] < tic_df.loc[0, col]:
            tic_df.loc[1, col] = tic_df.loc[0, col] + abs(tic_df.loc[1, col] - tic_df.loc[0, col])

    tic_df.to_csv(paths["out_tic_csv"], index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 4))
    color_map = {1: "green", 2: "blue", 3: "red"}
    label_map = {1: "Subregion 1", 2: "Subregion 2", 3: "Subregion 3"}
    marker_map = {1: "s", 2: "^", 3: "o"}  # 方形、三角、圆点

    for r in regions:
        plt.plot(
            tic_df["Phase"],
            tic_df[f"Region_{r}"],
            marker=marker_map[r],
            linewidth=2,
            color=color_map[r],
            label=label_map[r],
        )

    plt.xlabel("Phase", fontsize=18)
    plt.ylabel("Mean Signal Intensity", fontsize=20)
    plt.xticks(tic_df["Phase"], fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(paths["out_tic_png"], dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "case_tag": paths["case_tag"],
        "status": "ok",
        "changed_voxels": changed_voxels,
        "valid_voxels": all_valid_voxels,
        "changed_ratio": changed_ratio,
        "nifti_saved": int(has_change),
        "tic_csv": paths["out_tic_csv"],
        "tic_png": paths["out_tic_png"],
        "slice_png": paths["out_slice_png"],
        "mapping_csv": paths["out_map_csv"],
        "new_nifti": paths["out_new_nifti"] if has_change else "",
        "error": "",
    }


# =========================
# 4. 主流程：遍历 datas_clear_final.csv 所有肿瘤
# =========================
def main():
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"找不到元数据 CSV: {meta_csv}")

    df = pd.read_csv(meta_csv)
    required_cols = ["patient_name", "label_name", "grade", "nii_data"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"CSV 缺少必要列: {c}")

    print(f"总样本数: {len(df)}")

    logs = []
    for idx, row in df.iterrows():
        try:
            paths = resolve_case_paths(row)
            missing = check_inputs(paths)
            if missing:
                logs.append({
                    "case_tag": paths["case_tag"],
                    "status": "missing_input",
                    "changed_voxels": 0,
                    "valid_voxels": 0,
                    "changed_ratio": 0.0,
                    "nifti_saved": 0,
                    "tic_csv": "",
                    "tic_png": "",
                    "slice_png": "",
                    "mapping_csv": "",
                    "new_nifti": "",
                    "error": ";".join(missing),
                })
                print(f"[{idx+1}/{len(df)}] {paths['case_tag']} -> 跳过，缺少输入: {missing}")
                continue

            out = process_one_case(paths)
            logs.append(out)
            print(
                f"[{idx+1}/{len(df)}] {paths['case_tag']} -> 完成, "
                f"changed={out['changed_voxels']}/{out['valid_voxels']}, "
                f"nifti_saved={out['nifti_saved']}"
            )

        except Exception as e:
            case_tag = f"row_{idx}"
            try:
                case_tag = resolve_case_paths(row)["case_tag"]
            except Exception:
                pass
            logs.append({
                "case_tag": case_tag,
                "status": "error",
                "changed_voxels": 0,
                "valid_voxels": 0,
                "changed_ratio": 0.0,
                "nifti_saved": 0,
                "tic_csv": "",
                "tic_png": "",
                "slice_png": "",
                "mapping_csv": "",
                "new_nifti": "",
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"[{idx+1}/{len(df)}] {case_tag} -> 失败: {type(e).__name__}: {e}")

    log_df = pd.DataFrame(logs)
    summary_csv = os.path.join(out_csv_dir, "batch_process_summary.csv")
    log_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== 批处理结束 =====")
    print(f"成功: {(log_df['status'] == 'ok').sum()} | 缺输入: {(log_df['status'] == 'missing_input').sum()} | 错误: {(log_df['status'] == 'error').sum()}")
    print(f"汇总日志: {summary_csv}")


if __name__ == "__main__":
    main()
