import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 单病例参数区（只改这里）
# =========================
# 原始分区图（标签建议为 1/2/3）
HABITAT_FILE = r"E:\liuzhou_breastcancer\habitat_0403\nifti\Benign31_2-Untitled_new_region_by_score.nii"

# 9期 DCE 目录（结构：...\1\1.nii ... \9\9.nii）
DCE_CASE_DIR = r"E:\liuzhou_breastcancer\Benign_nii\31"

# 是否按“新分区”画曲线：
# True  -> 先用 washin/washout 计算 score 并重排分区，再统计曲线
# False -> 直接按原始分区统计曲线
USE_NEW_REGION = False

# 仅当 USE_NEW_REGION=True 时需要设置
WASHIN_FILE = r"E:\liuzhou_breastcancer\figure-res-1\washIn\Malignant53.nii"
WASHOUT_FILE = r"E:\liuzhou_breastcancer\figure-res-1\washOut\Malignant53.nii"

# 输出目录
OUTPUT_DIR = r"E:\liuzhou_breastcancer\habitat_0403\curve"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Benign31_single_tic.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Benign31_single_tic.png")


def build_new_region(habitat_data: np.ndarray, washin_data: np.ndarray, washout_data: np.ndarray):
    """根据 score=0.45*washin-0.55*washout 重排分区，返回新分区与映射。"""
    regions = [1, 2, 3]
    stats = []

    for r in regions:
        mask = (habitat_data == r)
        if np.any(mask):
            mean_washin = float(np.mean(washin_data[mask]))
            mean_washout = float(np.mean(washout_data[mask]))
            score = float(0.45 * mean_washin - 0.55 * mean_washout)
        else:
            score = np.nan

        stats.append({"old": r, "score": score})

    valid = [x for x in stats if not np.isnan(x["score"])]
    valid = sorted(valid, key=lambda x: x["score"])

    mapping = {}
    for new_label, item in enumerate(valid, start=1):
        mapping[item["old"]] = new_label

    for r in regions:
        if r not in mapping:
            mapping[r] = r

    new_region = np.array(habitat_data, copy=True)
    for old_label, new_label in mapping.items():
        if old_label != new_label:
            new_region[habitat_data == old_label] = new_label

    return new_region, mapping


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(HABITAT_FILE):
        raise FileNotFoundError(f"找不到 HABITAT_FILE: {HABITAT_FILE}")

    dce_files = [os.path.join(DCE_CASE_DIR, str(i), f"{i}.nii") for i in range(1, 10)]
    for f in dce_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到 DCE 文件: {f}")

    habitat_img = nib.load(HABITAT_FILE)
    habitat_data = habitat_img.get_fdata().astype(np.int32)

    # 选择用于统计曲线的分区图
    if USE_NEW_REGION:
        if not os.path.exists(WASHIN_FILE):
            raise FileNotFoundError(f"找不到 WASHIN_FILE: {WASHIN_FILE}")
        if not os.path.exists(WASHOUT_FILE):
            raise FileNotFoundError(f"找不到 WASHOUT_FILE: {WASHOUT_FILE}")

        washin_data = nib.load(WASHIN_FILE).get_fdata()
        washout_data = nib.load(WASHOUT_FILE).get_fdata()

        if washin_data.shape != habitat_data.shape or washout_data.shape != habitat_data.shape:
            raise ValueError(
                "washin/washout 与 habitat 尺寸不一致: "
                f"washin={washin_data.shape}, washout={washout_data.shape}, habitat={habitat_data.shape}"
            )

        region_data, mapping = build_new_region(habitat_data, washin_data, washout_data)
        print(f"使用新分区绘图，映射关系: {mapping}")
    else:
        region_data = habitat_data
        print("使用原分区绘图")

    regions = [1, 2, 3]
    rows = []

    for phase_idx, dce_file in enumerate(dce_files, start=1):
        dce_data = nib.load(dce_file).get_fdata()
        if dce_data.shape != habitat_data.shape:
            raise ValueError(
                f"第{phase_idx}期 DCE 与分区图尺寸不一致: "
                f"dce={dce_data.shape}, region={region_data.shape}"
            )

        row = {"Phase": phase_idx}
        for r in regions:
            mask = (region_data == r)
            row[f"Region_{r}"] = float(np.mean(dce_data[mask])) if np.any(mask) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # 保留你原来的修正规则
    for r in regions:
        col = f"Region_{r}"
        if df.loc[1, col] < df.loc[0, col]:
            df.loc[1, col] = df.loc[0, col] + abs(df.loc[1, col] - df.loc[0, col])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(5, 5), dpi=300)

    color_map = {1: "green", 2: "blue", 3: "red"}
    marker_map = {1: "s", 2: "^", 3: "o"}
    label_map = {1: "Subregion 1", 2: "Subregion 2", 3: "Subregion 3"}

    for r in regions:
        plt.plot(
            df["Phase"],
            df[f"Region_{r}"],
            marker=marker_map[r],
            linewidth=2,
            color=color_map[r],
            label=label_map[r],
        )

    plt.xlabel("Phase", fontsize=18)
    plt.ylabel("Mean Signal Intensity", fontsize=20)
    plt.xticks(df["Phase"], fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n===== 单病例平均信号结果 =====")
    print(df.to_string(index=False))
    print(f"CSV 已保存: {OUTPUT_CSV}")
    print(f"折线图已保存: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
