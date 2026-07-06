# =========================================================
# 0. 常用库
# =========================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")
# %matplotlib inline   # 在 Jupyter Notebook 中运行可取消注释


# =========================================================
# 1. 路径和参数设置
# =========================================================
in_datapath = r"E:\liuzhou_breastcancer\radiology\in_sub.csv"
out_datapath = r"E:\liuzhou_breastcancer\radiology\out_sub.csv"
filter_datapath = r"E:\liuzhou_breastcancer\datas_clear_final.csv"

RESULT_DIR = r"E:\liuzhou_breastcancer\feature_selection_results"

TEST_SIZE = 0.1

# 并行数：建议不要太大
# 如果你机器配置一般，先设为 4
# 如果还是慢，可以改成 2
N_JOBS_OUTER = 4

# LASSO 参数：这版是偏“快”的设置
LASSO_CV_FOLDS = 3
LASSO_N_ALPHAS = 50
LASSO_MAX_ITER = 10000
COEF_THRESHOLD = 0.01
# 输出文件
FEATURE_LONG_CSV = "feature_selection_long_format.csv"
FEATURE_MATRIX_CSV = "feature_selection_matrix.csv"
FEATURE_SUMMARY_CSV = "feature_selection_summary.csv"


# =========================================================
# 2. 固定的100个随机种子
# =========================================================
seed_list = [
    2537, 10880, 132453, 42539, 149191, 21903, 38176, 148376, 109335, 21932,
    85431, 172241, 102225, 297262, 266651, 48322, 41855, 20322, 165600, 22096,
    238541, 160128, 8170, 10876, 176597, 60977, 272441, 19873, 24623, 273843,
    284060, 41744, 158808, 49641, 234691, 11975, 126250, 129794, 79842, 168393,
    7094, 232214, 137044, 19597, 266215, 108991, 53031, 285336, 174992, 243942,
    140470, 57264, 1746, 23071, 192121, 191306, 287145, 101180, 52765, 180735,
    112074, 40083, 197733, 93286, 204889, 226555, 57712, 140084, 71943, 185676,
    279288, 190760, 45149, 73280, 259534, 87578, 42474, 60495, 108145, 20879,
    59695, 290404, 8987, 76984, 98109, 19759, 89914, 104079, 202196, 70586,
    25537, 240437, 59147, 75810, 129464, 143460, 293508, 30130, 158100, 125463
]


# =========================================================
# 3. 数据导入与预处理
# =========================================================

def filter_df_by_conditions(df, filter_df):
    merged = df.merge(
        filter_df[["patient_name", "label_name", "grade"]].drop_duplicates(),
        on=["patient_name", "label_name", "grade"],
        how="inner"
    )
    return merged


def load_and_prepare_data(in_datapath, out_datapath, filter_datapath):
    in_data = pd.read_csv(in_datapath)
    out_data = pd.read_csv(out_datapath)
    filter_df = pd.read_csv(filter_datapath)

    print("原始 in_data shape:", in_data.shape)
    print("原始 out_data shape:", out_data.shape)
    print("原始 filter_df shape:", filter_df.shape)

    required_merge_cols = ["patient_name", "label_name", "grade"]

    for df_name, df_ in [("in_data", in_data), ("out_data", out_data), ("filter_df", filter_df)]:
        for col in required_merge_cols:
            if col not in df_.columns:
                raise KeyError(f"{df_name} 中缺少必要列: {col}")

    # 统一类型
    for df_ in [in_data, out_data, filter_df]:
        df_["patient_name"] = df_["patient_name"].astype(str).str.strip()
        df_["label_name"] = df_["label_name"].astype(str).str.strip()
        df_["grade"] = pd.to_numeric(df_["grade"], errors="coerce")

    in_data = in_data.dropna(subset=["grade"]).copy()
    out_data = out_data.dropna(subset=["grade"]).copy()
    filter_df = filter_df.dropna(subset=["grade"]).copy()

    in_data["grade"] = in_data["grade"].astype(int)
    out_data["grade"] = out_data["grade"].astype(int)
    filter_df["grade"] = filter_df["grade"].astype(int)

    # 先分别按 filter_df 过滤
    in_data = filter_df_by_conditions(in_data, filter_df)
    out_data = filter_df_by_conditions(out_data, filter_df)

    print("筛选后 in_data shape:", in_data.shape)
    print("筛选后 out_data shape:", out_data.shape)

    # 基本信息列
    key_cols = ["patient_name", "label_name", "grade"]

    possible_meta_cols = {
        "Unnamed: 0", "index", "ID", "id",
        "patient_id", "series_id", "study_id",
        "hospital", "dataset", "cohort",
        "path", "image_path", "mask_path",
        "sex", "age", "label"
    }

    exclude_cols = set(key_cols) | {c for c in possible_meta_cols if c in in_data.columns or c in out_data.columns}

    # 分别取数值特征
    in_numeric_cols = in_data.select_dtypes(include=[np.number]).columns.tolist()
    out_numeric_cols = out_data.select_dtypes(include=[np.number]).columns.tolist()

    in_feature_cols = [c for c in in_numeric_cols if c not in exclude_cols]
    out_feature_cols = [c for c in out_numeric_cols if c not in exclude_cols]

    if len(in_feature_cols) == 0:
        raise ValueError("in_data 没有识别到可用数值特征列。")
    if len(out_feature_cols) == 0:
        raise ValueError("out_data 没有识别到可用数值特征列。")

    # 只保留键列 + 特征列
    in_feat = in_data[key_cols + in_feature_cols].copy()
    out_feat = out_data[key_cols + out_feature_cols].copy()

    # 加前缀，避免同名特征冲突
    in_feat = in_feat.rename(columns={c: f"W_in-{c}" for c in in_feature_cols})
    out_feat = out_feat.rename(columns={c: f"W_out-{c}" for c in out_feature_cols})

    # 两个表按键内连接，只保留两边都存在的样本
    merged_data = in_feat.merge(out_feat, on=key_cols, how="inner")

    print("合并后 merged_data shape:", merged_data.shape)

    # 标签分布
    grade_counts = merged_data["grade"].value_counts().sort_index()
    print("类别分布：")
    print(grade_counts)

    plt.figure(figsize=(5, 4))
    plt.bar(grade_counts.index.astype(str), grade_counts.values)
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.title("Distribution of Grades")
    plt.show()

    # 构建 x, y
    feature_cols = [c for c in merged_data.columns if c not in key_cols]
    x = merged_data[feature_cols].copy()
    y = merged_data["grade"].copy()

    print("原始特征矩阵 x shape:", x.shape)

    # 清理异常值
    x = x.dropna(axis=1, how="all")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(lambda col: col.fillna(col.median()), axis=0)

    # 去掉常数列
    nunique = x.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if len(constant_cols) > 0:
        print(f"去掉常数列 {len(constant_cols)} 个")
        x = x.drop(columns=constant_cols)

    print("清洗后 x shape:", x.shape)

    # 检查标签是否为二分类
    unique_y = sorted(pd.Series(y).dropna().unique().tolist())
    print("标签取值：", unique_y)
    if len(unique_y) != 2:
        raise ValueError(f"当前 grade 不是二分类标签，检测到取值为: {unique_y}")

    return x, y, merged_data


# =========================================================
# 4. 单次 Monte Carlo + LASSO 特征选择
# =========================================================
def evaluate_one_seed_feature_selection(seed, X_all, y_all, feature_names):
    X_train_df, _, y_train, _ = train_test_split(
        X_all,
        y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)

    lasso = Lasso(
        alpha=0.04,          # ⭐关键参数，后面我会教你怎么调
        max_iter=20000,
        tol=1e-3,
        random_state=seed
    )
    lasso.fit(X_train_scaled, y_train)

    coef = lasso.coef_.ravel()

    # 先按“非零”筛一遍，再按绝对值阈值进一步筛
    selected_mask = np.abs(coef) >= COEF_THRESHOLD

    # 兜底：如果一个特征都没选中，就保留绝对系数最大的1个
    if selected_mask.sum() == 0:
        max_idx = int(np.argmax(np.abs(coef)))
        selected_mask[max_idx] = True

    rows = []
    for feat, c, sel in zip(feature_names, coef, selected_mask):
        rows.append({
            "seed": seed,
            "feature_name": feat,
            "selected": int(sel),
            "lasso_coef": float(c),
            "lasso_alpha": float(lasso.alpha),
        })

    return rows


# =========================================================
# 5. 主函数：100次 Monte Carlo 特征统计
# =========================================================
def run_monte_carlo_lasso_feature_selection(x, y):
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name="label")

    X_all = x.copy()
    y_all = y.copy()
    feature_names = X_all.columns.tolist()

    print("X_all shape:", X_all.shape)
    print("特征数:", len(feature_names))
    print("样本数:", len(y_all))
    print(f"开始进行 {len(seed_list)} 次 Monte Carlo + LASSO 特征选择...")

    all_feature_rows = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=10)(
        delayed(evaluate_one_seed_feature_selection)(
            seed, X_all, y_all, feature_names
        )
        for seed in tqdm(seed_list, desc="Seeds")
    )

    feature_rows = [row for sublist in all_feature_rows for row in sublist]
    feature_df = pd.DataFrame(feature_rows)

    # 保存长表
    feature_long_path = os.path.join(RESULT_DIR, FEATURE_LONG_CSV)
    feature_df.to_csv(feature_long_path, index=False, encoding="utf-8")
    print(f"✅ 特征长表已保存到 {feature_long_path}")

    # 构建 0/1 选择矩阵
    seed_to_round = {seed: i + 1 for i, seed in enumerate(seed_list)}
    feature_df["round_id"] = feature_df["seed"].map(seed_to_round)

    selection_matrix = feature_df.pivot_table(
        index="feature_name",
        columns="round_id",
        values="selected",
        aggfunc="max",
        fill_value=0,
    )

    selection_matrix = selection_matrix.reindex(sorted(selection_matrix.columns), axis=1)
    selection_matrix.columns = [f"MC_{c}" for c in selection_matrix.columns]

    # 统计出现次数
    selection_matrix["selection_count"] = selection_matrix.sum(axis=1)

    # 计算被选中时的平均 LASSO 系数
    mean_coef_selected = (
        feature_df.loc[feature_df["selected"] == 1]
        .groupby("feature_name")["lasso_coef"]
        .mean()
    )
    selection_matrix["mean_lasso_coef_when_selected"] = selection_matrix.index.map(mean_coef_selected)
    selection_matrix["mean_lasso_coef_when_selected"] = selection_matrix["mean_lasso_coef_when_selected"].fillna(0.0)

    # 同时给一个平均绝对系数，便于排序
    mean_abs_coef_selected = (
        feature_df.loc[feature_df["selected"] == 1]
        .assign(abs_coef=lambda d: d["lasso_coef"].abs())
        .groupby("feature_name")["abs_coef"]
        .mean()
    )
    selection_matrix["mean_abs_lasso_coef_when_selected"] = selection_matrix.index.map(mean_abs_coef_selected)
    selection_matrix["mean_abs_lasso_coef_when_selected"] = selection_matrix["mean_abs_lasso_coef_when_selected"].fillna(0.0)

    # 排序
    selection_matrix = selection_matrix.sort_values(
        by=["selection_count", "mean_abs_lasso_coef_when_selected"],
        ascending=[False, False]
    )

    # 保存选择矩阵
    selection_matrix_path = os.path.join(RESULT_DIR, FEATURE_MATRIX_CSV)
    selection_matrix.to_csv(selection_matrix_path, encoding="utf-8")
    print(f"✅ 特征选择矩阵已保存到 {selection_matrix_path}")

    # 保存汇总表
    feature_summary = selection_matrix[[
        "selection_count",
        "mean_lasso_coef_when_selected",
        "mean_abs_lasso_coef_when_selected"
    ]].reset_index()

    feature_summary_path = os.path.join(RESULT_DIR, FEATURE_SUMMARY_CSV)
    feature_summary.to_csv(feature_summary_path, index=False, encoding="utf-8")
    print(f"✅ 特征汇总表已保存到 {feature_summary_path}")

    return {
        "feature_df": feature_df,
        "selection_matrix": selection_matrix,
        "feature_summary": feature_summary,
    }


# =========================================================
# 6. 运行
# =========================================================
x, y, merged_data = load_and_prepare_data(
    in_datapath=in_datapath,
    out_datapath=out_datapath,
    filter_datapath=filter_datapath
)

results = run_monte_carlo_lasso_feature_selection(x=x, y=y)