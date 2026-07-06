# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
from joblib import Parallel, delayed
from scipy.stats import levene, ttest_ind
from tqdm import tqdm
import torch 

# 确保 TabPFN 已安装：pip install tabpfn
from tabpfn import TabPFNClassifier

if torch.cuda.is_available():
    DEVICE = 'cuda'
    print("🎉 GPU (CUDA) is available and will be used for TabPFN!")
else:
    DEVICE = 'cpu'
    print("⚠️ GPU (CUDA) is not available. TabPFN will run on CPU.")

# --- 全局配置 ---
seed_list = [
    2537, 10880, 132453, 42539, 149191, 21903, 38176, 148376, 109335, 21932,
]
N_SEEDS = len(seed_list) # 随机种子数量，根据列表长度自动确定

TEST_SIZE = 0.3             # 测试集比例
CV_FOLDS = 5                # 交叉验证折数
RESULT_DIR = r"<PROJECT_ROOT>\eval_results-2" # 结果保存目录

# 确保结果保存目录存在
os.makedirs(RESULT_DIR, exist_ok=True)

# --- 数据加载和预处理 (修正部分) ---

# 定义合并函数：根据 patient_name, label_name, grade 进行内连接
def filter_df_by_conditions(df, filter_df):
    merged = df.merge(filter_df, on=['patient_name', 'label_name', 'grade'], how='inner')
    return merged

# 定义目标变量列名 (请根据你的实际情况修改)
TARGET_COLUMN = 'grade' 

# --- 1. 加载原始数据 ---
print("--- 正在加载和初步处理数据 ---")
in_datapath = r"<PROJECT_ROOT>\radiology\washin_radiology_2025-07-10.csv"
in_data = pd.read_csv(in_datapath)
filter_df = pd.read_csv(r'<PROJECT_ROOT>\datas_clear_final.csv')

val_datapath = r"<NEW_ROOT>\radiology\washin_radiology_2025-07-11.csv"
val_data = pd.read_csv(val_datapath)
filter_val_df = pd.read_csv(r'<NEW_ROOT>\datas_clear_pro.csv')

# 统一进行关键列的类型转换，确保合并和后续处理的正确性
for df_obj in [in_data, filter_df, val_data, filter_val_df]:
    if 'patient_name' in df_obj.columns:
        df_obj['patient_name'] = df_obj['patient_name'].astype(str)
    if 'label_name' in df_obj.columns:
        df_obj['label_name'] = df_obj['label_name'].astype(str)
    if TARGET_COLUMN in df_obj.columns:
        df_obj[TARGET_COLUMN] = df_obj[TARGET_COLUMN].astype(int)

# 执行合并操作
in_data = filter_df_by_conditions(in_data, filter_df)
val_data = filter_df_by_conditions(val_data, filter_val_df)

# 定义 ID 列 (不应作为特征)
ID_COLUMNS = ['patient_name', 'label_name'] 

# --- 2. 识别并排除所有非特征列 (ID, 目标变量, diagnostics_等) ---
# 从所有可用列中查找，以确保覆盖所有可能的 diagnostics 列
all_cols_in_data = set(in_data.columns)
all_cols_val_data = set(val_data.columns)
all_possible_diagnostics_cols = [col for col in (all_cols_in_data | all_cols_val_data) if col.startswith('diagnostics_')]

# 构建最终的排除列列表
EXCLUDE_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + list(set(all_possible_diagnostics_cols))
EXCLUDE_COLUMNS = list(set(EXCLUDE_COLUMNS)) # 去重

print(f"将从特征集中排除以下列: {EXCLUDE_COLUMNS}")

# --- 3. 确定初始的共同候选特征集 ---
# 从 in_data 和 val_data 中选择所有不是排除列的列作为候选特征
candidate_features_in = [col for col in in_data.columns if col not in EXCLUDE_COLUMNS]
candidate_features_val = [col for col in val_data.columns if col not in EXCLUDE_COLUMNS]

# 取两者的交集，确保两个数据集从一开始就处理相同的特征列
common_candidate_features = list(set(candidate_features_in) & set(candidate_features_val))

if not common_candidate_features:
    raise ValueError("错误：在排除ID、目标和diagnostics列后，训练集和验证集没有共同特征。请检查数据。")

# 提取初始的 X_all_raw 和 x_val_data_raw
X_all_raw = in_data[common_candidate_features].copy()
y_all = in_data[TARGET_COLUMN].copy()

x_val_data_raw = val_data[common_candidate_features].copy()
y_val_data = val_data[TARGET_COLUMN].copy()


# --- 4. 统一处理数据类型和缺失值 ---
# 定义一个函数来处理特征数据框：转换类型，删除非数值列，填充缺失值
def preprocess_features_df(df_features, name=""):
    # 尝试将对象类型转换为数值型，无法转换的变为 NaN
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        elif df_features[col].dtype == 'bool':
            df_features[col] = df_features[col].astype(int)
    
    # 显式删除仍然是非数值型的列 (例如，完全由非数字字符串组成的列)
    initial_cols_count = df_features.shape[1]
    df_features = df_features.select_dtypes(include=np.number)
    if df_features.shape[1] < initial_cols_count:
        dropped_cols = list(set(df_features.columns) ^ set(df_features.select_dtypes(include=np.number).columns))
        print(f"警告：从 {name} 中删除了非数值型列: {dropped_cols}")

    # 填充缺失值 (使用中位数，对异常值更鲁棒)
    for col in df_features.columns:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col].fillna(median_val, inplace=True)
    return df_features

print("\n--- 正在统一处理特征数据类型和缺失值 ---")
X_all_raw = preprocess_features_df(X_all_raw, name="X_all_raw")
x_val_data_raw = preprocess_features_df(x_val_data_raw, name="x_val_data_raw")

# 再次确保两个数据框在类型转换后仍然具有完全相同的列集
final_common_numeric_features = list(set(X_all_raw.columns) & set(x_val_data_raw.columns))
if not final_common_numeric_features:
    raise ValueError("错误：在数据类型转换和缺失值处理后，训练集和验证集没有共同的数值特征。请检查数据。")

X_all_raw = X_all_raw[final_common_numeric_features].copy()
x_val_data_raw = x_val_data_raw[final_common_numeric_features].copy()


# --- 5. 统计特征选择 (只在 X_all_raw 上进行，然后将结果应用于 x_val_data_raw) ---
columns_to_keep_after_stats = []
print("\n--- 正在进行统计特征选择 (基于训练集 X_all_raw) ---")
for column_name in tqdm(X_all_raw.columns, desc="Statistical Feature Selection"):
    try:
        group0 = X_all_raw[y_all == 0][column_name].dropna()
        group1 = X_all_raw[y_all == 1][column_name].dropna()

        # 确保每个组有足够的样本进行统计检验
        if len(group0) < 2 or len(group1) < 2:
            continue

        # Levene's test for equality of variances
        stat_levene, p_levene = levene(group0, group1)
        
        if p_levene > 0.05: # 方差相等，使用等方差T检验
            stat_ttest, p_ttest = ttest_ind(group0, group1, equal_var=True)
        else: # 方差不相等，使用不等方差T检验 (Welch's t-test)
            stat_ttest, p_ttest = ttest_ind(group0, group1, equal_var=False)
        
        if p_ttest < 0.05: # 如果p值小于0.05，则认为有显著差异，保留该特征
            columns_to_keep_after_stats.append(column_name)
    except Exception as ex:
        # print(f"警告：列 '{column_name}' 在统计检验中出现异常，已跳过: {ex}")
        continue

if not columns_to_keep_after_stats:
    raise ValueError("错误：统计特征选择后没有留下任何特征。请检查你的数据和统计筛选条件。")

# --- 6. 应用最终选择的特征集到 X_all 和 x_val_data ---
X_all = X_all_raw[columns_to_keep_after_stats].copy()
x_val_data = x_val_data_raw[columns_to_keep_after_stats].copy()

# --- 7. 最终验证 ---
print(f"\n--- 最终数据形状 ---")
print(f"X_all={X_all.shape}, y_all={y_all.shape}")
print(f"x_val_data={x_val_data.shape}, y_val_data={y_val_data.shape}")
print(f"X_all 中的特征数量: {X_all.shape[1]}")
print(f"x_val_data 中的特征数量: {x_val_data.shape[1]}")

if X_all.shape[1] == 0:
    raise ValueError("错误：X_all 在所有预处理后没有留下任何特征。请检查你的数据和筛选逻辑。")
if x_val_data.shape[1] == 0:
    raise ValueError("错误：x_val_data 在所有预处理后没有留下任何特征。请检查你的数据和筛选逻辑。")
if X_all.shape[1] != x_val_data.shape[1]:
    raise ValueError(f"错误：X_all ({X_all.shape[1]} features) 和 x_val_data ({x_val_data.shape[1]} features) 的特征数量不匹配！")
if not X_all.columns.equals(x_val_data.columns):
    raise ValueError("错误：X_all 和 x_val_data 的特征列名不匹配！")

print("\n--- 最终 X_all 列名列表 ---")
print(X_all.columns.tolist())
print("\n--- 最终 x_val_data 列名列表 ---")
print(x_val_data.columns.tolist())

print(f"将使用 {N_SEEDS} 个预设随机种子进行实验。")

# ------------------- 业务函数 (保持不变，或微调) -------------------

# 定义一个自定义的 specificity scorer，因为 sklearn 没有直接的 'specificity' 字符串
def specificity_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# 定义 cross_validate 使用的 scoring 字典
# 注意：这些键名必须与 compute_metrics 返回的键名匹配
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall', # recall 是敏感度
    'specificity': make_scorer(specificity_scorer),
    'precision': 'precision',
    'f1': 'f1',
    'auc': 'roc_auc'
}

def build_pipeline(seed):
    """
    构建包含 StandardScaler, Lasso特征选择 和 TabPFNClassifier 的 Pipeline。
    LassoCV 会在内部进行交叉验证来选择最佳的 alpha。
    SelectFromModel 会根据 LassoCV 的结果选择特征。
    """
    alpha_range = np.logspace(-3, 0, 150, base=5)

    lasso_cv_model = LassoCV(
        alphas=alpha_range,
        cv=5,
        max_iter=50000, # 增加 max_iter 以解决 ConvergenceWarning
        random_state=seed, # 传入 seed 确保每次运行的 LassoCV 可复现
        n_jobs=-1 # 保持并行处理
    )

    lasso_selector = SelectFromModel(
        estimator=lasso_cv_model,
        max_features=499 # 明确限制选择的特征数量，确保小于 500
    )

    tabpfn_clf = TabPFNClassifier(device=DEVICE)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", lasso_selector),
        ("tabpfn_classifier", tabpfn_clf)
    ])

def compute_metrics(y_true, y_pred, y_proba):
    """计算你需要的所有指标，返回 dict"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),   # 也叫召回率
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }

def evaluate_one_seed(seed, x_val_data_param, y_val_data_param): # 修改了参数名以避免与全局变量混淆
    """对单个随机种子完成划分、CV、Test、External 并返回结果列表"""

    # ---------- 1. 划分训练集和测试集 ----------
    # X_all 和 y_all 是经过全局预处理后的数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, 
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=seed,
    )

    # ---------- 2. 5-fold CV (在训练集上) ----------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    pipeline = build_pipeline(seed) # 为每个种子构建新的 Pipeline，确保内部随机性一致

    cv_scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,        # 保持 n_jobs=1，避免与外部 Parallel 冲突，TabPFN 内部可能也有并行
        return_estimator=False,
    )

    # ---------- 3. 在完整训练集上再次 fit (用于测试集和外部验证) ----------
    pipeline.fit(X_train, y_train)

    # ---------- 4. Test 集评估 ----------
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    # ---------- 5. External 验证 ----------
    external_metrics = None
    if x_val_data_param is not None and y_val_data_param is not None:
        y_ext_pred = pipeline.predict(x_val_data_param)
        y_ext_proba = pipeline.predict_proba(x_val_data_param)[:, 1]
        external_metrics = compute_metrics(y_val_data_param, y_ext_pred, y_ext_proba)

    # ---------- 6. 收集结果 ----------
    rows = []

    # CV 每折一行
    for fold_id in range(CV_FOLDS):
        rows.append({
            "seed": seed,
            "split_type": "CV",
            "repeat_id": 0,
            "fold_id": fold_id,
            "accuracy": cv_scores["test_accuracy"][fold_id],
            "sensitivity": cv_scores["test_sensitivity"][fold_id],
            "specificity": cv_scores["test_specificity"][fold_id],
            "precision": cv_scores["test_precision"][fold_id],
            "f1": cv_scores["test_f1"][fold_id],
            "auc": cv_scores["test_auc"][fold_id],
        })

    # Test 一行
    rows.append({
        "seed": seed,
        "split_type": "TEST",
        "repeat_id": np.nan,
        "fold_id": np.nan,
        **test_metrics,
    })

    # External 一行（若有）
    if external_metrics is not None:
        rows.append({
            "seed": seed,
            "split_type": "EXTERNAL",
            "repeat_id": np.nan,
            "fold_id": np.nan,
            **external_metrics,
        })

    return rows

# ------------------- 并行执行 -------------------
print(f"开始并行处理 {N_SEEDS} 个预设随机种子...")
all_rows = Parallel(n_jobs=-1, backend="loky", verbose=5)(
    delayed(evaluate_one_seed)(seed, x_val_data, y_val_data) # 将全局预处理后的 x_val_data 和 y_val_data 传递给函数
    for seed in tqdm(seed_list, desc="Processing Seeds")
)

# ------------------- 合并 & 保存 -------------------
flat_rows = [row for sublist in all_rows for row in sublist]
df = pd.DataFrame(flat_rows)

combined_path = os.path.join(RESULT_DIR, "combined_tabpfn_lasso.csv")
df.to_csv(combined_path, index=False, encoding="utf-8")
print(f"✅ 结果已保存到 {combined_path}")

# 打印一些汇总统计 (可选)
print("\n--- 汇总统计 (TEST集) ---")
test_df = df[df['split_type'] == 'TEST']
print(test_df.describe())

print("\n--- 汇总统计 (EXTERNAL集) ---")
external_df = df[df['split_type'] == 'EXTERNAL']
print(external_df.describe())

# ------------------- 后处理：生成 CV 汇总和整体拼接文件 -------------------
print("\n--- 开始生成 CV 汇总和整体拼接文件 ---")

# 定义需要统计的指标（与 compute_metrics 和 scoring 中保持一致）
metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]

# ① 依据 split_type == "CV" 取出所有 CV 折的记录
cv_df = df[df["split_type"] == "CV"].copy()

# ② 按 seed 分组，求 mean 与 std
cv_summary = (
    cv_df.groupby("seed")[metrics]
    .agg(["mean", "std"])                     # 生成 MultiIndex 列
    .reset_index()
)

# ③ 把 MultiIndex 的列名展平成 “metric_mean / metric_std”
cv_summary.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in cv_summary.columns.values
]

# ④ 保存到 CSV
cv_summary_path = os.path.join(RESULT_DIR, f"tabpfn_lasso_gpu_cv_summary_{N_SEEDS}_seeds.csv") # 动态文件名
cv_summary.to_csv(cv_summary_path, index=False, encoding="utf-8")
print(f"✅ {CV_FOLDS}-折 CV 均值/标准差已保存到 {cv_summary_path}")

# -------------------------------------------------
# 为了拼接，先把 test / external 的列名加上后缀，防止冲突
# ① Test
test_df_clean = (
    df[df["split_type"] == "TEST"] # 从原始df中获取，确保一致性
    .drop(columns=["repeat_id", "fold_id", "split_type"]) # 移除不必要的列
    .rename(columns=lambda c: f"{c}_test" if c not in ["seed"] else c)
)

# ② External（若提供）
ext_df_clean = None
if x_val_data is not None and y_val_data is not None:
    ext_df_clean = (
        df[df["split_type"] == "EXTERNAL"] # 从原始df中获取
        .drop(columns=["repeat_id", "fold_id", "split_type"]) # 移除不必要的列
        .rename(columns=lambda c: f"{c}_external" if c not in ["seed"] else c)
    )

# -------------------------------------------------
# ③ 合并：先左连接 CV summary → Test → External
combined_df_final = cv_summary.merge(test_df_clean, on="seed", how="left")

if ext_df_clean is not None:
    combined_df_final = combined_df_final.merge(ext_df_clean, on="seed", how="left")

# -------------------------------------------------
# ④ 保存最终的拼接文件
combined_path_final = os.path.join(RESULT_DIR, f"tabpfn_lasso_gpu_all_combined_{N_SEEDS}_seeds.csv") # 动态文件名
combined_df_final.to_csv(combined_path_final, index=False, encoding="utf-8")
print(f"✅ 所有信息已按列拼接并保存到 {combined_path_final}")

print("\n--- 所有任务完成！ ---")
