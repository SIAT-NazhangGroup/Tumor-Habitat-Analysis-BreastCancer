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
# 直接使用你提供的随机种子列表
# seed_list = [
#     2537, 10880, 132453, 42539, 149191, 21903, 38176, 148376, 109335, 21932,
#     85431, 172241, 102225, 297262, 266651, 48322, 41855, 20322, 165600, 22096,
#     238541, 160128, 8170, 10876, 176597, 60977, 272441, 19873, 24623, 273843,
#     284060, 41744, 158808, 49641, 234691, 11975, 126250, 129794, 79842, 168393,
#     7094, 232214, 137044, 19597, 266215, 108991, 53031, 285336, 174992, 243942,
#     140470, 57264, 1746, 23071, 192121, 191306, 287145, 101180, 52765, 180735,
#     112074, 40083, 197733, 93286, 204889, 226555, 57712, 140084, 71943, 185676,
#     279288, 190760, 45149, 73280, 259534, 87578, 42474, 60495, 108145, 20879,
#     59695, 290404, 8987, 76984, 98109, 19759, 89914, 104079, 202196, 70586,
#     25537, 240437, 59147, 75810, 129464, 143460, 293508, 30130, 158100, 125463
# ]

seed_list = [
    2537, 10880, 132453, 42539, 149191, 21903, 38176, 148376, 109335, 21932,
]

N_SEEDS = len(seed_list) # 随机种子数量，根据列表长度自动确定

TEST_SIZE = 0.3             # 测试集比例
CV_FOLDS = 5                # 交叉验证折数
RESULT_DIR = r"<PROJECT_ROOT>\eval_results-2" # 结果保存目录

# --- 模拟数据 (请替换为你的实际数据) ---
# 假设你有 X_all (特征), y_all (目标变量)
# 假设你有 x_val_data, y_val_data (外部验证数据)
# 这里创建一些随机数据作为示例

def filter_df_by_conditions(df, filter_df):
    # 合并方式：inner join（多列匹配）
    merged = df.merge(filter_df, on=['patient_name', 'label_name','grade'], how='inner')
    return merged

in_datapath = r"<PROJECT_ROOT>\radiology\washin_radiology_2025-07-10.csv"
in_data=pd.read_csv(in_datapath)

filter_df = pd.read_csv(r'<PROJECT_ROOT>\datas_clear_final.csv')  # 包含 patient_name, label_name, grade

for df in [in_data, filter_df]:
    df['patient_name'] = df['patient_name'].astype(str)
    df['label_name'] = df['label_name'].astype(str)
    df['grade'] = df['grade'].astype(int)

in_data = filter_df_by_conditions(in_data, filter_df)
X_all = in_data[in_data.columns[4:-5]]
y_all = in_data['grade']
columns_in = []
for column_name in X_all.columns[1:]:
    # print(column_name)
    print("\033[1;31;40m"+column_name+"\033[0m")
    try:
        if levene(X_all[y_all==0][column_name], X_all[y_all==1][column_name])[1] > 0.05:
            if ttest_ind(X_all[y_all==0][column_name], X_all[y_all==1][column_name],equal_var=True)[1] < 0.05:
                columns_in.append(column_name)
        else:
            if ttest_ind(X_all[y_all==0][column_name], X_all[y_all==1][column_name],equal_var=False)[1] < 0.05:
                columns_in.append(column_name)
    except Exception as ex:
        print("出现如下异常: %s"%ex)
        continue
X_all = X_all[columns_in]
print(X_all)

# 模拟外部验证数据

val_datapath = r"<NEW_ROOT>\radiology\washin_radiology_2025-07-11.csv"
val_data=pd.read_csv(val_datapath)

filter_val_df = pd.read_csv(r'<NEW_ROOT>\datas_clear_pro.csv')  # 包含 patient_name, label_name, grade

for df in [val_data, filter_val_df]:
    df['patient_name'] = df['patient_name'].astype(str)
    df['label_name'] = df['label_name'].astype(str)
    df['grade'] = df['grade'].astype(int)

val_data = filter_df_by_conditions(val_data, filter_val_df)
x_val_data = val_data[val_data.columns[4:-5]]
y_val_data = val_data['grade']

# columns_val = []
# for column_name in x_val_data.columns[1:]:
#     # print(column_name)
#     print("\033[1;31;40m"+column_name+"\033[0m")
#     try:
#         if levene(x_val_data[y_val_data==0][column_name], x_val_data[y_val_data==1][column_name])[1] > 0.05:
#             if ttest_ind(x_val_data[y_val_data==0][column_name], x_val_data[y_val_data==1][column_name],equal_var=True)[1] < 0.05:
#                 columns_val.append(column_name)
#         else:
#             if ttest_ind(x_val_data[y_val_data==0][column_name], x_val_data[y_val_data==1][column_name],equal_var=False)[1] < 0.05:
#                 columns_val.append(column_name)
#     except Exception as ex:
#         print("出现如下异常: %s"%ex)
#         continue
# x_val_data = x_val_data[columns_val]

print(f"模拟数据形状: X_all={X_all.shape}, y_all={y_all.shape}")
print(f"模拟外部验证数据形状: x_val_data={x_val_data.shape}, y_val_data={y_val_data.shape}")
print(f"将使用 {N_SEEDS} 个预设随机种子进行实验。")

# ------------------- 业务函数 -------------------

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
    # 1. 特征选择器: LassoCV 自动选择 alpha，SelectFromModel 根据系数选择特征
    # cv=5 是 LassoCV 内部的交叉验证折数
    # threshold='median' 表示选择系数绝对值大于中位数的特征。你也可以尝试 'mean' 或一个具体数值。
    alpha_range = np.logspace(-3, 0, 150, base=5)

    # --- 2. 实例化 LassoCV 模型 ---
    # 使用你的 alpha_range, cv, max_iter
    # 增加 max_iter 以解决 ConvergenceWarning
    # 传入 seed 给 random_state 确保可复现性
    # n_jobs=-1 保持并行计算
    lasso_cv_model = LassoCV(
        alphas=alpha_range,
        cv=5,
        max_iter=1000, # 从 1000 增加到 50000 或更高，以解决 ConvergenceWarning
        random_state=seed, # 传入 seed 确保每次运行的 LassoCV 可复现
        n_jobs=-1 # 保持并行处理
    )

    lasso_selector = SelectFromModel(
        estimator=lasso_cv_model,
        max_features=499 # 明确限制选择的特征数量，确保小于 500
        # threshold=None # 默认即可，或者如果你想更严格，可以尝试 'mean' 或一个固定值
    )

    # 2. TabPFN 分类器
    # device='cpu' 或 'cuda' (如果你有GPU且配置了PyTorch GPU支持)
    # N_ensemble_configurations 影响模型性能和运行时间，可以根据需要调整
    tabpfn_clf = TabPFNClassifier(device=DEVICE)

    # 构建 Pipeline:
    # 1. StandardScaler: 对数据进行标准化，这是 Lasso 和 TabPFN 通常需要的
    # 2. feature_selection: 使用 Lasso 选择特征
    # 3. tabpfn_classifier: 最终的 TabPFN 模型
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

def evaluate_one_seed(seed, x_val_data=None, y_val_data=None):
    """对单个随机种子完成划分、CV、Test、External 并返回结果列表"""

    # ---------- 1. 划分训练集和测试集 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=seed,
    )
    # 注意：StandardScaler 和特征选择现在都在 Pipeline 内部处理，无需在此处手动操作

    # ---------- 2. 5-fold CV (在训练集上) ----------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    pipeline = build_pipeline(seed) # 为每个种子构建新的 Pipeline，确保内部随机性一致

    # cross_validate 会在每个 CV 折叠中：
    # 1. 在训练子集上 fit pipeline (scaler, lasso, tabpfn)
    # 2. 在验证子集上 transform 并 predict
    cv_scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring, # 使用上面定义的 scoring 字典
        n_jobs=1,        # 保持 n_jobs=1，避免与外部 Parallel 冲突，TabPFN 内部可能也有并行
        return_estimator=False, # 不需要返回每个折叠训练的模型
    )

    # ---------- 3. 在完整训练集上再次 fit (用于测试集和外部验证) ----------
    # 这一步会用整个 X_train 重新训练 Pipeline：
    # 1. StandardScaler 在 X_train 上 fit 并 transform
    # 2. Lasso 在 X_train 上 fit 并选择特征，然后 transform
    # 3. TabPFN 在转换后的 X_train 上 fit
    pipeline.fit(X_train, y_train)

    # ---------- 4. Test 集评估 ----------
    # pipeline.predict/predict_proba 会自动对 X_test 进行缩放和特征选择
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    # ---------- 5. External 验证 ----------
    external_metrics = None
    if x_val_data is not None and y_val_data is not None:
        # pipeline.predict/predict_proba 会自动对 x_val_data 进行缩放和特征选择
        # 这些缩放和特征选择规则是在步骤3的完整训练集上学习到的
        y_ext_pred = pipeline.predict(x_val_data)
        y_ext_proba = pipeline.predict_proba(x_val_data)[:, 1]
        external_metrics = compute_metrics(y_val_data, y_ext_pred, y_ext_proba)

    # ---------- 6. 收集结果 ----------
    rows = []

    # CV 每折一行
    for fold_id in range(CV_FOLDS):
        rows.append({
            "seed": seed,
            "split_type": "CV",
            "repeat_id": 0, # 可以根据需要修改，这里设为0
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
    delayed(evaluate_one_seed)(seed, x_val_data, y_val_data)
    for seed in tqdm(seed_list, desc="Processing Seeds")
)

# ------------------- 合并 & 保存 -------------------
flat_rows = [row for sublist in all_rows for row in sublist]
df = pd.DataFrame(flat_rows)

os.makedirs(RESULT_DIR, exist_ok=True)
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
# 这里处理 MultiIndex 列名，确保第一层是 'seed' 时不加后缀
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
