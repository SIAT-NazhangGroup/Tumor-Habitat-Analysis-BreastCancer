import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_rel, wilcoxon

# ========= 手动指定7个CSV文件路径 =========
csv_paths = {
    "in": r"E:\liuzhou_breastcancer\eval_results-2\in_top100_seeds_all_combined.csv",
    "out": r"E:\liuzhou_breastcancer\eval_results-2\out_top100_seeds_all_combined.csv",
    "in_out": r"E:\liuzhou_breastcancer\eval_results-2\in+out_5000_seeds_all_combined.csv",
    "habitat_in": r"E:\liuzhou_breastcancer\eval_results-2\in_habitat_top100_seeds_all_combined.csv",
    "habitat_out": r"E:\liuzhou_breastcancer\eval_results-2\out_habitat_top100_seeds_all_combined.csv",
    "habitat_in_out": r"E:\liuzhou_breastcancer\eval_results-2\onlysub_top100_seeds_all_combined.csv",
    "combine": r"E:\liuzhou_breastcancer\eval_results-2\combined.csv"
}

# ========= 定义三类列名 =========
column_map = {
    "train": "auc_mean",
    "test": "auc_test",
    "external": "auc_external"
}

# ========= 读取数据 =========
auc_data = {k: {} for k in column_map.keys()}

for model_name, path in csv_paths.items():
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]  # 防止大小写问题

    for key, col in column_map.items():
        if col.lower() not in df.columns:
            print(f"⚠️ {model_name} 缺少列：{col}")
            continue
        auc_values = df[col.lower()].dropna().iloc[:100].values
        auc_data[key][model_name] = auc_values

print("✅ 所有CSV文件读取完成。\n")

# ========= 显著性检验函数 =========
def compare_auc_groups(auc_dict):
    models = list(auc_dict.keys())
    results = []
    mean_auc = {m: np.mean(auc_dict[m]) for m in models}

    for m1, m2 in combinations(models, 2):
        auc1, auc2 = np.array(auc_dict[m1]), np.array(auc_dict[m2])

        min_len = min(len(auc1), len(auc2))
        auc1, auc2 = auc1[:min_len], auc2[:min_len]

        stat, p = ttest_rel(auc1, auc2)

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        results.append({
            "Model_1": m1,
            "Model_2": m2,
            "Mean_AUC_1": mean_auc[m1],
            "Mean_AUC_2": mean_auc[m2],
            "Δ(AUC1-AUC2)": mean_auc[m1] - mean_auc[m2],
            "p_value": p,
            "Significance": sig
        })

    return pd.DataFrame(results).sort_values("p_value")

# ========= 分析并输出结果 =========
for set_type in ["train", "test", "external"]:
    data = auc_data[set_type]
    if not data:
        print(f"⚠️ 未检测到 {set_type} 数据，跳过。")
        continue

    print(f"📊 分析 {set_type.upper()} 集 AUC 差异...")
    df_result = compare_auc_groups(data)

    output_file = f"./t-test/auc_significance_{set_type}.csv"
    df_result.to_csv(output_file, index=False)
    print(f"✅ {set_type} 结果已保存为：{output_file}\n")

print("🎉 全部分析完成！")