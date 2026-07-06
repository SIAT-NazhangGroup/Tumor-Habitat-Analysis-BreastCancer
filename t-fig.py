# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 读取文件
file_path = r"<PROJECT_ROOT>\t-test\auc_significance_test.csv"
df = pd.read_csv(file_path)

# 模型列表
desired_order = [
    "in",
    "out",
    "in_out",
    "habitat_in",
    "habitat_out",
    "habitat_in_out",
    "combine"
]

# 从数据中只保留出现在表中的模型（防止有缺）
available_models = list({*df['Model_1'], *df['Model_2']})
models = [m for m in desired_order if m in available_models]


# 构建 p 值矩阵
p_matrix = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)
for _, row in df.iterrows():
    m1, m2, p = row['Model_1'], row['Model_2'], row['p_value']
    p_matrix.loc[m1, m2] = p
    p_matrix.loc[m2, m1] = p
np.fill_diagonal(p_matrix.values, np.nan)

# 显著性分类函数
def significance_label(p):
    if pd.isna(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# 生成符号矩阵
label_matrix = p_matrix.applymap(significance_label)

# 颜色映射
color_map = {'***': "#e64e46", '**': '#fc8d59', '*': '#fee090', 'ns': '#91bfdb', '': 'white'}
color_matrix = label_matrix.applymap(lambda s: color_map[s])

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
for i, row in enumerate(models):
    for j, col in enumerate(models):
        color = color_matrix.loc[row, col]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, ec='white'))

        label = label_matrix.loc[row, col]
        p_val = p_matrix.loc[row, col]

        # 绘制符号和p值
        if not pd.isna(p_val):
            if p_val < 0.001:
                p_text = "<0.001"
            else:
                p_text = f"{p_val:.3f}"
            ax.text(j + 0.5, i + 0.35, label, ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(j + 0.5, i + 0.65, p_text, ha='center', va='center', fontsize=10, color='black')

# 模型标签映射
label_map = {
    "in": "A",
    "out": "B",
    "in_out": "C",
    "habitat_in": "D",
    "habitat_out": "E",
    "habitat_in_out": "F",
    "combine": "G"
}


# 坐标轴设置
ax.set_xticks(np.arange(len(models)) + 0.5)
ax.set_yticks(np.arange(len(models)) + 0.5)
ax.set_xticklabels([label_map[m] for m in models], fontsize=13)
ax.set_yticklabels([label_map[m] for m in models], fontsize=13)
ax.invert_yaxis()
ax.set_xlim(0, len(models))
ax.set_ylim(len(models), 0)
ax.set_title('Result variability of models on the testing set', fontsize=14)

# 图例
legend_elements = [
    Patch(facecolor='#e64e46', edgecolor='white', label='*** (p < 0.001)'),
    Patch(facecolor='#fc8d59', edgecolor='white', label='** (0.001 < p < 0.01)'),
    Patch(facecolor='#fee090', edgecolor='white', label='* (0.01 < p < 0.05)'),
    Patch(facecolor='#91bfdb', edgecolor='white', label='ns (p ≥ 0.05)')
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Significance')

plt.tight_layout()
plt.show()
