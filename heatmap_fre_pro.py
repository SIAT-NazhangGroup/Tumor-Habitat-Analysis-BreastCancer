# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# =========================
# 字体设置
# =========================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']   # 英文数字
rcParams['font.sans-serif'] = ['SimSun']       # 中文宋体
rcParams['axes.unicode_minus'] = False         # 负号正常显示

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv(r'<PROJECT_ROOT>\feature_selection_simulated_matrix.csv')

# 如果有 Repeat 列则删除
if "Repeat" in df.columns:
    df = df.drop(columns=["Repeat"])

# =========================
# 2. 转置矩阵
# =========================
data = df.T
data.index.name = "Feature"
data.columns = [f"Iter_{i+1}" for i in range(data.shape[1])]

# =========================
# 3. 计算特征频率
# =========================
feature_counts = data.sum(axis=1)
max_iter = data.shape[1]
feature_freq = (feature_counts / max_iter) * 100

# =========================
# 4. 构造颜色矩阵
# =========================
color_matrix = np.zeros_like(data, dtype=float)

for i, freq in enumerate(feature_freq):
    color_matrix[i, data.iloc[i].values == 1] = freq

# =========================
# 5. 绘制热图
# =========================
plt.figure(figsize=(12, 8), dpi=300)

ax = sns.heatmap(
    color_matrix,
    cmap="coolwarm",
    vmin=0,
    vmax=100,
    cbar=True,
    linewidths=0.1,
    linecolor="lightgray",
    xticklabels=False,
    yticklabels=False,
    mask=(color_matrix == 0)
)

# =========================
# 6. 设置 y 轴
# =========================
y_ticks = np.arange(0, color_matrix.shape[0], 3)

ax.set_yticks(y_ticks + 0.5)
ax.set_yticklabels((y_ticks + 1).astype(str), fontsize=7)

# =========================
# 7. 设置 x 轴
# =========================
x_ticks = np.arange(9, color_matrix.shape[1], 10)

ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels((x_ticks + 1).astype(str), fontsize=7, rotation=0)

# =========================
# 8. 坐标轴边框
# =========================
for spine in ax.spines.values():
    spine.set_linewidth(0.2)
    spine.set_edgecolor("lightgray")

# =========================
# 9. colorbar 设置
# =========================
cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=7)

cbar.set_label("频率（次）", fontsize=10)

cbar.set_ticks(np.arange(0, 101, 10))

# =========================
# 10. 标题和标签
# =========================
plt.title("", fontsize=12)
plt.ylabel("", fontsize=10)
plt.xlabel("", fontsize=10)

plt.tight_layout()

# =========================
# 11. 保存图片
# =========================
plt.savefig("heatmap.pdf", bbox_inches='tight')
plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')

plt.show()