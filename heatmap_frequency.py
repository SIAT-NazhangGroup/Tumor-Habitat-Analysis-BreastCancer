import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize

# === 1. 读取数据 ===
df = pd.read_csv(r'E:\liuzhou_breastcancer\feature_selection_simulated_matrix.csv')  # 每列是特征，每行是迭代

# 如果有 Iteration 列，则去掉
if "Repeat" in df.columns:
    df = df.drop(columns=["Repeat"])

# === 2. 转置矩阵，使特征为行，循环为列 ===
data = df.T  # 转置后，每行=特征，每列=循环
data.index.name = "Feature"
data.columns = [f"Iter_{i+1}" for i in range(data.shape[1])]

# === 3. 计算每个特征出现的次数和频率 ===
feature_counts = data.sum(axis=1)
max_iter = data.shape[1]
feature_freq = (feature_counts / max_iter) *100   # 每个特征的出现频率 (0~1)

# === 4. 颜色矩阵：1处根据频率着色，0处白色 ===
color_matrix = np.zeros_like(data, dtype=float)
for i, freq in enumerate(feature_freq):
    color_matrix[i, data.iloc[i].values == 1] = freq  # 1的地方涂上频率值

# === 5. 绘制热图 ===
plt.figure(figsize=(12, 8), dpi=300)
ax = sns.heatmap(
    color_matrix,
    cmap="coolwarm",          # 红色系，频率越高越深
    vmin=0, vmax=100,       # 标准化颜色范围
    cbar=True,
    linewidths=0.1,
    linecolor="lightgray",
    xticklabels=False,
    yticklabels=False,
    mask=(color_matrix == 0)  # 0 的地方白色
)

# # === 6. 放大字体 ===
# plt.yticks(ticks=np.arange(0, color_matrix.shape[0], 2), fontsize=15)
# # 设置x轴显示所有标签，或可以根据需要调整间隔
# plt.xticks(fontsize=14)
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=12)     # colorbar刻度字体
# cbar.set_label("Feature Frequency (%)", fontsize=14)   # colorbar标签字体

# # === 7. 美化与展示 ===
# plt.title("Monte Carlo Feature Selection Heatmap (Color by Feature Frequency)", fontsize=16)
# plt.ylabel("Features", fontsize=20)
# plt.xlabel("Iterations", fontsize=15)
# plt.tight_layout()
# plt.show()

# === 设置 y 轴：从上到下 1~75，每隔2个显示一个 ===
y_ticks = np.arange(0, color_matrix.shape[0], 3)
ax.set_yticks(y_ticks + 0.5)  # 加 0.5 是为了对齐网格中心
ax.set_yticklabels((y_ticks + 1).astype(str), fontsize=7)

# === 设置 x 轴：从左到右 1~100，每隔5个显示一个 ===
x_ticks = np.arange(9, color_matrix.shape[1], 10)
ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels((x_ticks + 1).astype(str), fontsize=7, rotation=0)

# === 设置坐标轴线样式 ===
for spine in ax.spines.values():
    spine.set_linewidth(0.2)
    spine.set_edgecolor("lightgray")

# === colorbar 字体放大 ===
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Frequency", fontsize=10)
# 设置 colorbar 的刻度从 0 到 100，每隔 10 显示一次
cbar.set_ticks(np.arange(0, 101, 10))  # 刻度位置为 0, 10, 20, ..., 100

# === 标题与标签 ===
plt.title("Monte Carlo Feature Selection Heatmap", fontsize=12)
plt.ylabel("Feature Index", fontsize=10)
plt.xlabel("Iterations", fontsize=10)
plt.tight_layout()
plt.show()

plt.savefig("heatmap.pdf", bbox_inches='tight')
plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')




