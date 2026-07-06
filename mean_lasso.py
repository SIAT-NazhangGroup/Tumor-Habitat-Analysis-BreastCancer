import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 读取CSV
df = pd.read_csv(r"E:\liuzhou_breastcancer\feature_mean_LASSOcoef.csv")


# Feature index（1~76）
feature_index = np.arange(1, len(df) + 1)

# 根据系数正负设置颜色
colors = df["MEAN"].apply(lambda x: "#d62728" if x > 0 else "#1f77b4")

# 创建图
plt.figure(figsize=(10,12))

plt.barh(
    feature_index,
    df["MEAN"],
    color=colors
)

# 0参考线
plt.axvline(0, color="black", linewidth=1)

# 反转y轴（1在上）
plt.gca().invert_yaxis()

# y轴刻度：5、10、15...
yticks = np.arange(5, len(df)+1, 5)
plt.yticks(yticks, fontsize=17)

# 添加水平虚线辅助线
for y in yticks:
    plt.axhline(y, linestyle="--", linewidth=0.6, color="gray", alpha=0.5)

# 标签
plt.xlabel("Mean LASSO Coefficient", fontsize=19)
plt.ylabel("Feature Index", fontsize=19)
plt.title("Mean LASSO Coefficients Across 100 Monte Carlo Cross-Validations", fontsize=20)

plt.xticks(fontsize=17)

plt.tight_layout()

# 保存
plt.savefig(
    r"E:\test_fig\lasso_coefficients_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()