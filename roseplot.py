import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ======== 示例数据 ========
n_features = 76
f = pd.read_csv(r'E:\liuzhou_breastcancer\filtered_result.csv')
features = [i + 1 for i in range(n_features)]
weights = f['MEAN'].tolist()

df = pd.DataFrame({"Feature": features, "Weight": weights})

# ======== 参数配置 ========
clockwise = True
use_redwhiteblue = True
title = "Feature Weights Rose Chart"

# ======== 计算角度 ========
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
# if clockwise:
#     angles = np.flip(angles)

# ======== 颜色方案 ========
if use_redwhiteblue:
    cmap = cm.get_cmap('RdBu_r')
    norm = mcolors.TwoSlopeNorm(vmin=min(weights), vcenter=0, vmax=max(weights))
    colors = cmap(norm(weights))
else:
    cmap = cm.get_cmap('Set2')
    colors = [cmap(i / n_features) for i in range(n_features)]

# ======== 绘图 ========
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# ======== 绘制条形 ========
bars = ax.bar(
    angles,
    np.abs(df["Weight"]),  # 柱高取绝对值
    width=2 * np.pi / n_features * 0.9,
    bottom=np.where(df["Weight"] >= 0, 0, df["Weight"]),  # 负值从0往内画
    color=colors,
    edgecolor='white',
    linewidth=1.0
)

# ======== 美化样式 ========
ax.set_theta_zero_location("N")  # 顶部为起点
ax.set_theta_direction(-1 if clockwise else 1)
ax.set_xticks(angles)
ax.set_xticklabels(df["Feature"], fontsize=12)
ax.set_facecolor("white")
ax.set_rlabel_position(90)  # 把半径刻度文字移到右侧（90° 方向）
ax.grid(alpha=0.3)

# ======== 设置半径刻度（含负值） ========
r_max = np.ceil(max(abs(df["Weight"])) * 100) / 100
r_ticks = np.linspace(-r_max, r_max, 8)  # 从 -max 到 +max
ax.set_ylim(-r_max, r_max)
ax.set_yticks(r_ticks)
# ax.set_yticklabels([f"{r:.2f}" for r in r_ticks], fontsize=8, color="gray")
ax.set_yticklabels([])

# 让半径刻度标签稍偏移，避免与柱子重叠
ax.set_rlabel_position(180 / n_features)

# ======== 中心线（代表 0 半径） ========
ax.plot(np.linspace(0, 2 * np.pi, 400), [0]*400, color='black', linewidth=1, alpha=0.6)

# ======== 添加标题 ========
ax.set_title(title, va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig("radar_chart_300dpi.png", dpi=300, bbox_inches="tight")
plt.show()
