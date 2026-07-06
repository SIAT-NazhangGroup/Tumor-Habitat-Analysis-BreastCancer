# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


# 你想绘图的数据列名
column_name = "value"

# 自定义绘图函数
def plot_kde_counts(data, color, label, bw=1.0):
    if len(data) < 2:
        return  # KDE 需要至少两个点
    kde = gaussian_kde(data, bw_method=bw)
    x_grid = np.linspace(data.min(), data.max(),100)
    dx = x_grid[1] - x_grid[0]  # 每个小区间宽度
    y = kde(x_grid) * len(data) * dx  # 把密度 × 样本数 => 转换为频数
    plt.plot(x_grid, y, label=label, color=color, alpha=0.7)

# 读取两个 CSV 文件
df1 = pd.read_csv(r"<NEW_ROOT>\wash-out-mean.csv")  # 替换为实际路径
df2 = pd.read_csv(r"<PROJECT_ROOT>\wash-out-mean-1.csv")

# 设置图形风格（可选）
sns.set(style="whitegrid")

# 设置颜色顺序（可自定义）
colors = ['skyblue', 'palevioletred', 'steelblue',  'crimson']

# 创建画布
plt.figure(figsize=(12, 6))

# 分别绘制 grade=0 和 grade=1 的 KDE
for i, grade_val in enumerate([0,1]):
    # 过滤 df1 和 df2 中 grade == grade_val 的数据
    data1 = pd.to_numeric(df1[df1['grade'] == grade_val][column_name], errors='coerce').dropna()
    data2 = pd.to_numeric(df2[df2['grade'] == grade_val][column_name], errors='coerce').dropna()

    # 计算统计值
    mean1, median1, std1= data1.mean(), data1.median(), data1.std()
    mean2, median2, std2 = data2.mean(), data2.median(), data2.std()

    # 设置对应的权重：每个样本的权重 = 1
    weights1 = [1] * len(data1)
    weights2 = [1] * len(data2)

    if grade_val==0:
        grade = 'benign'
        print('============ benign ============')
        print(f"组1 均值: {mean1:.3f} ± {std1:.3f}，中位数: {median1:.3f}")
        print(f"组2 均值: {mean2:.3f} ± {std2:.3f}，中位数: {median2:.3f}")
        # KDE 曲线绘图
        plot_kde_counts(data1, color=colors[2 * i], label=f'External Validation - {grade}', bw=0.2)
        # plt.hist(data1, bins=30, alpha=0.3, density=False, color=colors[2 * i], label=f'External Validation - {grade} Hist')
        plot_kde_counts(data2, color=colors[2 * i + 1], label=f'Main Dataset - {grade}', bw=0.2)
    else:
        grade = 'malignant'
        print('============ malignant ============')
        print(f"组1 均值: {mean1:.3f} ± {std1:.3f}，中位数: {median1:.3f}")
        print(f"组2 均值: {mean2:.3f} ± {std2:.3f}，中位数: {median2:.3f}")

        # # KDE 曲线绘图
        # sns.kdeplot(
        #     data1, alpha=0.5, bw_adjust=1.0, fill=False,
        #     weights=weights1,
        #     label=f'External Validation - {grade}', color=colors[2 * i]
        # )
        # sns.kdeplot(
        #     data2, alpha=0.5, bw_adjust=1.0, fill=False,
        #     weights=weights2,
        #     label=f'Main Dataset - {grade}', color=colors[2 * i + 1]
        # )
        plot_kde_counts(data1, color=colors[2 * i], label=f'External Validation - {grade}', bw=0.2)
        plot_kde_counts(data2, color=colors[2 * i + 1], label=f'Main Dataset - {grade}', bw=0.2)


    

    plt.axvline(mean1, color=colors[2 * i], linestyle='--', linewidth=1.5, label=f'Average: {mean1:.3f} ± {std1:.3f}')
    # plt.axvline(median1, color=colors[2 * i], linestyle=':', linewidth=1.5, label=f'Madian: {median1:.3f}')
    plt.axvline(mean2, color=colors[2 * i + 1], linestyle='--', linewidth=1.5, label=f'Average: {mean2:.3f} ± {std2:.3f}')
    # plt.axvline(median2, color=colors[2 * i + 1], linestyle=':', linewidth=1.5, label=f'Madian: {median2:.3f}')

# 添加图形标题和标签
plt.title("Wash-out Value Distribution")
plt.xlim(-0.5, 1.0)  # 将 x 轴范围设置为 0 到 1，根据你的数据调整
plt.xlabel('Value')
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
