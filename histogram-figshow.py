# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# 设置图像风格
sns.set(style="whitegrid")

# 读取两个 CSV 文件
df1 = pd.read_csv(r"<NEW_ROOT>\wash-out-mean.csv")  # 替换为实际路径
df2 = pd.read_csv(r"<PROJECT_ROOT>\wash-out-mean.csv")

# 设置分析的列名（需要两张表中都有该列）
column_name = 'value'  # ← 请替换为你的实际列名

# 提取并清洗数据（去除空值并确保为数值类型）
data1 = pd.to_numeric(df1[column_name], errors='coerce').dropna()
data2 = pd.to_numeric(df2[column_name], errors='coerce').dropna()

# # 替换小于0的为0，并去除缺失值
# data1 = pd.to_numeric(df1[column_name], errors='coerce').dropna().clip(lower=0)
# data2 = pd.to_numeric(df2[column_name], errors='coerce').dropna().clip(lower=0)

print("数据1 最小值:", data1.min())
print("数据2 最小值:", data2.min())


# 计算统计值
mean1, median1, std1= data1.mean(), data1.median(), data1.std()
mean2, median2, std2 = data2.mean(), data2.median(), data2.std()

# 绘图
plt.figure(figsize=(12, 6))
# sns.kdeplot(data1, fill=True, label='External Validation', color='blue', alpha=0.5)
# sns.kdeplot(data2, fill=True, label='Main Dataset', color='red', alpha=0.5)
sns.kdeplot(data1, fill=True, label='External Validation', color='skyblue', alpha=0.5, bw_adjust=0.4)
sns.kdeplot(data2, fill=True, label='Main Dataset', color='palevioletred', alpha=0.5, bw_adjust=0.4)


# 添加均值和中位数线条与标签
plt.axvline(mean1, color='skyblue', linestyle='--', linewidth=1.5, label=f'Average: {mean1:.3f} ± {std1:.3f}')
plt.axvline(median1, color='skyblue', linestyle=':', linewidth=1.5, label=f'Madian: {median1:.3f}')
plt.axvline(mean2, color='palevioletred', linestyle='--', linewidth=1.5, label=f'Average: {mean2:.3f} ± {std2:.3f}')
plt.axvline(median2, color='palevioletred', linestyle=':', linewidth=1.5, label=f'Madian: {median2:.3f}')

plt.title('Wash-out Value Distribution')
plt.xlabel(column_name)
plt.ylabel('')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 差异显著性分析
t_stat, t_p = ttest_ind(data1, data2)
u_stat, u_p = mannwhitneyu(data1, data2, alternative='two-sided')

# 打印统计结果
print("=== 差异分析结果 ===")
print(f"t 检验 p 值: {t_p:.10f}")
print(f"Mann-Whitney U 检验 p 值: {u_p:.10f}")
print(f"组1 均值: {mean1:.3f} ± {std1:.3f}，中位数: {median1:.3f}")
print(f"组2 均值: {mean2:.3f} ± {std2:.3f}，中位数: {median2:.3f}")