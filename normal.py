# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
# 特征归一化
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

in_datapath=r'<PROJECT_ROOT>\radiology\washin_radiology_2024-10-30.csv'
x = pd.read_csv(in_datapath)

# 识别数值型特征
num_features = x.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
# 对数值型特征进行归一化
num_features_scaled = scaler.fit_transform(num_features)

# 创建新的 DataFrame
num_features_scaled_df = pd.DataFrame(num_features_scaled, columns=num_features.columns)

differences = {}
for column in num_features_scaled_df.columns:
    if not num_features_scaled_df[column].equals(num_features[column]):
        differences[column] = {
            'df1': num_features_scaled_df[column].tolist(),
            'df2': num_features[column].tolist()
        }

# 输出结果
if differences:
    for col, vals in differences.items():
        print(f"列 '{col}' 的值不一样:")
        print(f"DataFrame1: {vals['df1']}")
        print(f"DataFrame2: {vals['df2']}")
else:
    print("两个 DataFrame 的所有对应列的值相同。")

print(num_features)
print(num_features_scaled_df.head())