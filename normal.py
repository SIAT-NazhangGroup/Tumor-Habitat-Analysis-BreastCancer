# 特征归一化
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

in_datapath=r'E:\liuzhou_breastcancer\radiology\washin_radiology_2024-10-30.csv'
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