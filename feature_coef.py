# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd

# === 第一步：读取五个待合并的 CSV ===
df1 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_1211.csv")
df2 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_2.csv")
df3 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_3.csv")
df4 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_4.csv")
df5 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_5.csv")
df6 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_6.csv")
df7 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_7.csv")
df8 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_8.csv")
df9 = pd.read_csv(r"<PROJECT_ROOT>\in_out_sub_feature_weight_9.csv")

# === 统一第一列名（假设它们第一列都是匹配用的 Key）===
key = df1.columns[0]

# 给每个 DataFrame 的第二列取独立名字，避免重复冲突
df1.columns = [key, "value_1"]
df2.columns = [key, "value_2"]
df3.columns = [key, "value_3"]
df4.columns = [key, "value_4"]
df5.columns = [key, "value_5"]
df6.columns = [key, "value_6"]
df7.columns = [key, "value_7"]
df8.columns = [key, "value_8"]
df9.columns = [key, "value_9"]  

# === 第二步：依次按 key 合并五个表 ===
merged = df1
for df in [df2, df3, df4, df5, df6, df7, df8, df9]:
    merged = pd.merge(merged, df, on=key, how="outer")

# === 第三步：读取筛选文件 ===
df_filter = pd.read_csv(r"<PROJECT_ROOT>\voxel_features_random_distribution.csv")

# 假设筛选列名是第六个 CSV 的第一列（可改为你需要的列名）
filter_col = df_filter.columns[0]

# === 第四步：根据 filter.csv 中的值筛选 merged ===
# === 第四步：根据 filter.csv 中的值筛选 merged，并保持顺序 ===

# 取筛选文件中的顺序列表
result = pd.merge(df_filter, merged, left_on=filter_col, right_on=key, how="left")

# === 去掉重复列（如果 key 重复出现） ===
if key != filter_col:
    result = result.drop(columns=[key])

# === 保持第六个文件原始顺序 ===
result.index = range(len(result))  # 重置行号
# 不需要额外排序，因为 left join 已自动保持左表顺序

# === 第五步：保存结果 ===
result.to_csv(r"<PROJECT_ROOT>\filtered_result.csv", index=False)

print("✅ 合并并筛选完成，结果已保存为 filtered_result.csv")