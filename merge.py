# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import pandas as pd

# 读取两个CSV文件
file1 = pd.read_excel(r"<PROJECT_ROOT>\radiology\良恶性名单.xlsx")
file2 = pd.read_csv(r"<PROJECT_ROOT>\datas_rename.csv")

print("file1 中的前几行：")
print(file1[['patient_name', 'label_name']].head())

print("file2 中的前几行：")
print(file2[['patient_name', 'label_name', 'Patient_ID']].head())


# 根据 'patient_name' 和 'label_name' 两列进行合并
merged_df = pd.merge(file1, file2[['index', 'Patient_ID']], 
                     on=['index'], 
                     how='left')

print(merged_df)

# 保存为一个新的CSV文件
merged_df.to_csv(r'<PROJECT_ROOT>\selected_rename.csv', index=False)

print("合并完成，新的CSV文件已保存")
