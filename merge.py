import pandas as pd

# 读取两个CSV文件
file1 = pd.read_excel(r"E:\liuzhou_breastcancer\radiology\良恶性名单.xlsx")
file2 = pd.read_csv(r"E:\liuzhou_breastcancer\datas_rename.csv")

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
merged_df.to_csv(r'E:\liuzhou_breastcancer\selected_rename.csv', index=False)

print("合并完成，新的CSV文件已保存")
