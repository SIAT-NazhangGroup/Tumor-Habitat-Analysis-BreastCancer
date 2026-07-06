# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import re
import os
import numpy as np
import pandas as pd
# 只使用一次
count=0
columns = ['index','patient_name','label_name','grade','nii_data','dcm_data','label_path']
pd_data = pd.DataFrame(columns=columns)

dcm_path=r'<DCM_ROOT>\benign-DCE-221subs\part1-175subs+part2'
nii_path=r'<PROJECT_ROOT>\Benign_nii'
grade=0

for patient_name in os.listdir(dcm_path):
    print(patient_name)
    patient_path=os.path.join(dcm_path,patient_name)
    labels_name=[label for label in os.listdir(patient_path) if re.search('.nii',label)]
    for label_name in labels_name:
        print(label_name)
        pd_data.loc[count,:]=[count,patient_name,label_name,grade,os.path.join(nii_path,patient_name),patient_path,os.path.join(patient_path,label_name)]
        count=count+1

dcm_path=r'<DCM_ROOT>\malignant-DCE-451subs\part1-288subs+part2'
nii_path=r'<PROJECT_ROOT>\Malignant_nii'
grade=1

for patient_name in os.listdir(dcm_path):
    print(patient_name)
    patient_path=os.path.join(dcm_path,patient_name)
    labels_name=[label for label in os.listdir(patient_path) if re.search('.nii',label)]
    for label_name in labels_name:
        print(label_name)
        pd_data.loc[count,:]=[count,patient_name,label_name,grade,os.path.join(nii_path,patient_name),patient_path,os.path.join(patient_path,label_name)]
        count=count+1
print(pd_data)
pd_data.to_csv(r'<PROJECT_ROOT>\datas_path.csv',index=False)