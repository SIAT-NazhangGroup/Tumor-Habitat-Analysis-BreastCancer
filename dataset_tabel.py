import re
import os
import numpy as np
import pandas as pd
# 只使用一次
count=0
columns = ['index','patient_name','label_name','grade','nii_data','dcm_data','label_path']
pd_data = pd.DataFrame(columns=columns)

dcm_path=r'E:\DCESummary_2019-202004\benign-DCE-221subs\part1-175subs+part2'
nii_path=r'E:\liuzhou_breastcancer\Benign_nii'
grade=0

for patient_name in os.listdir(dcm_path):
    print(patient_name)
    patient_path=os.path.join(dcm_path,patient_name)
    labels_name=[label for label in os.listdir(patient_path) if re.search('.nii',label)]
    for label_name in labels_name:
        print(label_name)
        pd_data.loc[count,:]=[count,patient_name,label_name,grade,os.path.join(nii_path,patient_name),patient_path,os.path.join(patient_path,label_name)]
        count=count+1

dcm_path=r'E:\DCESummary_2019-202004\malignant-DCE-451subs\part1-288subs+part2'
nii_path=r'E:\liuzhou_breastcancer\Malignant_nii'
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
pd_data.to_csv(r'E:\liuzhou_breastcancer\datas_path.csv',index=False)