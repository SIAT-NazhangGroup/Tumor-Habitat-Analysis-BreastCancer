import numpy as np
import cv2
import pandas as pd
import nibabel as nib
# import matplotlib.pyplot as plt
import colorsys
import random
import os
import gc

# # 读取两个 CSV 文件
# csv1 = pd.read_csv(r"E:\breastcancer_new\radiology\patient.csv")
# csv2 = pd.read_csv(r"E:\breastcancer_new\datas_path.csv")

# # 按照三列进行合并（inner：只保留能匹配的行）
# merged = pd.merge(csv1, csv2[['patient_name', 'label_name', 'grade', 'nii_data', 'dcm_data', 'label_path']],
#                   on=['patient_name', 'label_name', 'grade'], how='left')

# merged = merged[~merged['label_name'].isin(['ground_truth.nii', 'ground_truth.nii.gz'])]
# # 保存结果到新的 CSV 文件
# merged.to_csv(r'E:\breastcancer_new\datas_clear.csv', index=False)



def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

# B,G,R 格式排序
# 标签地址
label_csv = r'E:\liuzhou_breastcancer\datas_clear.csv'


#%%
# 根据label读取病人的地址
pd_csv=pd.read_csv(label_csv)
# 删除指定值所在行
# df_filtered = pd_csv[~pd_csv['label_name'].isin(['ground_truth.nii', 'ground_truth.nii.gz'])]

# # 保存为新的 CSV
# df_filtered.to_csv('datas_clear.csv', index=False)

datalist=np.array(pd_csv)
print(datalist)
type = 'washIn'
# type = 'washOut'

missing_patien = []


result = []

for datalist0 in range(datalist.shape[0]):
    # if datalist0 == 9:
    #     continue
    prefix = 'Benign' if datalist[datalist0, 3] == 0 else 'Malignant'
    # 创建新的字符串
    max = 0
    patient = datalist[datalist0, 1]
    new_string = f'{prefix}{datalist[datalist0, 1]}.nii'
    if datalist[datalist0, 3] == 0:
        file_path = os.path.join(r'E:\liuzhou_breastcancer\figure-res-0', type, new_string)
    else:
        file_path = os.path.join(r'E:\liuzhou_breastcancer\figure-res-1', type, new_string)

    # 提取mask的名称，有.nii和.nii.gz两种后缀，分开处理
    mask_name = datalist[datalist0, 2]
    if mask_name.endswith(".nii.gz"):
        mask_name = datalist[datalist0, 2][:-7]  # .nii.gz 长度是 7
    elif mask_name.endswith(".nii"):
        mask_name = datalist[datalist0, 2][:-4]  # .nii 长度是 4

    if os.path.exists(file_path):
        data = nib.load(file_path).get_fdata()
        label = nib.load(datalist[datalist0, 7]).get_fdata()
        # print(label.shape[2])
        # print(data)
        # print(label)

        try:
            mean_value = data[label == 1].mean()
            # print(mean_value)
        except IndexError:
            print(f'{patient} : boolean index did not match indexed array')
            continue
        except TypeError:
            print(f'{patient} : Raise TypeError')
            continue

        result.append([patient,mask_name, datalist[datalist0, 3], mean_value])
        print(f'{patient} {mask_name} value = {mean_value}')


df = pd.DataFrame(result, columns = ['patient_name', 'label_name', 'grade', 'value'])
print(df)

df.to_csv(r'E:\liuzhou_breastcancer\wash-in-mean-1.csv')



# %%
