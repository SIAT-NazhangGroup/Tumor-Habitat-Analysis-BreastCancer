# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
from __future__ import print_function
import nibabel as nib
import collections
import logging

import numpy as np
import os  # needed navigate the system to get the input data
import datetime
import gc
import radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import pandas as pd
paramPath=os.path.join(r'<PROJECT_ROOT>\Setting', 'Params.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

labels_file=r'<NEW_ROOT>\datas_clear.csv'
datas0_folder=r'<NEW_ROOT>\figure-res-0\washIn'
datas1_folder=r'<NEW_ROOT>\figure-res-1\washIn'
# datas_folder=r'<PROJECT_ROOT>\habitat_out'

#label
datalist=pd.read_csv(labels_file)
# print(pd_labels_file)
patient_name=[]
# for i in range(pd_labels_file.shape[0]):
    # pd_labels_file.iloc[i,1]=pd_labels_file.iloc[i,0].split('_')[0]
# pd_labels_file['patient_name']=[pd_labels_file.iloc[i,2].split('_')[0] for i in range(pd_labels_file.shape[1])]
datalist.head()

# data
nii_datas=[]
# for nii_data in os.listdir(datas0_folder):
#     nii_data_name = nii_data.replace("Benign", "")
#     nii_data_name = nii_data_name[:-4]
#     grade = 0
#     print(nii_data_name)
#     nii_data_path=os.path.join(datas0_folder,nii_data)
#     nii_datas.append([nii_data_name, grade, nii_data_path])
#     # print(nii_data_path)

# for nii_data in os.listdir(datas1_folder):
#     nii_data_name = nii_data.replace("Malignant", "")

#     nii_data_name = nii_data_name[:-4]
#     grade = 1
#     print(nii_data_name)
#     nii_data_path=os.path.join(datas1_folder, nii_data)
#     nii_datas.append([nii_data_name, grade, nii_data_path])
#     # print(nii_data_path)

for datalist0 in range(datalist.shape[0]):
        # 读取数据

    nii_data_name = datalist.loc[datalist0, 'patient_name']
    grade = datalist.loc[datalist0, 'grade']
    label_name = str(datalist.loc[datalist0, 'label_name'])
    if label_name.endswith(".nii.gz"):
        mask_name = datalist.iloc[datalist0, 2][:-7]  # .nii.gz 长度是 7
    elif label_name.endswith(".nii"):
        mask_name = datalist.iloc[datalist0, 2][:-4]  # .nii 长度是 4
    if grade == 1:
        grade_name = r'Malignant'
        nii_data_path = os.path.join(datas1_folder, f"{grade_name}{nii_data_name}.nii")
    else:
        grade_name = r'Benign'
        nii_data_path = os.path.join(datas0_folder, f"{grade_name}{nii_data_name}.nii")
    # nii_data_path = os.path.join(datas_folder, f"{grade_name}{nii_data_name}_{mask_name}.nii")
    print(nii_data_path)
    if os.path.exists(nii_data_path):
        nii_datas.append([nii_data_name, grade, label_name, nii_data_path])


print(len(nii_datas))    



pd_nii_datas=pd.DataFrame(nii_datas)
pd_nii_datas.columns=['patient_name','grade', 'label_name','data_path']
pd_nii_datas.head()

pd_all=pd.merge(datalist,pd_nii_datas,left_on=["patient_name", "grade", 'label_name'],right_on=["patient_name", "grade", 'label_name'],how="left")
# pd_all.loc[13,:]
print(pd_all.shape)
print(pd_all)

# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

list_result=[]
for i in range(pd_all.shape[0]):
    dic_result=collections.OrderedDict({"index":pd_all.loc[i,'index'], "patient_name":pd_all.loc[i,'patient_name'],
                                        "label_name":pd_all.loc[i,'label_name'], "grade":pd_all.loc[i,'grade']})
    # print(dic_result)


    try:
        select=i
        imagePath=pd_all.loc[select, 'data_path']
        image=nib.load(imagePath)
    except:
        print(pd_all.loc[i, 'patient_name'], ' read image has error')
        continue

    try:
        maskPath=pd_all.loc[select,'label_path']
        mask=nib.Nifti1Image(nib.load(maskPath).get_fdata(),image.affine)
    except:
        print(pd_all.loc[i,'patient_name'],' read label has error')
        continue

    # tempMask='../data/temp/'+'mask.nii'
    # nib.save(mask,tempMask)

    # tempImage='../data/temp/'+'image.nii'
    tempMask=r'<NEW_ROOT>\temp'+'mask.nii'
    nib.save(mask,tempMask)

    tempImage=r'<NEW_ROOT>\temp'+'image.nii'
    image_trans=np.array(image.get_fdata())
    # image_trans[image_trans==19]=0
    nib.save(nib.Nifti1Image(image_trans,image.affine),tempImage)


    # print('image shape: ',image.shape,' mask shape: ',mask.shape)
    # result = extractor.execute(tempImage, tempMask)
    try:
        result = extractor.execute(tempImage, tempMask)
    except:
        print(pd_all.loc[i,'patient_name'],'err in extract radioligy feature')
        continue

    dic_result.update(result)
    list_result.append(dic_result)
    # print(dic_result)
    print('grade:',pd_all.loc[i,'grade'], ' ---  patient:', pd_all.loc[i,'patient_name'], ' is ok')


    # print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
    # print('')
    # print('Calculated features')
    # for key, value in six.iteritems(dic_result):
    #     print('\t', key, ':', value)
    # break

pd_result = pd.DataFrame(list_result)

today = datetime.date.today()
pd_result.to_csv(r'<NEW_ROOT>\radiology\in_radiology_habitat2_{}.csv'.format(today),index=False)

gc.collect()