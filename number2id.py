import os
import pydicom
import pandas as pd
import cupy as cp

# 设置根路径
datas_path = r'E:\liuzhou_breastcancer\datas_path.csv'
paths = pd.read_csv(datas_path)
mempool = cp.get_default_memory_pool()

ids = []

# 遍历所有病人文件夹（当前为数字命名）
for i in range(paths.shape[0]):
    useful_flag = True

    patient_id = r'Error'
    patient_number = paths.loc[i, 'patient_name']

    print(f'======================= {patient_number} is processing =======================')

    folder_path = paths.loc[i, 'dcm_data']
    if not os.path.isdir(folder_path):
        print(f"[跳过] {patient_number}：{folder_path}该路径无效")
        ids.append(patient_id)
        continue

    if not os.path.isdir(folder_path):
        ids.append(patient_id)
        continue

    sub_dce1_path = os.path.join(folder_path, "1")
    if not os.path.isdir(sub_dce1_path):
        print(f"[跳过] {folder_path}：无 1 文件夹")
        ids.append(patient_id)
        continue
    

    series_path = None
    # 查找 Series 子目录
    series_dirs = [d for d in os.listdir(sub_dce1_path)
                   if os.path.isdir(os.path.join(sub_dce1_path, d))]
    # print(series_dirs)
    if not series_dirs:
        for file in os.listdir(sub_dce1_path):
            file_path = os.path.join(sub_dce1_path, file)
    else:
        series_path = os.path.join(sub_dce1_path, series_dirs[0])

    if series_path:

    # 遍历该 Series 文件夹，寻找 DICOM 图像
    # patient_id = None
        if os.path.exists(series_path):
            for file in os.listdir(series_path):
                file_path = os.path.join(series_path, file)
            
    print(file_path)


    if os.path.isfile(file_path):
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            patient_id = ds.PatientID
            ids.append(patient_id)
            continue
        except Exception as e:
            print(f"[跳过] {patient_number}：找不到合法 DICOM")
            continue
    
    # if not patient_id:
    #     patient_id = r'Error'
    #     print(f"[跳过] {patient_number}：找不到合法 DICOM")
    #     ids.append(patient_id)
    #     # ids.append('Error')

    #     continue
    

paths["Patient_ID"] = ids

# 保存新的 CSV 文件
paths.to_csv(r'E:\liuzhou_breastcancer\datas_rename.csv', index=False)





    