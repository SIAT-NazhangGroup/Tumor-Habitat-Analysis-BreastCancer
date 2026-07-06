
import os
import re
import cupy as cp
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from datetime import datetime

from nibabel.filebasedimages import ImageFileError

tic_csv_save_path=r'E:\liuzhou_breastcancer\set\Tic'
set_csv_save_path=r'E:\liuzhou_breastcancer\set\Set'
datas_path = r'E:\liuzhou_breastcancer\datas_path.csv'
paths = pd.read_csv(datas_path)
mempool = cp.get_default_memory_pool()

for i in range(paths.shape[0]):
    useful_flag = True

    # csv的命名
    if paths.loc[i, 'grade'] == 0:
        grade = 'Benign'
    else:
        grade = 'Malignant'

    csv_name = grade + '_' + paths.loc[i, 'patient_name'] + '_' + paths.loc[i, 'label_name'] + '.csv'

    if os.path.exists(os.path.join(tic_csv_save_path, csv_name)):
        print(csv_name, ' is exist')
        continue
    else:
        print(paths.loc[i, 'label_name'], ' is on processing')

    # 读取label获得label信息
    try:
        label = nib.load(paths.loc[i, 'label_path']).get_fdata()
    except ImageFileError as err:
        useful_flag = False
        print("ImageFileError: {0}".format(err))
        continue

    label_gpu = cp.array(label)
    # print(label_gpu.shape)
    locs = label_gpu.nonzero()
    d = label_gpu[label_gpu != 0]
    if len(d) is 0:
        print('label have no signal, move to next')
        continue
    rows = locs[0]
    columns = locs[1]
    slices = locs[2]

    # stages = [int(stage) for stage in os.listdir(paths.loc[i, 'nii_data']) if not re.search('nii', stage)]
    stages = []
    for stage in os.listdir(paths.loc[i, 'nii_data']):
        if not re.search('nii', stage):
            try:
                stage_int = int(stage)
                stages.append(stage_int)
            except ValueError:
                print(f"Ignoring non-integer value: {stage}")
    stages.sort()
    stages_path = [os.path.join(paths.loc[i, 'nii_data'], str(stage)) for stage in stages]
    dcm_paths = [os.path.join(paths.loc[i, 'dcm_data'], str(stage)) for stage in stages]

    csv_columns = [str(stage) for stage in stages]
    # 因为期数不一样 所以浮动提取
    [csv_columns.append(name) for name in ['rows', 'columns', 'slices']]
    csv = pd.DataFrame(columns=csv_columns)
    data = []
    labels = []

    # 配置csv同理
    set_columns = ['filp_angle', 'TR']
    set = []

    # 遍历读取每个阶段的值
    for i in range(len(stages)):
        stage_path = os.path.join(stages_path[i], os.listdir(stages_path[i])[0])
        # print(stages[i],' path: ',stage_path)
        dcms = os.path.join(dcm_paths[i], os.listdir(dcm_paths[i])[0])
        # print(stages[i],' path: ',dcms)
        # 如果是目录就遍历目录下的所有子目录和文件
        if os.path.isdir(dcms):
            for session in os.listdir(dcms):
                session_dir = os.path.join(dcms, session)
                ds = pydicom.read_file(session_dir)
        else:
            ds = pydicom.read_file(dcms)

        if i == 0:
            # 读取固定的属性
            # filp angle
            filp_angle = ds[0X0018, 0X1314].value
            # print(filp_angle)
            set.append(filp_angle
                       )
            TR = ds[0X0018, 0X0080].value
            # print(TR)
            set.append(TR)

        #有trigger time的时候用这个
        #str_time = ds[0X0018, 0X1060].value
        #没有trigger time用acquisition time的时候用这个
        str_time = ds[0X0008, 0X0032].value[0:6]
        # print(str_time)
        # time_various='time_'+str(i+1)+'=time'
        set_columns.append('time_' + str(i + 1))
        time=str_time
        time = datetime.strftime(datetime.strptime(str_time, "%H%M%S"), "%H:%M:%S")

        # print(time_various)
        # 扫描时间
        # exec(time_various)
        set.append(time)
        stage_data = nib.load(stage_path).get_fdata()
        stage_data_gpu = cp.array(stage_data)

        # 尝试label 和data维度是否可用
        # label_temp = label_gpu
        try:
            label2data = stage_data_gpu[label_gpu != 0]  # 1,0

        except IndexError as err:
            useful_flag = False
            print("IndexError: {0}".format(err))
            break

        data.append(label2data)
        labels.append(label_gpu)

    # 保存配置文件
    if not useful_flag:
        print('move to next patient')
        continue
    del label2data
    del stage_data_gpu
    # del label_temp
    set_csv = np.array([set])
    # print(set_csv)
    pd_set = pd.DataFrame(set_csv, columns=set_columns)
    pd_set.to_csv(os.path.join(set_csv_save_path, csv_name), index=False)

    # 保存TIC文件
    # cp.get_default_memory_pool().used_bytes()
    cp_data = cp.array(data)
    # print(cp_data.nbytes)  # 120560400
    # print(mempool.used_bytes())  # 120560640
    # print(mempool.total_bytes())
    # cp_labels=cp.array(labels)
    # print(cp_labels.nbytes)  # 120560400
    # print(mempool.used_bytes())  # 120560640
    # print(mempool.total_bytes())
    # result_data = cp_data[cp_labels != 0]
    test = cp.reshape(cp_data, (-1, len(rows)))
    TIC = test.T  # 时间维度在第一维，所以要转置
    result = cp.c_[TIC, rows, columns, slices]
    csv = pd.DataFrame(result, columns=csv_columns)
    csv.to_csv(os.path.join(tic_csv_save_path, csv_name), index=False)
    # del cp_data
    # del cp_labels
# paths = pd.read_csv(datas_path)
# mempool = cp.get_default_memory_pool()
#
# for i in range(paths.shape[0]):
#     useful_flag = True
#
#     # csv的命名
#     if paths.loc[i, 'grade'] == 0:
#         grade = 'Benign'
#     else:
#         grade = 'Malignant'
#
#     csv_name = grade + '_' + paths.loc[i, 'patient_name'] + '_' + paths.loc[i, 'label_name'] + '.csv'
#
#     if os.path.exists(os.path.join(tic_csv_save_path, csv_name)):
#         print(csv_name, ' is exist')
#         continue
#     else:
#         print(paths.loc[i, 'label_name'], ' is on processing')
#
#     # 读取label获得label信息
#     try:
#         label = nib.load(paths.loc[i, 'label_path']).get_fdata()
#     except ImageFileError as err:
#         useful_flag = False
#         print("ImageFileError: {0}".format(err))
#         continue
#
#     label_gpu = cp.array(label)
#     # print(label_gpu.shape)
#     locs = label_gpu.nonzero()
#     d = label_gpu[label_gpu != 0]
#     if len(d) is 0:
#         print('label have no signal, move to next')
#         continue
#     rows = locs[0]
#     columns = locs[1]
#     slices = locs[2]
#
#     stages = [int(stage) for stage in os.listdir(paths.loc[i, 'nii_data']) if not re.search('nii', stage)]
#     stages.sort()
#     stages_path = [os.path.join(paths.loc[i, 'nii_data'], str(stage)) for stage in stages]
#     dcm_paths = [os.path.join(paths.loc[i, 'dcm_data'], str(stage)) for stage in stages]
#
#     csv_columns = [str(stage) for stage in stages]
#     # 因为期数不一样 所以浮动提取
#     [csv_columns.append(name) for name in ['rows', 'columns', 'slices']]
#     csv = pd.DataFrame(columns=csv_columns)
#     data = []
#     labels = []
#
#     # 配置csv同理
#     set_columns = ['filp_angle', 'TR']
#     set = []
#
#     # 遍历读取每个阶段的值
#     for i in range(len(stages)):
#         stage_path = os.path.join(stages_path[i], os.listdir(stages_path[i])[0])
#         # print(stages[i],' path: ',stage_path)
#         dcms = os.path.join(dcm_paths[i], os.listdir(dcm_paths[i])[0])
#         # print(stages[i],' path: ',dcms)
#         ds = pydicom.read_file(dcms)
#
#         if i == 0:
#             # 读取固定的属性
#             # filp angle
#             filp_angle = ds[0X0018, 0X1314].value
#             # print(filp_angle)
#             set.append(filp_angle
#                        )
#             TR = ds[0X0018, 0X0080].value
#             # print(TR)
#             set.append(TR)
#
#         str_time = ds[0X0008, 0X0032].value[0:6]
#         # print(str_time)
#         # time_various='time_'+str(i+1)+'=time'
#         set_columns.append('time_' + str(i + 1))
#         time = datetime.strftime(datetime.strptime(str_time, "%H%M%S"), "%H:%M:%S")
#         # print(time_various)
#         # 扫描时间
#         # exec(time_various)
#         set.append(time)
#         stage_data = nib.load(stage_path).get_fdata()
#         stage_data_gpu = cp.array(stage_data)
#         # 尝试label 和data维度是否可用
#         # label_temp = label_gpu
#         try:
#             label2data = stage_data_gpu[label_gpu != 0]  # 1,0
#         except IndexError as err:
#             useful_flag = False
#             print("IndexError: {0}".format(err))
#             break
#
#         data.append(label2data)
#         labels.append(label_gpu)
#
#     # 保存配置文件
#     if not useful_flag:
#         print('move to next patient')
#         continue
#     del label2data
#     del stage_data_gpu
#     # del label_temp
#     set_csv = np.array([set])
#     # print(set_csv)
#     pd_set = pd.DataFrame(set_csv, columns=set_columns)
#     pd_set.to_csv(os.path.join(set_csv_save_path, csv_name), index=False)
#
#     # 保存TIC文件
#     # cp.get_default_memory_pool().used_bytes()
#     cp_data = cp.array(data)
#     # print(cp_data.nbytes)  # 120560400
#     # print(mempool.used_bytes())  # 120560640
#     # print(mempool.total_bytes())
#     # cp_labels=cp.array(labels)
#     # print(cp_labels.nbytes)  # 120560400
#     # print(mempool.used_bytes())  # 120560640
#     # print(mempool.total_bytes())
#     # result_data = cp_data[cp_labels != 0]
#     test = cp.reshape(cp_data, (-1, len(rows)))
#     TIC = test.T  # 时间维度在第一维，所以要转置
#     result = cp.c_[TIC, rows, columns, slices]
#     csv = pd.DataFrame(result, columns=csv_columns)
#     csv.to_csv(os.path.join(tic_csv_save_path, csv_name), index=False)
#     # del cp_data
#     # del cp_labels

    