
import os
import re
import cupy as cp
import pandas as pd
import numpy as np
from datetime import datetime

tic_csv_save_path=r'E:\liuzhou_breastcancer\set\Tic'
set_csv_save_path=r'E:\liuzhou_breastcancer\set\Set'
datas_path = r'E:\liuzhou_breastcancer\datas_path.csv'
res_csv_save_path=r'E:\liuzhou_breastcancer\set\res'
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
    file_path = os.path.join(set_csv_save_path, csv_name)

    if os.path.exists(file_path):
        pass
    else:
        print(file_path, ' is wrong, move to next patient')
        continue

    # 读取时间和峰值
    df_time = pd.read_csv(os.path.join(set_csv_save_path, csv_name))
    df_peak = pd.read_csv(os.path.join(tic_csv_save_path, csv_name))

    if os.path.exists(os.path.join(res_csv_save_path, csv_name)):
        print(csv_name, ' is exist')
        continue
    else:
        print(csv_name, ' is on processing')

    # print(df_time.shape)
    # print(df_time.head())
    # 时间标准化处理
    # 由于第一次扫描时未打药，与后几次扫描的时间差有差异，故需重新计算时间
    second_row = df_time.iloc[0]
    # 获取达峰时间和末期的时间值
    peak_time = datetime.strptime(second_row.iloc[4], "%H:%M:%S")
    end_time = datetime.strptime(second_row.iloc[-1], "%H:%M:%S")
    # 计算单次时间差和峰值-末期时间差
    peak2end = end_time - peak_time
    # peak2end = peak2end.total_seconds()
    # print(peak2end)
    zero2peak = peak2end.total_seconds() / 60 / (df_time.shape[1] - 5) * 2

    wash_in = (df_peak.iloc[:, 2] - df_peak.iloc[:, 0]) / df_peak.iloc[:, 0] / zero2peak
    wash_out = (df_peak.iloc[:, -4] - df_peak.iloc[:, 2]) / df_peak.iloc[:, 2]

    res = pd.DataFrame({
        'wash-in': wash_in,
        'wash-out': wash_out
    })

    csv = pd.concat([res, df_peak.iloc[:, -3:]], axis=1)
    csv.to_csv(os.path.join(res_csv_save_path, csv_name), index=False)

