# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
# from ClassifyFun import *
import cupy as cp
import os
import numpy as np
import nibabel as nib
import re
import pandas as pd
from datetime import datetime
# 根据不同类别定义区分度大的颜色
import colorsys
import random
import os
import cv2 as cv

mempol = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

# BGR ->RGB
colors = [[102, 41, 0], [179, 71, 0], [0, 102, 0], [0, 153, 0], [0, 0, 102], [0, 0, 153],
          [255, 102, 0], [255, 148, 77], [0, 230, 0], [26, 255, 26], [0, 0, 230], [26, 26, 255],
          [255, 194, 153], [255, 224, 204], [102, 255, 102], [153, 255, 153], [102, 102, 255], [153, 153, 255],
          [177, 0, 179]]


# B,G,R 格式排序
def denoise_mask(input_data):
    data = np.array(input_data)
    data = np.array((data / np.max(data)) * 255).astype(np.uint8)
    result = np.zeros(data.shape)
    # 加权突出
    for i in range(data.shape[-1]):
        grayimg = cv.equalizeHist(data[:, :, i])
        denoising_image = cv.medianBlur(grayimg, 11)
        grayimg2 = cv.equalizeHist(denoising_image)
        ret1, th1 = cv.threshold(grayimg2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        result[:, :, i] = th1
    return result

Label_choice='Benign'
# Label_choice='Malignant'
Patient_path = r'<PROJECT_ROOT>\Benign_nii'
# Patient_path = r'<PROJECT_ROOT>\Malignant_nii'
#SET_Path = r'H:\Breast Cancer Project\yby\TypeClassify\result_data\Set'
# Patient_path=r'H:\Breast Cancer Project\yby'+'\\'+Label_choice+'_data'
SET_Path = r'<PROJECT_ROOT>\set\Set'
Save_Path = r'<PROJECT_ROOT>\figure-res-0'
# Save_Path = r'<PROJECT_ROOT>\figure-res-1'
label_pathFile=r'<PROJECT_ROOT>\datas_path.csv'
pd_labelFile=pd.read_csv(label_pathFile)

error_patients = []  # 记录数据形状不一致的患者


if not os.path.exists(Save_Path):
    os.makedirs(Save_Path)

for p in os.listdir(Patient_path):
    s_time = datetime.now()
    mempol.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # save_path = os.path.join(Save_Path, 'Malignant' + p + '.nii')
    save_path_in = os.path.join(Save_Path, 'washIn', Label_choice + p + '.nii')
    save_path_out = os.path.join(Save_Path, 'washOut', Label_choice + p + '.nii')

    # if os.path.exists(save_path):
    #     print('---------',p,' is already exists -----------')
    #     mempol.free_all_blocks()
    #     pinned_mempool.free_all_blocks()
    #     continue

    patient_path = os.path.join(Patient_path, p)
    print('---------', p, ' is on processing-----------')

    stages = [int(stage) for stage in os.listdir(patient_path) if not re.search('nii', stage)]
    stages.sort()
    # set_path = [file for file in os.listdir(SET_Path) if
    #             (file.split('_')[0] == 'Malignant') and (re.search(patient_path.split('\\')[-1], file))]
    set_path = [file for file in os.listdir(SET_Path) if
                (file.split('_')[0] == Label_choice) and (re.search(patient_path.split('\\')[-1], file))]
    # print(set_path)
    try:
        set_path = set_path[0]
    except:
        print('SET_path have wrong on' + p)
        continue

    # 检查数据形状是否正确
    shape_mismatch = False  # 用于标记是否出现形状不匹配
    data_shape = None  # [row,column,slice]
    for stage_dir in stages:
        stage_file_name = os.listdir(os.path.join(patient_path, str(stage_dir)))[0]
        # print(stage_file_name)
        stage_data = nib.load(os.path.join(patient_path, str(stage_dir), stage_file_name))
        # print(stage_data)
        stage_data_gpu = cp.array(stage_data.get_fdata())
        if stage_dir == 1:
            data_shape = stage_data_gpu.shape
        elif stage_data_gpu.shape != data_shape:
            print(f'患者 {p} 数据形状对不上: {stage_dir} 的数据形状是 {stage_data_gpu.shape}，期望的形状是 {data_shape}')
            error_patients.append((p, stage_dir))
            shape_mismatch = True
            break

    # 如果形状不匹配，记录该患者并跳过后续处理
    if shape_mismatch:
        continue  # 跳过当前患者，处理下一个患者

    data_gpu = np.zeros(data_shape)
    print(data_gpu.shape)
    img = cp.zeros((data_gpu.shape[0], data_gpu.shape[1], data_gpu.shape[2], len(stages)))
    stage_data_1_affine = None
    # break

    for stage_dir in stages:
        stage_file_name = os.listdir(os.path.join(patient_path, str(stage_dir)))[0]
        stage_data = nib.load(os.path.join(patient_path, str(stage_dir), stage_file_name))
        if stage_dir == 1:
            stage_data_1_affine = stage_data
        stage_data = stage_data.get_fdata()
        stage_data_gpu = cp.array(stage_data)
        img[:, :, :, stage_dir - 1] = stage_data_gpu
        del stage_data_gpu

    # 计算半定量参数
    np_img = img.get()
    # img[:, :, :, 0]=cp.where(img[:, :, :, 0]==0, 1, img[:, :, :, 0])    # 将第一个时间点的图像中的0设置为1

    # # 数据归一化
    # for s in range(img.shape[2]):
    #     temp = (img[:, :, s, :]) / img[:, :, s, 0].reshape(img.shape[0], img.shape[1], 1)
    #     # print(temp)
    #     img[:, :, s, :] = temp
    #     del temp
    #
    maxERs = img[:, :, :, 2]
    # maxERlocs = cp.argmax(img[:, :, :, 1:3], axis=3)+1
    # #print('maxERlocs:',maxERlocs,', initalERs:',initalERs.shape,', maxERs:',maxERs.shape,', lastERs:',lastERs.shape)

    # 读取Tp信息
    pd_set = pd.read_csv(os.path.join(SET_Path, set_path))
    Times = pd_set.iloc[0, 2:]  # 从第二个位置开始是时间
    start_time = pd_set.loc[0, 'time_1']
    startTime = datetime.strptime(start_time, "%H:%M:%S")

    # 初始化时间差列表
    time_objects = []
    differ_time = []

    # 将时间字符串转为 datetime 对象
    for time in Times:
        time_objects.append(datetime.strptime(time, "%H:%M:%S"))

    # 计算相邻时间点的平均时间差
    pairwise_diffs = []
    for k in range(1, len(time_objects) - 1):  # 从第二个时间点到倒数第二个
        diff = (time_objects[k + 1] - time_objects[k]).seconds
        pairwise_diffs.append(diff)

    # 计算相邻时间点的平均时间差
    average_diff = sum(pairwise_diffs) / len(pairwise_diffs) if len(pairwise_diffs) > 0 else 0

    # 重新设置各时间点到第一时间点的时间差
    for k in range(1, len(time_objects)):
        # if k == 0:      #第一个时间点和第一个时间点之间时间差为0
        #     diff = 0
        if k == 1:      # 将第一个时间点到第二个时间点的时间差设为平均值
            diff = average_diff
        else:           # 从第三个时间点开始，时间差为平均值加上相应点到第二个点的时间差
            additional_diff = (time_objects[k] - time_objects[1]).seconds  # 当前时间点到第二个时间点的时间差
            diff = average_diff + additional_diff
        differ_time.append(diff)

    # 转换为 Cupy 数组并输出
    differ_time = cp.array(differ_time)
    print(differ_time)


    # 计算半定量参数
    peak_time = differ_time[1]      # 把第三个时间点设为达峰时间
    peak_time = peak_time / 60
    # washin_diff = cp.where((maxERs - img[:, :, :, 0]) < 0, 0, (maxERs - img[:, :, :, 0]))
    washin_stage = cp.array((maxERs - img[:, :, :, 0]) / img[:, :, :, 0]) / peak_time
    washin_stage[img[:, :, :, 0]==0] = 0
    washin_stage[washin_stage > 3] = 3

    del peak_time
    print('washin_stage', cp.max(washin_stage), cp.min(washin_stage), cp.mean(washin_stage))
    # print('washin_stage array', cp.array(washin_stage))

    washout_stage = cp.array((img[:, :, :, -1] - maxERs) / maxERs)
    washout_stage[maxERs==0]=0
    washout_stage[washout_stage > 3] = 3
    print('washout_stage', cp.max(washout_stage), cp.min(washout_stage), cp.mean(washout_stage))
    # temp = img[maxERlocs]


    denoise = denoise_mask(np_img[:,:,:,2])
    washin_cpu = washin_stage.get()
    washout_cpu = washout_stage.get()
    result_in = [np.array(np.where(denoise[:, :, i] != 0, washin_cpu[:, :, i], 0)) for i in range(denoise.shape[-1])]
    result_out = [np.array(np.where(denoise[:, :, i] != 0, washout_cpu[:, :, i], 0)) for i in range(denoise.shape[-1])]
    result_in = np.array(result_in)
    result_out = np.array(result_out)
    # result=np.dot(denoise,Type19.get())

    # # data_gpu[:,:,i]=Type19
    # plt.imshow(result)
    # plt.show()
    # break
    # print(i)
    del data_gpu,washout_stage,washin_stage
    in_gpu = np.array(result_in)
    in_gpu = np.swapaxes(in_gpu, 0, 1)
    in_gpu = np.swapaxes(in_gpu, 1, 2)

    out_gpu = np.array(result_out)
    out_gpu = np.swapaxes(out_gpu, 0, 1)
    out_gpu = np.swapaxes(out_gpu, 1, 2)

    # plt.imshow(data_gpu[:,:,i])
    # plt.show()
    # print(Type19.max(),Type19.min())

    # 再转回nii
    in_image = nib.Nifti1Image(in_gpu, stage_data_1_affine.affine)
    out_image = nib.Nifti1Image(out_gpu, stage_data_1_affine.affine)

    del result_in, result_out
    c_time = datetime.now()
    ss_time = c_time - s_time
    print(ss_time)
    # break
    nib.save(in_image, save_path_in)
    nib.save(out_image, save_path_out)
    mempol.free_all_blocks()
    pinned_mempool.free_all_blocks()
    # break

# 将列表转换为 pandas DataFrame
df = pd.DataFrame(error_patients, columns=['Patient ID', 'Stage'])

# 将 DataFrame 保存为 CSV 文件
csv_filename = 'error_patients.csv'
df.to_csv(csv_filename, index=False)  # index=False 防止写入行索引

