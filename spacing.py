# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import os
import nibabel as nib
import numpy as np
import pandas as pd

def analyze_spacing(data_dirs, center_names):
    all_spacing_data = []

    for data_dir, center in zip(data_dirs, center_names):
        for fname in os.listdir(data_dir):
            if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                try:
                    img = nib.load(os.path.join(data_dir, fname))
                    spacing = img.header.get_zooms()  # (x, y, z)
                    print(f'{fname} : {spacing}')
                    all_spacing_data.append({
                        'center': center,
                        'file': fname,
                        'spacing_x': spacing[0],
                        'spacing_y': spacing[1],
                        'spacing_z': spacing[2],
                    })
                except Exception as e:
                    print(f"Failed to read {fname}: {e}")

    df = pd.DataFrame(all_spacing_data)
    print("\n📊 全部图像 spacing 概况统计：\n")
    print(df.groupby('center')[['spacing_x', 'spacing_y', 'spacing_z']].describe())

    return df

# 示例用法
data_dirs = [r'<PROJECT_ROOT>\figure-res-1\washIn', r'<NEW_ROOT>\figure-res-1\washIn']  # 替换成你的文件夹路径
center_names = ['Center_SZ', 'Center_GZ']

spacing_df = analyze_spacing(data_dirs, center_names)

# 可选：保存为 CSV 文件方便查看
spacing_df.to_csv("spacing_statistics_malignant.csv", index=False)
