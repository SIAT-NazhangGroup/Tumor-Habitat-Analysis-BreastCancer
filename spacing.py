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
data_dirs = [r'E:\liuzhou_breastcancer\figure-res-1\washIn', r'E:\breastcancer_new\figure-res-1\washIn']  # 替换成你的文件夹路径
center_names = ['Center_SZ', 'Center_GZ']

spacing_df = analyze_spacing(data_dirs, center_names)

# 可选：保存为 CSV 文件方便查看
spacing_df.to_csv("spacing_statistics_malignant.csv", index=False)
