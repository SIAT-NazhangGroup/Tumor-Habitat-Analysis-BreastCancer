# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
import cupy as cp
import cv2
import pandas as pd
import nibabel as nib
import colorsys
import random
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

# 标签地址
label_csv = r'<REDACTED_PATH>'' #原本这个文件是<PROJECT_ROOT>\data_path.csv

# 结果保存地址
pic_save = r'<PROJECT_ROOT>\res_pic\result_fig1'

file_path = r'<PROJECT_ROOT>\figure-res-1\washIn\Malignant1.nii'
label_path = r'<DCM_ROOT>\malignant-DCE-451subs\part1-288subs+part2\1\Untitled.nii.gz'
label = nib.load(label_path).get_fdata()
data_all = nib.load(file_path).get_fdata()
data = data_all[:, :, 144]
data = np.where(label[:, :, 144] == 0, 0, data)

# # 假设数据范围在 0 到 3 之间
# data = np.random.uniform(cp.min(data), 3, (100, 100))

# 将数据归一化到 0-255
normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 应用 jet 映射
jet_mapped_data = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

# 创建带有 Alpha 通道的图像 (RGBA)
rgba_image = cv2.cvtColor(jet_mapped_data, cv2.COLOR_BGR2BGRA)

# 将原始数据为 0 的区域的 alpha 值设置为 0（完全透明）
rgba_image[data == 0, 3] = 0

# 保存结果为 PNG（支持透明度）
cv2.imwrite('malignant_1.png', rgba_image)

print("归一化后图像已保存为 'malignant_1.png'")


# # 读取背景图和前景图（前景图有透明度 alpha 通道）
# background = cv2.imread(r'<PROJECT_ROOT>\res_pic\result_fig1\Benignsz_38440\data.png')
# foreground = cv2.imread('sz_38440_.png', cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取透明通道
#
# # 检查两张图像大小是否一致，不一致则调整大小
# if background.shape[:2] != foreground.shape[:2]:
#     foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
#
# # 分离前景图像的 BGR 和 alpha 通道
# bgr_foreground = foreground[:, :, :3]  # 获取 BGR 通道
# alpha_foreground = foreground[:, :, 3] / 255.0  # 获取 alpha 通道并归一化到 0-1
#
# # 将 alpha 扩展为三通道，方便应用到 BGR 上
# alpha_foreground = cv2.merge([alpha_foreground, alpha_foreground, alpha_foreground])
#
#
# # 计算前景和背景的混合结果
# blended_image = alpha_foreground * bgr_foreground + (1 - alpha_foreground) * background
#
# # 保存并显示结果
# cv2.imwrite('sz_38440_combined11.png', blended_image.astype(np.uint8))
# # cv2.imshow('Blended Image with Alpha', blended_image.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 使用 matplotlib 生成颜色条并保存为图像
fig, ax = plt.subplots(figsize=(1, 5))  # 调整 figsize 以控制颜色条的大小
fig.subplots_adjust(left=0.5, right=0.6, top=0.95, bottom=0.05)

# 创建一个颜色条
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=data.min(), vmax=data.max())), cax=ax)
cbar.set_label('Value')  # 颜色条的标签（可选）
plt.savefig('colorbar_.png', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 读取保存的颜色条图像
colorbar = cv2.imread('colorbar_.png')

# 调整颜色条的高度，使其与 jet 映射图像的高度相同
colorbar = cv2.resize(colorbar, (colorbar.shape[1], rgba_image.shape[0]))

# 确保 colorbar 的通道数为 4
if colorbar.shape[2] == 3:
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2BGRA)

# 拼接颜色条和图像
combined_image = np.hstack((rgba_image, colorbar))

# 显示拼接后的图像
cv2.imshow('Jet Image with Colorbar', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 保存拼接后的图像
cv2.imwrite('malignant1_with_colorbar11.png', combined_image)