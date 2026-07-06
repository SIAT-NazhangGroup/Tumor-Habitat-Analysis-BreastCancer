# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import numpy as np
import cv2
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import colorsys
import random
import os
import gc



# peak_path = r"<PROJECT_ROOT>\Malignant_nii\P005_NOR\3\3.nii"
# peak_data = nib.load(peak_path).get_fdata()

pic_save = r'<REDACTED_PATH>''
# label_path = r'<DCM_ROOT>\malignant-DCE-451subs\part1-288subs+part2\100\Untitled.nii.gz'
label_path = r'<DCM_ROOT>\benign-DCE-221subs\part1-175subs+part2\12\Untitled.nii.gz'
label = nib.load(label_path).get_fdata()
file_path = (r"<PROJECT_ROOT>\figure-res-0\washOut\Benign12.nii")
data = nib.load(file_path).get_fdata()

WL = 1700
WW = 3400
max = 0
sclise=0

for i in range(label.shape[2]):
    i_size = len(label[:, :, i][label[:, :, i] != 0])
    if i_size > max:
        max = i_size
        sclise = i



min = WL - WW / 2
max = WL + WW / 2
s_data = data[:, :, sclise]
s_data = np.where(label[:, :, sclise] == 0, 0, s_data)
# 将数据归一化到 0-255

min_value = -0.5  # 设置你希望的最小值
max_value = 1  # 设置你希望的最大值
# # 如果数据的最大值和最小值不是你期望的范围，先进行剪切
s_data[s_data>max_value] = max_value
s_data[s_data<min_value] = min_value
# normalized_data = np.clip(s_data, min_value, max_value)
# print(normalized_data.min(), normalized_data.max())
# # normalized_data = cv2.normalize(normalized_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# 归一化 s_data 到 [0, 255] 之间，使用数据范围 [-0.1, 1.2]
# 然后将归一化后的数据线性映射到 [0, 255]
normalized_data = (s_data - min_value) / (max_value-min_value) * 255
normalized_data = normalized_data.astype(np.uint8)  # 转换为 uint8 类型

# 应用 jet 映射
jet_mapped_data = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
# 创建带有 Alpha 通道的图像 (RGBA)
rgba_image = cv2.cvtColor(jet_mapped_data, cv2.COLOR_BGR2BGRA)
# 将原始数据为 0 的区域的 alpha 值设置为 0（完全透明）
rgba_image[s_data == 0, 3] = 0

m_path = os.path.join(pic_save, 'B12_-05-1.png')
m1_path = os.path.join(pic_save, 'B12_-05-1_com.png')
cv2.imwrite(m_path, rgba_image)

# 读取背景图和前景图（前景图有透明度 alpha 通道）
background = cv2.imread(r'<PROJECT_ROOT>\res_pic\result_fig\Benign12\third.png')
foreground = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取透明通道

# 检查两张图像大小是否一致，不一致则调整大小
if background.shape[:2] != foreground.shape[:2]:
    foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

# 分离前景图像的 BGR 和 alpha 通道
bgr_foreground = foreground[:, :, :3]  # 获取 BGR 通道
alpha_foreground = foreground[:, :, 3] / 255.0  # 获取 alpha 通道并归一化到 0-1

# 将 alpha 扩展为三通道，方便应用到 BGR 上
alpha_foreground = cv2.merge([alpha_foreground, alpha_foreground, alpha_foreground])

# 计算前景和背景的混合结果
blended_image = alpha_foreground * bgr_foreground + (1 - alpha_foreground) * background

cv2.imwrite(m1_path, blended_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

fig, ax = plt.subplots(figsize=(3, 6))  # 调整 figsize 以控制颜色条的大小
fig.subplots_adjust(left=0.5, right=0.6, top=0.95, bottom=0.05)
# 创建一个颜色条
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-0.5, vmax=1.5)),
                            cax=ax)
cbar.set_label('')  # 颜色条的标签（可选）
cbar.ax.tick_params(labelsize=22)  # 调整刻度字体大小
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.savefig(os.path.join(pic_save, 'M_-05_1.png'), bbox_inches='tight', pad_inches=0.1)
plt.close()

# 读取保存的颜色条图像
# colorbar = cv2.imread(os.path.join(pic_save, 'temp.png'))




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import ttest_ind
#
# # 设置随机种子
# np.random.seed(42)
#
# # 生成虚拟数据
# n_samples = 200
# n_features = 4
# X_0 = np.random.normal(loc=0.0, scale=1.0, size=(n_samples//2, n_features))
# X_1 = np.random.normal(loc=1.0, scale=1.0, size=(n_samples//2, n_features))
# X = np.vstack((X_0, X_1))
# y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
# columns = ['diagnostics_Image-original_Mean', 'wavelet-HLL_firstorder_Skewness', 'original_shape_Sphericity', 'wavelet-HHL_glcm_Id']
# columns1 = ['Benign', 'Malignant']
# X_train = pd.DataFrame(X, columns=columns)
# y_train = pd.Series(y)
#
# # T 检验特征筛选
# selected_features = []
# for column in X_train.columns:
#     t_stat, p_value = ttest_ind(X_train[y_train == 0][column], X_train[y_train == 1][column])
#     if p_value < 0.05:
#         selected_features.append(column)
#
# print("筛选后显著特征:", selected_features)
#
# sns.set_context("notebook", font_scale=1.5)  # 1.5 倍字体
# # 可视化
# plt.figure(figsize=(12, 11))
# for i, feature in enumerate(selected_features):
#     plt.subplot(2, len(selected_features) // 2, i + 1)
#     sns.boxplot(x=y_train, y=X_train[feature], palette='Set2')
#     plt.ylabel('')
#     # plt.xlabel(fontsize=16)
#     plt.title(feature)
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib as mpl
#
# # 创建一个假数据来展示 colorbar
# data = np.linspace(-0.5, 3, 100).reshape(10, 10)
#
# # 创建一个新的 figure 和 axis
# fig, ax = plt.subplots(figsize=(12, 12))  # 设置长条形的 figure
#
# # 创建一个 colormap（'jet'）并使用 Normalize 来设置 colorbar 范围
# cmap = mpl.cm.jet
# norm = mpl.colors.Normalize(vmin=-0.75, vmax=2)
#
# # 创建 colorbar
# cbar = ax.imshow(data, cmap=cmap, norm=norm)
#
# # 添加竖直方向的 colorbar
# cbar_obj = fig.colorbar(cbar, ax=ax, orientation='vertical', pad=0.05)
#
# # 增大 colorbar 字体大小
# cbar_obj.ax.tick_params(labelsize=26)  # 调整刻度字体大小
# cbar_obj.set_label(' ', fontsize=26, fontweight='bold')  # 设置 colorbar 标签的字体大小
#
# # 设置字体为加粗
# for label in cbar_obj .ax.get_yticklabels():
#     label.set_fontweight('bold')
# # 显示图像
# plt.show()