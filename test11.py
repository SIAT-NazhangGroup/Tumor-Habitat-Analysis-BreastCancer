import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 假设你的 CSV 文件路径
csv_file_path = r'E:\liuzhou_breastcancer\set\res\Benign_1_Untitled.nii.gz.csv'
pic_save = r'E:\liuzhou_breastcancer\res_pic\result_fig'

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

# 假设 CSV 文件有 'x', 'y', 'z' 和 'value' 列
# 将数据转换为三维数组
x_coords = df['rows'].values.astype(int)
y_coords = df['columns'].values.astype(int)
z_coords = df['slices'].values.astype(int)
values = df['wash-in'].values

# 找到数组的最大尺寸
x_max = (x_coords.max() + 1).astype(int)
y_max = (y_coords.max() + 1).astype(int)
z_max = (z_coords.max() + 1).astype(int)

# 创建三维数组来存储值
data_array = np.full((x_max, y_max, z_max), np.nan)

# 将值填充到三维数组中
data_array[x_coords, y_coords, z_coords] = values

# 归一化数据到 0 到 1 之间

# 筛选出包含最多有效数值的切片
max_valid_pixels = 0
slice_index = 0

for i in range(z_max):
    valid_pixels = np.count_nonzero(~np.isnan(data_array[:, :, i]))
    if valid_pixels > max_valid_pixels:
        max_valid_pixels = valid_pixels
        slice_index = i

# 选择最佳切片进行可视化
data_slice = data_array[:, :, slice_index]

# 将NaN值设为0以显示为黑色
data_slice = np.nan_to_num(data_slice, nan=0.0)

# 创建图像
plt.figure(figsize=(256/100, 256/100), dpi=100)  # 设置图像大小为256x256像素
plt.imshow(data_slice, cmap='Blues', origin='lower')
plt.axis('off')  # 关闭坐标轴
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整图像填充
plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0, dpi=100)  # 保存图像
plt.show()
