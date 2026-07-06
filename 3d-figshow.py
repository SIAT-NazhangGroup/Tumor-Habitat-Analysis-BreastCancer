# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
import matplotlib.colors as mcolors

# --- 加载数据 & 预处理 ---
img = nib.load(r"<PROJECT_ROOT>\habitat_out\Malignant114_Untitled.nii")
seg_data = img.get_fdata().astype(int)
zooms = img.header.get_zooms()  # 获取真实体素间隔

def normalize_color(r, g, b, a=1.0):
    return (r/255, g/255, b/255, a)

# 定义颜色 (按标签)
colors = {
    0: (0.0, 0.0, 0.0, 0.0),   # 背景透明
    1: normalize_color(255, 0, 0),    # 红色
    2:normalize_color(0, 255, 0),    # 绿色
    3:normalize_color(0, 0, 255),    # 蓝色
    4:normalize_color(255, 165, 0),  # 橙色
    5:normalize_color(75, 0, 130),   # 紫色
    6:normalize_color(255, 255, 0),   # 黄色
    7:normalize_color(0, 255, 255),  # 青色
    8:normalize_color(255, 20, 147), # 深粉色
    9:normalize_color(0, 128, 128)   # 绿色蓝色
}

# --- 创建3D画布 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 遍历每个标签，生成网格并渲染
for label in np.unique(seg_data):
    if label == 0:
        continue  # 跳过背景
    
    # 生成三维网格表面
    mask = (seg_data == label).astype(float)
    verts, faces, _, _ = marching_cubes(mask, spacing=zooms, level=0.5)
    
    # 调整表面坐标真实间距
    verts *= zooms
    
    # 添加面片到坐标系
    mesh = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color=colors[label][:9],
        alpha=colors[label][3],
        edgecolor='none'
    )

# --- 调整视角和坐标轴 ---
ax.view_init(elev=20, azim=0)  # Right 45°视角 (Elev俯仰角，Azim方位角)
ax.set_xlabel('X (voxel)', fontsize=10)
ax.set_ylabel('Y (voxel)', fontsize=10)
ax.set_zlabel('Z (voxel)', fontsize=10)

# 添加色标示例
for label in colors:
    if label == 0:
        continue
    ax.scatter([], [], [], color=colors[label], label=f'Label {label}')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# save figure
plt.savefig('output.png')  
