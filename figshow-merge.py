import numpy as np
import cv2
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorsys
import random
import os
import gc
import builtins

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

# B,G,R 格式排序
# 标签地址
label_csv = r'E:\liuzhou_breastcancer\datas_clear.csv'

# 结果保存地址
pic_save = r'E:\liuzhou_breastcancer\res_pic\result_fig'
in_save = r'E:\liuzhou_breastcancer\res_pic\wash_in'
out_save = r'E:\liuzhou_breastcancer\res_pic\wash_out'
com_save = r'E:\liuzhou_breastcancer\res_pic\combined_pic'

# 选择label 其实同时会对应到label对应的病人 所以要做一下数据预处理 按照label_csv这个文件整理
select_label = 'Untitled.nii.gz'
# 窗宽窗高 这个得用一些查看的软件找到合适的像素值范围，期数一般选取对比度较高的一期作为底图
WL = 0.5 #washin 1 #190 #1700
WW = 2 #washin 2 #380 #3400
#%%
# 根据label读取病人的地址
pd_csv=pd.read_csv(label_csv)
datalist=np.array(pd_csv)
print(datalist)
# type = 'washIn'
# type = 'washOut'

missing_patien = []

sclise=0

try:
    del max  # 如果你之前覆盖了 max
except:
    pass

for datalist0 in range(datalist.shape[0]):
    # if datalist0 == 9:
    #     continue
    prefix = 'Benign' if datalist[datalist0, 3] == 0 else 'Malignant'
    # 创建新的字符串
    max = 0
    new_string = f'{prefix}{datalist[datalist0, 1]}.nii'

    # 提取mask的名称，有.nii和.nii.gz两种后缀，分开处理
    mask_name = datalist[datalist0, 2]
    if mask_name.endswith(".nii.gz"):
        mask_name = datalist[datalist0, 2][:-7]  # .nii.gz 长度是 7
    elif mask_name.endswith(".nii"):
        mask_name = datalist[datalist0, 2][:-4]  # .nii 长度是 4

    # 读图像用
    result_path = os.path.join(com_save,f'{prefix}{datalist[datalist0, 4]}_{mask_name}.png')
    if os.path.exists(result_path):
        print(result_path + ' is exist.')
        continue

    end_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'end.png')
    first_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'first.png')
    peak_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'third.png')
    in_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'washIn.png')
    out_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'washOut.png')
    in_com_path = os.path.join(in_save, f'{prefix}{datalist[datalist0, 1]}_{mask_name}.png')
    out_com_path = os.path.join(out_save, f'{prefix}{datalist[datalist0, 1]}_{mask_name}.png')
        
    # 读取六张图片
    img_paths = [
        first_path, in_path, in_com_path,
        peak_path, out_path, out_com_path
    ]

    imgs = []
    
    # 确保所有图片尺寸相同（可选择 resize）
    for path in img_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"❌ 读取失败：{path}")
                # missing_patien.append([f'{prefix}{datalist[datalist0, 1]}',mask_name])
                # continue
            else:
                print(f"✅ 读取成功：{path}")
            
            print(f"{path}: {img.shape[1]} x {img.shape[0]}")
            imgs.append(img)
        except:
            print(f"{prefix}{datalist[datalist0, 1]} - {mask_name} path is not exist")
    
    if not imgs:
        missing_patien.append([f'{prefix}{datalist[datalist0, 1]}',mask_name])
        continue

    # 检查所有图像高度是否一致
    heights = [img.shape[0] for img in imgs]
    assert all(h == heights[0] for h in heights), "所有图像高度必须一致"

    # 第一行（img1, img2, img3）
    row1 = np.hstack(imgs[0:3])

    # 第二行（img4, img5, img6）
    row2 = np.hstack(imgs[3:6])

    # 找到最大宽度
    max_width = builtins.max(row1.shape[1], row2.shape[1])

    # 统一行宽度（在右边补白）
    def pad_row_to_width(row, target_width):
        height, width = row.shape[:2]
        if width < target_width:
            pad_width = target_width - width
            return cv2.copyMakeBorder(
                row, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        return row

    row1_padded = pad_row_to_width(row1, max_width)
    row2_padded = pad_row_to_width(row2, max_width)

    # 检查 row1 和 row2 宽度是否一致，才能垂直拼接
    assert row1_padded.shape[1] == row2_padded.shape[1], "每行拼接后的总宽度必须一致，才能使用 vstack"
    # 垂直拼接
    final_image = np.vstack((row1_padded, row2_padded))

    # 拼接成最终图像

    # result_path = os.path.join(com_save,f'{prefix}{datalist[datalist0, 1]}_{mask_name}.png')
    # 保存或显示
    cv2.imwrite(result_path, final_image)
    print(f'{prefix}{datalist[datalist0, 1]} - {mask_name} is saved.')
        # 读取原始图像用的
        # 已有原始图的话这段代码可以去掉
        # %%%
        # origin_path = os.path.join(datalist[datalist0,4], r'1\1.nii')
        # origin_data = nib.load(origin_path).get_fdata()
        # peak_path = os.path.join(datalist[datalist0, 4], r'3\3.nii')
        # peak_data = nib.load(peak_path).get_fdata()
        # data1_list = origin_path.split('\\')[:-2]
        # data1_path = os.path.join('\\'.join(i for i in data1_list), '8', '8.nii')

        # if not os.path.exists(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}')):
        #     os.mkdir(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}'))
        # cv2.imwrite(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'first.png'), s_odata,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'third.png'), s_opdata,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'end.png'), s_odata1,
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # %%%
        # 以上是已有原始图像可注释内容

        # %%
        # s_data = np.where(label[:, :, sclise] == 0, 0, s_data)
        # # %%
        # # # 根据type19的结果赋予不同的颜色
        # # mask_color = np.zeros([s_data.shape[0], s_data.shape[1], 3])
        # # for row in range(s_data.shape[0]):
        # #     for column in range(s_data.shape[1]):
        # #         mask_color[row, column, :] = get_color(int(s_data[row, column]))
        # # m_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', str(sclise) + '.png')
        # # cv2.imwrite(m_path, mask_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # # 根据wash-in/out结果赋予不同颜色
        # # 将数据归一化到 0-255
        # normalized_data = cv2.normalize(s_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # 应用 jet 映射
        # jet_mapped_data = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
        # # 创建带有 Alpha 通道的图像 (RGBA)
        # rgba_image = cv2.cvtColor(jet_mapped_data, cv2.COLOR_BGR2BGRA)
        # # 将原始数据为 0 的区域的 alpha 值设置为 0（完全透明）
        # rgba_image[s_data == 0, 3] = 0

        # # 根据图像是wash-in或wash-out注释
        # m_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', mask_name + '_in.png')
        # # m_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', mask_name + '_out.png')
        # cv2.imwrite(m_path, rgba_image)
        # # 只有病灶内是彩色的图像
        # m1_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', mask_name + '_in_com.png')
        # # m1_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', mask_name + '_out_com.png')

        # p_path = os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'third.png')


        # # img1 = cv2.imread(o_path)
        # # img2 = cv2.imread(m_path)
        # # img_label = cv2.imread(l_path)
        # # mask_color = np.array(img1)

        # # 读取背景图和前景图（前景图有透明度 alpha 通道）
        # background = cv2.imread(p_path)
        # foreground = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 读取透明通道

        # # 检查两张图像大小是否一致，不一致则调整大小
        # try:
        #     if background.shape[:2] != foreground.shape[:2]:
        #         foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
        # except AttributeError:
        #     print('error')
        #     missing_patien.append(datalist[datalist0,1])
        #     continue

        # # 分离前景图像的 BGR 和 alpha 通道
        # bgr_foreground = foreground[:, :, :3]  # 获取 BGR 通道
        # alpha_foreground = foreground[:, :, 3] / 255.0  # 获取 alpha 通道并归一化到 0-1

        # # 将 alpha 扩展为三通道，方便应用到 BGR 上
        # alpha_foreground = cv2.merge([alpha_foreground, alpha_foreground, alpha_foreground])

        # # 计算前景和背景的混合结果
        # blended_image = alpha_foreground * bgr_foreground + (1 - alpha_foreground) * background

        # cv2.imwrite(m1_path, blended_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # 使用 matplotlib 生成颜色条并保存为图像
        # fig, ax = plt.subplots(figsize=(1, 5))  # 调整 figsize 以控制颜色条的大小
        # fig.subplots_adjust(left=0.5, right=0.6, top=0.95, bottom=0.05)
        # # 创建一个颜色条
        # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=s_data.min(), vmax=s_data.max())),
        #                     cax=ax)
        # cbar.set_label('Value')  # 颜色条的标签（可选）
        # plt.savefig(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'colorbar_in.png'), bbox_inches='tight', pad_inches=0.1)
        # plt.close()

        # # 读取保存的颜色条图像
        # colorbar = cv2.imread(os.path.join(pic_save, f'{prefix}{datalist[datalist0, 1]}', 'colorbar_in.png'))

        # # 调整颜色条的高度，使其与 jet 映射图像的高度相同
        # colorbar = cv2.resize(colorbar, (int(colorbar.shape[1] * blended_image.shape[0] / colorbar.shape[0]), blended_image.shape[0]))

        # # 拼接颜色条和图像
        # combined_image = np.hstack((blended_image, colorbar))

        # # # 保存拼接后的图像
        # # mkdir(out_save)
        # m2_path = os.path.join(in_save, f'{prefix}{datalist[datalist0, 1]}' + '_' + mask_name + '.png')
        # # m2_path = os.path.join(out_save, f'{prefix}{datalist[datalist0, 1]}' + '_' + mask_name + '.png')
        # cv2.imwrite(m2_path, combined_image)
        # print(m2_path, ' is processed.')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # del colorbar, combined_image, blended_image, rgba_image
        # del s_opdata, s_odata, s_odata1
    gc.collect()


    # print(f"Path does not exist: {file_path}. Skipping to the next step.")
    # missing_patien.append(f'{prefix}{datalist[datalist0, 1]}')

print(missing_patien)

#%%


# 保存归一化后的数据

#%%
# 根据不同类别定义区分度大的颜色
# Type19的彩色可视化函数





