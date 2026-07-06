#加载必要包
import os
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg'，不使用 Qt 相关后端


os.environ["OMP_NUM_THREADS"] = "1"  # 将并行线程数限制为1

import gc
from threadpoolctl import threadpool_limits
import cv2
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import SimpleITK as sitk
from skimage import segmentation
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import hdbscan
import pandas as pd
import warnings
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.stats import zscore

warnings.filterwarnings("ignore", message=".*force_all_finite.*")


# signal处理超时办法，由于signal不适合windows系统故改用threading
def timeout_handler(signum, frame):
    raise TimeoutError("HDBSCAN聚类超时")


def timeout_wrapper(func, timeout):
    """超时装饰器核心逻辑"""
    class InterruptableThread(Thread):
        def __init__(self):
            super().__init__()
            self.result = None
            self.exception = None

        def run(self):
            try:
                self.result = func()
            except Exception as e:
                self.exception = e

    thread = InterruptableThread()
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Windows无法强制终止线程，但主流程可继续
        return None  # 用特殊值触发降级
    elif thread.exception:
        raise thread.exception
    else:
        return thread.result


def assign_noise_to_clusters(features, labels):
    """ 将噪声点（标签-1）分配到最近的簇 """
    if not np.any(labels == -1):
        return labels
    
    non_noise_mask = (labels != -1)
    if non_noise_mask.sum() == 0:  # 全噪声，无法处理，需依赖后续重试
        return labels
    
    # 仅当存在有效簇时分配噪声
    from sklearn.neighbors import NearestNeighbors
    
    # 提取非噪声特征和标签
    non_noise_features = features[non_noise_mask]
    non_noise_labels = labels[non_noise_mask]
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(non_noise_features)
    noise_features = features[~non_noise_mask]
    
    if noise_features.shape[0] > 0:
        _, indices = nbrs.kneighbors(noise_features)
        labels[~non_noise_mask] = non_noise_labels[indices.flatten()]  # 替换为最近簇的标签
    
    return labels

def visualize_largest_cluster_slice(cluster_nifti_path, mask_nifti_path, save_png_path):
    """
    可视化聚类的最大病灶slice，并按RGB颜色标注三类
    Args:
        cluster_nifti_path (str): 聚类结果nii路径（标签值 1~3）
        mask_nifti_path (str): 掩码nii路径（非零为有效区域）
        save_png_path (str): 保存PNG的路径
    """

    # 读取图像和掩码
    cluster_img = sitk.ReadImage(cluster_nifti_path)
    cluster_array = sitk.GetArrayFromImage(cluster_img)  # [z,y,x]

    mask_img = sitk.ReadImage(mask_nifti_path)
    mask_array = sitk.GetArrayFromImage(mask_img)

    if cluster_array.shape != mask_array.shape:
        raise ValueError("聚类图像和掩码尺寸不一致")

    # 找到掩码中病灶最大的slice（z方向）
    slice_sums = np.sum(mask_array > 0, axis=(1, 2))
    max_slice_idx = np.argmax(slice_sums)

    # 提取该层图像
    cluster_slice = cluster_array[max_slice_idx]
    mask_slice = mask_array[max_slice_idx]

    # 创建RGB图像
    rgb = np.zeros((cluster_slice.shape[0], cluster_slice.shape[1], 3), dtype=np.uint8)

    # 标注每一类（只在掩码区域内）
    for i in range(1, 4):
        class_mask = (cluster_slice == i) & (mask_slice > 0)
        if i == 1:
            rgb[class_mask] = [255, 0, 0]  # 红色
        elif i == 2:
            rgb[class_mask] = [0, 255, 0]  # 绿色
        elif i == 3:
            rgb[class_mask] = [0, 0, 255]  # 蓝色

    # 可选：叠加mask边缘（白色轮廓）
    contours = cv2.findContours((mask_slice > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(rgb, contours, -1, (255, 255, 255), 1)

    # 保存图像
    plt.imsave(save_png_path, rgb)
    print(f"已保存PNG图像至：{save_png_path}")




def plot_evaluation_curves(img_name, mask_name, k_values, ch_scores, silhouettes, composite, best_k):
    """三合一评估可视化函数"""
    plt.figure(figsize=(14, 6))
    
    # -- CH值子图 --
    plt.subplot(1, 3, 1)
    plt.plot(k_values, ch_scores, 'b-o')
    plt.axvline(best_k, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title(f'CH Peak at K={best_k}')
    plt.grid(True)

    # -- 轮廓系数子图 --
    plt.subplot(1, 3, 2)
    plt.plot(k_values, silhouettes, 'g-s')
    plt.axvline(best_k, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Trend')
    plt.grid(True)

    # -- 复合评分子图 --
    plt.subplot(3, 3, 3)
    plt.plot(k_values, composite, 'm-*')
    plt.scatter(best_k, np.max(composite), c='red', s=200, edgecolors='k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Composite Score')
    plt.title(f'Optimal K={best_k} (Composite)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path,img_name + '_' + mask_name + '.png'), dpi=300)

def remap_labels_by_washin_washout(labels, in_values, out_values, alpha=0.45, beta=0.55):
    """
    根据每个聚类的平均 wash-in / wash-out 计算综合得分，并将标签按恶性度排序重映射。
    labels: 聚类后的原始标签
    in_values: wash-in 特征数组
    out_values: wash-out 特征数组
    alpha, beta: 权重参数
    """
    unique_labels = np.unique(labels)
    scores = []
    for label in unique_labels:
        idx = (labels == label)
        mean_in = in_values[idx].mean()
        mean_out = out_values[idx].mean()
        score = alpha * mean_in - beta * mean_out  # wash-out 越小(负值)越恶性
        scores.append(score)
    
    # 排序：得分低 → 标签0 (最良性)，得分高 → 标签2 (最恶性)
    sorted_idx = np.argsort(scores)
    mapping = {old: new for new, old in zip(unique_labels[sorted_idx], range(len(unique_labels)))}
    
    remapped_labels = np.array([mapping[l] for l in labels])
    return remapped_labels, mapping

def slic_supervoxel(img_name, mask_name, image_path1, image_path2, mask_path, smooth_sigma=1.0):
    """
    使用 K-means 聚类进行超体素分割。结合 SimpleITK 和 KMeans 聚类对 3D 图像进行处理，
    仅对 mask 为 True 的区域进行聚类，并将结果保存为 NIfTI 文件。
    Args:
        image_path (str): 输入 3D 图像的路径，支持 NIfTI 格式 (如 .nii.gz)。
        mask (numpy.ndarray or None): 二值掩码图像，指定只对掩码区域进行聚类，默认 None 表示对整个图像进行处理。
        n_clusters (int): 超体素的数量（即聚类数量），默认 100。
        smooth_sigma (float): 高斯平滑的标准差，默认 1.0。
        output_path (str or None): 输出文件的路径，保存为 .nii.gz 格式。如果为 None，则不保存。
    Returns:
        np.ndarray: 超体素标签图像。
    """
    # 读取图像
    image1 = sitk.ReadImage(image_path1)        # wash in image
    image2 = sitk.ReadImage(image_path2)        # wash out image
    
    # 预处理图像：高斯平滑
    smoothed_image_in = sitk.SmoothingRecursiveGaussian(image1, sigma=smooth_sigma)
    smoothed_image_out = sitk.SmoothingRecursiveGaussian(image2, sigma=smooth_sigma)
    
    # 转换为 numpy 数组，方便处理
    in_array = sitk.GetArrayFromImage(smoothed_image_in)
    out_array = sitk.GetArrayFromImage(smoothed_image_out)
    segmented_image = np.zeros(out_array.shape, dtype=np.int32)
    
    # 如果提供了 mask，则只对 mask 为 True 的区域进行聚类

    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    if image2.GetSize() != mask.GetSize()  or image1.GetSize() != mask.GetSize():
        print(f"尺寸不匹配：wash-out 图像尺寸 {image2.GetSize()} or , 掩码图像尺寸 {mask.GetSize()}，跳过该样本")
        return None  # 或引发自定义异常

    # ====== 新增掩码有效性检查 ======
    if not np.any(mask_array):
        print(f"错误：掩码 {mask_path} 完全为空 → 跳过处理")
        return None  # 中止函数并返回空值

    
    # in_array = in_array * mask_array  # 只保留掩码区域
    # out_array = out_array * mask_array
    
    # 提取特征：体素的灰度值和空间坐标
    # z, y, x = np.indices(image_array.shape)
    # pixel_values = image_array.flatten()
    # spatial_coordinates = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    # z_coords, y_coords, x_coords = np.where(mask_array > 0)  # ⭐ 筛选非零点，假设值大于0有效
    # in_values = in_array[z_coords, y_coords, x_coords]  # 直接筛选，避免展开整个数组

    # out_values = out_array[z_coords, y_coords, x_coords] 
    # washout_values = out_values.reshape(-1,1)

    # --- 合并特征 ---
    # 提取坐标位置
    z_coords, y_coords, x_coords = np.where(mask_array > 0)
    if len(z_coords) == 0:
        print("No valid voxels found in mask → Skip")
        return None

    # 提取 wash-in / wash-out 原始值
    in_values = in_array[z_coords, y_coords, x_coords]
    out_values = out_array[z_coords, y_coords, x_coords]

    # 派生强度特征
    delta    = out_values - in_values
    ratio    = out_values / (in_values + 1)
    log_in   = np.log1p(in_values) #( log(wash-in + 1) )
    out_tanh = np.tanh(out_values) # tanh(wash-out)

    # === 局部均值特征 ===

    # # 3x3 平面局部均值（对每个 slice 单独算）
    # local2d_in = uniform_filter(in_array, size=(1,3,3), mode='nearest')
    # local2d_out = uniform_filter(out_array, size=(1,3,3), mode='nearest')
    # local2d_in_vals = local2d_in[z_coords, y_coords, x_coords]
    # local2d_out_vals = local2d_out[z_coords, y_coords, x_coords]

    # 3x3x3 空间局部均值
    local3d_in = uniform_filter(in_array, size=3, mode='nearest')
    local3d_out = uniform_filter(out_array, size=3, mode='nearest')
    local3d_in_vals = local3d_in[z_coords, y_coords, x_coords]
    local3d_out_vals = local3d_out[z_coords, y_coords, x_coords]

    # === 合并所有特征 ===
    features = np.column_stack([
        in_values,
        out_values,
        delta,
        ratio,
        log_in,
        out_tanh,
        local3d_in_vals,
        local3d_out_vals
    ])


    # Kmeans 聚类
    # KMeans 聚类 - 固定为3类
    try:
        fixed_k = 3
        final_kmeans = MiniBatchKMeans(
            n_clusters=fixed_k,
            batch_size=1024,
            n_init='auto',
            init="k-means++",
            random_state=3407
        )
        labels = final_kmeans.fit_predict(features)

    except Exception as e:
        print(f"异常降级处理: {str(e)}")
        labels = np.zeros(features.shape[0], dtype=int)

    labels, mapping = remap_labels_by_washin_washout(labels, in_values, out_values)
    print(f"标签重映射关系: {mapping}")

    # 获取非噪声点的坐标和标签
    if np.any(labels == -1):
        non_noise_mask = (labels != -1)
        
        # 确保非噪声点存在（否则无法找邻居）
        if non_noise_mask.sum() > 0:  
            labels = assign_noise_to_clusters(features, labels)  # 处理噪声
            # else: 无噪声点需要处理（可能已被其他条件过滤）
        else:
            # 极端情况：所有点都被聚类器标记为噪声（例如min_cluster_size过大）
            is_all_noise = non_noise_mask.sum()==0
            if is_all_noise:
                print("全噪声警告: 启动参数降级重试...")
                return None


    # 生成超体素标签图像
    if mask is not None:
        segmented_image[mask_array > 0] = labels + 1  # 对掩码区域赋标签，避免标签从 0 开始
    else:
        segmented_image = labels.reshape(mask_array.shape)  # 将标签重塑为图像形状
    
    del image2, mask, labels
    gc.collect()  # 异常时尝试回收
    # 保存结果
    return segmented_image


#加载数据
# 使用示例

# 调用函数
# segmented_image = slic_supervoxel(image_path, mask=mask, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)


# image_path = r"E:\liuzhou_breastcancer\figure-res-0\washIn\Benign1.nii" #替换为你的图像路径
# mask_path= r'E:\DCESummary_2019-202004\benign-DCE-221subs\part1-175subs+part2\1\Untitled.nii.gz' #替换为你的掩码路径
datapath = r"E:\liuzhou_breastcancer\datas_clear.csv"
output_dir = r"E:\liuzhou_breastcancer\habitat_test"
fig_path = r"E:\liuzhou_breastcancer\class_rank_in"
modpath = r"E:\liuzhou_breastcancer\modified_datas.csv"

all_intensities = []
datalist = pd.read_csv(datapath)
mod = pd.read_csv(modpath)

datalist = pd.merge(datalist, mod, on=['patient_name','Patient_ID','label_name','grade'], how='inner')

error_records = []

silhouette = []

# for datalist0 in range(datalist.shape[0]):
#     folder_path = r'E:\liuzhou_breastcancer\figure-res-0' if datalist.loc[datalist0, 'grade'] == 0 else r'E:\liuzhou_breastcancer\figure-res-1'
#     grade = 'Benign' if datalist.loc[datalist0, 'grade'] == 0 else 'Malignant'

#     img_name = grade + str(datalist.loc[datalist0, 'patient_name']) + '.nii'

#     #根据wash-in和wash-out注释
#     img_path = os.path.join(folder_path, r'washIn/' + img_name)
#     # img_path = os.path.join(folder_path, r'washOut/' + img_name)

#     if not os.path.exists(img_path):
#         continue

#     img = sitk.ReadImage(img_path)  
#     mask = sitk.ReadImage(datalist.iloc[datalist0, 6])
        
#     # 获取mask内的强度值
#     img_array = sitk.GetArrayFromImage(img)  # shape (z,y,x)
#     mask_array = sitk.GetArrayFromImage(mask).astype(bool)
    
    
#     if img_array.shape != mask_array.shape:
#         error_msg = f"Shape mismatch: image {img_array.shape}; mask {mask_array.shape}"
#         # 记录不匹配的图像名称和错误信息
#         error_records.append({
#             'image': img_path, 
#             'mask': mask_path, 
#             'error': error_msg
#         })
#         print(f"Skipping {img_path}: {error_msg}")
#         continue  # 跳过当前循环
#     roi_intensities = img_array[mask_array]  # 获取mask区域体素

#     all_intensities.append(roi_intensities.flatten())
#     print(f'{img_name}' + ' is collected.')

# global_thresholds = train_global_model(all_intensities)

if error_records:
    pd.DataFrame(error_records).to_csv("shape_mismatch_errors.csv", index=False)
    print("保存错误文件到 shape_mismatch_errors.csv")


os.makedirs(r"E:\liuzhou_breastcancer\habitat_vis", exist_ok=True)

for datalist0 in range(datalist.shape[0]):
        # 读取数据

    if datalist.loc[datalist0, 'grade'] == 0 :
        folder_path = r'E:\liuzhou_breastcancer\figure-res-0'
        grade = 'Benign'
    else : 
        folder_path = r'E:\liuzhou_breastcancer\figure-res-1'
        grade = 'Malignant'

    img_name = grade + str(datalist.loc[datalist0, 'patient_name'])

    mask_name = datalist.iloc[datalist0, 2]
    if mask_name.endswith(".nii.gz"):
        mask_name = datalist.iloc[datalist0, 2][:-7]  # .nii.gz 长度是 7
    elif mask_name.endswith(".nii"):
        mask_name = datalist.iloc[datalist0, 2][:-4]  # .nii 长度是 4

    print('------------------ ' + f'{img_name} + {mask_name}' +' is processing. ------------------')

    #根据wash-in和wash-out注释
    img_path1 = os.path.join(folder_path, r'washIn/' + img_name + '.nii')
    img_path2 = os.path.join(folder_path, r'washOut/' + img_name + '.nii')
    mask_path = datalist.iloc[datalist0, 7]
    # img_path = os.path.join(folder_path, r'washOut/' + img_name)

    if not os.path.exists(img_path1) or not os.path.exists(img_path2):
    # 更详细的错误提示（推荐）：
        print(f"文件缺失：{img_path1} → 跳过处理")
        continue  # 直接进入下一个循环迭代

    # mask = sitk.ReadImage(datalist.iloc[datalist0, 6])
    # img_array = sitk.GetArrayFromImage(img)
    # mask_array = sitk.GetArrayFromImage(mask).astype(bool)
        
    # 应用聚类       
    cluster_img = slic_supervoxel(img_name, mask_name, img_path1, img_path2, mask_path, smooth_sigma=1.0)
    
    if cluster_img is None:
        print(f"处理 {img_name} 时发生错误，跳过保存")
        continue

        # 保存结果
    output_img = sitk.GetImageFromArray(cluster_img)
    # output_img.CopyInformation(img)  # 保持空间属性
    sitk.WriteImage(output_img, f"{output_dir}/{img_name}_{mask_name}.nii")
    print(f'{img_name} - {mask_name}' + ' is saved.')

        # ============ 可视化处理 ============

    # 创建输出目录

    # 读取掩码
    mask_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)

    # 找出掩码中病灶最大的 slice（z轴）
    slice_sums = np.sum(mask_array > 0, axis=(1, 2))
    max_slice_idx = np.argmax(slice_sums)

    cluster_slice = cluster_img[max_slice_idx]
    mask_slice = mask_array[max_slice_idx]

    # 创建RGB图像
    rgb = np.zeros((*cluster_slice.shape, 3), dtype=np.uint8)
    for i in range(1, 4):  # 聚类标签值：1, 2, 3
        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)][i - 1]
        mask = (cluster_slice == i) & (mask_slice > 0)
        rgb[mask] = color

    # # 添加掩码轮廓（白色）
    # contours = cv2.findContours((mask_slice > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # cv2.drawContours(rgb, contours, -1, (255, 255, 255), 1)

    # 保存PNG图像
    vis_output_path = os.path.join(r"E:\liuzhou_breastcancer\habitat_vis", f"{img_name}_{mask_name}.png")
    plt.imsave(vis_output_path, rgb)
    print(f"PNG 可视化图已保存至：{vis_output_path}")




# slic_supervoxel(image_path, mask_path, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)

