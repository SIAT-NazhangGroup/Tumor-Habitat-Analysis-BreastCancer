# 路径占位符说明（运行前请全局替换）：
#   <PROJECT_ROOT>  -> 原数据/中间结果根目录（如 wash-in/out 图、habitat 输出）
#   <NEW_ROOT>      -> 原二期数据根目录
#   <DCM_ROOT>      -> 原 DICOM 原始数据根目录
#   <FIG_ROOT>      -> 原图表输出根目录
#   <REDACTED_PATH> -> 已脱敏的零散绝对路径，请按需替换
#加载必要包
import os
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
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import hdbscan
import pandas as pd
import warnings
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor

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


def slic_supervoxel(image_path1, image_path2, mask_path, smooth_sigma=1.0):
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
    segmented_image = np.zeros(in_array.shape, dtype=np.int32)
    
    # 如果提供了 mask，则只对 mask 为 True 的区域进行聚类

    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    if image1.GetSize() != mask.GetSize() or image2.GetSize() != mask.GetSize():
        print(f"尺寸不匹配：wash-in 图像尺寸 {image1.GetSize()} or wash-out 图像尺寸 {image2.GetSize()}, 掩码图像尺寸 {mask.GetSize()}，跳过该样本")
        return None  # 或引发自定义异常

    # ====== 新增掩码有效性检查 ======
    if not np.any(mask_array):
        print(f"错误：掩码 {mask_path} 完全为空 → 跳过处理")
        return None  # 中止函数并返回空值

    
    in_array = in_array * mask_array  # 只保留掩码区域
    out_array = out_array * mask_array
    
    # 提取特征：体素的灰度值和空间坐标
    # z, y, x = np.indices(image_array.shape)
    # pixel_values = image_array.flatten()
    # spatial_coordinates = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    z_coords, y_coords, x_coords = np.where(mask_array > 0)  # ⭐ 筛选非零点，假设值大于0有效
    in_values = in_array[z_coords, y_coords, x_coords]  # 直接筛选，避免展开整个数组
    out_values = out_array[z_coords, y_coords, x_coords] 

    # --- 合并特征 ---
    spatial_coords = np.column_stack((x_coords, y_coords, z_coords))  # 形状 (N_nonzero, 3), dtype=int
    if spatial_coords.shape[0] == 0:
        print(f"No valid voxels found in mask → Skip")
        return None
    spacing = image1.GetSpacing() 
    physical_coords = spatial_coords.astype(float) * spacing

    # 缩放坐标以减少权重（例如缩小10倍）
    scaler = MinMaxScaler(feature_range=(0, 1))
    spatial_coords = scaler.fit_transform(physical_coords)
    scaling_factor = [0.01, 0.01, 0.02]
    spatial_coords = spatial_coords * scaling_factor
    features = np.column_stack((spatial_coords, in_values.reshape(-1,1), out_values.reshape(-1,1)))  # 最终形状 (N_nonzero, 4)
    # features = np.concatenate([spatial_coordinates, pixel_values[:, np.newaxis]], axis=-1)


    # hdbscan聚类
    # HDBSCAN 聚类
    if features.shape[0] < 1000:
        clusters_size = max(8, int(features.shape[0]**0.3))  # 对小数区友好
    else:
        clusters_size = int(features.shape[0] / 50)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size= clusters_size,
        min_samples = max(2, int(clusters_size * 0.125)),  # 防止为0
        cluster_selection_method="eom",  # "eom"（默认）或 "leaf"
        cluster_selection_epsilon=0.01,
        allow_single_cluster=False
    )

    # 打印当前线程配置
    print("OMP线程数:", os.environ.get("OMP_NUM_THREADS"))
    print("MKL线程数:", os.environ.get("MKL_NUM_THREADS"))  
    with threadpool_limits(limits=1):  # 关键线程控制

        # 使用带自毁保护的线程池
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(clusterer.fit, features)

        try:
            labels = future.result(timeout=300).labels_
        except (MemoryError, TimeoutError) as e:
            future.cancel()
            print(f"切换至MiniBatchKMeans，原因：{e}")
            # 确保关闭alarm
            # 回收内存
            del clusterer, future
            gc.collect()
            # 估算n_clusters
            # clusters_size之前在HDBSCAN的设置中使用
            estimated_n_clusters = max(2, min(10, int(features.shape[0] / clusters_size)))
            print(f"估计簇数为: {estimated_n_clusters}")
            mbk = MiniBatchKMeans(
                n_clusters=estimated_n_clusters,
                batch_size=1024,
                n_init='auto',
                random_state=3417
            )
            mbk.fit_predict(features)
            labels = mbk.labels_
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            mbk = MiniBatchKMeans(n_clusters=10)
            mbk.fit_predict(features)
            labels = mbk.labels_ #二维数据不需要reshape（-1，1），只有wash-in/out单个特征需要
        finally:
            executor.shutdown(wait=False)

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
                # 重试参数：降低min_cluster_size，放宽限制
                new_min_size = max(5, int(clusters_size * 0.5))  # 至少为2
                clusterer_retry = hdbscan.HDBSCAN(
                    min_cluster_size=new_min_size,
                    min_samples=2,
                    cluster_selection_method="leaf"  # 更加宽松的方法
                )
                labels = clusterer_retry.fit_predict(features)
                if np.any(labels == -1):
                    labels = assign_noise_to_clusters(features, labels)  # 处理噪声
                if (np.unique(labels) == -1).all():
                    labels[:] = 0  # 选择将所有点归入第0簇
    
    # 生成超体素标签图像
    
    if mask is not None:
        segmented_image[mask_array > 0] = labels + 1  # 对掩码区域赋标签，避免标签从 0 开始
    else:
        segmented_image = labels.reshape(mask_array.shape)  # 将标签重塑为图像形状
    
    del image1, image2, mask, labels
    gc.collect()  # 异常时尝试回收
    # 保存结果
    return segmented_image

# def train_global_model(all_intensities):
#     intensities_2d = np.concatenate(all_intensities).reshape(-1, 1)
    
#     # 增加标准化步骤（关键！）
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(intensities_2d)  # 标准化后数据

#     best_score = 0
#     best_i = 0
#     # scores = [0]
#     for i in range(2,20):
#     # K-means训练 + 根据中心排序生成固定分割阈值
#         kmeans = KMeans(n_clusters=i, random_state=42)
#         kmeans.fit(scaled_data)
#         score = calinski_harabasz_score(scaled_data,kmeans.labels_)
#         print('数据聚'+ f"{i}" + '类calinski_harabaz指数为:' + f"{score}")
#         if score > best_score:
#             best_score = score
#             best_i = i
    
#     kmeans = KMeans(n_clusters=best_i, random_state=42).fit(scaled_data)
#     # 按中心排序后确定区间分割点（全局固定）
#     sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
#     thresholds = (sorted_centers[:-1] + sorted_centers[1:]) / 2  # 中间点为分割阈值
#     return thresholds  # 返回全局分割阈值

# def classify_voxels(voxel_intensities, thresholds):
#     # 例如：阈值=[100,200]，分割区间为 (-inf,100), [100,200), [200,+inf)
#     return np.digitize(voxel_intensities, thresholds) 

# HDBSCAN算法
# def train_global_model(all_intensities): 
#     # 合并数据并标准化
#     intensities_flatten = np.concatenate(all_intensities).reshape(-1,1)
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(intensities_flatten)
    
#     # 创建HDBSCAN对象（关键参数设置）
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=50,       # 最小簇的样本数（根据数据量调整）
#         min_samples=20,            # 核心点的邻域样本数
#         cluster_selection_method='eom',  # 更倾向分割清晰簇
#         allow_single_cluster=False,      # 禁止所有点成为一个大簇
#         gen_min_span_tree=True
#     )
    
#     # 训练模型
#     cluster_labels = clusterer.fit_predict(scaled_data)
    
#     # 提取非噪声点（噪声标签-1）的阈值
#     valid_mask = (cluster_labels != -1)
#     valid_data = scaled_data[valid_mask] 
#     valid_labels = cluster_labels[valid_mask]
    
#     # 计算每个聚类的区间分割点（替代KMeans中心）
#     thresholds = []
#     unique_labels = np.unique(valid_labels)
#     for label in np.sort(unique_labels):
#         # 提取当前簇的数据点
#         cluster_points = valid_data[valid_labels == label].flatten()
        
#         # 计算当前簇的分割阈值（示例用分位点）
#         lower = np.percentile(cluster_points, 5)  # 下边界
#         upper = np.percentile(cluster_points, 95) # 上边界
        
#         thresholds.extend([lower, upper])         # 保存边界点
        
#     # 去重排序并转换到原尺度
#     thresholds = np.unique(thresholds)
#     sorted_thresholds = scaler.inverse_transform(thresholds.reshape(-1,1)).flatten()
    
#     return sorted_thresholds, clusterer, scaler

# def classify_voxels(voxel_intensities, scaler, thresholds):
#     scaled_data = scaler.transform(voxel_intensities.reshape(-1, 1))
#     return np.digitize(scaled_data.flatten(), thresholds)


#加载数据
# 使用示例

# 调用函数
# segmented_image = slic_supervoxel(image_path, mask=mask, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)


image_path = r"<PROJECT_ROOT>\figure-res-0\washIn\Benign1.nii" #替换为你的图像路径
mask_path= r'<DCM_ROOT>\benign-DCE-221subs\part1-175subs+part2\1\Untitled.nii.gz' #替换为你的掩码路径
datapath = r"<PROJECT_ROOT>\datas_path" \
".csv"
output_dir = r"<PROJECT_ROOT>\habitat_in"

all_intensities = []
datalist = pd.read_csv(datapath)

error_records = []

# for datalist0 in range(datalist.shape[0]):
#     folder_path = r'<PROJECT_ROOT>\figure-res-0' if datalist.loc[datalist0, 'grade'] == 0 else r'<PROJECT_ROOT>\figure-res-1'
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

for datalist0 in range(datalist.shape[0]):
        # 读取数据

    if datalist.loc[datalist0, 'grade'] == 0 :
        folder_path = r'<PROJECT_ROOT>\figure-res-0'
        grade = 'Benign'
    else : 
        folder_path = r'<PROJECT_ROOT>\figure-res-1'
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
    mask_path = datalist.iloc[datalist0, 6]
    # img_path = os.path.join(folder_path, r'washOut/' + img_name)

    if not os.path.exists(img_path1) or not os.path.exists(img_path2):
    # 更详细的错误提示（推荐）：
        print(f"文件缺失：{img_path1} 和/或 {img_path2} → 跳过处理")
        continue  # 直接进入下一个循环迭代

    # mask = sitk.ReadImage(datalist.iloc[datalist0, 6])
    # img_array = sitk.GetArrayFromImage(img)
    # mask_array = sitk.GetArrayFromImage(mask).astype(bool)
        
    # 应用聚类       
    cluster_img = slic_supervoxel(img_path1, img_path2, mask_path, smooth_sigma=1.0)
    
    if cluster_img is None:
        print(f"处理 {img_name} 时发生错误，跳过保存")
        continue

        # 保存结果
    output_img = sitk.GetImageFromArray(cluster_img)
    # output_img.CopyInformation(img)  # 保持空间属性
    sitk.WriteImage(output_img, f"{output_dir}/{img_name}_{mask_name}.nii")
    print(f'{img_name} - {mask_name}' + ' is saved.')



# slic_supervoxel(image_path, mask_path, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)

