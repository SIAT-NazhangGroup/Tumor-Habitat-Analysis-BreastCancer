#加载必要包
import os
import cv2
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import SimpleITK as sitk
from skimage import segmentation
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import hdbscan
from hdbscan.validity import validity_index
import pandas as pd
import gc

# KNN聚类
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
def train_global_model(all_intensities): 
    # 合并数据并标准化
    intensities_flatten = np.concatenate(all_intensities).reshape(-1,1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(intensities_flatten)
    
    # 创建HDBSCAN对象（关键参数设置）
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=500,       # 最小簇的样本数（根据数据量调整）
        min_samples=100,            # 核心点的邻域样本数
        cluster_selection_method='eom',  # 更倾向分割清晰簇
        allow_single_cluster=False,      # 禁止所有点成为一个大簇
        cluster_selection_epsilon=0.1,
        alpha = 1.5, 
        gen_min_span_tree=False,
        approx_min_span_tree=True  # 启用近似算法替代精确计算
    )
    
    # 训练模型
    cluster_labels = clusterer.fit_predict(scaled_data)
    
    # 提取非噪声点（噪声标签-1）的阈值
    valid_mask = (cluster_labels != -1)
    valid_data = scaled_data[valid_mask] 
    valid_labels = cluster_labels[valid_mask]
    
    # 计算每个聚类的区间分割点（替代KMeans中心）
    thresholds = []
    unique_labels = np.unique(valid_labels)
    for label in np.sort(unique_labels):
        # 提取当前簇的数据点
        cluster_points = valid_data[valid_labels == label].flatten()
        
        # 计算当前簇的分割阈值（示例用分位点）
        lower = np.percentile(cluster_points, 5)  # 下边界
        upper = np.percentile(cluster_points, 95) # 上边界
        
        thresholds.extend([lower, upper])         # 保存边界点
        
    # 去重排序并转换到原尺度
    thresholds = np.unique(thresholds)
    sorted_thresholds = scaler.inverse_transform(thresholds.reshape(-1,1)).flatten()

    # ============== 新增DBCV评估 ================
    # 关键修改：启用内存优化
    np.random.seed(42)
    sample_size = min(50000, len(scaled_data))
    sample_idx = np.random.choice(len(scaled_data), sample_size, replace=False)
    
    # 抽取评估子集（保持双精度）
    sample_data = scaled_data[sample_idx]  # float64，无需astype转换！
    sample_labels = cluster_labels[sample_idx]  # 使用第一次训练的标签（可选）
    
    # 可能优化点：对第二次聚类评估单独采样（降低内存消耗）
    clusterer_v2 = hdbscan.HDBSCAN(
        min_cluster_size=50,
        memory='./hdbscan_cache/',
        core_dist_n_jobs=1
    ).fit(sample_data)  # 在小样本上训练
    
    labels_v2 = clusterer_v2.labels_.astype(np.int16)
    
    # 计算DBCV（必须保证输入数据为float64）
    try:
        dbcv_score = validity_index(sample_data, labels_v2)  # float64已满足
    except MemoryError:
        # 最后一次防御：极端情况下降低精度（需全局允许float32处理）
        sample_data_float32 = sample_data.astype(np.float32)
        dbcv_score = validity_index(sample_data_float32, labels_v2)
    
    # 及时释放资源（关键！）
    del sample_data, labels_v2, clusterer_v2
    gc.collect()
    # ============================================

    print(f'训练生成阈值数量: {len(sorted_thresholds)}个, 区间为{sorted_thresholds}')
    # 原后续处理及返回（增加返回评分）
    return sorted_thresholds, clusterer, scaler, dbcv_score
    
    


def classify_voxels(voxel_intensities, scaler, thresholds):
    scaled_data = scaler.transform(voxel_intensities.reshape(-1, 1))
    print(f'分类输入强度范围:[{voxel_intensities.min()}, {voxel_intensities.max()}]', 
      f'标准化后范围:[{scaled_data.min():.2f}, {scaled_data.max():.2f}]')
    return np.digitize(scaled_data.flatten(), scaler.transform(thresholds.reshape(-1,1)).flatten())


#加载数据
# 使用示例

# 调用函数
# segmented_image = slic_supervoxel(image_path, mask=mask, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)


image_path = r"E:\liuzhou_breastcancer\figure-res-0\washIn\Benign1.nii" #替换为你的图像路径
mask_path= r'E:\DCESummary_2019-202004\benign-DCE-221subs\part1-175subs+part2\1\Untitled.nii.gz' #替换为你的掩码路径
datapath = r"E:\liuzhou_breastcancer\datas_path_test.csv"
output_dir = r"E:\liuzhou_breastcancer\habitat"

all_intensities = []
datalist = pd.read_csv(datapath)

error_records = []

for datalist0 in range(datalist.shape[0]):
    folder_path = r'E:\liuzhou_breastcancer\figure-res-0' if datalist.loc[datalist0, 'grade'] == 0 else r'E:\liuzhou_breastcancer\figure-res-1'
    grade = 'Benign' if datalist.loc[datalist0, 'grade'] == 0 else 'Malignant'

    img_name = grade + str(datalist.loc[datalist0, 'patient_name']) + '.nii'

    #根据wash-in和wash-out注释
    img_path = os.path.join(folder_path, r'washIn/' + img_name)
    # img_path = os.path.join(folder_path, r'washOut/' + img_name)

    if not os.path.exists(img_path):
        continue

    img = sitk.ReadImage(img_path)  
    mask = sitk.ReadImage(datalist.iloc[datalist0, 6])
        
    # 获取mask内的强度值
    img_array = sitk.GetArrayFromImage(img)  # shape (z,y,x)
    mask_array = sitk.GetArrayFromImage(mask).astype(bool)
    
    
    if img_array.shape != mask_array.shape:
        error_msg = f"Shape mismatch: image {img_array.shape}; mask {mask_array.shape}"
        # 记录不匹配的图像名称和错误信息
        error_records.append({
            'image': img_path, 
            'mask': mask_path, 
            'error': error_msg
        })
        print(f"Skipping {img_path}: {error_msg}")
        continue  # 跳过当前循环
    roi_intensities = img_array[mask_array]  # 获取mask区域体素

    all_intensities.append(roi_intensities.flatten())
    print(f'{img_name}' + ' is collected.')

global_thresholds, hdbscan_model, scaler, dbcv_score = train_global_model(all_intensities)

if error_records:
    pd.DataFrame(error_records).to_csv("shape_mismatch_errors.csv", index=False)
    print("保存错误文件到 shape_mismatch_errors.csv")

for datalist0 in range(datalist.shape[0]):
        # 读取数据
    if datalist.loc[datalist0, 'grade'] == 0 :
        folder_path = r'E:\liuzhou_breastcancer\figure-res-0'
        grade = 'Benign'
    else : 
        folder_path = r'E:\liuzhou_breastcancer\figure-res-1'
        grade = 'Malignant'

    img_name = grade + str(datalist.loc[datalist0, 'patient_name']) + '.nii'

    #根据wash-in和wash-out注释
    img_path = os.path.join(folder_path, r'washIn/' + img_name)
    # img_path = os.path.join(folder_path, r'washOut/' + img_name)

    if os.path.exists(img_path):
        img = sitk.ReadImage(img_path)  
    else:
        continue


    mask = sitk.ReadImage(datalist.iloc[datalist0, 6])
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask).astype(bool)
    
    if img_array.shape != mask_array.shape:
        print(f"Shape mismatch: Skip {img_name}")
        continue  # 直接跳过不处理

    # 应用聚类
    cluster_labels = np.zeros_like(img_array, dtype=np.int16)
    roi_intensities = img_array[mask_array]
        
    # 使用训练好的scaler和阈值进行分类
    cluster_ids = classify_voxels(roi_intensities, thresholds=global_thresholds, scaler=scaler)
    
    cluster_labels[mask_array] = cluster_ids + 1  # HDBSCAN标签可能有负数
        # 保存结果
    output_img = sitk.GetImageFromArray(cluster_labels)
    output_img.CopyInformation(img)  # 保持空间属性
    sitk.WriteImage(output_img, f"{output_dir}/{str(datalist.loc[datalist0, 'patient_name'])}.nii")



# slic_supervoxel(image_path, mask_path, n_clusters=n_clusters, smooth_sigma=smooth_sigma, output_path=output_path)

