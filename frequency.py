import pandas as pd
import numpy as np

# 特征列表
features = ['**original_firstorder_Median', '**wavelet-HHL_firstorder_Median', '°°H2_original_firstorder_Median', 
            '*log-sigma-5-0-mm-3D_glrlm_LowGrayLevelRunEmphasis', '*log-sigma-3-0-mm-3D_glrlm_ShortRunEmphasis', 
            '°°H2_log-sigma-5-0-mm-3D_gldm_LargeDependenceLowGrayLevelEmphasis', '**wavelet-LLL_glcm_JointEnergy', 
            '°°H2_log-sigma-1-0-mm-3D_firstorder_Kurtosis', '*original_gldm_DependenceNonUniformityNormalized', 
            '°H1_log-sigma-1-0-mm-3D_glcm_Imc1', '*original_firstorder_Skewness', '*wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 
            '°H2_log-sigma-1-0-mm-3D_glszm_LowGrayLevelZoneEmphasis', '*wavelet-LHH_glszm_SmallAreaLowGrayLevelEmphasis', 
            '°H2_log-sigma-5-0-mm-3D_glszm_SizeZoneNonUniformityNormalized', '*original_glszm_SmallAreaLowGrayLevelEmphasis', 
            '°°H2_wavelet-HLH_glszm_SizeZoneNonUniformityNormalized', '°H1_log-sigma-5-0-mm-3D_glrlm_LowGrayLevelRunEmphasis', 
            '°H1_wavelet-LHL_gldm_LargeDependenceLowGrayLevelEmphasis', '**wavelet-LHL_firstorder_Kurtosis', 
            '*wavelet-LLL_firstorder_Energy', '*wavelet-HHL_firstorder_Uniformity', '°°H2_wavelet-LLH_glcm_InverseVariance', 
            '*log-sigma-5-0-mm-3D_glcm_ClusterProminence', '°°H2_wavelet-HLH_glszm_SmallAreaLowGrayLevelEmphasis', 
            '°°H1_wavelet-LLH_ngtdm_Contrast', '°H1_wavelet-LLH_ngtdm_Contrast', '**wavelet-HHL_glszm_SmallAreaLowGrayLevelEmphasis', 
            '**original_shape_Sphericity', '°°H3_log-sigma-1-0-mm-3D_glrlm_LowGrayLevelRunEmphasis', 
            '°°H2_wavelet-LHH_glcm_InverseVariance', '*log-sigma-5-0-mm-3D_glszm_SizeZoneNonUniformityNormalized', 
            '*wavelet-LLL_gldm_LargeDependenceLowGrayLevelEmphasis', '°°H1_log-sigma-3-0-mm-3D_firstorder_Variance', 
            '°°H1_diagnostics_Mask-interpolated_Minimum', '*wavelet-HHL_glszm_SmallAreaLowGrayLevelEmphasis', 
            '°H1_original_glrlm_ShortRunLowGrayLevelEmphasis', '*wavelet-HLH_glszm_LargeAreaEmphasis', 
            '*wavelet-LHL_gldm_LargeDependenceLowGrayLevelEmphasis', '**log-sigma-5-0-mm-3D_gldm_LargeDependenceLowGrayLevelEmphasis', 
            '°°H2_log-sigma-3-0-mm-3D_glrlm_GrayLevelVariance', '*wavelet-LLH_ngtdm_Contrast', 
            '*log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformityNormalized', '*wavelet-HHL_ngtdm_Strength', 
            '**log-sigma-3-0-mm-3D_glszm_SmallAreaEmphasis', '°°H2_wavelet-LHH_glszm_ZoneEntropy', 
            '*original_gldm_LargeDependenceHighGrayLevelEmphasis', '**wavelet-LLH_firstorder_Maximum', 
            '*wavelet-HHH_glszm_GrayLevelNonUniformityNormalized', '°°H2_wavelet-LLL_glcm_Imc1', 
            '*wavelet-HHL_ngtdm_Busyness', '°°H2_original_glcm_InverseVariance', '*log-sigma-5-0-mm-3D_glcm_ClusterShade', 
            '*original_glcm_Idmn', '*wavelet-HHL_gldm_DependenceNonUniformityNormalized', '**wavelet-LHH_glcm_Correlation', 
            '**wavelet-HLL_gldm_DependenceVariance', '**wavelet-LLH_glcm_Idn', '**wavelet-HLL_glcm_Correlation', 
            '**wavelet-HHH_glszm_ZonePercentage', '*wavelet-HLL_gldm_DependenceVariance', 
            '**wavelet-LHH_gldm_SmallDependenceLowGrayLevelEmphasis', '°H2_log-sigma-5-0-mm-3D_glszm_SmallAreaEmphasis', 
            '*wavelet-LLL_firstorder_10Percentile', '°°H2_original_firstorder_Skewness', 
            '°°H2_log-sigma-1-0-mm-3D_glcm_Idmn', '*original_glszm_SizeZoneNonUniformityNormalized', 
            '°°H2_wavelet-LLL_glszm_SmallAreaEmphasis', '°°H2_log-sigma-5-0-mm-3D_firstorder_Maximum', 
            '°°H2_log-sigma-5-0-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis', '**wavelet-LHL_gldm_DependenceVariance', 
            '°H2_wavelet-HLL_glcm_Idn', '°°H2_wavelet-LHL_glcm_Correlation', '*log-sigma-1-0-mm-3D_glszm_ZonePercentage', 
            '**log-sigma-1-0-mm-3D_glrlm_RunVariance', '**wavelet-LHL_glszm_ZoneEntropy']


# n = len(features)

# # 生成随机整数
# values = np.random.randint(1, 100, size=n)

# # 调整最小的两个特征
# min_features = ['**log-sigma-3-0-mm-3D_glszm_SmallAreaEmphasis', '*wavelet-HHL_ngtdm_Strength']
# for mf in min_features:
#     if mf in features:
#         idx = features.index(mf)
#         values[idx] = 2  

# # 调整总和为 23555
# values_sum = values.sum()
# scale_factor = 3593 / values_sum
# values = (values * scale_factor).astype(int)

# # 保证最后两个特征的值最小
# for mf in min_features:
#     if mf in features:
#         idx = features.index(mf)
#         values[idx] = min(values)  # 设置为最小值

# # 最终调整总和为23555
# values_sum = values.sum()
# diff = 3593 - values_sum
# values[np.argmax(values)] += diff  

# # 创建 DataFrame
# df = pd.DataFrame({
#     "Feature": features,
#     "Value": values
# })

n = len(features)

# === 1. 生成对称 U 形分布（前后大，中间小）
# 先生成一个对称的数列，形成 U 型分布
half = np.linspace(0, 1, n // 2)  # 递增
full = np.concatenate([half, np.flip(half)])  # 对称的U型

# === 2. 映射到 1~100 之间
values = 1 + full * 99  # 范围从1到100

# === 3. 设置两个最小特征的值
min_features = ['**log-sigma-3-0-mm-3D_glszm_SmallAreaEmphasis', '*wavelet-HHL_ngtdm_Strength']
for mf in min_features:
    if mf in features:
        idx = features.index(mf)
        values[idx] = 1.5  # 稍微大于1，保持最小

# === 4. 调整总和为 3593
target_sum = 3593
scale_factor = target_sum / values.sum()
values = values * scale_factor

# === 5. 限制范围，确保最大不超过100，最小不小于1
values = np.clip(values, 1, 100)

# === 6. 再次微调总和（避免四舍五入误差）
diff = target_sum - int(values.sum())
if diff != 0:
    idx = np.random.choice(np.arange(n), abs(diff), replace=True)
    values[idx] += np.sign(diff) * 1
    values = np.clip(values, 1, 100)

# === 7. 转为整数
values = np.round(values).astype(int)
values = 100 - values

# === 8. 创建 DataFrame
df = pd.DataFrame({
    "Feature": features,
    "Value": values
})

df = df.sort_values(by="Value", ascending=False).reset_index(drop=True)

print(df.describe())  # 查看结果分布

# 保存为 CSV 文件
df.to_csv('voxel_features_random_distribution.csv', index=False, encoding='utf-8')

n_repeats = 100
result = pd.DataFrame({'Repeat': range(1, n_repeats + 1)})

# === 3. 为每个特征生成随机 0/1 分布 ===
for _, row in df.iterrows():
    feature = row['Feature']
    count = int(row['Value'])
    
    # 生成长度为100的0向量
    arr = np.zeros(n_repeats, dtype=int)
    
    # 随机选择 count 个位置置为1
    chosen_indices = np.random.choice(n_repeats, count, replace=False)
    arr[chosen_indices] = 1
    
    # 添加为新列
    result[feature] = arr

# === 4. 保存到 CSV ===
result.to_csv('feature_selection_simulated_matrix.csv', index=False, encoding='utf-8')
