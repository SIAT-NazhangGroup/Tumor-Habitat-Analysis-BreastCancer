import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import friedmanchisquare
from matplotlib.colors import LinearSegmentedColormap, Normalize

def friedman_posthoc(df, cols, subject_col, padjust="holm", plot=True):
    """
    运行 Friedman 检验 + 事后两两比较（非参数） + 可视化
    
    参数
    ----
    df : pd.DataFrame
        包含数据的 DataFrame
    cols : list
        需要比较的列名（不同方法/处理条件）
    subject_col : str
        被试/个体的唯一标识列
    padjust : str, 默认 'holm'
        多重比较校正方法，支持 'bonf', 'holm', 'fdr_bh' 等
    plot : bool, 默认 True
        是否绘制图形（小提琴 + 箱型图 和 p值热图）
    
    返回
    ----
    dict:
        {
            "friedman": (stat, p值),
            "posthoc": DataFrame 事后比较结果
        }
    """
    
    # ========= 1. Friedman (scipy) =========
    arrays = [df[c].values for c in cols]
    stat, p = friedmanchisquare(*arrays)
    
    # ========= 2. 转换成长格式 & 事后比较 (pingouin) =========
    df_long = df.melt(id_vars=[subject_col], value_vars=cols,
                      var_name="method", value_name="score")
    
    posthoc = pg.pairwise_tests(
        dv="score",
        within="method",
        subject=subject_col,
        data=df_long,
        parametric=False,
        padjust=padjust
    )
    
    # ========= 3. 可视化 (小提琴 + 箱型图) =========
    if plot:
        plt.figure(figsize=(7,5))
        sns.violinplot(x="method", y="score", data=df_long, inner=None, alpha=0.6)
        sns.boxplot(x="method", y="score", data=df_long, width=0.2, showcaps=True,
                    boxprops={"facecolor":"white", "zorder":2},
                    showfliers=False, whiskerprops={"linewidth":2})
        
        plt.title(f"Friedman test: chi2={stat:.3f}, p={p:.4f}")
        plt.xticks(rotation=45, ha='right') # 旋转x轴标签，避免重叠
        plt.tight_layout()
        plt.show()

        # ========= 4. p值热图 (使用 sns.clustermap) =========
        # 创建一个空的方阵来存储p值
        p_value_matrix = pd.DataFrame(np.ones((len(cols), len(cols))),
                                      index=cols,
                                      columns=cols)
        
        # 填充调整后的p值
        for _, row in posthoc.iterrows():
            method_a = row['A']
            method_b = row['B']
            p_corr = row['p-corr']
            p_value_matrix.loc[method_a, method_b] = p_corr
            p_value_matrix.loc[method_b, method_a] = p_corr # 对称填充

        # 对角线设置为0，表示方法与自身的比较，这将映射到深红色
        np.fill_diagonal(p_value_matrix.values, 0) 

        # --- 自定义颜色映射 ---
        # 定义颜色节点和对应的p值
        # p=0 -> 深红色
        # p=0.05 -> 白色
        # p=1 -> 深蓝色
        
        colors_for_cmap = [
            (0.0, 'darkred'),    # p=0 对应深红色
            (0.05, 'white'),     # p=0.05 对应白色
            (1.0, 'darkblue')    # p=1 对应深蓝色
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_p_values", colors_for_cmap, N=256)

        # 使用 sns.clustermap 绘制热图
        # clustermap 会自动进行层次聚类，并显示树状图
        g = sns.clustermap(p_value_matrix,
                           cmap=custom_cmap,
                           annot=False,     # 不显示p值
                           fmt=".3f",       # 格式化p值，虽然不显示但保留
                           linewidths=0,    # 无框线。如果需要白色框线，改为 linewidths=0.5, linecolor='white'
                           figsize=(len(cols) * 0.8 + 2, len(cols) * 0.8 + 2), # 动态调整图大小
                           cbar_kws={'label': f'Adjusted p-value ({padjust})'},
                           # 以下参数控制聚类方式，可根据数据特点调整
                           method='average', # 链接方法：'single', 'average', 'complete', 'ward'等
                           metric='euclidean' # 距离度量：'euclidean', 'correlation', 'cityblock'等
                          )
        
        # 设置主图标题
        g.fig.suptitle(f'Pairwise Adjusted p-values Heatmap ({padjust} correction)', y=1.02) # y调整标题位置

        # 旋转 x 轴标签
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        # 旋转 y 轴标签 (通常不需要旋转)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

        # --- 自定义颜色条刻度 ---
        # g.cbar_ax 是颜色条的轴对象
        g.ax_cbar.yaxis.set_ticks([0, 0.05, 1])
        g.ax_cbar.yaxis.set_ticklabels(['p=0 (Red)', 'p=0.05 (White)', 'p=1 (Blue)'])

        
        # 调整布局以适应所有元素
        g.fig.tight_layout()
        plt.show()
    
    return {"friedman": (stat, p), "posthoc": posthoc}


# 文件路径
files = {
    'in': r"E:\liuzhou_breastcancer\eval_results-2\in_top100_seeds_all_combined.csv",
    'out': r"E:\liuzhou_breastcancer\eval_results-2\out_top100_seeds_all_combined.csv",
    'in+out': r"E:\liuzhou_breastcancer\eval_results-2\in+out_5000_seeds_all_combined.csv",
    'habitat in': r"E:\liuzhou_breastcancer\eval_results-2\in_habitat_top100_seeds_all_combined.csv",
    'habitat out': r"E:\liuzhou_breastcancer\eval_results-2\out_habitat_top100_seeds_all_combined.csv",
    'habitat in+out': r"E:\liuzhou_breastcancer\eval_results-2\onlysub_top100_seeds_all_combined.csv",
    'combine': r"E:\liuzhou_breastcancer\eval_results-2\combined-copy-2.csv"
}

# 读取并合并
data = pd.DataFrame()
for name, path in files.items():
    df = pd.read_csv(path)   # 默认第一行是表头
    # 如果列名不是 auc_mean，请根据实际情况修改
    data[name] = df['auc_mean']

print("原始数据前5行:")
print(data.head())  # 每列就是一组特征下的准确率

data = data.iloc[:100, :]
print(f"\n使用前 {len(data)} 行数据进行分析。")

# 列出所有要比较的方法
methods_to_compare = list(files.keys())

# 执行初步的Friedman检验
stat_pre, p_pre = friedmanchisquare(*[data[col].values for col in methods_to_compare])
print(f"\n初步Friedman检验结果: chi2 = {stat_pre:.3f}, p = {p_pre:.4f}")

data["subject"] = np.arange(len(data))
results = friedman_posthoc(data, cols=methods_to_compare, subject_col='subject', padjust='fdr_bh') # 更改校正方法示例

print(f"\nFriedman 检验结果 (通过函数): chi2 = {results['friedman'][0]:.3f}, p = {results['friedman'][1]:.4f}")
print(f"\n事后比较结果列名: {results['posthoc'].columns.tolist()}")
print("\n事后比较结果：")
print(results["posthoc"][['A','B','p-unc','p-corr','p-adjust']])
