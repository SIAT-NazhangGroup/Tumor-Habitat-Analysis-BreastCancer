import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import scikit_posthocs as sp
import pingouin as pg
from scipy.stats import friedmanchisquare
# from statannotations.Annotator import Annotator

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
        是否绘制图形（小提琴 + 箱型图）
    
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
    
    # ========= 3. 可视化 =========
    if plot:
        plt.figure(figsize=(7,5))
        sns.violinplot(x="method", y="score", data=df_long, inner=None, alpha=0.6)
        sns.boxplot(x="method", y="score", data=df_long, width=0.2, showcaps=True,
                    boxprops={"facecolor":"white", "zorder":2},
                    showfliers=False, whiskerprops={"linewidth":2})
        # sns.swarmplot(x="method", y="score", data=df_long, color="k", alpha=0.7)
        
        plt.title(f"Friedman test: chi2={stat:.3f}, p={p:.4f}")
        plt.tight_layout()
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
    # 如果列名不是 accuracy，可以改成 df.iloc[:,0] 取第一列
    data[name] = df['auc_mean']

print(data.head())  # 每列就是一组特征下的准确率

data = data.iloc[:100, :]

stat, p = friedmanchisquare(data['in'], data['out'], data['in+out'], data['habitat in'], data['habitat out'], data['habitat in+out'], data['combine'])
print("10-Fold Friedman Result: chi2 = %.3f, p = %.4f" % (stat, p))

data["subject"] = np.arange(len(data))
results = friedman_posthoc(data, cols=['in','out','in+out', 'habitat in', 'habitat out', 'habitat in+out', 'combine'], subject_col = 'subject')
print("Friedman 检验结果: chi2 = %.3f, p = %.4f" % results["friedman"])
print(results["posthoc"].columns)
print("\n事后比较结果：")
print(results["posthoc"][['A','B','p-unc','p-corr','p-adjust']])