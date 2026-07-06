from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from tabpfn import TabPFNClassifier
import numpy as np
import torch  # 新增导入torch用于检查CUDA可用性
from scipy.stats import levene, ttest_ind

# 检查CUDA是否可用（可选提示）
print("CUDA available:", torch.cuda.is_available())

data_path = r"E:\liuzhou_breastcancer\radiology\washin_radiology_2024-10-30.csv"

data = pd.read_csv(data_path)
# Load data

X = data[data.columns[4:]]
y = data['grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6677, stratify=y)

#通过T检验从106个特征筛选

counts = 0
columns_index =[]
# print(X_train[y_train==0])
for column_name in X_train.columns[1:]:
    # print(column_name)
    # print("\033[1;31;40m"+column_name+"\033[0m")
    try:
        if levene(X_train[y_train==0][column_name], X_train[y_train==1][column_name])[1] > 0.05:
            if ttest_ind(X_train[y_train==0][column_name], X_train[y_train==1][column_name],equal_var=True)[1] < 0.05:
                columns_index.append(column_name)
        else:
            if ttest_ind(X_train[y_train==0][column_name], X_train[y_train==1][column_name],equal_var=False)[1] < 0.05:
                columns_index.append(column_name)
    except Exception as ex:
        print("出现如下异常: %s"%ex)
        continue

print("筛选后剩下的特征数：{}个".format(len(columns_index)))

X_train = X_train[columns_index]

# help(TabPFNClassifier)  # 检查 "__init__" 部分参数定义
# 初始化分类器时指定设备 -----------------------------------------------------------------
clf = TabPFNClassifier(
    random_state = 1028,     
    softmax_temperature=0.8,                # 提高预测锐度
    balance_probabilities=True, 
    memory_saving_mode=True,    
    device='cuda'
)  # 关键改动：添加 device='cuda' 参数
# ---------------------------------------------------------------------------------

columnNames = X_train.columns

lassoCV_x = X_train.astype(np.float32)#把x数据转换成np.float格式
lassoCV_y = y_train

standardscaler = StandardScaler()
lassoCV_x = standardscaler.fit_transform(lassoCV_x)#对x进行均值-标准差归一化
lassoCV_x = pd.DataFrame(lassoCV_x,columns=columnNames)#转 DataFrame 格式

# 形成5为底的指数函数
# 5**（-3） ~  5**（-2）
alpha_range = np.logspace(-4,-2,50,base=5)
#alpha_range在这个参数范围里挑出aplpha进行训练，cv是把数据集分5分，进行交叉验证，max_iter是训练1000轮
lassoCV_model = LassoCV(alphas=alpha_range,cv=5,max_iter=100)
#进行训练
lassoCV_model.fit(lassoCV_x,lassoCV_y)

#打印训练找出来的入值
# print(lassoCV_model.alpha_)
# print("Coefficient of the model:{}".format(lassoCV_model.coef_) )
# print("intercept of the model:{}".format(lassoCV_model.intercept_))

coef = pd.Series(lassoCV_model.coef_, index=columnNames)
print("从原来{}个特征，筛选剩下{}个".format(len(columnNames),sum(coef !=0)))
# print("分别是以下特征")
# print(coef[coef !=0])
index = coef[coef !=0].index
lassoCV_x = lassoCV_x[index]
X_train = lassoCV_x


clf.fit(X_train, y_train)

# Predict probabilities
X_test = X_test[index]
X_test = standardscaler.fit_transform(X_test)
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
