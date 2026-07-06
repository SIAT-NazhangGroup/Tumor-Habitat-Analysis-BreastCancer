import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_ind, levene
# from pytorch_tabnet.tab_model import TabNetClassifier

# ==== 1. 数据准备 ====
df = pd.read_csv(r"E:\liuzhou_breastcancer\radiology\washin_radiology_2025-07-10.csv")  # 替换为你的文件路径

df_features = df.iloc[:, 5:]

# 3. 分离标签列
y = df["grade"]

# 4. 去掉目标变量列
X = df_features.drop(columns=["grade"], errors="ignore")
# X = X.select_dtypes(include=['number', 'float64', 'int64'])

# 5. 进行 t 检验（根据方差齐性判断 equal_var 参数）

counts = 0
columns_index =[]
# print(X_train[y_train==0])
for column_name in X.columns[1:]:
    # print(column_name)
    print("\033[1;31;40m"+column_name+"\033[0m")
    try:
        if levene(X[y==0][column_name], X[y==1][column_name])[1] > 0.05:
            if ttest_ind(X[y==0][column_name], X[y==1][column_name],equal_var=True)[1] < 0.05:
                columns_index.append(column_name)
        else:
            if ttest_ind(X[y==0][column_name], X[y==1][column_name],equal_var=False)[1] < 0.05:
                columns_index.append(column_name)
    except Exception as ex:
        print("出现如下异常: %s"%ex)
        continue

print("筛选后剩下的特征数：{}个".format(len(columns_index)))


# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==== 2. 特征选择 pipeline ====
# 1. 方差过滤
var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X)

# 2. 相关性过滤（简单相关系数 > 0.95 的去除）
X_df = pd.DataFrame(X_var)
corr_matrix = X_df.corr().abs()
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix.iloc[column, :][upper[column]] > 0.9)]
X_corr = X_df.drop(X_df.columns[to_drop], axis=1).values

# 3. SelectKBest（选择前 k 个重要特征）
k_best = SelectKBest(score_func=f_classif, k=500)  # 可以调节 k
X_selected = k_best.fit_transform(X_corr, y)

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 准备 float32 格式
X_train_32 = X_train.astype(np.float32)
X_test_32 = X_test.astype(np.float32)

# ==== 3. LightGBM ====
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

# ==== 4. XGBoost ====
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# ==== 5. MLP with training curve ====
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = MLP(X_train.shape[1]).to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train_32, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_32, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

loss_list = []
acc_list = []

for epoch in range(30):
    mlp_model.train()
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Eval during training
    mlp_model.eval()
    with torch.no_grad():
        outputs_eval = mlp_model(X_train_tensor)
        preds = torch.argmax(outputs_eval, dim=1)
        acc = (preds == y_train_tensor).float().mean().item()
        loss_list.append(loss.item())
        acc_list.append(acc)

# Final eval
mlp_model.eval()
with torch.no_grad():
    outputs = mlp_model(X_test_tensor)
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    mlp_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    mlp_auc = roc_auc_score(y_test, probs)

# ==== 6. TabNet ====
# tabnet_model = TabNetClassifier(verbose=0)
# tabnet_model.fit(
#     X_train_32, y_train,
#     eval_set=[(X_test_32, y_test)],
#     eval_metric=['auc'],
#     max_epochs=50,
#     patience=10,
#     batch_size=256
# )
# tabnet_pred = tabnet_model.predict(X_test_32)
# tabnet_auc = roc_auc_score(y_test, tabnet_model.predict_proba(X_test_32)[:, 1])

# ==== 7. 汇总结果 ====
results = {
    "LightGBM": (accuracy_score(y_test, lgb_pred), lgb_auc),
    "XGBoost": (accuracy_score(y_test, xgb_pred), xgb_auc),
    "MLP": (accuracy_score(y_test, mlp_pred), mlp_auc),
    # "TabNet": (accuracy_score(y_test, tabnet_pred), tabnet_auc)
}

# 可视化 MLP 训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Training Loss")
plt.title("MLP Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label="Training Accuracy")
plt.title("MLP Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

print(results)
