# Tumor-Habitat-Analysis-BreastCancer

基于多期相 DCE-MRI 的乳腺癌肿瘤**生境（habitat）分析**与良恶性分类流水线。通过对 wash-in / wash-out 动态增强图进行体素级聚类，将肿瘤分割为若干生理子区域（habitat），提取每个子区域的影像组学特征，再用机器学习方法对良性 / 恶性进行分类。

> 本仓库由中国科学院深圳先进技术研究院（SIAT）Na Zhang 课题组维护。仅发布**方法代码**，不发布患者数据。

---

## 目录

- [模型 / 工具介绍](#模型--工具介绍)
- [仓库结构](#仓库结构)
- [环境依赖](#环境依赖)
- [数据格式](#数据格式)
- [使用流程](#使用流程)
- [输出说明](#输出说明)
- [患者数据与模型权重](#患者数据与模型权重)
- [注意事项](#注意事项)
- [引用 / Citation](#引用--citation)
- [许可证](#许可证)

---

## 模型 / 工具介绍

- **目标**：用 DCE-MRI 多期相动态增强信息刻画肿瘤内部异质性，并基于 habitat 影像组学特征做良恶性二分类。
- **适用模态**：多期相 DCE-MRI（动态对比增强 T1）。需要 mask 内的 wash-in、wash-out 体素值。
- **方法概要**：
  1. 从 DICOM 多期相序列提取每个 mask 内体素的**时间-强度曲线（TIC）**；
  2. 计算 **wash-in / wash-out** 灰度图；
  3. 对每个肿瘤，以 `(物理坐标, wash-in, wash-out)` 为特征做体素聚类（HDBSCAN，超时/异常降级为 MiniBatchKMeans），得到 **habitat 子区域**；
  4. 用 **pyradiomics** 对每个 habitat 提取影像组学特征；
  5. **Monte Carlo + LASSO**（100 次随机种子）做特征选择与稳定性统计；
  6. 用 **LightGBM / XGBoost / MLP / TabPFN** 做良恶性分类，并做 DeLong 检验、t 检验、Wilcoxon 检验等统计比较。

---

## 仓库结构

```
Tumor-Habitat-Analysis-BreastCancer/
├── README.md
├── LICENSE                  # MIT
├── env.yml                  # conda 环境锁定文件（Windows，环境名 ckx）
├── .gitignore
│
├── # ===== 流水线核心脚本 =====
├── get_TIC.py               # 从多期相 DICOM 提取体素时间-强度曲线（CuPy GPU 加速）
├── calculat_wash.py         # 计算 wash-in / wash-out 灰度图（CuPy）
├── washin_map.py            # wash-in 图可视化 / 灰度窗
├── dce_gray_window.py       # DCE 灰度窗处理
├── subregion.py             # habitat 分割（HDBSCAN → MiniBatchKMeans 降级），wash-in + wash-out 双特征
├── subregion-3class.py      # habitat 分割，3 类版本
├── subregion-new.py         # habitat 分割，迭代版
├── Radiomics.py             # pyradiomics 影像组学特征提取（单序列）
├── Radiomics-3class.py      # pyradiomics 影像组学特征提取（3 habitat）
├── feature.py               # Monte Carlo + LASSO 特征选择（100 次随机种子，并行）
├── in.py / in_test.py       # wash-in 特征建模：LassoCV + SelectFromModel + TabPFN/LightGBM 等
├── model.py                 # LightGBM / XGBoost / MLP 对比基线
│
├── # ===== 统计检验 =====
├── f-test.py                # 方差齐性 + t 检验
├── t-test.py                # t 检验
├── t-fig.py                 # t 检验可视化
├── wilcoxon.py              # Wilcoxon 秩和检验
├── p value.py               # p 值汇总
├── DelongTest.ipynb         # DeLong 检验比较 AUC
│
├── # ===== 可视化 =====
├── habitat_fig.py           # habitat 子区域叠图
├── habitat_single_fig.py    # 单病例 habitat 可视化
├── single_tumor_tic_curve.py# 单肿瘤 TIC 曲线绘制
├── figshow.py / figshow-1.py / figshow-merge.py  # 结果图绘制
├── 3d-figshow.py            # 3D 可视化
├── histogram.py / histogram-figshow.py / histogram-figshow2.py  # 直方图
├── heatmap_frequency.py / heatmap_fre_pro.py     # 特征选择频次热图
├── roseplot.py              # 玫瑰图
├── color.py                 # 配色工具
│
├── # ===== 数据/特征整理工具 =====
├── number2id.py             # 病例编号 → ID 映射
├── merge.py                 # 表格合并
├── normal.py                # 特征归一化
├── mean_lasso.py            # LASSO 系数平均与可视化
├── feature_coef.py          # 特征系数分析
├── frequency.py             # 选定特征列表与频次
├── dataset_tabel.py         # 数据集统计表
├── spacing.py               # 体素间距统计
├── SubAnonymized.py         # 数据匿名化
├── liu_PK2nii.py            # DICOM → nii 转换
├── TIC_habitat.py           # 按 habitat 提取 TIC
│
├── # ===== 端到端实验 notebook（按特征组合命名） =====
├── in_only.ipynb            # 仅 wash-in 特征
├── out.ipynb / out+sub.ipynb
├── sub_only.ipynb           # 仅 habitat 子区特征
├── in_sub.ipynb
├── in+out.ipynb             # wash-in + wash-out
├── in+out+sub.ipynb / in+out+sub-1.ipynb / in+out+sub-2.ipynb
├── in+out+sub-pro.ipynb     # 含预后
├── BenignAndMalignant.ipynb # 良恶性主分析
└── DelongTest.ipynb         # AUC DeLong 检验
```

> `aaa.py`、`test.py`、`test11.py`、`untitle.py` 为早期探索性脚本，保留作参考，非流水线必经环节。

---

## 环境依赖

- **Python**：3.10
- **关键库**：CuPy（GPU 加速，需 CUDA 12.x）、SimpleITK、nibabel、pydicom、pyradiomics、scikit-learn、LightGBM、XGBoost、hdbscan、torch（+CUDA）、tabpfn、imbalanced-learn、opencv、scikit-image、statsmodels、pingouin
- **完整锁定**：见 `env.yml`（Windows conda 环境，名为 `ckx`）。

快速安装（按需精简）：

```bash
conda create -n ckx python=3.10 -y
conda activate ckx
# GPU 版本请按你的 CUDA 版本安装 cupy / torch，下面仅作示例
pip install SimpleITK nibabel pydicom pyradiomics scikit-learn lightgbm xgboost \
            hdbscan opencv-python scikit-image statsmodels pingouin imbalanced-learn \
            seaborn pandas matplotlib tqdm joblib tabpfn
```

---

## 数据格式

### 输入格式约定

| 项目        | 说明                                                                   |
| --------- | -------------------------------------------------------------------- |
| 图像来源      | 多期相 DCE-MRI（DICOM 序列）                                                 |
| 模态        | 动态对比增强 T1（DCE）                                                        |
| 派生图       | wash-in / wash-out 灰度图（`.nii`），由 `calculat_wash.py` 从多期相 DCE 计算得到      |
| 掩码        | 肿瘤 ROI 二值 mask（`.nii` / `.nii.gz`），值为非零即有效                            |
| 体素间距      | 保留原始 spacing，聚类时用 `image.GetSpacing()` 转物理坐标                            |
| 病例 ID 约定  | `patient_name` + `label_name`（病灶名），良恶性在 `grade` 列：`0`=Benign，`1`=Malignant |
| 数据清单      | `datas_path.csv` / `datas_clear_final.csv`（含 `patient_name, label_name, grade, nii_data, dcm_data, label_path`，**托管在 HF 私仓，见下**） |

目录结构示例：

```
<PROJECT_ROOT>/
├── datas_path.csv                       # 病例清单（HF 私仓）
├── figure-res-0/washIn/Benign<x>.nii     # 良性 wash-in 图
├── figure-res-0/washOut/Benign<x>.nii    # 良性 wash-out 图
├── figure-res-1/washIn/Malignant<x>.nii  # 恶性 wash-in 图
├── figure-res-1/washOut/Malignant<x>.nii # 恶性 wash-out 图
└── <mask 路径>/...                         # 各病例 ROI mask
```

### 输出格式约定

| 输出                       | 格式           | 含义                                |
| ------------------------ | ------------ | --------------------------------- |
| TIC 表                    | CSV          | 每个 mask 内体素在各期相的信号强度 + 体素坐标        |
| wash-in / wash-out 图     | `.nii`       | 计算后的增强动力学参数图                      |
| habitat 分割图              | `.nii`       | 体素标签 = habitat 编号（从 1 起，0 为背景）    |
| 影像组学特征表                  | CSV          | 每行一个病例，列为 pyradiomics 特征          |
| 特征选择矩阵                  | CSV          | 100 次随机种子下每个特征是否被 LASSO 选中（0/1）    |
| 模型评估结果                  | CSV          | 多种子下 AUC / Accuracy / F1 等指标      |

**Habitat 标签值约定**（3 类版本）：

| 体素值 | 含义                  |
| --- | ------------------- |
| 0   | 背景（mask 外）          |
| 1   | habitat 1（特征前缀 `H1`） |
| 2   | habitat 2（特征前缀 `H2`） |
| 3   | habitat 3（特征前缀 `H3`） |

**良恶性标签约定**：

| grade | 含义       |
| ----- | -------- |
| 0     | Benign   |
| 1     | Malignant |

**影像组学特征命名约定**：特征名带前缀标识来源 habitat，例如 `°H1_original_firstorder_Median`、`°°H2_wavelet-HHL_glszm_SmallAreaLowGrayLevelEmphasis`、`°°H3_log-sigma-1-0-mm-3D_glrlm_LowGrayLevelRunEmphasis`；不带 H 前缀的为整瘤特征。

---

## 使用流程

> 所有脚本中的绝对路径已脱敏为占位符，**运行前请全局替换**：
>
> | 占位符              | 含义                       |
> | ---------------- | ------------------------ |
> | `<PROJECT_ROOT>` | 数据 / 中间结果根目录（wash-in/out 图、habitat 输出等） |
> | `<NEW_ROOT>`     | 二期数据根目录                  |
> | `<DCM_ROOT>`     | DICOM 原始数据根目录            |
> | `<FIG_ROOT>`     | 图表输出根目录                  |
> | `<REDACTED_PATH>`| 已脱敏的零散绝对路径，按需替换          |

典型运行顺序（在 `conda activate ckx` 后）：

```bash
# 1. 从多期相 DICOM 提取 TIC（GPU）
python get_TIC.py

# 2. 计算 wash-in / wash-out 灰度图
python calculat_wash.py

# 3. habitat 分割（HDBSCAN + 降级策略）
python subregion-3class.py      # 或 subregion.py

# 4. 提取影像组学特征
python Radiomics-3class.py      # 或 Radiomics.py

# 5. Monte Carlo + LASSO 特征选择（100 次随机种子）
python feature.py

# 6. 建模与评估（wash-in 特征为例）
python in.py                    # 含 LassoCV + SelectFromModel + TabPFN / LightGBM
python model.py                 # LightGBM / XGBoost / MLP 基线对比

# 7. 统计检验
jupyter nbconvert --to notebook --execute DelongTest.ipynb   # AUC DeLong 检验
python t-test.py
python wilcoxon.py

# 8. 可视化
python habitat_fig.py
python heatmap_frequency.py
python roseplot.py
```

端到端实验可直接运行 `in_only.ipynb`、`in+out+sub.ipynb`、`BenignAndMalignant.ipynb` 等 notebook（按特征组合命名）。

---

## 输出说明

运行后典型输出目录结构：

```
<PROJECT_ROOT>/
├── set/Tic/                # TIC 表 CSV（每病例一个）
├── set/Set/                # 扫描参数表（flip angle, TR, 各期时间）
├── habitat_in/             # habitat 分割 NIfTI（基于 wash-in）
├── habitat_test/           # habitat 分割（3 类版）
├── habitat_vis/            # habitat 可视化 PNG
├── radiology/              # 影像组学特征 CSV（in_radiology_*.csv / out_radiology_*.csv）
├── feature_selection_results/
│   ├── feature_selection_long_format.csv
│   ├── feature_selection_matrix.csv
│   └── feature_selection_summary.csv
└── eval_results-2/         # 多种子评估结果 CSV（AUC 等）
```

---

## 患者数据与模型权重

**患者数据不发布在 GitHub**。本仓库涉及的病例清单、影像组学特征表等含患者相关信息的数据已迁移至 Hugging Face **私有**数据集仓库：

- **数据地址**：<https://huggingface.co/datasets/SIAT-NazhangGroup/Tumor-Habitat-Analysis-BreastCancer>
- **可见性**：Private（仅授权成员可访问）

下载方式（需有访问权限）：

```bash
# 方式1：huggingface_hub Python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="SIAT-NazhangGroup/Tumor-Habitat-Analysis-BreastCancer",
                         repo_type="dataset", local_dir="./data")

# 方式2：hf CLI
hf download SIAT-NazhangGroup/Tumor-Habitat-Analysis-BreastCancer --repo-type dataset --local-dir ./data
```

下载后数据位于 `./data/data/` 下，包含 `datas_path.csv`、`datas_clear_final.csv`、影像组学特征表、特征选择矩阵等。将其放入上文 `<PROJECT_ROOT>` 对应位置后即可运行脚本。

**模型权重**：本流水线为影像组学 + 经典机器学习，**无深度学习权重**需托管；如未来新增基于神经网络的分割 / 分类权重，将统一托管至 <https://huggingface.co/SIAT-NazhangGroup>。

---

## 注意事项

1. **隐私**：本仓库不含任何患者可识别信息（DICOM / nii / 含 ID 的清单）。涉及患者数据的处理与上传须遵循伦理审查与数据使用协议。
2. **硬编码路径**：原仓库中的 Windows 绝对路径已全部脱敏为占位符（`<PROJECT_ROOT>` 等），运行前需全局替换为你本地的实际路径。请勿再次提交本地绝对路径。
3. **GPU**：`get_TIC.py`、`calculat_wash.py` 使用 CuPy 做 GPU 加速；`in.py` / `model.py` 中的 TabPFN / MLP 优先使用 CUDA。无 GPU 时部分脚本会回退到 CPU，但速度会显著下降。
4. **HDBSCAN 超时**：`subregion.py` 中 HDBSCAN 设有 300 秒超时，超时 / 内存不足 / 全噪声时会自动降级为 MiniBatchKMeans；极端情况下所有点归入同一簇。
5. **评估口径**：`feature.py` 做 100 次 Monte Carlo 特征选择，`in.py` 做多种子训练 / 评估；`model.py` 为单次划分基线，报告 AUC 时请注意口径差异。
6. **Python 环境**：`env.yml` 为 Windows 下 conda 锁定文件，跨平台复现时请按需调整 CuPy / CUDA 版本。

---

## 引用 / Citation

如本仓库对你的研究有帮助，请引用相关论文（待补充 BibTeX）。

---

## 许可证

代码采用 **MIT License**（见 `LICENSE`）。患者数据不对外发布，如需用于研究合作请联系课题组获取数据使用许可。

- 课题组：中国科学院深圳先进技术研究院（SIAT）
- GitHub 组织：[SIAT-NazhangGroup](https://github.com/SIAT-NazhangGroup)
- Hugging Face 组织：[SIAT-NazhangGroup](https://huggingface.co/SIAT-NazhangGroup)
