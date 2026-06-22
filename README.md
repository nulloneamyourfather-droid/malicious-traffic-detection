# 🛡️ 恶意流量检测系统

> 基于 CIC-IDS2017 数据集，用机器学习自动识别网络中的恶意流量。

## 项目简介

**自动分析网络流量数据**，判断哪些是**正常流量（BENIGN）**，哪些是**恶意攻击流量（Malicious）**。

支持识别的攻击类型：
- **DDoS / DoS** — 分布式/拒绝服务攻击（Hulk、GoldenEye、slowloris 等）
- **PortScan** — 端口扫描
- **Brute Force** — 暴力破解（FTP、SSH 密码爆破）
- **Web Attack** — Web 攻击（SQL 注入、XSS、暴力破解）
- **Bot** — 僵尸网络
- **Infiltration** — 渗透攻击
- **Heartbleed** — OpenSSL 心脏出血漏洞


### 1 依赖

```bash
cd e:\Malicious_traffic_detection
pip install -r requirements.txt
```

### 2️ 完整训练 + 评估

```bash
python main.py --mode train
```

这会依次完成：
1. 加载数据（约 280 万条流量记录）
2. 随机分层采样划分训练/测试集（70%/30%）
3. 特征工程（标准化处理）
4. 训练 3 个模型（LightGBM、Random Forest、Logistic Regression）
5. 评估并生成图表

### 3️ 对新数据进行预测

```bash
python main.py --mode predict --input 你的流量文件.csv
```

结果会生成一个 `你的流量文件_predicted.csv`，里面多了一列 `Prediction`（BENIGN/Malicious）和一列 `Malicious_Probability`（恶意概率）。

### 4️ 只看数据分析报告

```bash
python main.py --mode eda
```

##  模型效果

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
| **Random Forest** | **99.84%** | **99.41%** | **99.78%** | **99.60%** | **0.9999** |
| **LightGBM**  | **99.79%** | **99.61%** | **99.35%** | **99.48%** | **0.9997** |
| Logistic Regression | 93.62% | 78.97% | 92.17% | 85.06% | 0.9743 |

> ** Random Forest**，综合效果最好，对 Bot 和 Web 攻击的检测能力也最强。
>
>  **LightGBM **，训练只要几秒钟，部署到生产环境最合适。

### 各类攻击检测率（Random Forest）

| 攻击类型 | 检测率 |
|---------|:------:|
| PortScan | **99.99%** |
| DoS Hulk | **99.97%** |
| DDoS | **99.92%** |
| FTP-Patator | **99.87%** |
| DoS slowloris | **99.71%** |
| DoS GoldenEye | **99.61%** |
| DoS Slowhttptest | **99.45%** |
| SSH-Patator | **98.59%** |
| Web Attack - Brute Force | **98.45%** |
| Web Attack - XSS | **92.86%** |
| Bot | **64.24%** |

> Infiltration、SQL Injection、Heartbleed 因为样本太少（不到 10 条），检测率偏低，属于正常情况。

## 文件结构

```
e:\Malicious_traffic_detection\
├── data/
│   ├── MachineLearningCSV/       # 原始数据（8 个 CSV 文件）
│   ├── GeneratedLabelledFlows/   # 备份数据（未使用）
│   └── processed/                # 处理后缓存
├── models/                       # 训练好的模型文件
│   ├── lgb.joblib                #   LightGBM 模型
│   ├── rf.joblib                 #   Random Forest 模型
│   ├── lr.joblib                 #   Logistic Regression 模型
│   ├── scaler.joblib             #   标准化器
│   └── feature_columns.txt       #   特征列名列表
├── results/                      #   评估结果图表
│   ├── lgb_confusion_matrix.png  #   混淆矩阵
│   ├── roc_curves_comparison.png #   ROC 曲线对比
│   ├── lgb_feature_importance.png#   特征重要性
│   ├── rf_detection_by_type.png  #   按攻击类型检测率
│   └── model_comparison.png      #   模型对比
├── src/
│   ├── data_loader.py            # 加载数据
│   ├── preprocessing.py          # 特征工程
│   ├── train.py                  # 模型训练
│   ├── evaluate.py               # 模型评估
│   ├── eda.py                    # 数据分析
│   └── predict.py                # 预测推理
├── main.py                       # 主入口
└── requirements.txt              # 依赖清单
```

## 进阶用法

### 指定要训练的模型

```bash
# 只训练 LightGBM（最快）
python main.py --mode train --models lgb

# 只训练 Random Forest
python main.py --mode train --models rf

# 训练多个
python main.py --mode train --models lgb rf
```

### 使用按时间划分（模拟零日攻击）

```bash
python main.py --mode train --split time
```

> 这种方式用周一~周四的数据训练，用周五的数据测试。
> 因为周五的攻击类型（Bot、PortScan、DDoS）在训练集里没出现过，
> 可以模拟真实场景中"遇到新型攻击"的效果。

### 调整测试集比例

```bash
# 测试集占 20%，训练集占 80%
python main.py --mode train --test_size 0.2
```

## 常见问题

**Q: 模型检测准确率有多高？**
A: Random Forest 和 LightGBM 的 F1 分数都在 **99.5% 以上**，效果非常好。

**Q: 为什么有些攻击类型检测不到？**
A: 像 Heartbleed（3 条）、SQL Injection（6 条）这类样本太少，模型学不到足够特征。这是数据本身的问题，不是模型的错。

**Q: 我能拿它检测实时流量吗？**
A: 目前只支持分析 CSV 格式的流量文件。要用于实时检测，需要额外开发流量抓取 → 特征提取 → 预测的流水线。

**Q: 数据是从哪里来的？**
A: 使用加拿大网络安全研究所的 [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) 公开数据集，含 280 万条真实流量记录。

## 技术说明

- **数据规模**：2,830,743 条记录，77 个特征维度
- **特征类型**：流持续时间、包数量、包长度统计、IAT 时间间隔、Flag 计数等
- **不平衡处理**：使用 `class_weight='balanced'` 和 `scale_pos_weight` 处理正负样本不均
- **概率阈值**：自动在验证集上找最优 F1 阈值（不硬用 0.5）
- **评估指标**：Accuracy、Precision、Recall、F1-Score、ROC-AUC
