# SentimentFlow — BERT 分支

> 基于 **BERT**（`hfl/chinese-roberta-wwm-ext`）的中文情感分析全栈项目，支持多数据集融合训练、FastAPI 后端推理与 Next.js 前端展示。

---

## 目录

1. [项目概览](#项目概览)
2. [目录结构](#目录结构)
3. [技术栈](#技术栈)
4. [快速开始](#快速开始)
   - [环境要求](#环境要求)
   - [安装依赖](#安装依赖)
   - [训练模型](#训练模型)
   - [运行推理](#运行推理)
5. [BERT 模块详解](#bert-模块详解)
   - [config.py — 配置中心](#configpy--配置中心)
   - [model.py — 模型定义](#modelpy--模型定义)
   - [text_processing.py — 文本处理](#text_processingpy--文本处理)
   - [dataset.py — 数据集](#datasetpy--数据集)
   - [data_sources.py — 多数据源构建](#data_sourcespy--多数据源构建)
   - [trainer.py — 训练流程](#trainerpy--训练流程)
   - [evaluate.py — 验证评估](#evaluatepy--验证评估)
   - [checkpoint.py — 模型保存与加载](#checkpointpy--模型保存与加载)
   - [pipeline.py — 加载或训练调度](#pipelinepy--加载或训练调度)
   - [inference.py — 单条推理](#inferencepy--单条推理)
   - [main.py — 脚本入口](#mainpy--脚本入口)
6. [支持的数据集](#支持的数据集)
7. [环境变量参考](#环境变量参考)
8. [后端 API](#后端-api)
   - [API 路由](#api-路由)
   - [请求 / 响应格式](#请求--响应格式)
9. [前端](#前端)
10. [Docker 部署](#docker-部署)
11. [BERT vs LSTM 对比](#bert-vs-lstm-对比)
12. [常见问题](#常见问题)

---

## 项目概览

SentimentFlow 是一个中文情感分析系统。**BERT 分支**用预训练的中文 RoBERTa 模型替换了原有的 LSTM 字符哈希方案，显著提升了短句、口语化表达的判断能力。

核心特性：

| 特性 | 说明 |
|------|------|
| 预训练骨干 | `hfl/chinese-roberta-wwm-ext`（可通过环境变量替换） |
| 多数据源融合 | 一次配置，自动合并多个 Hugging Face 数据集 |
| 平衡采样 | 对类别不均衡数据集（如 DMSC）自动做 force-balance |
| 早停 | 基于验证集 Macro-F1 的 patience 早停机制 |
| 加权损失 | 可选的类别权重交叉熵，缓解 0-5 评分分布失衡 |
| 梯度累积 | 在小显存 GPU 上模拟更大有效 batch |
| 混合精度 | CUDA 下自动启用 float16 加速 |
| 短句增强 | 支持合成短句数据集与从长文本提取短句的混合训练 |
| 全栈集成 | FastAPI 后端 + Next.js 前端 + Docker Compose 一键部署 |

---

## 目录结构

```
SentimentFlow/
├── BERT/                          # BERT 训练包（核心）
│   ├── __init__.py
│   ├── config.py                  # 超参数与环境变量配置
│   ├── model.py                   # SentimentBertModel 定义
│   ├── text_processing.py         # Tokenizer 加载与文本编码
│   ├── dataset.py                 # CsvStreamDataset（支持 CSV / HF Dataset）
│   ├── data_sources.py            # 多数据源加载、标签清洗、数据集别名
│   ├── trainer.py                 # 完整训练主流程
│   ├── evaluate.py                # 验证集 Accuracy / Macro-F1 计算
│   ├── checkpoint.py              # 模型与 tokenizer 的保存 / 加载
│   ├── pipeline.py                # load_or_train 调度入口
│   ├── inference.py               # 单条文本推理
│   ├── main.py                    # run() 脚本入口
│   ├── sample_texts.py            # 演示预测示例文本
│   ├── custom_test_cases.py       # 自定义测试用例（质量验证）
│   ├── generate_synthetic_data.py # 合成短句数据生成
│   ├── extract_short_sentences.py # 从长评论提取短句
│   └── env_utils.py               # .env 文件解析工具
│
├── BERT.py                        # 顶层训练入口（python BERT.py）
├── LSTM.py                        # LSTM 基线实现
├── data.py                        # CSV 训练数据生成脚本
│
├── backend/                       # FastAPI 后端
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py                # FastAPI app 入口
│       ├── api/
│       │   ├── predict.py         # POST /api/predict/
│       │   ├── auth.py
│       │   ├── admin.py
│       │   └── stats.py
│       ├── models/
│       │   ├── BERT/
│       │   │   └── executor.py    # BERT 推理封装（供 service 层调用）
│       │   └── LSTM/
│       │       └── executor.py
│       ├── schemas/
│       │   └── predict.py         # PredictRequest / PredictResponse
│       ├── services/
│       │   └── predict_service.py # 模型分发逻辑
│       └── core/
│           └── config.py          # 后端环境变量加载
│
├── frontend/                      # Next.js 前端
│   ├── app/
│   │   ├── page.tsx               # 主页面
│   │   └── api/
│   │       ├── integration/health/route.ts
│   │       └── integration/predict/route.ts
│   └── components/
│       ├── integration-test-panel.tsx
│       └── theme-toggle.tsx
│
├── docker-compose.yml             # 一键启动全栈
└── .gitignore
```

---

## 技术栈

**训练侧**

| 组件 | 版本要求 |
|------|----------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 |
| Transformers (Hugging Face) | ≥ 4.35 |
| Datasets (Hugging Face) | ≥ 2.14 |
| pandas | ≥ 2.0 |

**后端**

| 组件 | 说明 |
|------|------|
| FastAPI | REST API 框架 |
| Uvicorn | ASGI 服务器 |
| Pydantic | 请求/响应校验 |
| jieba | 可选的分词工具 |

**前端**

| 组件 | 说明 |
|------|------|
| Next.js 14 | React 全栈框架 |
| TypeScript | 类型安全 |
| Tailwind CSS + shadcn/ui | UI 样式组件 |

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- （推荐）CUDA 兼容 GPU，至少 8 GB 显存
- Node.js ≥ 20（仅运行前端时需要）

### 安装依赖

```bash
# 克隆仓库并切换到 BERT 分支
git clone https://github.com/Cthaat/SentimentFlow.git
cd SentimentFlow
git checkout BERT

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装 Python 依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pandas jieba
```

### 训练模型

```bash
# 使用默认数据集（二分类旧数据 + 1-5 星级评分数据）训练 BERT 0-5 评分模型
python BERT.py

# 强制重新训练（忽略已有 checkpoint）
BERT_FORCE_RETRAIN=1 python BERT.py

# 自定义数据集与超参数
BERT_TRAIN_DATASETS="lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/JD_review,BerlinWang/DMSC" \
BERT_EPOCHS=3 \
BERT_TRAIN_BATCH_SIZE=32 \
python BERT.py
```

训练完成后，模型会保存到 `./bert_sentiment_model/`（HuggingFace 格式，含 `config.json`、`pytorch_model.bin`、tokenizer 配置及 `training_meta.json`）。

### 运行推理

```python
from BERT.pipeline import load_or_train
from BERT.inference import predict_text

model, device = load_or_train()   # 自动加载已有 checkpoint

result = predict_text("这个产品质量太差了，完全不推荐！", model, device)
print(result)
# {'text': '...', 'score': 0, 'label': 'extremely_negative', 'label_zh': '极端负面',
#  'confidence': 0.997, 'probabilities': [0.997, 0.0, 0.0, 0.0, 0.0, 0.003]}
```

---

## BERT 模块详解

### config.py — 配置中心

所有超参数均可通过环境变量覆盖，代码中不需要硬编码。

```python
MAX_LEN = int(os.getenv("BERT_MODEL_MAX_LEN", "128"))       # 最大 token 长度
EPOCHS  = int(os.getenv("BERT_EPOCHS", "5"))                 # 训练轮数
CHECKPOINT_PATH = os.getenv("BERT_CHECKPOINT_PATH", "./bert_sentiment_model")
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")
```

`get_runtime_settings(device_type)` 函数根据当前设备（cuda / cpu）生成 `RuntimeSettings` 数据类，包含 `batch_size`、`num_workers`、`grad_accum_steps`、`learning_rate` 等运行时参数，全部支持环境变量覆盖（见 [环境变量参考](#环境变量参考)）。

---

### model.py — 模型定义

```python
class SentimentBertModel(nn.Module):
    def __init__(self, model_name=BERT_MODEL_NAME):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=6
        )

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
```

- 直接使用 `AutoModelForSequenceClassification`，HuggingFace 自动接上 6 分类输出头。
- 支持任意兼容的中文 BERT/RoBERTa 模型，只需修改 `BERT_MODEL_NAME`。

---

### text_processing.py — 文本处理

```python
@lru_cache(maxsize=2)
def get_tokenizer(model_name=BERT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)

def encode_text(text, max_len):
    tokenizer = get_tokenizer()
    encoded = tokenizer(str(text or ""), max_length=max_len,
                        truncation=True, padding="max_length",
                        return_tensors="pt")
    return {"input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)}
```

- `get_tokenizer` 通过 `lru_cache` 缓存，避免多次加载，节省内存与启动时间。
- 在 DataLoader 的 `collate_fn`（`bert_collate_fn`）中对整个 batch 批量 tokenize，相比逐条处理性能提升 100 倍以上。

---

### dataset.py — 数据集

`CsvStreamDataset` 实现了 `torch.utils.data.IterableDataset`，支持两种数据源：

1. **CSV 文件**：按 `chunk_size` 流式读取，适合超大文件，避免内存爆炸。
2. **HuggingFace Dataset 对象**：直接索引访问，支持多 worker 分片。

**关键设计**：Dataset 只返回 `(text, label)` 原始字符串，不在单条样本层面 tokenize。批量 tokenize 由 DataLoader 的 `collate_fn` 统一完成，大幅提升吞吐量。

`label_map` 参数可将原始标签映射到 `{0, 1, 2, 3, 4, 5}`，`-1` 表示跳过该样本。旧二分类数据自动按 `0 -> 0`、`1 -> 5` 迁移。

CSV 数据的旧二分类识别按整份文件判断，而不是按单个 chunk 判断：只有整份 CSV 的标签集合确认为 `{0, 1}` 时才会自动迁移，真实 0-5 CSV 中的 `1` 分会保持为 `1`。若你的数据集只有 `0/1` 两档但语义是 0-5 评分子集，可设置 `MIGRATE_LEGACY_BINARY_LABELS=0` 关闭自动迁移。

---

### data_sources.py — 多数据源构建

`build_train_split_and_val_split()` 是数据准备的核心函数，流程如下：

1. **数据集名称解析**：通过 `DATASET_ALIASES` 将缩写映射到 HuggingFace 标准名称。
2. **数据集加载**：通过 `datasets.load_dataset()` 下载，自动处理 `train / validation / test` 分割；若数据集无验证集，则按 `TRAIN_VAL_RATIO` 自动切分。
3. **标签清洗**（`_normalize_split_columns`）：
   - 自动识别文本列和标签列（支持多种列名格式）。
   - 将多元标签统一转换为 0-5 情感评分；1-5 星级数据映射到评分档位，旧 0/1 数据迁移到 0/5。
4. **平衡采样**（`_force_balance_score_split`）：对 DMSC 等评分分布不均的数据集，按存在的评分类别做下采样均衡。
5. **多数据集合并**：用 `datasets.concatenate_datasets` 拼接，再整体 shuffle。
6. **短句增强**（可选）：
   - `USE_EXTRACTED_SHORT_SENTENCES=1`：加载 `BERT/extracted_short_sentences.csv` 混入短句。
   - `USE_SYNTHETIC_DATA=1`：使用 `generate_synthetic_data.py` 生成合成短句。

---

### trainer.py — 训练流程

`train_model()` 完整训练流程：

```
初始化设备 (CUDA / CPU)
→ 获取运行时配置 (RuntimeSettings)
→ 加载并合并多数据集
→ 统计 0-5 样本分布，构建加权 CrossEntropyLoss（可选）
→ 构建 DataLoader（批量 tokenize via collate_fn）
→ 初始化 SentimentBertModel，AdamW + LinearWarmup 调度器
→ 混合精度训练循环（GradScaler）+ 梯度累积
→ 每 epoch 结束后在验证集评估 Accuracy / Macro-F1 / Weighted-F1 / MAE / QWK
→ 保存最优 checkpoint（基于 val_f1）
→ 早停（patience 轮 val_f1 不提升则提前结束）
→ 加载最优 checkpoint 返回
```

关键优化点：

| 优化 | 说明 |
|------|------|
| 批量 tokenize | `collate_fn` 一次处理整批文本，避免逐条 tokenize |
| 混合精度 | `torch.amp.GradScaler` + `autocast`，CUDA 下自动启用 |
| 梯度累积 | `BERT_TRAIN_ACCUM_STEPS` 控制，小显存下模拟大 batch |
| 加权损失 | `BERT_TRAIN_WEIGHTED_LOSS=1` 时按 0-5 各评分样本比例加权 |
| 早停 | `BERT_EARLY_STOP_PATIENCE` 控制容忍轮数 |
| 最优 checkpoint | 按验证集 Macro-F1 保存最优，训练结束后自动恢复 |

---

### evaluate.py — 验证评估

`evaluate(model, split, device, batch_size, max_len)` 在验证集上计算：

- **Accuracy**：`(TP + TN) / Total`
- **Macro-F1**：各存在评分类别 F1 的算术平均，更能反映类别不平衡下的真实性能

采用与训练相同的 `bert_collate_fn`，保证评估和训练的 tokenize 行为完全一致。

---

### checkpoint.py — 模型保存与加载

**保存**（`save_checkpoint`）：

```
bert_sentiment_model/
├── config.json          # HuggingFace 模型配置
├── pytorch_model.bin    # 模型权重（或 safetensors）
├── tokenizer_config.json
├── vocab.txt
└── training_meta.json   # 训练元信息（max_len, model_name, best_val_f1, best_epoch）
```

**加载**（`load_checkpoint`）：从 `training_meta.json` 恢复 `model_name` 和 `max_len`，保证推理时与训练时参数完全一致。

---

### pipeline.py — 加载或训练调度

`load_or_train()` 实现"有则加载、无则训练"的策略：

```python
if not force_retrain:
    model = load_checkpoint(device)
    if model is not None:
        return model, device
return train_model()
```

`BERT_FORCE_RETRAIN=1`（或 `FORCE_RETRAIN=1`）可强制跳过加载直接训练。

---

### inference.py — 单条推理

`predict_text(text, model, device, max_len)` 返回结构化预测结果：

```python
{
    "text":            "这个产品非常好用",
    "score":           5,
    "label":           "extremely_positive",
    "label_zh":        "极端正面",
    "confidence":      0.998,
    "probabilities":   [0.001, 0.0, 0.0, 0.001, 0.0, 0.998],
    "reasoning":       "模型将文本情感强度判定为 5 分（极端正面）。",
}
```

使用 `torch.inference_mode()` 关闭梯度计算，并通过 `softmax` 将 6 维 logits 转换为概率，取概率最高的评分作为最终预测。

---

### main.py — 脚本入口

`run()` 函数是训练入口，执行以下步骤：

1. 加载 `.env` 文件（不覆盖已有环境变量）
2. 打印当前配置（数据集、模型名、是否强制重训）
3. 根据 `BERT_FORCE_RETRAIN` 决定训练或加载模型
4. 对 `DEFAULT_SAMPLES` 中所有示例文本逐条预测并打印
5. 执行 `CUSTOM_TEST_CASES` 质量验证，输出通过率

---

## 支持的数据集

| 数据集 | HuggingFace 名称 | 类型 | 说明 |
|--------|-----------------|------|------|
| ChnSentiCorp | `lansinuote/ChnSentiCorp` | 默认 | 旧二分类数据，自动 0→0 / 1→5 |
| 外卖评论 | `XiangPan/waimai_10k` | 默认 | 旧二分类数据，自动 0→0 / 1→5 |
| 微博情感 | `dirtycomputer/weibo_senti_100k` | 可选 | 旧二分类数据，自动 0→0 / 1→5 |
| JD 商品评论 | `dirtycomputer/JD_review` | 默认 | 京东评论，1-5 星映射到 0-5 评分 |
| NLPCC14-SC | `ndiy/NLPCC14-SC` | 可选 | NLPCC 2014 情感评测数据 |
| 酒店评论 | `dirtycomputer/ChnSentiCorp_htl_all` | 可选 | ChnSentiCorp 酒店完整版 |
| DMSC | `BerlinWang/DMSC` | 默认 | 豆瓣影评，1-5 星，自动平衡 |

**使用别名**（`DATASET_ALIASES`）时，可以用简写名：

```bash
BERT_TRAIN_DATASETS="dmsc,jd_reviews,weibo_senti" python BERT.py
```

---

## 环境变量参考

所有训练参数均可通过 `.env` 文件或 Shell 环境变量配置，无需修改代码。

### 模型与训练

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_MODEL_NAME` | `hfl/chinese-roberta-wwm-ext` | 预训练模型名称（HuggingFace Hub 或本地路径） |
| `BERT_MODEL_MAX_LEN` | `128` | 最大 token 序列长度 |
| `BERT_EPOCHS` | `5` | 最大训练轮数 |
| `BERT_CHECKPOINT_PATH` | `./bert_sentiment_model` | Checkpoint 保存目录 |
| `BERT_FORCE_RETRAIN` | `0` | `1` = 忽略已有 checkpoint，强制重新训练 |

### 运行时参数

| 变量 | 默认值（GPU/CPU） | 说明 |
|------|-----------------|------|
| `BERT_TRAIN_BATCH_SIZE` | `32` / `16` | 训练 batch 大小 |
| `BERT_EVAL_BATCH_SIZE` | `64` / `32` | 验证 batch 大小 |
| `BERT_TRAIN_NUM_WORKERS` | `2` / `0` | DataLoader worker 数量 |
| `BERT_TRAIN_ACCUM_STEPS` | `1` | 梯度累积步数 |
| `BERT_TRAIN_CHUNK_SIZE` | `batch_size * 8` | CSV 流式读取块大小 |
| `BERT_TRAIN_LR` | `2e-5` | AdamW 学习率 |
| `BERT_TRAIN_WEIGHTED_LOSS` | `1` | `1` = 启用加权交叉熵损失 |

### 早停

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_EARLY_STOP_PATIENCE` | `2` | 容忍验证集 F1 不提升的轮数（0 = 关闭早停） |
| `BERT_EARLY_STOP_MIN_DELTA` | `0.0005` | F1 提升的最小门槛 |

### 数据集

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_TRAIN_DATASETS` | `lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/JD_review,BerlinWang/DMSC` | 逗号分隔的数据集名称列表，也支持本地 CSV 路径 |
| `TRAIN_VAL_RATIO` | `0.1` | 无 validation split 时自动切分比例 |
| `TRAIN_MAX_SAMPLES` | `0`（不限） | 训练集最大样本数 |
| `TRAIN_MAX_VAL_SAMPLES` | `0`（不限） | 验证集最大样本数 |
| `MIGRATE_LEGACY_BINARY_LABELS` | `auto` | 自动识别旧 0/1 标签并迁移到 0/5；设为 `0` 可关闭 |

### 半监督 0/1 细分

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SEMI_SUPERVISED_01_TO_05` | `auto` | `0` 关闭；其他值表示尝试用 teacher 将旧 0/1 数据细分到 0-5 |
| `PSEUDO_LABEL_TEACHER_PATH` | 空 | 通用 teacher checkpoint 路径 |
| `LSTM_PSEUDO_LABEL_TEACHER_PATH` | 空 | LSTM 专用 teacher `.pt` 路径 |
| `BERT_PSEUDO_LABEL_TEACHER_PATH` | 空 | BERT 专用 teacher 目录 |
| `PSEUDO_LABEL_MIN_CONFIDENCE` | `0.45` | teacher 伪标签最低置信度 |
| `PSEUDO_LABEL_FALLBACK_TO_ENDPOINT` | `1` | 低置信度时保留弱标签端点：`0->0`、`1->5` |
| `PSEUDO_LABEL_BATCH_SIZE` | LSTM `128` / BERT `64` | teacher 推理 batch 大小 |
| `PSEUDO_LABEL_VALIDATION_SPLIT` | `0` | 是否也对 validation split 打伪标签；默认只处理训练集 |

半监督细分会保留 0/1 弱标签的极性约束：原始 `0` 只会被细分为 `0/1/2`，原始 `1` 只会被细分为 `4/5`。这避免 teacher 把弱负面样本误打成正面，或把弱正面样本误打成负面。

### 短句增强

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `USE_EXTRACTED_SHORT_SENTENCES` | `1` | `1` = 混入 `extracted_short_sentences.csv` |
| `EXTRACTED_SHORT_SENTENCES_MAX` | `0`（不限） | 短句最大数量 |
| `SHORT_SENTENCES_RATIO` | `0.3` | 短句在训练集中的比例 |
| `USE_SYNTHETIC_DATA` | `0` | `1` = 生成合成短句混入训练 |
| `SYNTHETIC_DATA_SIZE` | `5000` | 合成短句数量 |

历史版本生成的 `extracted_short_sentences.csv` 可能仍是旧二分类标签，其中 `1` 表示正面。训练管道会先检查整份短句 CSV 的标签集合：若只有 `{0, 1}`，按旧二分类迁移到 `0/5`；若已经包含 0-5 多档标签，则保持原始评分。已经用错误短句标签训练出的 checkpoint 应删除、切换到健康模型，或重新训练。

### 后端

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREDICT_MODEL_TYPE` | `lstm` | `bert` 或 `lstm`，决定后端使用哪个模型推理 |

---

## 后端 API

### 启动后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 Swagger 文档：http://localhost:8000/docs

### API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，返回 `{"status": "ok"}` |
| `POST` | `/api/predict/` | 情感预测（核心接口） |
| `POST` | `/api/auth/register` | 用户注册 |
| `POST` | `/api/auth/login` | 用户登录 |
| `GET` | `/api/admin/...` | 管理接口 |
| `GET` | `/api/stats/...` | 统计接口 |

### 请求 / 响应格式

**POST `/api/predict/`**

请求体（`application/json`）：

```json
{
  "text": "这个手机的电池续航很不错！",
  "model": "bert"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | `string` | ✅ | 待分析文本（1 ~ 2000 字符） |
| `model` | `"lstm" \| "bert" \| null` | ❌ | 指定模型，不传则使用 `PREDICT_MODEL_TYPE` |

响应体：

```json
{
  "text":   "这个手机的电池续航很不错！",
  "score":  4,
  "label":  "slightly_positive",
  "label_zh": "略微正面",
  "confidence": 0.8731,
  "probabilities": [0.002, 0.006, 0.018, 0.081, 0.873, 0.020],
  "reasoning": "模型将文本情感强度判定为 4 分（略微正面）。",
  "source": "bert"
}
```

| 字段 | 说明 |
|------|------|
| `text` | 原始输入文本 |
| `score` | 情感评分，整数 0-5 |
| `label` | 机器可读标签，如 `"slightly_positive"` |
| `label_zh` | 中文展示标签 |
| `confidence` | 预测评分的置信度（0 ~ 1） |
| `probabilities` | 长度为 6 的概率数组，索引即评分 |
| `reasoning` | 简短解释文本 |
| `source` | 模型来源：`"bert"` 或 `"lstm"` |

---

## 前端

```bash
cd frontend
yarn install
yarn dev
```

访问 http://localhost:3000

前端提供：
- 情感分析主界面（文本输入 → 实时预测结果展示）
- 联调测试面板（`IntegrationTestPanel`）：批量发送预测请求，验证后端连通性
- 深色/浅色主题切换（`ThemeToggle`）

前端通过 Next.js 的 API 路由（`/app/api/integration/predict/route.ts`）转发请求到后端，避免跨域问题。

---

## Docker 部署

项目根目录提供 `docker-compose.yml`，一条命令启动完整全栈：

```bash
# 在项目根目录（含 .env 文件）执行
docker compose up --build
```

服务说明：

| 服务 | 端口 | 说明 |
|------|------|------|
| `backend` | `8000` | FastAPI 后端，挂载 `./backend` 目录，支持热重载 |
| `frontend` | `3000` | Next.js 前端，Node.js 20 官方镜像 |

**注意**：BERT 模型文件（`bert_sentiment_model/`）体积较大，建议在启动 Docker 前先完成训练，并将 checkpoint 目录挂载到容器内，或通过 `BERT_CHECKPOINT_PATH` 指向共享卷。

---

## BERT vs LSTM 对比

| 维度 | LSTM 分支 | BERT 分支 |
|------|-----------|-----------|
| 骨干网络 | 字符哈希 Embedding + 2 层 LSTM | 预训练 RoBERTa（chinese-roberta-wwm-ext） |
| 词表 | 哈希映射（65536 槽位，无需预统计） | HuggingFace WordPiece 分词（21128 词表） |
| 训练数据 | 自动生成合成 CSV | 多源真实中文 NLP 数据集 |
| 模型参数 | ~12 M | ~102 M |
| 短句理解 | 较弱（字符哈希信息损失大） | 较强（预训练语言模型捕获语义） |
| 训练速度 | 快（CPU 可训练） | 慢（推荐 GPU，至少 8 GB 显存） |
| Checkpoint 格式 | PyTorch `.pt` 单文件 | HuggingFace 标准目录（含 tokenizer） |
| 推理延迟 | ~1 ms | ~20 ms（CPU）/ ~5 ms（GPU） |

---

## 常见问题

**Q：训练时显存不足（CUDA OOM），怎么办？**

减小 batch size 并增大梯度累积步数，保持有效 batch 大小不变：

```bash
BERT_TRAIN_BATCH_SIZE=8 BERT_TRAIN_ACCUM_STEPS=4 python BERT.py
```

**Q：如何使用其他中文 BERT 模型？**

修改 `BERT_MODEL_NAME` 即可，例如使用 BERT-base：

```bash
BERT_MODEL_NAME=bert-base-chinese python BERT.py
```

**Q：在 Windows 上多 worker 训练崩溃？**

代码已自动将 Windows 下 `BERT_TRAIN_NUM_WORKERS` 上限截断为 4。如果仍崩溃，尝试：

```bash
BERT_TRAIN_NUM_WORKERS=0 python BERT.py
```

**Q：如何只评估模型，不重新训练？**

```python
from BERT.pipeline import load_or_train
from BERT.evaluate import evaluate
import torch

model, device = load_or_train()   # BERT_FORCE_RETRAIN=0 时直接加载

# 加载验证集
from BERT.data_sources import build_train_split_and_val_split
_, _, val_split, _ = build_train_split_and_val_split()

metrics = evaluate(model, val_split, device, batch_size=64, max_len=128)
print(
    f"Accuracy: {metrics.accuracy:.4f}, Macro-F1: {metrics.macro_f1:.4f}, "
    f"MAE: {metrics.mae:.4f}, QWK: {metrics.quadratic_weighted_kappa:.4f}"
)
```

**Q：`extracted_short_sentences.csv` 是什么？如何生成？**

该文件由 `BERT/extract_short_sentences.py` 从多个数据集的长评论中提取 5-20 字的短句生成，用于增强模型对短句的判断能力。运行：

```bash
python -m BERT.extract_short_sentences
```

生成后放置于 `BERT/extracted_short_sentences.csv`，训练时通过 `USE_EXTRACTED_SHORT_SENTENCES=1` 启用（默认启用）。

**Q：后端如何切换使用 BERT 模型推理？**

在 `backend/.env` 中设置：

```
PREDICT_MODEL_TYPE=bert
```

或在 API 请求体中指定 `"model": "bert"` 逐次覆盖。
