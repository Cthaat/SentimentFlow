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
   - [model.py — 多分类模型定义](#modelpy--多分类模型定义)
   - [text_processing.py — 文本处理](#text_processingpy--文本处理)
   - [dataset.py — 数据集](#datasetpy--数据集)
   - [data_sources.py — 多数据源构建](#data_sourcespy--多数据源构建)
   - [trainer.py — 训练流程](#trainerpy--训练流程)
   - [evaluate.py — 验证评估](#evaluatepy--验证评估)
   - [checkpoint.py — 模型保存与加载](#checkpointpy--模型保存与加载)
   - [pipeline.py — 加载或训练调度](#pipelinepy--加载或训练调度)
   - [inference.py — 批量推理与部署导出](#inferencepy--批量推理与部署导出)
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
| 两阶段训练 | Teacher 仅用 DMSC/JD_review 真实多分类数据；Student 混合真实标签与高置信伪标签 |
| 软伪标签 | Teacher 对二分类数据生成 0-5 soft pseudo labels，结合弱极性候选桶过滤，低置信样本直接丢弃 |
| 早停 | 默认基于验证集 QWK 的 patience 早停机制 |
| 六分类模型 | BERT/RoBERTa backbone + 6 维 score logits |
| 多分类损失 | Class-balanced focal CrossEntropy + expected-score 距离惩罚 |
| 类别不平衡 | Effective-number class weights、logit adjustment、分布感知过采样，保留全量数据 |
| 梯度累积 | 在小显存 GPU 上模拟更大有效 batch |
| 混合精度 | 支持 fp16 / bf16，可选 Accelerate |
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
│   ├── evaluate.py                # Accuracy / F1 / MAE / RMSE / QWK / Spearman
│   ├── checkpoint.py              # 模型与 tokenizer 的保存 / 加载
│   ├── pipeline.py                # load_or_train 调度入口
│   ├── inference.py               # 批量推理、softmax argmax 评分、ONNX 导出
│   ├── export.py                  # checkpoint -> ONNX CLI
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
pip install transformers datasets pandas jieba accelerate
```

### 训练模型

```bash
# Stage 1: 仅用真实多分类数据训练 Teacher
BERT_TRAINING_STAGE=teacher \
BERT_CHECKPOINT_PATH=models/bert_teacher \
python BERT.py

# Stage 2: Teacher 生成 soft pseudo labels，并训练 Student
BERT_TRAINING_STAGE=student \
BERT_TEACHER_CHECKPOINT_PATH=models/bert_teacher \
PSEUDO_LABEL_PATH=pseudo_labels.jsonl \
BERT_CHECKPOINT_PATH=models/bert_student \
python BERT.py

# 强制重新训练（忽略已有 checkpoint）
BERT_FORCE_RETRAIN=1 python BERT.py

# 自定义超参数
BERT_EPOCHS=3 \
BERT_TRAIN_BATCH_SIZE=32 \
BERT_TRAINING_STAGE=teacher \
python BERT.py
```

训练完成后，模型会保存到 `./bert_sentiment_model/`（HuggingFace backbone + 自定义评分头，含 `config.json`、`model.safetensors`/`pytorch_model.bin`、`sentiment_model_state.pt`、tokenizer 配置及 `training_meta.json`）。

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

### model.py — 多分类模型定义

```python
class SentimentBertModel(nn.Module):
    def __init__(self, model_name=BERT_MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_size, 6)

    def forward(self, input_ids, attention_mask, return_dict=False):
        ...
        return {"logits": class_logits}
```

- 默认架构为 `BERT_MODEL_ARCHITECTURE=multiclass`，输出 6 类评分 logits。
- 训练目标使用 softmax CrossEntropy，并额外加入 expected-score 距离正则，缓解“预测 5 错成 4”和“预测 5 错成 0”惩罚相同的问题。
- `BERT_MODEL_ARCHITECTURE=sequence` 保留旧 checkpoint 兼容路径。
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

**关键设计**：BERT Dataset 返回标准训练记录 `{text, label, soft_labels, sample_weight, label_source}`，不在单条样本层面 tokenize。批量 tokenize 由 DataLoader 的 `collate_fn` 统一完成，大幅提升吞吐量。

`label_map` 参数可将原始标签映射到 `{0, 1, 2, 3, 4, 5}`，`-1` 表示跳过该样本。CSV/legacy 兼容路径仍可把旧二分类数据按 `0 -> 0`、`1 -> 5` 迁移，但两阶段 BERT 主流程不会把这种端点映射样本直接用于 Teacher。

CSV 数据的旧二分类识别按整份文件判断，而不是按单个 chunk 判断：只有整份 CSV 的标签集合确认为 `{0, 1}` 时才会自动迁移，真实 0-5 CSV 中的 `1` 分会保持为 `1`。两阶段 BERT 训练不会把二分类数据直接映射到 0/5 参与 Teacher 训练；二分类数据只在 Student 阶段由 Teacher 生成 soft pseudo labels 后使用。

---

### data_sources.py — 多数据源构建

`build_train_split_and_val_split()` 是数据准备的核心函数，流程如下：

1. **数据集名称解析**：通过 `DATASET_ALIASES` 将缩写映射到 HuggingFace 标准名称。
2. **数据集加载**：通过 `datasets.load_dataset()` 下载，自动处理 `train / validation / test` 分割；若数据集无验证集，则按 `TRAIN_VAL_RATIO` 自动切分。
3. **标签清洗**（`_normalize_split_columns`）：
   - 自动识别文本列和标签列（支持多种列名格式）。
   - 将多元标签统一转换为 0-5 情感评分；1-5 星级数据映射到评分档位；旧 0/1 端点迁移只保留在 legacy 兼容路径。
4. **阶段路由**：
   - `teacher`：只加载 `BerlinWang/DMSC` 与 `dirtycomputer/JD_review`。
   - `student`：加载真实多分类数据，并读取/生成 `pseudo_labels.jsonl`。
   - `legacy`：保留旧的直接多数据集训练入口，仅用于兼容。
5. **伪标签生成**：Teacher 输出 logits，经 temperature scaling + softmax 得到概率；结合二分类弱极性候选桶选择 0-5 分，仅保留 `confidence >= 0.75` 的样本。
6. **多数据集合并**：真实标签与伪标签用统一 schema 拼接，伪标签默认 `sample_weight=0.3`。
7. **短句增强**（legacy 可选）：
   - `USE_EXTRACTED_SHORT_SENTENCES=1`：加载 `BERT/extracted_short_sentences.csv` 混入短句。
   - `USE_SYNTHETIC_DATA=1`：使用 `generate_synthetic_data.py` 生成合成短句。

---

### trainer.py — 训练流程

`train_model()` 完整训练流程：

```
初始化设备 (CUDA / CPU)
→ 获取运行时配置 (RuntimeSettings)
→ 加载并合并多数据集
→ 统计 0-5 样本分布，构建 class weights（可选）
→ 构建 DataLoader（批量 tokenize via collate_fn）
→ 初始化 SentimentBertModel，AdamW
→ 使用 DistanceAwareOrdinalLoss（soft CE + expected-score distance）
→ multiclass loss：class-balanced focal CrossEntropy + SmoothL1 score distance
→ 混合精度训练循环（fp16/bf16）+ 梯度累积 + cosine warmup
→ 每 epoch 结束后在验证集评估 Accuracy / Macro-F1 / Weighted-F1 / MAE / RMSE / QWK / Spearman
→ 保存最优 checkpoint（默认基于 QWK）
→ 早停（patience 轮 selection metric 不提升则提前结束）
→ 加载最优 checkpoint 返回
```

关键优化点：

| 优化 | 说明 |
|------|------|
| 批量 tokenize | `collate_fn` 一次处理整批文本，避免逐条 tokenize |
| 混合精度 | `torch.amp.GradScaler` + `autocast`，CUDA 下自动启用 |
| 梯度累积 | `BERT_TRAIN_ACCUM_STEPS` 控制，小显存下模拟大 batch |
| Distance-aware multiclass loss | 6 类 softmax CE + 评分距离正则 |
| Focal + class-balanced loss | `FOCAL_GAMMA`、`CLASS_BALANCED_BETA` 抑制多数类主导 |
| Logit adjustment | `LOGIT_ADJUSTMENT_WEIGHT` 按类别先验修正偏置 |
| 缺失类插值 | `BERT_INTERPOLATE_MISSING_LABELS=1` 为 score=2 等缺失档生成低权重 soft labels |
| 分布感知过采样 | `BERT_DISTRIBUTION_AWARE_OVERSAMPLING=1` 只追加少数类样本，不裁剪多数类 |
| Layer-wise LR decay | `BERT_LAYERWISE_LR_DECAY` 让底层 backbone 更稳定 |
| Cosine warmup | `BERT_SCHEDULER=cosine`、`BERT_WARMUP_RATIO=0.06` |
| Resume | `BERT_RESUME_FROM_CHECKPOINT` 加载权重与 `trainer_state.pt` |
| 早停 | `BERT_EARLY_STOP_PATIENCE` 控制容忍轮数 |
| 最优 checkpoint | 默认按验证集 QWK 保存最优，训练结束后自动恢复 |

---

### evaluate.py — 验证评估

`evaluate(model, split, device, batch_size, max_len)` 在验证集上计算：

- **Accuracy**：`(TP + TN) / Total`
- **Macro-F1**：各存在评分类别 F1 的算术平均，更能反映类别不平衡下的真实性能
- **Weighted-F1 / Per-class F1**：观察多数类和少数类表现是否分裂
- **MAE / RMSE**：评分距离误差
- **QWK**：核心选模指标，适合 0-5 有序评分
- **Spearman**：预测排序与真实评分排序的一致性

采用与训练相同的 `bert_collate_fn`，并通过 6 类 softmax argmax 计算最终离散评分，保证评估和推理使用同一评分逻辑。

---

### checkpoint.py — 模型保存与加载

**保存**（`save_checkpoint`）：

```
bert_sentiment_model/
├── config.json          # HuggingFace 模型配置
├── pytorch_model.bin    # 模型权重（或 safetensors）
├── sentiment_model_state.pt # multiclass 分类头完整权重
├── tokenizer_config.json
├── vocab.txt
└── training_meta.json   # 训练元信息（max_len, model_name, architecture, QWK, MAE 等）
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

### inference.py — 批量推理与部署导出

`predict_text(text, model, device, max_len)` 和 `predict_batch(texts, model, device)` 返回结构化预测结果：

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

推理使用 6 类 softmax 概率，并通过 `argmax(dim=1)` 得到最终 0-5 分。可选优化：

- `BERT_INFERENCE_BATCH_SIZE=64`：批量推理动态 padding。
- `BERT_INFERENCE_TORCH_COMPILE=1`：启用 `torch.compile` 推理。
- `BERT_DYNAMIC_QUANTIZATION=1`：CPU 上启用 Linear 动态 int8 量化。
- ONNX 导出：

```bash
python -m BERT.export models/bert_student exports/bert_student.onnx --max-len 128
```

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
| DMSC | `BerlinWang/DMSC` | Teacher/Student real | 豆瓣影评，1-5 星映射到 0-5 评分 |
| JD 商品评论 | `dirtycomputer/JD_review` | Teacher/Student real | 京东评论，1-5 星映射到 0-5 评分 |
| ChnSentiCorp | `lansinuote/ChnSentiCorp` | Student pseudo | 二分类弱标签，只由 Teacher 生成 soft pseudo labels |
| 外卖评论 | `XiangPan/waimai_10k` | Student pseudo | 二分类弱标签，只由 Teacher 生成 soft pseudo labels |
| 微博情感 | `dirtycomputer/weibo_senti_100k` | Student pseudo | 二分类弱标签，只由 Teacher 生成 soft pseudo labels |
| NLPCC14-SC | `ndiy/NLPCC14-SC` | Student pseudo | 二分类弱标签，只由 Teacher 生成 soft pseudo labels |
| 酒店评论 | `dirtycomputer/ChnSentiCorp_htl_all` | Student pseudo | 二分类弱标签，只由 Teacher 生成 soft pseudo labels |

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
| `BERT_TRAINING_STAGE` | `auto` | `teacher` / `student` / `legacy`；`auto` 有 teacher path 时走 student，否则走 teacher |
| `BERT_RESUME_FROM_CHECKPOINT` | 空 | 从已有 BERT checkpoint 恢复模型权重，并尽量恢复 `trainer_state.pt` |
| `BERT_MODEL_ARCHITECTURE` | `multiclass` | `multiclass` = 6 类 softmax 分类；`sequence` 仅用于旧 checkpoint 兼容 |
| `BERT_SELECTION_METRIC` | `qwk` | 最优 checkpoint/早停选模指标，可设 `qwk` / `macro_f1` / `weighted_f1` / `mae` |

### 运行时参数

| 变量 | 默认值（GPU/CPU） | 说明 |
|------|-----------------|------|
| `BERT_TRAIN_BATCH_SIZE` | `32` / `16` | 训练 batch 大小 |
| `BERT_EVAL_BATCH_SIZE` | `64` / `32` | 验证 batch 大小 |
| `BERT_TRAIN_NUM_WORKERS` | `2` / `0` | DataLoader worker 数量 |
| `BERT_TRAIN_ACCUM_STEPS` | `1` | 梯度累积步数 |
| `BERT_TRAIN_CHUNK_SIZE` | `batch_size * 8` | CSV 流式读取块大小 |
| `BERT_TRAIN_LR` | `2e-5` | AdamW 学习率 |
| `BERT_WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `BERT_LAYERWISE_LR_DECAY` | `0.9` | backbone 层级学习率衰减 |
| `BERT_HEAD_LR_MULTIPLIER` | `2.0` | 分类头学习率倍率 |
| `BERT_SCHEDULER` | `cosine` | `cosine` / `linear` / `none` |
| `BERT_WARMUP_RATIO` | `0.06` | scheduler warmup 比例 |
| `BERT_TRAIN_WEIGHTED_LOSS` | `1` | `1` = 启用 class-weighted multiclass loss |
| `BERT_GRADIENT_CHECKPOINTING` | `1` | `1` = 启用 backbone gradient checkpointing |
| `BERT_MIXED_PRECISION` | `fp16` | `fp16` / `bf16` |
| `BERT_USE_ACCELERATE` | `0` | `1` = 使用 HuggingFace Accelerate（需通过 accelerate launch 启动） |
| `BERT_FUSED_ADAMW` | `1` | CUDA 下优先使用 fused AdamW |
| `BERT_TORCH_COMPILE` | `0` | 实验性训练编译，默认关闭以保证 checkpoint/resume 稳定 |

### Loss 与类别不平衡

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORDINAL_CE_WEIGHT` | `1.0` | 6 类 soft CE 权重 |
| `ORDINAL_DISTANCE_WEIGHT` | `0.35` | 期望评分 SmoothL1 距离损失权重 |
| `ORDINAL_LABEL_SMOOTHING` | `0.05` | 真实标签 smoothing |
| `PSEUDO_LABEL_SMOOTHING` | `0.02` | 伪标签 smoothing |
| `FOCAL_GAMMA` | `1.5` | focal loss gamma |
| `CLASS_BALANCED_BETA` | `0.9999` | effective number class-balanced loss beta |
| `LOGIT_ADJUSTMENT_WEIGHT` | `0.3` | 类别先验 logit adjustment 权重 |
| `BERT_INTERPOLATE_MISSING_LABELS` | `1` | 对真实多分类数据中缺失的中间评分生成低权重 soft-label 样本 |
| `INTERPOLATED_LABEL_RATIO` | `0.15` | 缺失类插值样本占邻域样本比例 |
| `INTERPOLATED_LABEL_WEIGHT` | `0.2` | 插值样本训练权重 |
| `BERT_DISTRIBUTION_AWARE_OVERSAMPLING` | `1` | 追加少数类样本，不进行 force-balanced downsampling |
| `BERT_OVERSAMPLE_TEMPERATURE` | `0.5` | sqrt 风格过采样强度 |
| `BERT_OVERSAMPLE_MAX_ADDED` | `500000` | 过采样最多追加样本数 |
| `BERT_PSEUDO_CURRICULUM_EPOCHS` | `2` | Student 阶段伪标签/插值标签权重渐进升温 epoch 数 |
| `BERT_PSEUDO_CURRICULUM_START_SCALE` | `0.3` | 课程学习开始时伪标签/插值样本额外缩放系数 |

### 早停

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_EARLY_STOP_PATIENCE` | `2` | 容忍 selection metric 不提升的轮数（0 = 关闭早停） |
| `BERT_EARLY_STOP_MIN_DELTA` | `0.0005` | selection metric 提升的最小门槛 |

### 数据集

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_TEACHER_DATASETS` | `BerlinWang/DMSC,dirtycomputer/JD_review` | Teacher/Student 真实多分类数据白名单；传入其他数据会报错 |
| `BERT_BINARY_PSEUDO_DATASETS` | 五个二分类数据集 | Student 阶段用于生成伪标签的二分类数据白名单 |
| `BERT_TRAIN_DATASETS` | 多数据集列表 | 仅 `BERT_TRAINING_STAGE=legacy` 使用 |
| `TRAIN_VAL_RATIO` | `0.1` | 无 validation split 时自动切分比例 |
| `TRAIN_MAX_SAMPLES` | `0`（不限） | 训练集最大样本数 |
| `TRAIN_MAX_VAL_SAMPLES` | `0`（不限） | 验证集最大样本数 |
| `MIGRATE_LEGACY_BINARY_LABELS` | `auto` | 自动识别旧 0/1 标签并迁移到 0/5；设为 `0` 可关闭 |

### Teacher / Student 伪标签

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_TEACHER_CHECKPOINT_PATH` | 空 | Student 阶段必填；指向 Stage 1 Teacher checkpoint 目录 |
| `BERT_PSEUDO_LABEL_TEACHER_PATH` | 空 | BERT teacher 路径兼容别名 |
| `PSEUDO_LABEL_TEACHER_PATH` | 空 | 通用 teacher 路径兼容别名 |
| `PSEUDO_LABEL_PATH` | `pseudo_labels.jsonl` | 伪标签缓存文件，支持断点追加和重复跳过 |
| `PSEUDO_LABEL_MIN_CONFIDENCE` | `0.75` | teacher 伪标签最低置信度，低于阈值直接丢弃 |
| `PSEUDO_LABEL_TEMPERATURE` | `1.5` | logits temperature scaling |
| `PSEUDO_LABEL_WEIGHT` | `0.3` | Student 训练中伪标签样本权重 |
| `REAL_LABEL_WEIGHT` | `1.0` | Student 训练中真实标签样本权重 |
| `PSEUDO_LABEL_FALLBACK_TO_ENDPOINT` | `0` | 仅 legacy 兼容项；默认禁止低置信样本回退到 `0/5` |
| `PSEUDO_LABEL_BATCH_SIZE` | LSTM `128` / BERT `64` | teacher 推理 batch 大小 |
| `PSEUDO_LABEL_MAX_SAMPLES` | `0`（不限） | 伪标签生成样本上限，仅用于调试 |

`pseudo_labels.jsonl` 每行包含：

```json
{"text":"...", "score":4, "confidence":0.87, "probabilities":[0.01,0.02,0.03,0.07,0.82,0.05]}
```

Student 训练读取 `probabilities` 作为 soft labels，并通过 `sample_weight` 让伪标签贡献低于真实标签。

### 推理与部署

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BERT_INFERENCE_BATCH_SIZE` | `64` | BERT 批量推理 batch 大小 |
| `BERT_INFERENCE_TEMPERATURE` | `1.0` | 推理 softmax temperature |
| `BERT_INFERENCE_TORCH_COMPILE` | `0` | 后端加载 BERT 后启用 `torch.compile` 推理 |
| `BERT_DYNAMIC_QUANTIZATION` | `0` | CPU 推理时启用 Linear 动态 int8 量化 |

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
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

训练时不要使用 `--reload`。热重载会在 `.env`、代码或缓存文件变化时重启
`uvicorn` 进程，正在后台线程中运行的训练任务会被直接中断。

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
| `backend` | `8000` | FastAPI 后端；训练场景默认关闭热重载，避免重启中断训练 |
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
    f"MAE: {metrics.mae:.4f}, RMSE: {metrics.rmse:.4f}, "
    f"QWK: {metrics.quadratic_weighted_kappa:.4f}, Spearman: {metrics.spearman:.4f}"
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
