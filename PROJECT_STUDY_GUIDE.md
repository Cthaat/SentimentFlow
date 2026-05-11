# SentimentFlow 从零学习与答辩指南

这份文档的目标不是替代 README，而是帮助你从零理解 SentimentFlow，并在答辩时能清楚讲出：

- 这个项目解决什么问题。
- 前端、后端、训练代码、模型代码分别负责什么。
- 一次预测请求从浏览器到模型输出经历了哪些步骤。
- 一次训练任务从点击按钮到生成模型经历了哪些步骤。
- LSTM 和 BERT 两条模型路线有什么区别。
- 各个核心文件为什么存在、怎么协作。
- 老师可能追问的问题应该怎么回答。

## 1. 一句话介绍项目

SentimentFlow 是一个中文情感分析训练与推理平台。它把一段中文文本映射成 0-5 六档情感评分，并提供前端页面、FastAPI 后端、LSTM/BERT 训练代码、模型管理和在线预测能力。

可以这样答辩开场：

> 我的项目是一个中文情感分析全栈系统，叫 SentimentFlow。它不仅能对用户输入文本进行 0-5 六档情感评分，还支持在前端启动模型训练、查看训练进度、管理训练好的模型，并在后端统一封装 LSTM 和 BERT 两种模型的推理接口。

## 2. 项目整体架构

项目可以分成五层：

```text
用户浏览器
  ↓
Next.js 前端页面与 API Route 代理
  ↓
FastAPI 后端接口层
  ↓
Service 业务层和模型执行器
  ↓
LSTM/BERT 训练与推理代码、models/ 模型产物
```

更具体地说：

```text
frontend/
  app/page.tsx                         展示预测、训练、模型管理三个功能页
  app/api/*                            代理后端，解决跨域和容器地址问题
  components/*                         页面组件

backend/
  app/main.py                          创建 FastAPI 应用，注册路由
  app/api/*                            HTTP API 层
  app/services/*                       业务逻辑层
  app/models/*                         模型加载与推理执行层
  app/schemas/*                        Pydantic 请求响应结构

training/
  LSTM 训练包

BERT/
  BERT/RoBERTa 训练包

根目录共享文件
  sentiment_scale.py                   全项目统一 0-5 评分标准
  ordinal_loss.py                      序数距离感知损失函数
  models/                              本地模型产物
```

答辩时要强调：项目不是单独的模型脚本，而是完整的工程系统。模型只是其中一层，前端交互、后端 API、训练任务管理、模型管理和推理契约共同组成完整应用。

## 3. 最重要的统一契约：0-5 情感评分

`sentiment_scale.py` 是整个项目的核心共享契约。它规定所有模型、API、前端展示都必须使用同一套 0-5 标签。

| 分数 | 英文标签 | 中文标签 | 含义 |
| --- | --- | --- | --- |
| 0 | `extremely_negative` | 极端负面 | 非常强烈的负面情绪 |
| 1 | `clearly_negative` | 明显负面 | 明确负面，但不一定极端 |
| 2 | `slightly_negative` | 略微负面 | 轻微不满意 |
| 3 | `neutral` | 中性 | 没有明显情绪倾向 |
| 4 | `slightly_positive` | 略微正面 | 比较满意 |
| 5 | `extremely_positive` | 极端正面 | 非常强烈的正面情绪 |

这个文件还负责：

- 把旧二分类标签 `0/1` 映射成 `0/5`。
- 把 1-5 星级评分映射成 0-5 情感分。
- 把概率数组转换成统一预测结果。
- 计算 Accuracy、Macro-F1、Weighted-F1、MAE、RMSE、QWK、Spearman 等指标。
- 为伪标签选择提供二分类弱标签约束。

答辩讲法：

> 我把 0-5 评分抽成了项目级契约，避免训练、推理和前端展示各自定义标签导致不一致。所有模型最后都输出长度为 6 的 logits 或概率数组，索引就是情感分数。

## 4. 一次预测请求的完整链路

当用户在前端输入文本并点击预测时，链路如下：

```text
frontend/components/integration-test-panel.tsx
  ↓ fetch("/api/integration/predict")
frontend/app/api/integration/predict/route.ts
  ↓ proxyToBackend("/api/predict/")
backend/app/api/predict.py
  ↓ app.services.predict_service.predict_text()
backend/app/services/predict_service.py
  ↓ 根据 model 参数或 PREDICT_MODEL_TYPE 选择 LSTM/BERT
backend/app/models/LSTM/executor.py 或 backend/app/models/BERT/executor.py
  ↓ 加载模型并推理
sentiment_scale.py
  ↓ 概率转 score/label/label_zh/reasoning
返回 JSON 给前端展示
```

### 4.1 前端预测页面做什么

文件：`frontend/components/integration-test-panel.tsx`

它负责：

- 保存用户输入文本。
- 调用 `/api/integration/health` 检查后端状态。
- 调用 `/api/integration/predict` 发送预测请求。
- 展示评分方块、情感标签、置信度、6 档概率条。
- 展示当前活跃模型名称和原始 JSON 响应。
- 如果后端返回 404，提示用户先训练模型。

### 4.2 前端 API Route 为什么存在

文件：`frontend/app/api/integration/predict/route.ts`

它不是直接做模型推理，而是代理请求到后端：

```text
POST /api/integration/predict
  -> POST backend /api/predict/
```

这样做的好处：

- 浏览器只请求同源的 Next.js 服务，避免 CORS 问题。
- 本地开发、Docker 容器和不同后端地址可以统一处理。
- 后端地址探测逻辑集中在 `frontend/lib/api-proxy.ts`。

### 4.3 后端预测 API 做什么

文件：`backend/app/api/predict.py`

职责很窄：

- 用 `PredictRequest` 校验请求体。
- 调用 service 层的 `predict_text()`。
- 把结果包装成 `PredictResponse`。
- 模型不存在时返回 404。

这体现了分层设计：API 层不直接加载模型，也不写业务细节。

### 4.4 Service 层如何选择模型

文件：`backend/app/services/predict_service.py`

核心函数是：

```python
predict_text(text: str, model_type: str | None = None) -> PredictResult
```

它的逻辑：

1. 读取 `.env` 和环境变量。
2. 如果请求体指定了 `model`，优先使用请求体。
3. 否则使用 `PREDICT_MODEL_TYPE`。
4. 检查对应模型是否存在。
5. 存在就调用 LSTM 或 BERT executor。
6. 如果模型推理运行时异常，降级到关键词基线。
7. 如果模型文件不存在，直接抛出 `FileNotFoundError`，提醒先训练。

这里还有一个关键词冲突保护：

- 如果模型高置信预测与明显关键词完全冲突，就用关键词基线兜底。
- 这是为了防止污染 checkpoint 输出明显错误的高置信结果。

### 4.5 LSTM 推理执行器

文件：`backend/app/models/LSTM/executor.py`

职责：

- 根据 `MODEL_PATH` 加载 LSTM `.pt` checkpoint。
- 自动缓存模型，避免每次请求重复加载。
- 自动检查 checkpoint 修改时间，文件变化后重新加载。
- 调用 `app.utils.tokenizer.encode_text()` 把文本编码成固定长度 id。
- 输出 6 类 softmax 概率，并返回预测分数和置信度。

### 4.6 BERT 推理执行器

文件：`backend/app/models/BERT/executor.py`

职责：

- 确保项目根目录在 `sys.path` 中，这样后端能导入根目录 `BERT` 包。
- 调用 `BERT.pipeline.load_or_train()` 加载有效 checkpoint。
- 调用 `BERT.inference.prepare_inference_model()` 启用可选推理优化。
- 调用 `BERT.inference.predict_text()` 执行单条文本推理。

注意：BERT checkpoint 必须是微调后的目录，且包含 `training_meta.json`。代码会拒绝只有基础预训练权重的目录，避免没训练的分类头参与预测。

## 5. 一次训练任务的完整链路

当用户在前端点击开始训练时，链路如下：

```text
frontend/components/training-panel.tsx
  ↓ fetch("/api/training/start")
frontend/app/api/training/start/route.ts
  ↓ proxyToBackend("/api/training/start")
backend/app/api/training.py
  ↓ TrainingManager.start_training()
backend/app/services/training_service.py
  ↓ 后台线程执行 LSTM 或 BERT 训练
training.trainer.train_model() 或 BERT.trainer.train_model()
  ↓ 保存到 models/{type}_{timestamp}/
  ↓ set_active_model()
frontend 通过 SSE /api/training/stream/{jobId} 接收日志和指标
```

### 5.1 前端训练页面做什么

文件：`frontend/components/training-panel.tsx`

它负责：

- 选择模型类型：`lstm` 或 `bert`。
- 展示和编辑超参数。
- 选择训练数据集。
- BERT 训练时根据阶段自动拆分真实多分类数据和二分类伪标签数据。
- 提交训练请求。
- 建立 SSE 连接接收实时日志。
- SSE 断开时自动切换为状态轮询。
- 刷新页面后从 `localStorage` 恢复训练任务。
- 展示 loss、Accuracy、Macro-F1、Weighted-F1、MAE、RMSE、QWK、Spearman、Best Metric 等指标。

### 5.2 后端训练 API

文件：`backend/app/api/training.py`

主要接口：

| 接口 | 作用 |
| --- | --- |
| `POST /api/training/start` | 启动训练 |
| `GET /api/training/status/{job_id}` | 查询训练状态 |
| `GET /api/training/stream/{job_id}` | SSE 实时输出状态和日志 |
| `GET /api/training/jobs` | 查看训练任务列表 |
| `POST /api/training/cancel/{job_id}` | 取消训练 |

API 层只处理 HTTP 和 Pydantic 响应，实际任务管理在 service 层。

### 5.3 TrainingManager 的作用

文件：`backend/app/services/training_service.py`

`TrainingManager` 是后台训练任务管理器，核心职责：

- 保证同一时间只有一个训练任务运行。
- 为每个任务生成 `job_id`。
- 在后台线程执行训练，避免阻塞 API 请求。
- 捕获训练过程中的 stdout 日志。
- 用正则解析日志中的 epoch、step、loss、验证指标。
- 支持取消训练。
- 训练完成后把模型设置为活跃模型。

训练状态保存在内存里，所以后端进程重启后任务记录不会持久化。这是答辩时可以主动说明的一个限制。

### 5.4 模型保存路径

训练任务会统一输出到：

```text
models/{model_type}_{UTC时间戳}/
```

LSTM：

```text
models/lstm_YYYYMMDD_HHMMSS/
├── model.pt
└── training_meta.json
```

BERT：

```text
models/bert_YYYYMMDD_HHMMSS/
├── config.json
├── model.safetensors
├── sentiment_model_state.pt
├── tokenizer.json
├── tokenizer_config.json
├── trainer_state.pt
└── training_meta.json
```

## 6. LSTM 模型路线

LSTM 是轻量基线模型，对应根目录 `LSTM.py` 和 `training/` 包。

### 6.1 LSTM 处理流程

```text
原始文本
  ↓
jieba 分词
  ↓
CRC32 哈希到固定词表 id
  ↓
padding/truncation 到固定长度
  ↓
Embedding
  ↓
LSTM
  ↓
取最后一个非 padding token 的隐藏状态
  ↓
Linear 输出 6 类 logits
  ↓
softmax 得到 6 档概率
```

### 6.2 LSTM 关键文件

| 文件 | 作用 |
| --- | --- |
| `LSTM.py` | 兼容入口，导入 `training` 包中的真实实现 |
| `training/config.py` | LSTM 默认超参数和环境变量读取 |
| `training/text_processing.py` | jieba 分词和 CRC32 哈希编码 |
| `training/dataset.py` | 流式 Dataset，支持 CSV 和 HuggingFace Dataset |
| `training/data_sources.py` | 加载、清洗和合并训练/验证数据集 |
| `training/model.py` | `SentimentLSTMModel` 模型结构 |
| `training/trainer.py` | 训练循环、DataLoader、loss、optimizer、早停、保存最优模型 |
| `training/evaluate.py` | 验证集指标计算 |
| `training/checkpoint.py` | LSTM checkpoint 保存和加载 |
| `training/inference.py` | LSTM 单条文本推理 |
| `training/pipeline.py` | 有 checkpoint 就加载，否则训练 |
| `training/main.py` | 命令行入口，训练后打印样例预测和测试用例 |

### 6.3 LSTM 的优点和局限

优点：

- 训练快，模型小。
- 不需要下载大型预训练模型。
- 适合演示完整训练闭环。

局限：

- 哈希词表有冲突。
- 对语义、上下文和反讽表达理解较弱。
- 泛化能力通常弱于 BERT。

答辩讲法：

> LSTM 分支在项目里作为轻量级 baseline。它能快速跑通训练到推理的流程，但由于没有预训练语义知识，复杂中文表达下效果不如 BERT。

## 7. BERT 模型路线

BERT 是项目主力模型，对应根目录 `BERT.py` 和 `BERT/` 包。

### 7.1 BERT 推理流程

```text
原始文本
  ↓
AutoTokenizer 分词
  ↓
input_ids + attention_mask
  ↓
Chinese RoBERTa/BERT backbone
  ↓
pooler_output 或 CLS hidden state
  ↓
Dropout
  ↓
Linear 分类头
  ↓
6 类 logits
  ↓
softmax + argmax 得到 0-5 分
```

### 7.2 BERT 关键文件

| 文件 | 作用 |
| --- | --- |
| `BERT.py` | BERT 命令行入口，调用 `BERT.main.run()` |
| `BERT/config.py` | BERT 模型名、最大长度、训练超参数、HF 环境配置 |
| `BERT/model.py` | `SentimentBertModel`，封装 backbone 和 6 类分类头 |
| `BERT/text_processing.py` | tokenizer 加载与缓存 |
| `BERT/dataset.py` | BERT Dataset，返回原始文本和训练元信息，不逐条 tokenize |
| `BERT/data_sources.py` | BERT 数据集加载、teacher/student 阶段、伪标签、过采样、插值 |
| `BERT/trainer.py` | BERT 训练主流程 |
| `BERT/evaluate.py` | BERT 验证评估 |
| `BERT/checkpoint.py` | 保存/加载 HF 兼容 checkpoint |
| `BERT/inference.py` | BERT 批量推理、概率转换、ONNX 导出 |
| `BERT/export.py` | checkpoint 导出 ONNX 的 CLI |
| `BERT/pipeline.py` | 加载或训练调度 |
| `BERT/main.py` | 命令行训练和样例推理 |
| `BERT/generate_synthetic_data.py` | 生成短句合成数据 |
| `BERT/extract_short_sentences.py` | 从长评论提取短句 |

### 7.3 BERT 的关键优化点

1. **批量 tokenize**

   Dataset 不在 `__iter__` 中逐条调用 tokenizer，而是在 `collate_fn` 中对整个 batch 批量 tokenize，降低 CPU 开销。

2. **DistanceAwareOrdinalLoss**

   情感评分是有顺序的。预测 5 错成 4 比预测 5 错成 0 更接近真实情绪，所以 loss 加入 expected-score 距离惩罚。

3. **类别不平衡处理**

   BERT 使用 effective-number class weights、focal loss、logit adjustment、分布感知过采样等策略缓解数据不平衡。

4. **Teacher/Student 两阶段训练**

   避免直接把二分类数据粗暴映射成 0/5 导致模型只学极端情绪。

5. **QWK 选模**

   默认用 Quadratic Weighted Kappa 作为 BERT 最优模型选择指标，更适合 0-5 有序评分。

## 8. BERT Teacher/Student 训练怎么讲

这是项目中最值得讲的算法设计。

### 8.1 为什么需要两阶段

很多中文情感数据集是二分类，只告诉你正面或负面；但项目目标是 0-5 六档评分。如果直接把二分类：

```text
负面 0 -> 0
正面 1 -> 5
```

模型会被大量极端标签污染，容易只预测 0 或 5。

所以项目采用两阶段：

```text
Teacher 阶段
  只使用真实多分类数据，例如 DMSC、JD_review
  学习 0-5 六档评分能力

Student 阶段
  用 Teacher 给二分类数据生成 soft pseudo labels
  只保留高置信样本
  合并真实多分类数据和伪标签数据
  训练更强的 Student 模型
```

### 8.2 Teacher 阶段

入口：`BERT.data_sources._teacher_dataset_names()`

只允许：

- `BerlinWang/DMSC`
- `dirtycomputer/JD_review`

原因：这两个数据集有更接近真实多档评分的信息。

### 8.3 Student 阶段

入口：`BERT.data_sources._generate_pseudo_labels_if_needed()`

可用于伪标签的数据：

- `lansinuote/ChnSentiCorp`
- `XiangPan/waimai_10k`
- `dirtycomputer/weibo_senti_100k`
- `ndiy/NLPCC14-SC`
- `dirtycomputer/ChnSentiCorp_htl_all`

Student 阶段会：

1. 加载 Teacher checkpoint。
2. 遍历二分类数据。
3. Teacher 输出 6 类概率。
4. 根据二分类弱标签限制候选分数：
   - 负面只能变成 0/1/2。
   - 正面只能变成 4/5。
5. 低于 `PSEUDO_LABEL_MIN_CONFIDENCE` 的样本丢弃。
6. 写入 `pseudo_labels.jsonl`。
7. 读取伪标签作为 `soft_labels` 参与训练。

答辩讲法：

> 二分类标签本身不能区分轻微正面和极端正面，所以我没有直接端点映射，而是先用真实多评分数据训练 Teacher，再让 Teacher 对二分类数据做受约束的软伪标签生成。这样既利用了大量二分类数据，又减少了标签噪声。

## 9. 损失函数为什么这样设计

文件：`ordinal_loss.py`

核心类：`DistanceAwareOrdinalLoss`

它由两部分组成：

```text
总损失 = CE 部分 + expected-score 距离部分
```

### 9.1 CE 部分

CE 让模型学会把文本分类到正确的 0-5 档位。

支持：

- one-hot 标签。
- soft labels。
- label smoothing。
- pseudo label smoothing。
- class weights。
- focal loss。

### 9.2 距离部分

先根据 softmax 概率计算模型的期望分数：

```text
predicted_score = sum(probability[i] * i)
```

再和目标期望分数做 SmoothL1Loss。

这样模型知道：

- 预测 5 错成 4 是小错。
- 预测 5 错成 0 是大错。

答辩讲法：

> 普通 CrossEntropy 不关心类别顺序，但情感分数是有序的。我在 CE 外加了 expected-score 距离项，让模型不仅关注分类对错，也关注预测分数离真实分数有多远。

## 10. 后端分层架构

后端目录：

```text
backend/app/
├── main.py
├── api/
├── services/
├── schemas/
├── models/
├── core/
├── db/
└── utils/
```

### 10.1 `backend/app/main.py`

职责：

- 加载 `.env`。
- 创建 FastAPI app。
- 注册 CORS。
- 注册所有 router。
- 在 lifespan 中后台预加载活跃模型，降低第一次预测延迟。
- 提供 `/health`。

### 10.2 `backend/app/api/`

API 层，只处理 HTTP：

| 文件 | 说明 |
| --- | --- |
| `predict.py` | 预测接口 |
| `training.py` | 训练任务接口 |
| `models.py` | 模型列表、激活、删除 |
| `auth.py` | 登录占位接口 |
| `admin.py` | 管理占位接口 |
| `stats.py` | 统计占位接口 |

答辩时可以说明：占位接口代表预留扩展方向，不是当前核心功能。

### 10.3 `backend/app/services/`

Service 层，处理业务逻辑：

| 文件 | 说明 |
| --- | --- |
| `predict_service.py` | 统一预测入口、模型选择、模型存在性检查、关键词兜底 |
| `training_service.py` | 训练任务管理、后台线程、日志捕获、指标解析、取消训练 |
| `user_service.py` | 当前为空，预留用户服务 |

### 10.4 `backend/app/schemas/`

Pydantic 数据结构：

| 文件 | 说明 |
| --- | --- |
| `predict.py` | `PredictRequest`、`PredictResponse` |
| `training.py` | 训练启动、状态、任务列表响应 |
| `models.py` | 模型列表和活跃模型响应 |
| `user.py` | 当前为空，预留用户结构 |

### 10.5 `backend/app/models/`

模型执行层：

```text
backend/app/models/
├── common.py                  # 设备选择、checkpoint state_dict 处理
├── loader.py                  # 旧导入路径兼容
├── LSTM/
│   ├── architecture.py        # LSTM 推理结构
│   ├── executor.py            # LSTM 模型加载和批量推理
│   └── training.py            # 旧版 LSTM 训练兼容代码
└── BERT/
    └── executor.py            # BERT 模型加载和推理
```

注意：后端的模型执行器和根目录训练包有重叠，但职责不同：

- 根目录 `training/`、`BERT/` 偏训练和完整实验。
- `backend/app/models/*/executor.py` 偏后端服务中的加载和推理。

### 10.6 `backend/app/core/`

| 文件 | 说明 |
| --- | --- |
| `config.py` | 加载 `.env`、读取活跃模型配置、设置活跃模型 |
| `paths.py` | 统一计算项目根目录、后端目录和 `models/` 目录 |
| `security.py` | 当前为空，预留安全相关逻辑 |

### 10.7 `backend/app/utils/tokenizer.py`

用于 LSTM 推理：

- jieba 分词。
- CRC32 哈希。
- padding/truncation。

必须和 LSTM 训练侧编码方式保持一致，否则训练和推理输入分布会不一致。

## 11. 前端架构

前端目录：

```text
frontend/
├── app/
│   ├── page.tsx
│   ├── layout.tsx
│   ├── globals.css
│   └── api/
├── components/
├── components/ui/
├── lib/
├── package.json
└── tsconfig.json
```

### 11.1 `frontend/app/page.tsx`

首页，负责：

- 管理当前标签页：`predict`、`train`、`models`。
- 把标签页保存到 `localStorage`。
- 渲染三个主要组件：
  - `IntegrationTestPanel`
  - `TrainingPanel`
  - `ModelManagementPanel`

### 11.2 `frontend/app/layout.tsx`

全局布局：

- 设置 HTML 语言为 `zh-CN`。
- 加载 Geist 字体。
- 使用 `ThemeProvider` 包裹页面。
- 设置页面 metadata。

### 11.3 `frontend/lib/api-proxy.ts`

前端代理核心：

- 读取 `BACKEND_API_URL`。
- 没配置时按顺序尝试多个后端地址。
- 记住上一次成功的后端地址。
- 每个请求带超时控制。

答辩讲法：

> 前端不是硬编码一个后端地址，而是通过代理工具兼容本地启动和 Docker 网络，避免跨域和地址切换问题。

### 11.4 主要组件

| 文件 | 作用 |
| --- | --- |
| `integration-test-panel.tsx` | 预测页面，健康检查、文本输入、结果展示 |
| `training-panel.tsx` | 训练页面，配置参数、启动训练、SSE 日志、指标展示 |
| `model-management-panel.tsx` | 模型管理页面，列表、启用、删除 |
| `theme-toggle.tsx` | 明暗主题切换 |

### 11.5 UI 基础组件

| 文件 | 作用 |
| --- | --- |
| `components/ui/button.tsx` | Button 组件和 variant 配置 |
| `components/ui/card.tsx` | Card 组件 |
| `components/ui/input.tsx` | Input 组件 |
| `components/ui/select.tsx` | Select 组件 |
| `components/ui/textarea.tsx` | Textarea 组件 |
| `components/ui/badge.tsx` | Badge 组件 |

这些组件让页面风格统一，减少重复 CSS。

## 12. 模型管理怎么讲

模型管理链路：

```text
frontend/components/model-management-panel.tsx
  ↓ /api/models
frontend/app/api/models/route.ts
  ↓ backend /api/models/
backend/app/api/models.py
  ↓ 扫描 models/ 目录
  ↓ 判断模型类型、读取 training_meta.json、计算大小
```

后端如何判断模型类型：

- 目录有 `config.json` 和 `model.safetensors` 或 `pytorch_model.bin`：认为是 BERT。
- 目录里有 `.pt` 文件：认为是 LSTM。

切换活跃模型时：

```text
PUT /api/models/active
  -> set_active_model()
  -> 更新进程内 active config
  -> 同步写入 os.environ
```

删除模型时：

- 先扫描 `models/`。
- 找到目标 `model_id`。
- 确认真实路径在 `models/` 目录内部。
- 再删除目录或文件。

这个路径检查是安全设计，防止误删项目外文件。

## 13. 数据集处理怎么讲

项目支持 HuggingFace Dataset 和本地 CSV。

数据处理核心目标：

1. 找到文本列。
2. 找到标签列。
3. 把不同数据集的标签统一转成 0-5。
4. 过滤无效标签。
5. 合并多个数据集。
6. 打乱并限制样本量。

常见文本列候选：

```text
text, review, content, Comment, comment, sentence
```

常见标签列候选：

```text
label, sentiment, Star, score, rating
```

标签映射例子：

- 旧二分类 `0/1`：映射到 `0/5`，但 BERT teacher/student 中不会粗暴用于 Teacher。
- DMSC 星级 `1/2/3/4/5`：映射到 `0/1/3/4/5`。
- JD_review rating：同样做星级映射。
- 10 档评分：映射到 0-5。

## 14. 评估指标怎么讲

项目不仅看 Accuracy，还看多种指标：

| 指标 | 含义 | 为什么需要 |
| --- | --- | --- |
| Accuracy | 预测完全正确比例 | 最直观 |
| Macro-F1 | 各类别 F1 平均 | 关注少数类表现 |
| Weighted-F1 | 按样本数加权 F1 | 反映整体加权表现 |
| MAE | 平均绝对误差 | 适合有序分数 |
| RMSE | 均方根误差 | 对大偏差更敏感 |
| QWK | 二次加权 Kappa | 适合有序评分一致性 |
| Spearman | 排序相关性 | 看情绪强弱排序是否一致 |

答辩讲法：

> 对 0-5 评分任务，预测错 1 分和错 5 分严重程度不同，所以我除了分类指标，还加入 MAE、RMSE、QWK、Spearman 这些更适合有序评分的指标。BERT 默认用 QWK 选择最佳模型。

## 15. 环境和部署怎么讲

### 15.1 本地开发

两个服务：

```text
FastAPI 后端: http://localhost:8000
Next.js 前端: http://localhost:3000
```

本地启动后，浏览器访问前端。前端通过 API Route 请求后端。

### 15.2 Docker 部署

`docker-compose.yml` 定义两个服务：

| 服务 | 镜像/构建 | 作用 |
| --- | --- | --- |
| `backend` | `./backend` Dockerfile | FastAPI 后端 |
| `frontend` | `node:20` | Next.js 前端 |

容器内前端使用：

```text
BACKEND_API_URL=http://backend:8000
```

这是 Docker Compose 服务名通信。

## 16. 测试怎么讲

测试文件：`tests/test_sentiment_scale.py`

主要测试：

- 旧二分类标签迁移。
- 星级评分映射。
- 概率数组必须是 6 类。
- 分类指标和有序指标计算。
- LSTM 模型前向、loss、评估、推理形状。
- BERT soft pseudo label 读取。
- BERT teacher/student 数据集选择。
- CSV 文件级别判断旧二分类，避免按 chunk 误判。

运行：

```powershell
python -m unittest tests.test_sentiment_scale
```

答辩讲法：

> 我把最关键的契约写成了测试，包括标签映射、概率输出、伪标签、数据集选择和模型输出形状。这样可以保证重构后训练和推理仍然遵守统一的 0-5 评分协议。

## 17. 文件作用速查表

### 17.1 根目录文件

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目说明、运行、训练、API、部署 |
| `PROJECT_STUDY_GUIDE.md` | 答辩学习文档 |
| `docker-compose.yml` | 一键启动前后端 |
| `.gitignore` | 忽略环境、缓存、模型、构建产物 |
| `BERT.py` | BERT 命令行入口 |
| `LSTM.py` | LSTM 兼容入口 |
| `sentiment_scale.py` | 全项目统一情感评分标准 |
| `ordinal_loss.py` | 距离感知有序多分类损失 |
| `test1.py` | DataLoader/IterableDataset 学习示例 |
| `test2.py` | 简单 LSTM 训练学习示例 |

### 17.2 `training/`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包标识 |
| `config.py` | LSTM 训练配置 |
| `text_processing.py` | jieba 分词和哈希编码 |
| `dataset.py` | LSTM 流式数据集 |
| `data_sources.py` | LSTM 数据集加载和标签清洗 |
| `model.py` | LSTM 网络结构 |
| `trainer.py` | LSTM 训练主流程 |
| `evaluate.py` | LSTM 验证指标 |
| `checkpoint.py` | LSTM checkpoint 读写 |
| `pipeline.py` | 加载或训练调度 |
| `inference.py` | LSTM 推理 |
| `main.py` | LSTM 命令行流程 |
| `sample_texts.py` | 样例文本 |
| `custom_test_cases.py` | 自定义质量验证样例 |
| `generate_synthetic_data.py` | 合成短句数据 |
| `extract_short_sentences.py` | 从长评论提取短句 |
| `env_utils.py` | `.env` 加载工具 |

### 17.3 `BERT/`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包标识 |
| `config.py` | BERT 配置和 HuggingFace 环境变量 |
| `model.py` | BERT/RoBERTa + 分类头模型 |
| `text_processing.py` | tokenizer 缓存和编码 |
| `dataset.py` | BERT 流式 Dataset，返回原始文本和 soft label 信息 |
| `data_sources.py` | BERT 数据集、teacher/student、伪标签、插值、过采样 |
| `trainer.py` | BERT 训练主流程 |
| `evaluate.py` | BERT 验证指标 |
| `checkpoint.py` | BERT checkpoint 保存/加载 |
| `pipeline.py` | 加载或训练调度 |
| `inference.py` | BERT 批量推理、概率转换、ONNX 导出 |
| `export.py` | ONNX 导出命令行入口 |
| `main.py` | BERT 命令行流程 |
| `sample_texts.py` | 样例文本 |
| `custom_test_cases.py` | 自定义质量验证样例 |
| `generate_synthetic_data.py` | 合成短句数据 |
| `extract_short_sentences.py` | 提取短句数据 |
| `env_utils.py` | `.env` 加载工具 |

### 17.4 `backend/app/`

| 文件/目录 | 作用 |
| --- | --- |
| `main.py` | FastAPI 应用入口 |
| `api/predict.py` | 预测接口 |
| `api/training.py` | 训练接口 |
| `api/models.py` | 模型管理接口 |
| `api/auth.py` | 登录占位 |
| `api/admin.py` | 管理占位 |
| `api/stats.py` | 统计占位 |
| `services/predict_service.py` | 预测业务逻辑 |
| `services/training_service.py` | 训练任务管理 |
| `schemas/predict.py` | 预测请求响应模型 |
| `schemas/training.py` | 训练请求响应模型 |
| `schemas/models.py` | 模型管理响应模型 |
| `models/common.py` | checkpoint 和设备通用工具 |
| `models/LSTM/architecture.py` | 后端 LSTM 推理结构 |
| `models/LSTM/executor.py` | 后端 LSTM 加载和推理 |
| `models/BERT/executor.py` | 后端 BERT 加载和推理 |
| `core/config.py` | 后端环境和活跃模型配置 |
| `core/paths.py` | 项目路径工具 |
| `utils/tokenizer.py` | LSTM 推理文本编码 |
| `db/database.py` | 当前为空，预留数据库连接 |
| `db/crud.py` | 当前为空，预留 CRUD |

### 17.5 `frontend/`

| 文件/目录 | 作用 |
| --- | --- |
| `app/page.tsx` | 首页和三标签页切换 |
| `app/layout.tsx` | 全局布局和主题 Provider |
| `app/globals.css` | Tailwind 和主题变量 |
| `app/api/integration/health/route.ts` | 健康检查代理 |
| `app/api/integration/predict/route.ts` | 预测代理 |
| `app/api/training/start/route.ts` | 启动训练代理 |
| `app/api/training/status/[jobId]/route.ts` | 状态查询代理 |
| `app/api/training/stream/[jobId]/route.ts` | SSE 训练日志代理 |
| `app/api/training/jobs/route.ts` | 训练任务列表代理 |
| `app/api/training/cancel/route.ts` | 取消训练代理 |
| `app/api/models/route.ts` | 模型列表和删除代理 |
| `app/api/models/active/route.ts` | 活跃模型代理 |
| `components/integration-test-panel.tsx` | 预测界面 |
| `components/training-panel.tsx` | 训练界面 |
| `components/model-management-panel.tsx` | 模型管理界面 |
| `components/theme-toggle.tsx` | 明暗主题切换 |
| `components/ui/*` | 基础 UI 组件 |
| `lib/api-proxy.ts` | 后端代理工具 |
| `lib/theme.tsx` | 主题状态管理 |
| `lib/utils.ts` | className 合并工具 |
| `package.json` | 前端依赖和脚本 |
| `tsconfig.json` | TypeScript 配置 |
| `next.config.ts` | Next.js 配置 |

## 18. 答辩演示建议

建议演示顺序：

1. 打开前端首页，说明三大模块：预测、训练、模型管理。
2. 在 `模型管理` 展示已有 LSTM/BERT 模型，说明模型都保存在 `models/`。
3. 切换一个活跃模型。
4. 回到 `情感预测` 输入正面文本和负面文本，展示 0-5 分、概率条和 JSON。
5. 打开后端 Swagger，展示 `/api/predict/`、`/api/training/*`、`/api/models/*`。
6. 打开代码讲架构：
   - `frontend/app/page.tsx`
   - `backend/app/main.py`
   - `backend/app/services/predict_service.py`
   - `sentiment_scale.py`
   - `BERT/trainer.py`
7. 解释 LSTM 和 BERT 的区别。
8. 解释 BERT teacher/student 设计。
9. 解释测试文件保证了哪些关键契约。

## 19. 三分钟讲解稿

可以按下面这段讲：

> SentimentFlow 是一个中文情感分析全栈系统。它的目标是把用户输入的中文文本转换成 0-5 六档情感评分。项目分为前端、后端、训练代码和模型产物四部分。前端用 Next.js 实现，包括情感预测、模型训练和模型管理三个页面；后端用 FastAPI 实现，提供预测、训练任务、模型管理等接口；模型层支持 LSTM 和 BERT 两种方案。
>
> 整个项目最核心的契约是 `sentiment_scale.py`，它统一定义了 0-5 分数、中文标签、英文标签、概率转换和评估指标。这样 LSTM、BERT、后端 API 和前端展示不会出现标签不一致的问题。
>
> 用户预测时，前端先请求 Next.js 的 API Route，再由代理转发到 FastAPI 的 `/api/predict/`。后端 service 层根据请求参数或环境变量选择 LSTM 或 BERT，加载对应模型，输出六类概率，再转换成统一响应返回前端。
>
> 用户训练时，前端把模型类型、数据集和超参数提交给后端。后端 `TrainingManager` 创建后台线程执行训练，并通过 SSE 把日志、epoch、loss 和验证指标实时推给前端。训练完成后模型保存到 `models/`，并自动成为活跃模型。
>
> 模型方面，LSTM 是轻量基线，使用 jieba 分词和哈希词表；BERT 是主力模型，基于中文 RoBERTa，并且支持 teacher/student 两阶段训练。Teacher 只使用真实多分类数据，Student 再用 Teacher 给二分类数据生成高置信软伪标签。这样可以利用更多数据，同时避免把二分类标签粗暴映射成极端 0/5。

## 20. 老师可能追问的问题

### Q1：为什么不用普通二分类，而做 0-5 六档？

答：

> 二分类只能区分正负，表达能力比较弱。0-5 六档可以表示情绪强度，比如略微负面和极端负面是不同的。这个项目希望输出更细粒度的情感评分，所以采用六分类。

### Q2：为什么要有 `sentiment_scale.py`？

答：

> 因为训练、推理、API、前端都要使用同一套标签。如果每个模块自己定义，很容易出现标签含义不一致。`sentiment_scale.py` 相当于全项目的数据契约，统一评分、标签、概率转换和指标计算。

### Q3：LSTM 和 BERT 有什么区别？

答：

> LSTM 是轻量 baseline，训练快、部署成本低，但语义理解能力有限。BERT 使用预训练中文语言模型，能更好理解上下文和短句语义，但训练和推理成本更高。项目保留两者，是为了既能快速演示训练闭环，也能提供更强的主力模型。

### Q4：为什么 BERT 要 teacher/student？

答：

> 因为很多中文情感数据集只有正负二分类，如果直接映射到 0/5，会让六分类任务退化成极端二分类。Teacher 先用真实多评分数据学习六档评分，再给二分类数据生成软伪标签，Student 用真实标签加高置信伪标签训练，能更稳地利用二分类数据。

### Q5：为什么使用 SSE 而不是普通轮询？

答：

> 训练日志是持续产生的，SSE 可以让后端主动推送日志和指标，延迟低、实现简单，适合单向实时更新。如果 SSE 断开，前端还会自动切换到轮询，提高稳定性。

### Q6：为什么前端要通过 Next API Route 代理？

答：

> 这样浏览器只访问同源的 Next 服务，避免 CORS 问题。同时本地开发和 Docker 部署的后端地址不同，代理层可以统一处理地址探测和超时。

### Q7：模型文件怎么管理？

答：

> 所有训练产物统一放到根目录 `models/`，每次训练按模型类型和时间戳生成一个子目录。后端模型管理 API 会扫描这个目录，判断模型类型，读取 `training_meta.json` 展示指标，并支持启用和删除模型。

### Q8：项目目前有什么不足？

答：

> 第一，训练任务状态保存在内存里，后端重启后不会持久化。第二，登录、管理和统计接口目前是占位实现，没有完整权限系统。第三，BERT 模型较大，对显存和训练时间有要求。后续可以加数据库持久化任务、用户权限和更完善的实验记录。

### Q9：怎么保证训练和推理预处理一致？

答：

> LSTM 的训练和后端推理都使用相同的分词和哈希编码逻辑；BERT 的训练和推理都使用 checkpoint 对应的 HuggingFace tokenizer。BERT 加载 checkpoint 时还会恢复 `max_len` 和模型路径，避免训练推理参数不一致。

### Q10：为什么要加入有序距离损失？

答：

> 因为 0-5 是有序分数。普通 CrossEntropy 认为预测 5 错成 4 和错成 0 都只是分类错误，但实际严重程度不同。距离损失可以让模型学习预测分数离真实分数越近越好。

## 21. 你最应该熟悉的 10 个文件

答辩前优先把这些文件看懂：

1. `sentiment_scale.py`
2. `ordinal_loss.py`
3. `backend/app/main.py`
4. `backend/app/services/predict_service.py`
5. `backend/app/services/training_service.py`
6. `frontend/app/page.tsx`
7. `frontend/components/training-panel.tsx`
8. `frontend/components/integration-test-panel.tsx`
9. `BERT/data_sources.py`
10. `BERT/trainer.py`

如果时间有限，先理解：

```text
sentiment_scale.py
  -> 统一评分契约
predict_service.py
  -> 预测怎么选模型
training_service.py
  -> 训练任务怎么跑
BERT/trainer.py
  -> 主力模型怎么训练
frontend/components/*
  -> 用户怎么操作系统
```

## 22. 最后总结

SentimentFlow 的核心不是某一个模型，而是完整工程闭环：

```text
数据集处理
  -> 模型训练
  -> 模型保存
  -> 模型管理
  -> 后端推理
  -> 前端展示
  -> 测试保障
```

答辩时只要围绕这条主线讲，就能把项目讲清楚。
