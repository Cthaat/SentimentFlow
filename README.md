# SentimentFlow

SentimentFlow 是一个中文情感分析全栈项目，提供从模型训练、模型管理到在线推理的完整闭环。项目支持两类模型：

- **LSTM**：轻量级基线模型，使用 jieba 分词、CRC32 哈希词表、Embedding + LSTM + Linear 完成 0-5 情感评分。
- **BERT/RoBERTa**：主力模型，基于 `hfl/chinese-roberta-wwm-ext`，支持 teacher/student 两阶段训练、软伪标签、类别不平衡处理和序数距离感知损失。

前端使用 Next.js 构建训练、模型管理和预测页面；后端使用 FastAPI 提供 REST API、后台训练任务和 SSE 实时日志。

## 功能特性

- 0-5 六档中文情感评分：极端负面、明显负面、略微负面、中性、略微正面、极端正面。
- 前后端分离：Next.js 前端通过 API Route 代理访问 FastAPI 后端。
- 在线预测：输入文本后返回评分、中文标签、英文标签、置信度、6 档概率和解释文本。
- 后台训练：支持从前端启动 LSTM 或 BERT 训练，训练进度通过 SSE 实时返回。
- 模型管理：扫描 `models/` 目录，展示模型类型、大小、指标，支持切换活跃模型和删除模型。
- BERT 两阶段训练：Teacher 只使用真实多分类数据，Student 结合真实标签和高置信二分类伪标签。
- 统一评分契约：`sentiment_scale.py` 统一维护标签、概率、指标和 0-5 评分转换规则。
- 序数距离感知损失：`ordinal_loss.py` 在 CrossEntropy 基础上加入 expected-score 距离惩罚。
- Docker Compose 一键启动前后端。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Next.js 16.2.3, React 19.2.4, TypeScript, Tailwind CSS 4, shadcn/ui 风格组件, lucide-react |
| 后端 | FastAPI, Uvicorn, Pydantic |
| 训练与推理 | PyTorch, Transformers, Datasets, Accelerate, pandas, jieba |
| 模型 | LSTM, Chinese RoBERTa/BERT |
| 部署 | Docker Compose, Python 3.11 slim, Node.js 20 |

## 项目结构

```text
SentimentFlow/
├── README.md                         # 项目使用与维护说明
├── PROJECT_STUDY_GUIDE.md            # 从零学习与答辩讲解指南
├── docker-compose.yml                # 前后端 Docker Compose 编排
├── .gitignore                        # 忽略环境、缓存、模型和构建产物
├── BERT.py                           # BERT 训练/推理命令行入口
├── LSTM.py                           # LSTM 兼容入口，实际转发到 training 包
├── sentiment_scale.py                # 0-5 情感评分统一契约
├── ordinal_loss.py                   # 序数距离感知多分类损失
├── test1.py / test2.py               # 早期 PyTorch/DataLoader 学习示例
├── tests/
│   └── test_sentiment_scale.py       # 情感评分、训练数据、推理契约测试
├── training/                         # LSTM 训练包
├── BERT/                             # BERT 训练包
├── backend/                          # FastAPI 后端
├── frontend/                         # Next.js 前端
└── models/                           # 本地训练产物目录，模型大文件不入库
```

## 快速开始

### 1. 环境要求

- Python 3.10+，推荐 Python 3.11。
- Node.js 20+。
- 训练 BERT 推荐使用 CUDA GPU；CPU 可运行但训练会很慢。
- Windows PowerShell、Linux shell 或 Docker 均可运行。

### 2. 后端本地启动

```powershell
cd C:\Code\SentimentFlow\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8846
```

启动后访问：

- 健康检查：http://127.0.0.1:8846/health
- Swagger 文档：http://127.0.0.1:8846/docs

### 3. 前端本地启动

```powershell
cd C:\Code\SentimentFlow\frontend
yarn install
yarn dev
```

前端默认访问：

```text
http://localhost:3000
```

前端页面包含三个标签页：

- `情感预测`：输入文本并调用当前活跃模型。
- `模型训练`：选择 LSTM/BERT、数据集和超参数，启动后台训练。
- `模型管理`：查看 `models/` 下已训练模型，切换活跃模型或删除模型。

### 4. Docker Compose 启动

```powershell
cd C:\Code\SentimentFlow
docker compose up
```

`docker-compose.yml` 已为前后端配置 `pull_policy: build`，执行 `docker compose up` 时会触发镜像构建；依赖文件未变化时会复用 Docker 缓存，避免每次重新安装依赖。后台运行可使用：

```powershell
docker compose up -d
```

服务端口：

| 服务 | 地址 |
| --- | --- |
| 前端 | http://localhost:30008 |
| 后端 | http://localhost:8846 |

`docker-compose.yml` 会把项目根目录挂载到后端容器 `/workspace`，前端使用生产构建镜像，并设置：

- `SENTIMENTFLOW_PROJECT_ROOT=/workspace`
- 前端 `BACKEND_API_URL=http://backend:8846`

## 配置说明

项目会从根目录 `.env` 或进程环境变量读取配置。不要把真实 token、密钥或私有路径提交到仓库。

### 后端与推理配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PREDICT_MODEL_TYPE` | `lstm` | 默认预测模型类型，可设为 `lstm` 或 `bert` |
| `MODEL_PATH` | 自动扫描 | LSTM checkpoint 文件或目录 |
| `BERT_CHECKPOINT_PATH` | `./bert_sentiment_model` | BERT checkpoint 目录 |
| `MODEL_MAX_LEN` | `100` | LSTM 推理最大序列长度 |
| `MODEL_VOCAB_SIZE` | `65536` | LSTM 哈希词表大小 |
| `SENTIMENTFLOW_PROJECT_ROOT` | 后端目录父级 | 后端定位项目根目录和 `models/` 的路径 |

### 前端代理配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_API_URL` | 空 | 指定后端地址，设置后只访问该地址 |
| `NEXT_SERVER_API_BASE_URL` | 空 | 后端地址兼容变量 |
| `BACKEND_API_TIMEOUT_MS` | `10000` | 前端 API Route 访问后端的超时时间 |

如果没有指定后端地址，前端会依次尝试：

```text
http://127.0.0.1:8846
http://localhost:8846
http://backend:8846
```

### LSTM 训练配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TRAIN_DATASETS` | 多个 HuggingFace 数据集 | LSTM 训练数据集列表 |
| `MODEL_PATH` | `./sentiment_model.pt` | LSTM 训练输出 checkpoint |
| `EPOCHS` | `25` | 训练轮数 |
| `TRAIN_BATCH_SIZE` | GPU: `256`, CPU: `128` | batch size |
| `TRAIN_LR` | `0.0005` | 学习率 |
| `TRAIN_NUM_WORKERS` | GPU: `1`, CPU: `0` | DataLoader worker 数量 |
| `TRAIN_ACCUM_STEPS` | `1` | 梯度累积步数 |
| `TRAIN_WEIGHTED_LOSS` | `1` | 是否启用类别加权损失 |
| `EARLY_STOP_PATIENCE` | `2` | 早停 patience |
| `EARLY_STOP_MIN_DELTA` | `0.0005` | 早停最小提升阈值 |

### BERT 训练配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BERT_MODEL_NAME` | `hfl/chinese-roberta-wwm-ext` | HuggingFace 预训练模型名或本地路径 |
| `BERT_MODEL_MAX_LEN` | `128` | 最大 token 长度 |
| `BERT_CHECKPOINT_PATH` | `./bert_sentiment_model` | BERT checkpoint 输出目录 |
| `BERT_TRAINING_STAGE` | `auto` | `teacher`、`student` 或 `legacy` |
| `BERT_EPOCHS` | `5` | 训练轮数 |
| `BERT_TRAIN_BATCH_SIZE` | GPU: `32`, CPU: `16` | 训练 batch size |
| `BERT_EVAL_BATCH_SIZE` | GPU: `64`, CPU: `32` | 验证 batch size |
| `BERT_TRAIN_LR` | `2e-5` | 学习率 |
| `BERT_WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `BERT_SELECTION_METRIC` | `qwk` | 选择最优 checkpoint 的指标 |
| `BERT_TEACHER_CHECKPOINT_PATH` | 空 | Student 阶段使用的 Teacher checkpoint |
| `PSEUDO_LABEL_PATH` | `pseudo_labels.jsonl` | 伪标签缓存文件 |
| `PSEUDO_LABEL_MIN_CONFIDENCE` | `0.75` | Teacher 伪标签最低置信度 |
| `PSEUDO_LABEL_WEIGHT` | `0.3` | 伪标签样本权重 |
| `BERT_GRADIENT_CHECKPOINTING` | `1` | 是否启用梯度 checkpoint |
| `BERT_MIXED_PRECISION` | `fp16` | 混合精度类型，支持 `fp16` / `bf16` |

## 模型训练

### 通过前端训练

1. 启动后端和前端。
2. 打开 http://localhost:3000（本地开发）或 http://localhost:30008（Docker）。
3. 切换到 `模型训练`。
4. 选择 `LSTM` 或 `BERT (RoBERTa)`。
5. 选择数据集和超参数。
6. 点击 `开始训练`。

后端会创建一个 `job_id`，训练输出保存到：

```text
models/{model_type}_{UTC时间戳}/
```

例如：

```text
models/lstm_20260508_074918/model.pt
models/bert_20260509_064555/
```

训练完成后，后端会自动把该模型设置为活跃模型。

### 通过命令行训练 LSTM

```powershell
cd C:\Code\SentimentFlow
$env:MODEL_PATH="models\lstm_manual\model.pt"
$env:FORCE_RETRAIN="1"
python LSTM.py
```

`LSTM.py` 是兼容入口，实际调用 `training.main.run()`。

### 通过命令行训练 BERT Teacher

```powershell
cd C:\Code\SentimentFlow
$env:BERT_TRAINING_STAGE="teacher"
$env:BERT_CHECKPOINT_PATH="models\bert_teacher"
$env:BERT_FORCE_RETRAIN="1"
python BERT.py
```

Teacher 阶段只允许真实多分类数据集：

- `BerlinWang/DMSC`
- `dirtycomputer/JD_review`

### 通过命令行训练 BERT Student

```powershell
cd C:\Code\SentimentFlow
$env:BERT_TRAINING_STAGE="student"
$env:BERT_TEACHER_CHECKPOINT_PATH="models\bert_teacher"
$env:BERT_CHECKPOINT_PATH="models\bert_student"
$env:PSEUDO_LABEL_PATH="pseudo_labels.jsonl"
$env:BERT_FORCE_RETRAIN="1"
python BERT.py
```

Student 阶段流程：

1. 加载 Teacher checkpoint。
2. 对二分类数据集生成 0-5 soft pseudo labels。
3. 丢弃低置信伪标签。
4. 合并真实多分类样本和伪标签样本。
5. 用样本权重和课程学习控制伪标签影响。

## 模型推理

### 通过前端推理

1. 打开前端首页。
2. 在 `情感预测` 标签页输入文本。
3. 点击 `发送预测请求`。
4. 页面展示评分、标签、置信度、概率条和原始 JSON 响应。

### 通过后端 API 推理

请求：

```http
POST /api/predict/
Content-Type: application/json
```

```json
{
  "text": "这个产品非常好用，体验超出预期。",
  "model": "bert"
}
```

`model` 可选，不传时使用 `PREDICT_MODEL_TYPE`。

响应：

```json
{
  "text": "这个产品非常好用，体验超出预期。",
  "score": 5,
  "label": "extremely_positive",
  "label_zh": "极端正面",
  "confidence": 0.9321,
  "probabilities": [0.001, 0.002, 0.004, 0.018, 0.043, 0.932],
  "reasoning": "模型将文本情感强度判定为 5 分（极端正面）。",
  "source": "bert",
  "model_name": "bert_20260509_064555"
}
```

## 后端 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/predict/` | 文本情感预测 |
| `POST` | `/api/training/start` | 启动训练任务 |
| `GET` | `/api/training/status/{job_id}` | 查询训练状态 |
| `GET` | `/api/training/stream/{job_id}` | SSE 实时训练日志 |
| `GET` | `/api/training/jobs` | 查看训练任务列表 |
| `POST` | `/api/training/cancel/{job_id}` | 取消训练任务 |
| `GET` | `/api/models/` | 扫描模型列表 |
| `GET` | `/api/models/active` | 获取活跃模型 |
| `PUT` | `/api/models/active` | 设置活跃模型 |
| `DELETE` | `/api/models/{model_id}` | 删除模型目录 |
| `POST` | `/api/auth/login` | 登录占位接口 |
| `GET` | `/api/admin/users` | 管理占位接口 |
| `GET` | `/api/stats/overview` | 统计占位接口 |

说明：`auth`、`admin`、`stats` 当前是占位实现，项目核心功能集中在预测、训练和模型管理。

## 前端 API Route

前端不直接从浏览器请求 FastAPI，而是通过 Next.js API Route 做后端代理：

| 前端路径 | 转发到后端 |
| --- | --- |
| `/api/integration/health` | `/health` |
| `/api/integration/predict` | `/api/predict/` |
| `/api/training/start` | `/api/training/start` |
| `/api/training/status/[jobId]` | `/api/training/status/{job_id}` |
| `/api/training/stream/[jobId]` | `/api/training/stream/{job_id}` |
| `/api/training/jobs` | `/api/training/jobs` |
| `/api/training/cancel` | `/api/training/cancel/{job_id}` |
| `/api/models` | `/api/models/` |
| `/api/models/active` | `/api/models/active` |

这样可以避免浏览器跨域问题，并让前端在本地、Docker 和容器网络中复用同一套调用代码。

## 核心模型设计

### 0-5 情感评分契约

`sentiment_scale.py` 是全项目共享的评分标准：

| 分数 | 英文标签 | 中文标签 |
| --- | --- | --- |
| 0 | `extremely_negative` | 极端负面 |
| 1 | `clearly_negative` | 明显负面 |
| 2 | `slightly_negative` | 略微负面 |
| 3 | `neutral` | 中性 |
| 4 | `slightly_positive` | 略微正面 |
| 5 | `extremely_positive` | 极端正面 |

所有训练、评估、推理和 API 响应都围绕这个契约实现。

### LSTM 分支

LSTM 训练代码在 `training/`：

```text
文本 -> jieba 分词 -> CRC32 哈希 id -> Embedding -> LSTM -> Linear -> 6 类 logits
```

特点：

- 训练和推理成本低。
- 不需要维护显式词表。
- 使用 `DistanceAwareOrdinalLoss` 处理 0-5 有序评分。
- 适合作为基线模型和答辩演示中的轻量模型。

### BERT 分支

BERT 训练代码在 `BERT/`：

```text
文本 -> HuggingFace tokenizer -> RoBERTa/BERT backbone -> dropout -> Linear -> 6 类 logits
```

特点：

- 使用预训练中文语言模型，语义理解能力更强。
- 支持 teacher/student 两阶段训练。
- 使用 soft labels、伪标签权重、类别均衡、logit adjustment、focal loss、QWK 选模等策略。
- checkpoint 保存为 HuggingFace 兼容目录，并额外保存 `sentiment_model_state.pt` 和 `training_meta.json`。

## 模型产物

模型统一保存在项目根目录 `models/`。该目录被 `.gitignore` 忽略大文件，只保留目录结构。

LSTM 产物示例：

```text
models/lstm_YYYYMMDD_HHMMSS/
├── model.pt
└── training_meta.json
```

BERT 产物示例：

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

`training_meta.json` 供模型管理 API 快速读取指标，不需要加载模型大权重。

## 测试与校验

运行 Python 测试：

```powershell
cd C:\Code\SentimentFlow
python -m unittest tests.test_sentiment_scale
```

运行前端 lint：

```powershell
cd C:\Code\SentimentFlow\frontend
yarn lint
```

运行前端构建：

```powershell
cd C:\Code\SentimentFlow\frontend
yarn build
```

测试覆盖重点：

- 0/1 二分类旧标签迁移到 0/5。
- 1-5 星级数据映射到 0-5。
- 6 类概率输出契约。
- LSTM/BERT Dataset 标签处理。
- BERT soft pseudo label 加载。
- BERT teacher/student 数据集选择逻辑。

## 常见问题

### 预测提示没有模型怎么办？

先到 `模型训练` 页面完成一次训练，或手动设置：

```powershell
$env:MODEL_PATH="C:\Code\SentimentFlow\models\lstm_xxx\model.pt"
$env:BERT_CHECKPOINT_PATH="C:\Code\SentimentFlow\models\bert_xxx"
$env:PREDICT_MODEL_TYPE="bert"
```

后端也会自动扫描 `models/` 目录，找到最新可用的同类型模型并激活。

### BERT checkpoint 目录存在但无法加载怎么办？

BERT 推理只接受包含 `training_meta.json` 的微调 checkpoint。只有基础预训练权重而没有训练元信息时，代码会拒绝加载，避免用未训练分类头进行预测。

处理方式：

1. 删除无效 checkpoint 目录。
2. 重新训练 BERT。
3. 或把 `BERT_CHECKPOINT_PATH` 指向有效的 `models/bert_xxx/`。

### 前端训练日志断开怎么办？

前端优先使用 SSE 实时日志。如果 SSE 断开，会自动切换到 `/api/training/status/{job_id}` 轮询。刷新页面后，前端会从 `localStorage` 恢复最近的训练任务。

### 为什么 BERT Teacher 不能直接使用二分类数据集？

项目目标是 0-5 六档评分。如果直接把二分类 0/1 映射成 0/5 训练 Teacher，模型容易退化成只会预测极端负面或极端正面。当前设计是：

- Teacher 只学真实多分类数据。
- Student 使用 Teacher 对二分类数据生成高置信 0-5 soft pseudo labels。

### 模型删除是否安全？

后端删除模型时会先确认目标路径位于 `models/` 目录内部，避免误删项目外文件。

## 开发注意事项

- 不要提交 `.env`、模型权重、缓存目录、`node_modules` 或 `.venv`。
- 训练 BERT 时不要使用会频繁重启进程的热重载方式，否则后台训练线程会被中断。
- 后台训练任务保存在进程内存中，后端进程重启后任务状态不会持久化。
- `auth`、`admin`、`stats` 目前是占位接口，不能当作完整用户系统或权限系统。
- BERT 模型文件很大，复制、删除、提交前都要确认路径。
