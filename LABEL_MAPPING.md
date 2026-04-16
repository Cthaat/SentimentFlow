# 标签映射系统说明

本文档介绍如何为多类分类数据集添加二分类标签映射支持。

## 现有支持的数据集

### 二分类数据集（无需映射）

- `lansinuote/ChnSentiCorp` - 标签: [0, 1]
- `XiangPan/waimai_10k` - 标签: [0, 1]
- `dirtycomputer/weibo_senti_100k` - 标签: [0, 1]

### 多类数据集（需要标签映射）

- `ttxy/online_shopping_10_cats` - 原始标签: [1-10] → 映射为 [0, 1]
    - 标签 1-5 级（较差评价）→ 负面 (0)
    - 标签 6-10 级（较好评价）→ 正面 (1)
    - 样本数：~6.2 万条电商评论

## 架构设计

标签映射系统由以下模块组成：

### 1. 映射规则定义 (`training/data_sources.py`)

```python
def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """获取数据集的标签映射规则。"""
    if dataset_name == "ttxy/online_shopping_10_cats":
        return {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0,  # 负面
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1  # 正面
        }
    return None
```

**如何添加新的多类数据集：**

1. 在 `get_label_map()` 中添加条件判断
2. 定义原始标签到二分类标签的映射字典
3. 返回映射规则

### 2. 数据加载 (`training/dataset.py`)

`CsvStreamDataset` 支持可选的 `label_map` 参数：

```python
dataset = CsvStreamDataset(
    source,
    chunk_size=settings.chunk_size,
    max_len=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    label_map=label_map,  # 可选的标签映射
)
```

**映射逻辑：**

```python
if self.label_map and label in self.label_map:
    mapped_label = self.label_map[label]
else:
    mapped_label = label  # 保持原值（用于二分类数据集）
```

### 3. 标签分布计算 (`training/data_sources.py`)

```python
def get_label_distribution(split, label_map: dict | None = None) -> Tuple[int, int]:
    """统计二分类标签分布，支持映射。"""
    if label_map:
        mapped_labels = [label_map.get(int(x), int(x)) for x in labels]
    # ... 计数代码
```

### 4. 训练和评估集成 (`training/trainer.py`)

- `build_train_loader()` 接收 `label_map` 参数
- `evaluate()` 函数应用标签映射进行验证

## 配置方式

### 方法 1：通过环境变量（推荐）

编辑 `.env` 文件：

```bash
# 包含所有数据集（自动应用映射）
TRAIN_DATASETS=lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats

# 仅使用二分类数据集
TRAIN_DATASETS=lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k
```

### 方法 2：代码中硬编码

编辑 `training/data_sources.py` 的 `build_train_split_and_val_split()` 函数：

```python
dataset_names = [
    "lansinuote/ChnSentiCorp",
    "XiangPan/waimai_10k",
    "dirtycomputer/weibo_senti_100k",
    "ttxy/online_shopping_10_cats",  # 添加新数据集
]
```

## 添加新的多类数据集的步骤

假设要添加一个 10 类的新数据集 `example/dataset_name`：

### 第 1 步：在 `get_label_map()` 中定义映射

```python
def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    # ... 现有代码 ...
    if dataset_name == "example/dataset_name":
        return {
            # 负面标签 (class 0-4)
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
            # 正面标签 (class 5-9)
            5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
        }
    return None
```

### 第 2 步：添加到数据集列表

编辑 `.env` 或代码中的 `TRAIN_DATASETS`：

```
TRAIN_DATASETS=lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats,example/dataset_name
```

### 第 3 步：运行训练

```bash
python LSTM.py
```

## 注意事项

1. **标签验证**：数据加载时会自动验证映射后的标签是否在 [0, 1] 范围内
2. **混合数据集**：支持混合二分类和多类数据集，自动应用相应的映射
3. **性能**：对于大数据集（>100k），初始化 DataLoader 可能需要 20-30 秒
4. **跳过无效样本**：任何映射后标签超出范围的样本会触发错误

## 故障排查

### 错误：`Invalid mapped label value`

- **原因**：标签映射规则不完整或不正确
- **解决**：检查 `get_label_map()` 中的映射定义，确保所有原始标签都被正确映射到 [0, 1]

### 标签分布不平衡

- **原因**：源数据集本身标签不平衡
- **解决**：使用加权损失函数（已自动启用）：`TRAIN_WEIGHTED_LOSS=1`

### 训练初始化很慢

- **原因**：大型数据集的 DataLoader 初始化耗时
- **解决**：
    - 减少数据集数量
    - 使用 `TRAIN_MAX_SAMPLES` 限制样本数量进行快速测试

## 参考实现

查看以下文件了解完整实现：

- [training/data_sources.py](../../training/data_sources.py) - 数据集加载和映射规则
- [training/dataset.py](../../training/dataset.py) - 数据集类和映射应用逻辑
- [training/trainer.py](../../training/trainer.py) - 训练循环集成
- [.env](./.env) - 环境变量配置
