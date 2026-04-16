# 快速启用 ttxy/online_shopping_10_cats 指南

## 现状

当前配置 (`.env`) 包含 3 个数据集：

- lansinuote/ChnSentiCorp
- XiangPan/waimai_10k
- dirtycomputer/weibo_senti_100k
- **已禁用**: ttxy/online_shopping_10_cats

## 为什么 ttxy 被禁用？

ttxy/online_shopping_10_cats 是一个**多类分类数据集**（10 个评级类别），需要转换为二分类：

- **问题**：原始标签范围是 1-10，但模型期望二分类标签 [0, 1]
- **解决**：实现了自动标签映射机制
    - 评级 1-5（较差） → 负面 (0)
    - 评级 6-10（较好） → 正面 (1)

## 启用 ttxy 数据集

### 方法 1：编辑 .env 文件（推荐）

打开 `.env` 文件，找到这一行：

```
TRAIN_DATASETS=lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k
```

改为：

```
TRAIN_DATASETS=lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats
```

### 方法 2：命令行覆盖

```bash
# 在运行前设置环境变量
export TRAIN_DATASETS="lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats"
python LSTM.py
```

## 预期结果

启用 ttxy 后：

- **训练样本增加**：~128k → ~185k（增加 ~6.2 万条电商评论）
- **标签分布**：由于 ttxy 的评级分布，可能出现类不平衡（自动使用加权损失处理）
- **训练时间**：初始化增加 20-30 秒，每 epoch 增加 ~50%
- **模型性能**：电商数据可提升模型在电商场景的表现

## 实现细节

### 标签映射规则

定义在 `training/data_sources.py` 的 `get_label_map()` 函数：

```python
if dataset_name == "ttxy/online_shopping_10_cats":
    return {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0,  # 负面（评级 1-5 级）
        6: 1, 7: 1, 8: 1, 9: 1, 10: 1  # 正面（评级 6-10 级）
    }
```

### 数据流

1. **加载**：HuggingFace 加载原始数据集（标签 1-10）
2. **映射**：CsvStreamDataset 应用标签映射（1-10 → 0-1）
3. **验证**：在数据加载时验证映射后的标签有效性
4. **训练**：模型在二分类标签上训练

## 故障排查

### 训练卡住或很慢

**症状**：启用 ttxy 后，训练初始化阶段花费很长时间

**原因**：

- 数据集总大小增至 185k+ 样本
- HuggingFace 数据集加载和预处理耗时

**解决方案**：

1. **快速测试**：暂时禁用 ttxy，验证基础配置工作
2. **缩小规模**：使用 `TRAIN_MAX_SAMPLES` 限制样本数
3. **预加载**：首次下载后会缓存，后续运行更快

### 标签映射失败

**错误**：`Invalid mapped label value`

**检查清单**：

- [ ] ttxy 数据集在 `get_label_map()` 中定义了映射规则
- [ ] 映射覆盖了所有原始标签（1-10）
- [ ] 所有映射值都在 [0, 1] 范围内

### 类不平衡警告

**现象**：训练输出显示 `pos_ratio=0.1292`（类严重不平衡）

**原因**：ttxy 中负面评价（1-5 级）远多于正面（6-10 级）

**自动处理**：

- 加权损失函数已启用（`TRAIN_WEIGHTED_LOSS=1`）
- pos_weight 自动调整为 3.8691 以平衡类

## 添加其他多类数据集

遵循相同模式为任何多类数据集添加支持：

```python
# 在 training/data_sources.py 的 get_label_map() 中添加

if dataset_name == "owner/new_multiclass_dataset":
    return {
        # 将类 0-4 映射为负面
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
        # 将类 5-9 映射为正面
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    }
```

详见 [LABEL_MAPPING.md](./LABEL_MAPPING.md)

## 性能指标

与基础 3 数据集的对比（预期）：

| 指标       | 基础（3 数据集） | 含 ttxy（4 数据集）    |
| ---------- | ---------------- | ---------------------- |
| 训练样本   | 128,377          | ~185,000+              |
| Epoch 时间 | ~60s             | ~90-100s               |
| 初始化时间 | ~30s             | ~50-60s                |
| 类平衡度   | 平衡             | 不平衡（自动加权处理） |

## 参考文件

- [.env](./.env) - 主配置文件
- [training/data_sources.py](./training/data_sources.py) - 数据加载和映射规则
- [training/dataset.py](./training/dataset.py) - 数据集类
- [training/trainer.py](./training/trainer.py) - 训练逻辑
- [LABEL_MAPPING.md](./LABEL_MAPPING.md) - 完整技术文档
