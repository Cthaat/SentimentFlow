"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from typing import Tuple, Dict


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """获取数据集的标签映射规则。
    
    用于兼容多类数据集，将其转换为二分类。
    返回 None 表示无需映射（已是二分类）。
    """
    if dataset_name == "ttxy/online_shopping_10_cats":
        # 10类评分转二分类：1-5级(较差/负面) -> 0, 6-10级(较好/正面) -> 1
        return {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0,  # 负面
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1  # 正面
        }
    return None  # 其他数据集无需映射


def build_train_split_and_val_split():
    """加载并拼接多个训练/验证数据集。

    返回：(dataset_names, train_split, val_split, label_map)
    - label_map: 标签映射规则（用于多类数据集转二分类），None 表示无需映射
    
    行为说明：
    - 优先读取 validation split
    - 没有 validation 时退化到 test split
    - 两者都没有时，从 train 自动切分验证集
    """
    from datasets import concatenate_datasets, load_dataset

    dataset_names = [
        item.strip()
        # TRAIN_DATASETS 环境变量示例：lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats
        for item in os.getenv(
            "TRAIN_DATASETS",
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/weibo_senti_100k,ttxy/online_shopping_10_cats"
        ).split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    label_map = None  # 使用第一个需要映射的数据集的映射规则
    # TRAIN_VAL_RATIO 环境变量示例：0.1（表示 10% 的训练数据用于验证）
    val_ratio = float(os.getenv("TRAIN_VAL_RATIO", "0.1"))
    val_ratio = min(max(val_ratio, 0.01), 0.5)

    for name in dataset_names:
        ds = load_dataset(name)
        if "train" not in ds:
            raise ValueError(f"Dataset {name} has no train split.")

        # 检查是否需要标签映射
        current_label_map = get_label_map(name)
        if current_label_map and label_map is None:
            label_map = current_label_map
            print(f"Using label mapping for dataset: {name}")

        train_part = ds["train"]
        if "validation" in ds:
            val_part = ds["validation"]
        elif "test" in ds:
            val_part = ds["test"]
        else:
            if len(train_part) < 2:
                raise ValueError(f"Dataset {name} is too small to split train/val.")

            split_result = train_part.train_test_split(test_size=val_ratio, seed=42)
            train_part = split_result["train"]
            val_part = split_result["test"]
            print(
                f"Dataset {name} has no validation/test split; "
                f"auto-split train with TRAIN_VAL_RATIO={val_ratio:.2f}."
            )

        train_splits.append(train_part)
        val_splits.append(val_part)

    train_split = train_splits[0] if len(train_splits) == 1 else concatenate_datasets(train_splits)
    val_split = val_splits[0] if len(val_splits) == 1 else concatenate_datasets(val_splits)

    train_split = train_split.shuffle(seed=42)
    val_split = val_split.shuffle(seed=42)

    # TRAIN_MAX_SAMPLES 环境变量示例：10000（表示最多使用 10,000 个训练样本）
    max_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    if max_samples > 0:
        max_samples = min(max_samples, len(train_split))
        train_split = train_split.select(range(max_samples))

    # TRAIN_MAX_VAL_SAMPLES 环境变量示例：1000（表示最多使用 1,000 个验证样本）
    max_val_samples = int(os.getenv("TRAIN_MAX_VAL_SAMPLES", "0"))
    if max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_split))
        val_split = val_split.select(range(max_val_samples))

    return dataset_names, train_split, val_split, label_map


def get_label_distribution(split, label_map: dict | None = None) -> Tuple[int, int]:
    """统计二分类标签分布。
    
    如果提供 label_map，会在计数前应用映射（仅对 label_map 中的标签）。
    """
    labels = split["label"]
    if label_map:
        # 应用标签映射，只映射在 label_map 中的标签
        mapped_labels = [label_map.get(int(x), int(x)) for x in labels]
        neg = sum(1 for x in mapped_labels if x == 0)
        pos = sum(1 for x in mapped_labels if x == 1)
    else:
        # 直接计数原始标签
        neg = sum(1 for x in labels if int(x) == 0)
        pos = sum(1 for x in labels if int(x) == 1)
    return neg, pos
