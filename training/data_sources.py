"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from typing import Tuple, Dict


def _parse_label_map_from_env(raw: str) -> Dict[int, int]:
    """解析标签映射配置。

    形如 "0:1,1:0,2:-1,3:-1"：
    - 0/1 表示映射到二分类标签
    - -1 表示丢弃该原始标签样本
    """
    mapping: Dict[int, int] = {}
    for item in raw.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid label map item '{pair}'. Expected format like '0:1'."
            )
        src, dst = pair.split(":", 1)
        src_label = int(src.strip())
        dst_label = int(dst.strip())
        if dst_label not in (-1, 0, 1):
            raise ValueError(
                f"Invalid target label {dst_label} in '{pair}'. Expected -1, 0 or 1."
            )
        mapping[src_label] = dst_label
    if not mapping:
        raise ValueError("Parsed empty label map from environment variable.")
    return mapping


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """获取数据集的标签映射规则。
    
    用于兼容多类数据集，将其转换为二分类。
    返回 None 表示无需映射（已是二分类）。
    """
    if dataset_name == "ttxy/online_shopping_10_cats":
        # 10类评分转二分类。
        # 为了减少边界噪声，默认丢弃 5/6 分样本，仅保留更明确的两端样本。
        # 可通过环境变量覆盖，例如：TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP=1:0,2:0,3:0,4:0,5:-1,6:-1,7:1,8:1,9:1,10:1
        raw_map = os.getenv(
            "TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP",
            "1:0,2:0,3:0,4:0,5:-1,6:-1,7:1,8:1,9:1,10:1",
        )
        return _parse_label_map_from_env(raw_map)
    if dataset_name == "dirtycomputer/simplifyweibo_4_moods":
        # 4类情绪转二分类。默认规则：保留 0/1，丢弃 2/3（噪声类）。
        # 可通过环境变量覆盖，例如：SIMPLIFYWEIBO_4_MOODS_LABEL_MAP=0:1,1:0,2:-1,3:-1
        raw_map = os.getenv("SIMPLIFYWEIBO_4_MOODS_LABEL_MAP", "0:1,1:0,2:-1,3:-1")
        return _parse_label_map_from_env(raw_map)
    return None  # 其他数据集无需映射


def build_train_split_and_val_split():
    """加载并拼接多个训练/验证数据集。

    返回：(dataset_names, train_split, val_split, label_map)
    - label_map: 标签映射规则（用于多类数据集转二分类），None 表示无需映射
    
    行为说明：
    - 优先读取 validation split
    - 没有 validation 时退化到 test split
    - 两者都没有时，从 train 自动切分验证集
    - 如果数据集需要标签映射，在拼接前应用映射
    """
    from datasets import concatenate_datasets, load_dataset

    dataset_names = [
        item.strip()
        # TRAIN_DATASETS 环境变量示例：lansinuote/ChnSentiCorp,XiangPan/waimai_10k
        # 改为仅使用两个高质量数据集，避免多数据集混合导致的类别不平衡
        for item in os.getenv(
            "TRAIN_DATASETS",
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k"
        ).split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    label_map = None  # 标记是否有任何数据集需要映射
    # TRAIN_VAL_RATIO 环境变量示例：0.1（表示 10% 的训练数据用于验证）
    val_ratio = float(os.getenv("TRAIN_VAL_RATIO", "0.1"))
    val_ratio = min(max(val_ratio, 0.01), 0.5)

    for name in dataset_names:
        ds = load_dataset(name)
        if "train" not in ds:
            raise ValueError(f"Dataset {name} has no train split.")

        # 检查是否需要标签映射
        current_label_map = get_label_map(name)
        if current_label_map:
            label_map = current_label_map  # 记录映射规则
            print(f"Using label mapping for dataset: {name}")
            print(f"Label map: {current_label_map}")

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

        # 如果需要标签映射，在拼接前应用。
        # 当映射目标为 -1 时，表示丢弃该标签样本（如噪声类）。
        if current_label_map:
            def apply_mapping(row, mapping=current_label_map):
                row_copy = dict(row)
                row_copy["label"] = mapping.get(int(row["label"]), -1)
                return row_copy
            
            train_part = train_part.map(apply_mapping)
            val_part = val_part.map(apply_mapping)

            train_before = len(train_part)
            val_before = len(val_part)
            train_part = train_part.filter(lambda row: int(row["label"]) in (0, 1))
            val_part = val_part.filter(lambda row: int(row["label"]) in (0, 1))
            print(
                f"After mapping filter for {name}: "
                f"train {len(train_part)}/{train_before}, val {len(val_part)}/{val_before}"
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

    # 可选：添加提取的真实短句数据来补充分布差异
    # USE_EXTRACTED_SHORT_SENTENCES 环境变量：1 表示启用（默认），0 表示禁用
    if os.getenv("USE_EXTRACTED_SHORT_SENTENCES", "1") == "1":
        try:
            from pathlib import Path
            csv_path = Path(__file__).resolve().parent.parent / "extracted_short_sentences.csv"
            
            if csv_path.exists():
                print(f"Loading extracted short sentences from {csv_path}...")
                
                # 直接读取CSV，避免pandas转换引起的类型问题
                short_texts = []
                short_labels = []
                
                with open(csv_path, 'r', encoding='utf-8') as f:
                    # 跳过header (text,label)
                    next(f)
                    for line in f:
                        line = line.rstrip('\n')
                        if not line:
                            continue
                        # 格式：text,label，找最后一个逗号来分割
                        last_comma = line.rfind(',')
                        if last_comma > 0:
                            text = line[:last_comma]
                            label_str = line[last_comma+1:]
                            try:
                                label = int(label_str)
                                if label in (0, 1):
                                    short_texts.append(text)
                                    short_labels.append(label)
                            except ValueError:
                                continue
                
                if short_texts:
                    from datasets import Dataset
                    short_sentences_ds = Dataset.from_dict({
                        'text': short_texts,
                        'label': short_labels
                    })
                    
                    # 可选限制数量
                    max_short_samples = int(os.getenv("EXTRACTED_SHORT_SENTENCES_MAX", "0"))
                    if max_short_samples > 0 and len(short_sentences_ds) > max_short_samples:
                        short_sentences_ds = short_sentences_ds.select(range(max_short_samples))
                    
                    # 混合：长句 + 短句
                    short_ratio = float(os.getenv("SHORT_SENTENCES_RATIO", "0.3"))  # 默认30%短句
                    short_samples_to_use = int(len(train_split) * short_ratio / (1 - short_ratio))
                    short_samples_to_use = min(short_samples_to_use, len(short_sentences_ds))
                    
                    if short_samples_to_use > 0:
                        short_sentences_ds = short_sentences_ds.select(range(short_samples_to_use))
                        train_split = concatenate_datasets([train_split, short_sentences_ds])
                        neg_short = sum(1 for l in short_labels[:short_samples_to_use] if l == 0)
                        pos_short = sum(1 for l in short_labels[:short_samples_to_use] if l == 1)
                        print(f"Added {len(short_sentences_ds)} extracted short sentences (neg={neg_short}, pos={pos_short}).")
                        print(f"  New train size: {len(train_split)} (long:short = {100*(1-short_ratio):.0f}%:{100*short_ratio:.0f}%)")
                else:
                    print("Warning: No valid short sentences found in CSV")
            else:
                print(f"Note: extracted_short_sentences.csv not found at {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to load short sentences: {e}")

    # 可选：添加合成短句数据来补充分布差异（不推荐，已证明无效）
    # USE_SYNTHETIC_DATA 环境变量：1 表示启用，0 表示禁用
    if os.getenv("USE_SYNTHETIC_DATA", "0") == "1":
        try:
            from .generate_synthetic_data import generate_short_sentence_dataset
            
            synthetic_size = int(os.getenv("SYNTHETIC_DATA_SIZE", "5000"))
            print(f"Generating {synthetic_size} synthetic short sentences...")
            synthetic_ds = generate_short_sentence_dataset(size=synthetic_size)
            
            # 将合成数据追加到训练集
            train_split = concatenate_datasets([train_split, synthetic_ds])
            print(f"Added {len(synthetic_ds)} synthetic samples. New train size: {len(train_split)}")
        except ImportError:
            print("Warning: Synthetic data generation not available (generate_synthetic_data module not found)")

    # 返回 None 作为 label_map，因为映射已在拼接前应用
    return dataset_names, train_split, val_split, None


def get_label_distribution(split, label_map: dict | None = None) -> Tuple[int, int]:
    """统计二分类标签分布。
    
    注意：如果标签映射在 build_train_split_and_val_split() 中应用过，label_map 参数会是 None。
    此函数仅统计最终的二分类标签（0 和 1）。
    """
    labels = split["label"]
    neg = sum(1 for x in labels if int(x) == 0)
    pos = sum(1 for x in labels if int(x) == 1)
    return neg, pos
