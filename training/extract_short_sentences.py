"""从长评论中提取短句子，保持原始标签。

策略：
1. 加载训练配置中的多源数据集。
2. 使用句号、感叹号、逗号等分割。
3. 保留 5-20 字的短句（过滤太短/太长）。
4. 将多源标签统一到二分类（0=负面, 1=正面）。
5. 导出混合后的短句数据集。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset


DATASET_NAMES = [
    "lansinuote/ChnSentiCorp",
    "XiangPan/waimai_10k",
    "dirtycomputer/weibo_senti_100k",
    "dirtycomputer/JD_review",
    "ndiy/NLPCC14-SC",
    "dirtycomputer/ChnSentiCorp_htl_all",
    "BerlinWang/DMSC",
]

TEXT_CANDIDATES = ["text", "review", "content", "Comment", "comment", "sentence"]
LABEL_CANDIDATES = ["label", "sentiment", "Star", "score", "rating"]


def extract_short_sentences_from_text(text: str, label: int, min_len: int = 5, max_len: int = 20):
    """从一条长文本中提取短句子。
    
    Args:
        text: 长文本
        label: 原始标签（0=负, 1=正）
        min_len: 最小字数（过滤过短的碎片）
        max_len: 最大字数（保持短句特性）
    
    Returns:
        [(短句, 标签), ...] 列表
    """
    result = []
    
    # 使用多种分隔符分割
    for sent in text.replace('。', '|').replace('！', '|').replace('，', '|').split('|'):
        sent = sent.strip()
        if min_len <= len(sent) <= max_len:
            result.append((sent, label))
    
    return result


def _to_binary_label(raw_label: Any, dataset_name: str, label_col: str) -> int:
    """将不同来源标签统一为二分类标签。返回 -1 表示应过滤。"""
    if dataset_name == "BerlinWang/DMSC" and label_col == "Star":
        try:
            star = int(raw_label)
        except (TypeError, ValueError):
            return -1
        if star <= 2:
            return 0
        if star >= 4:
            return 1
        return -1

    if dataset_name == "dirtycomputer/JD_review" and label_col == "rating":
        try:
            rating = int(float(raw_label))
        except (TypeError, ValueError):
            return -1
        if rating <= 2:
            return 0
        if rating >= 4:
            return 1
        return -1

    if isinstance(raw_label, bool):
        return int(raw_label)

    if isinstance(raw_label, (int, float)):
        value = int(raw_label)
        if value in (0, 1):
            return value
        if value == -1:
            return 0
        return -1

    text = str(raw_label).strip().lower()
    if text in {"1", "pos", "positive", "正面", "好评"}:
        return 1
    if text in {"0", "-1", "neg", "negative", "负面", "差评"}:
        return 0
    return -1


def _resolve_split_and_columns(ds_name: str):
    """加载数据集，并推断 split/text/label 字段。"""
    ds = load_dataset(ds_name)

    if "train" in ds:
        split = ds["train"]
        split_name = "train"
    elif "validation" in ds:
        split = ds["validation"]
        split_name = "validation"
    elif "test" in ds:
        split = ds["test"]
        split_name = "test"
    else:
        raise ValueError(f"Dataset {ds_name} has no train/validation/test split")

    columns = set(split.column_names)
    text_col = next((c for c in TEXT_CANDIDATES if c in columns), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Dataset {ds_name} unsupported schema: {sorted(columns)}"
        )

    return split_name, split, text_col, label_col


def _extract_from_dataset(ds_name: str) -> list[tuple[str, int]]:
    """从单个数据集提取短句。"""
    split_name, split, text_col, label_col = _resolve_split_and_columns(ds_name)
    print(f"Loading {ds_name} ({split_name}, size={len(split)})...")

    extracted: list[tuple[str, int]] = []
    skipped_labels = 0
    for item in split:
        text = str(item.get(text_col) or "").strip()
        if not text:
            continue

        label = _to_binary_label(item.get(label_col), ds_name, label_col)
        if label not in (0, 1):
            skipped_labels += 1
            continue

        extracted.extend(extract_short_sentences_from_text(text, label))

    print(
        f"  -> extracted={len(extracted)}, skipped_label_rows={skipped_labels}, "
        f"text_col={text_col}, label_col={label_col}"
    )
    return extracted


def process_dataset():
    """从HF数据集中提取短句子。"""

    short_sentences = []

    print(f"Active TRAIN_DATASETS ({len(DATASET_NAMES)}): {', '.join(DATASET_NAMES)}")

    for name in DATASET_NAMES:
        try:
            short_sentences.extend(_extract_from_dataset(name))
        except Exception as exc:
            print(f"Warning: failed to extract from {name}: {exc}")
    
    print(f"\nExtracted {len(short_sentences)} short sentences in total")
    
    # 统计
    pos_count = sum(1 for _, label in short_sentences if label == 1)
    neg_count = sum(1 for _, label in short_sentences if label == 0)
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    
    # 保存为CSV
    df = pd.DataFrame(short_sentences, columns=['text', 'label'])
    df = df.drop_duplicates(subset=['text', 'label'])
    output_path = 'extracted_short_sentences.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # 显示样本
    print("\nSample short sentences:")
    for text, label in short_sentences[:10]:
        label_str = "正面" if label == 1 else "负面"
        print(f"  [{label_str}] {text}")


if __name__ == "__main__":
    process_dataset()
