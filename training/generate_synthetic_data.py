"""生成 0-5 情感评分短句训练数据。"""

from __future__ import annotations

import random

from datasets import Dataset

from sentiment_scale import SENTIMENT_DISPLAY_LABELS, SENTIMENT_SCORES


SENTENCE_BANK: dict[int, list[str]] = {
    0: ["糟糕透顶", "完全不能接受", "差到离谱", "再也不会买了", "极度失望"],
    1: ["很差", "不太满意", "问题比较多", "体验很不好", "明显不推荐"],
    2: ["有点失望", "略微不舒服", "稍微有些卡顿", "不算太好", "还有点粗糙"],
    3: ["一般般", "中规中矩", "没有明显好坏", "还算普通", "符合基本预期"],
    4: ["还不错", "比较满意", "体验挺好", "基本符合预期", "值得考虑"],
    5: ["非常满意", "远超预期", "太棒了", "强烈推荐", "完美"],
}


def generate_short_sentence_dataset(size: int = 12000):
    """生成均衡的 0-5 短句数据集。"""
    random.seed(42)
    domains = ["这个产品", "这次体验", "这个功能", "这个服务", "整体表现", "这款产品"]
    per_class = max(1, size // len(SENTIMENT_SCORES))

    data = []
    for score in SENTIMENT_SCORES:
        for _ in range(per_class):
            prefix = random.choice(domains)
            sentence = random.choice(SENTENCE_BANK[score])
            data.append({"text": f"{prefix}{sentence}", "label": score})

    random.shuffle(data)
    return Dataset.from_dict({
        "text": [item["text"] for item in data],
        "label": [item["label"] for item in data],
    })


def add_synthetic_data_to_training():
    """生成合成数据并追加到训练集。"""
    from datasets import concatenate_datasets

    from .data_sources import build_train_split_and_val_split, get_label_distribution

    print("=" * 80)
    print("GENERATING SYNTHETIC 0-5 SHORT SENTENCE DATA")
    print("=" * 80)

    synthetic_ds = generate_short_sentence_dataset(size=12000)
    print(f"  Synthetic score counts: {get_label_distribution(synthetic_ds)}")

    _, train, val, _ = build_train_split_and_val_split()
    print(f"  Original train size: {len(train)}")
    print(f"  Original score counts: {get_label_distribution(train)}")

    combined_train = concatenate_datasets([train, synthetic_ds])
    print(f"  Combined train size: {len(combined_train)}")
    print(f"  Combined score counts: {get_label_distribution(combined_train)}")

    return combined_train, val


if __name__ == "__main__":
    ds = generate_short_sentence_dataset(size=120)
    print("生成的前10条数据：")
    for i in range(10):
        text = ds["text"][i]
        label = int(ds["label"][i])
        print(f"  [{label} {SENTIMENT_DISPLAY_LABELS[label]}] {text}")
    print(f"总量：{len(ds)} 条")
