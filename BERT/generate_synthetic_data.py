"""生成短句训练数据。"""

from __future__ import annotations

from datasets import Dataset


POSITIVE_SHORT_SENTENCES = [
    "很好", "不错", "好", "很棒", "太棒了", "非常好", "喜欢", "满意", "完美", "赞", "流畅", "清晰", "舒服", "新鲜",
]

NEGATIVE_SHORT_SENTENCES = [
    "很差", "差", "太差了", "糟糕", "很烂", "讨厌", "失望", "垃圾", "卡顿", "闪退", "很慢", "粗糙", "廉价", "不推荐",
]


def generate_short_sentence_dataset(size: int = 10000):
    """生成短句训练数据。"""
    import random

    random.seed(42)
    data = []
    domains = ["这个", "这个产品", "这款", "这个功能", "这件"]

    for _ in range(size // 2):
        prefix = random.choice(domains)
        sentence = random.choice(POSITIVE_SHORT_SENTENCES)
        full_text = f"{prefix}{sentence}" if len(sentence) < 5 else sentence
        data.append({"text": full_text, "label": 1})

    for _ in range(size // 2):
        prefix = random.choice(domains)
        sentence = random.choice(NEGATIVE_SHORT_SENTENCES)
        full_text = f"{prefix}{sentence}" if len(sentence) < 5 else sentence
        data.append({"text": full_text, "label": 0})

    random.shuffle(data)
    return Dataset.from_dict(
        {
            "text": [d["text"] for d in data],
            "label": [d["label"] for d in data],
        }
    )
