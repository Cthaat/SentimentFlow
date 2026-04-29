"""生成短句训练数据 - 填补训练数据与推理数据的分布差异。"""

import os
import json
from datasets import load_dataset, Dataset

# 高质量的短句正面例子
POSITIVE_SHORT_SENTENCES = [
    "很好", "不错", "不错的", "好", "很棒", "棒极了", "太棒了", "非常好",
    "非常不错", "不错呢", "好极了", "棒", "给力", "很不错", "相当不错",
    "爱死了", "太喜欢了", "爱好", "喜欢", "超喜欢", "非常喜欢", "很喜欢",
    "满意", "非常满意", "很满意", "完美", "太完美了", "极好", "很极好",
    "赞", "真赞", "太赞了", "大赞", "强", "很强", "太强了", "强到爆",
    "清晰", "很清晰", "非常清晰", "清晰度高", "清爽", "清新", "明亮",
    "流畅", "很流畅", "非常流畅", "顺畅", "快速", "很快", "很快的",
    "精致", "很精致", "非常精致", "精美", "很精美", "漂亮", "很漂亮",
    "舒适", "很舒适", "非常舒适", "舒服", "很舒服", "柔软", "很柔软",
    "新鲜", "很新鲜", "非常新鲜", "新", "很新", "最新", "时尚", "很时尚",
]

# 高质量的短句负面例子
NEGATIVE_SHORT_SENTENCES = [
    "很差", "差", "太差", "太差了", "差劲", "很差劲", "糟糕", "很糟糕",
    "太糟糕", "糟糕透了", "烂", "很烂", "太烂", "烂到家", "讨厌", "很讨厌",
    "不喜欢", "很不喜欢", "不爱", "讨厌死了", "反感", "很反感", "厌烦",
    "不满意", "很不满意", "不满足", "失望", "很失望", "太失望", "极度失望",
    "烂透了", "坏", "很坏", "太坏", "坏到家", "垃圾", "很垃圾", "太垃圾",
    "破", "很破", "破旧", "老化", "陈旧", "掉色", "褪色", "变形",
    "卡顿", "很卡顿", "太卡顿", "卡死", "闪退", "经常闪退", "总是闪退",
    "慢", "很慢", "太慢", "非常慢", "极慢", "龟速", "蜗牛速", "缓慢",
    "粗糙", "很粗糙", "太粗糙", "粗劣", "简陋", "廉价", "很廉价", "山寨",
    "漏水", "很漏水", "不防水", "生锈", "容易生锈", "褪色", "易褪色", "掉毛",
]

def generate_short_sentence_dataset(size=10000):
    """生成短句训练数据。"""
    import random
    random.seed(42)
    
    data = []
    
    # 各领域的前缀，增加多样性
    domains = ["这个", "这个产品", "这件", "这款", "这个功能", "这个地方", "这个部分"]
    
    # 生成正面句子
    for _ in range(size // 2):
        prefix = random.choice(domains)
        sentence = random.choice(POSITIVE_SHORT_SENTENCES)
        full_text = f"{prefix}{sentence}" if len(sentence) < 5 else sentence
        data.append({"text": full_text, "label": 1})
    
    # 生成负面句子
    for _ in range(size // 2):
        prefix = random.choice(domains)
        sentence = random.choice(NEGATIVE_SHORT_SENTENCES)
        full_text = f"{prefix}{sentence}" if len(sentence) < 5 else sentence
        data.append({"text": full_text, "label": 0})
    
    # 打乱顺序
    random.shuffle(data)
    
    return Dataset.from_dict({
        "text": [d["text"] for d in data],
        "label": [d["label"] for d in data]
    })

def add_synthetic_data_to_training():
    """生成合成数据并追加到训练集。"""
    from .data_sources import build_train_split_and_val_split
    from datasets import concatenate_datasets
    
    print("="*80)
    print("GENERATING SYNTHETIC SHORT SENTENCE DATA")
    print("="*80)
    
    # 生成合成数据
    print("Generating 10,000 synthetic short sentences...")
    synthetic_ds = generate_short_sentence_dataset(size=10000)
    print(f"  Positive: {sum(1 for x in synthetic_ds['label'] if x == 1)}")
    print(f"  Negative: {sum(1 for x in synthetic_ds['label'] if x == 0)}")
    
    # 加载原始训练数据
    print("\nLoading original training data...")
    n, train, val, _ = build_train_split_and_val_split()
    print(f"  Original train size: {len(train)}")
    print(f"  Positive: {sum(1 for x in train['label'] if x == 1)}")
    print(f"  Negative: {sum(1 for x in train['label'] if x == 0)}")
    
    # 合并
    combined_train = concatenate_datasets([train, synthetic_ds])
    print(f"\n✓ Combined train size: {len(combined_train)}")
    print(f"  Positive: {sum(1 for x in combined_train['label'] if x == 1)}")
    print(f"  Negative: {sum(1 for x in combined_train['label'] if x == 0)}")
    
    # 样本展示
    print("\nSample synthetic sentences:")
    for i in range(10):
        text = combined_train['text'][len(train) + i]
        label = combined_train['label'][len(train) + i]
        label_str = "正面" if label == 1 else "负面"
        print(f"  [{label_str}] {text}")
    
    return combined_train, val

if __name__ == "__main__":
    # 演示用途
    print("生成样本短句数据...")
    ds = generate_short_sentence_dataset(size=100)
    
    print(f"\n生成的前10条数据：")
    for i in range(10):
        text = ds['text'][i]
        label = ds['label'][i]
        label_str = "正面" if label == 1 else "负面"
        print(f"  [{label_str}] {text}")
    
    print(f"\n✓ 可用于训练的合成数据集创建完成！")
    print(f"  总量：{len(ds)} 条")
    print(f"  正面：{sum(1 for x in ds['label'] if x == 1)} 条")
    print(f"  负面：{sum(1 for x in ds['label'] if x == 0)} 条")
