"""从长评论中提取短句子，保持原始标签。

策略：
1. 加载原始长评论数据
2. 使用句号、感叹号、逗号分割
3. 保留 5-20 字的短句（过滤太短/太长）
4. 保持原始情感标签
5. 混合训练数据
"""

import jieba
from datasets import load_dataset
import pandas as pd


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


def process_dataset():
    """从HF数据集中提取短句子。"""
    
    print("Loading ChnSentiCorp...")
    ds_chn = load_dataset("lansinuote/ChnSentiCorp", split="train")
    
    print("Loading waimai_10k...")
    ds_wai = load_dataset("XiangPan/waimai_10k", split="train")
    
    short_sentences = []
    
    # 从 ChnSentiCorp 提取
    print("Extracting short sentences from ChnSentiCorp...")
    for item in ds_chn:
        text = item['text']
        label = item['label']
        shorts = extract_short_sentences_from_text(text, label)
        short_sentences.extend(shorts)
    
    # 从 waimai 提取
    print("Extracting short sentences from waimai_10k...")
    for item in ds_wai:
        text = item['text']
        label = item['label']  # 0=负面, 1=正面
        shorts = extract_short_sentences_from_text(text, label)
        short_sentences.extend(shorts)
    
    print(f"\nExtracted {len(short_sentences)} short sentences in total")
    
    # 统计
    pos_count = sum(1 for _, label in short_sentences if label == 1)
    neg_count = sum(1 for _, label in short_sentences if label == 0)
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    
    # 保存为CSV
    df = pd.DataFrame(short_sentences, columns=['text', 'label'])
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
