"""文本预处理模块。

这里采用"词级 + 哈希映射"策略：
- 使用 jieba 中文分词器
- 对短文本情感分类精度更高
- 能够捕捉词语语义关系
"""

from __future__ import annotations

import zlib

import jieba


def tokenize(text: str) -> list[str]:
    """词级切分，使用jieba分词器。"""
    return list(jieba.cut(text.strip()))


def encode_text(text: str, max_len: int, vocab_size: int) -> list[int]:
    """把文本编码成固定长度整数序列。

    说明：
    - 0 作为 padding id
    - 其他字符通过 crc32 哈希映射到 [1, vocab_size-1]
    """
    ids = [zlib.crc32(token.encode("utf-8")) % (vocab_size - 1) + 1 for token in tokenize(text)]

    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))
