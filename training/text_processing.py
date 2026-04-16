"""文本预处理模块。

这里采用“字符级 + 哈希映射”策略：
- 无需预先构建完整词表
- 对短文本情感分类足够实用
- 训练启动快、内存占用低
"""

from __future__ import annotations

import zlib


def tokenize(text: str) -> list[str]:
    """字符级切分，避免额外分词器依赖。"""
    return list(str(text).strip())


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
