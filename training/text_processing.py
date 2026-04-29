"""文本预处理模块。

这里采用"词级 + 哈希映射"策略：
- 使用 jieba 中文分词器
- 对短文本情感分类精度更高
- 能够捕捉词语语义关系
"""

from __future__ import annotations

import zlib

import jieba


_JIEBA_READY = False


def _ensure_jieba_ready() -> None:
    """在当前进程内仅初始化一次 jieba，避免重复加载词典。"""
    global _JIEBA_READY
    if _JIEBA_READY:
        return
    # 多进程 DataLoader 下禁用 jieba 内部并行，避免竞争和额外开销。
    jieba.disable_parallel()
    jieba.initialize()
    _JIEBA_READY = True


def tokenize(text: str) -> list[str]:
    """词级切分，使用jieba分词器。"""
    _ensure_jieba_ready()
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
