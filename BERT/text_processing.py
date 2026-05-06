"""BERT 文本处理工具。"""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

from .config import BERT_MODEL_NAME, get_model_name


@lru_cache(maxsize=2)
def get_tokenizer(model_name: str = BERT_MODEL_NAME):
    """延迟加载并缓存 tokenizer。"""
    return AutoTokenizer.from_pretrained(model_name)


def encode_text(text: str, max_len: int):
    """把文本编码为 BERT 所需输入。"""
    tokenizer = get_tokenizer(get_model_name())
    encoded = tokenizer(
        str(text or ""),
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }
