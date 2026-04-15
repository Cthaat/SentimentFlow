import zlib
from typing import List


def tokenize(text: str) -> List[str]:
	"""字符级切分。"""
	return list(str(text).strip())


def encode_text(text: str, max_len: int, vocab_size: int) -> List[int]:
	"""将文本编码为定长 token id 序列。

	编码规则与根目录训练脚本保持一致：
	- 使用 crc32 哈希映射到 [1, vocab_size-1]
	- 0 作为 padding id
	"""
	ids = [zlib.crc32(token.encode("utf-8")) % (vocab_size - 1) + 1 for token in tokenize(text)]
	if len(ids) >= max_len:
		return ids[:max_len]
	return ids + [0] * (max_len - len(ids))

