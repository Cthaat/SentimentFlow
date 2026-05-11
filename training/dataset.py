"""数据集模块。

支持两类数据源：
1. CSV 文件路径：按 chunk 流式读取，节省内存
2. HuggingFace Dataset：按索引遍历

此外内置“容错 CSV 解析”逻辑，处理文本含逗号但未正确加引号的历史数据。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from sentiment_scale import LEGACY_BINARY_TO_SCORE, validate_sentiment_score

from .text_processing import encode_text


class CsvStreamDataset(IterableDataset):
    """把文本样本流式编码成模型可消费的张量。
    
    支持标签映射，用于兼容历史或第三方数据集。例如：
    - 旧二分类 0/1 -> 0/5
    - 评分数据 -> 0-5 情感分
    - 通过 label_map 参数指定映射规则
    """

    def __init__(
        self,
        source,
        chunk_size: int,
        max_len: int,
        vocab_size: int,
        label_map: dict | None = None,
        labels_are_normalized: bool = False,
    ):
        super().__init__()
        # 数据来源 可以是 CSV 文件路径，也可以是内存中的列表/数组（如 HuggingFace Dataset）。Dataset 需要支持按索引访问和长度查询。
        self.source = source
        # 数据处理参数 一次读取多少行进行处理
        self.chunk_size = chunk_size
        # 文本最大长度（超过部分截断，不足部分 padding）
        self.max_len = max_len
        # 词汇表大小（哈希映射的模数）
        self.vocab_size = vocab_size
        # 标签映射规则 {original_label: score}，score 应为 0-5；-1 表示跳过该样本。
        self.label_map = label_map or {}
        self.labels_are_normalized = labels_are_normalized
        self._csv_legacy_binary_cache: dict[str, bool] = {}

    @staticmethod
    def _parse_label(value):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _iter_fallback_csv_rows(self, csv_path: Path) -> Iterable[tuple[str, int]]:
        """容错读取 CSV。

        兼容如下非规范情况：
        - 文本里有逗号但没加引号
        - label 在首列或末列
        
        解决这种烂数据：
        1,这个文本有,逗号,但是没加引号
        """
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                if not row:
                    continue

                # 自动跳过常见表头。
                if row_index == 0:
                    header = [str(item).strip().lower() for item in row]
                    if "label" in header and "text" in header:
                        continue

                if len(row) == 1:
                    continue

                label_first = self._parse_label(row[0])
                if label_first is not None:
                    yield ",".join(row[1:]).strip(), label_first
                    continue

                label_last = self._parse_label(row[-1])
                if label_last is not None:
                    yield ",".join(row[:-1]).strip(), label_last

    def _iter_csv_chunks(self, csv_path: Path):
        """优先用 pandas 高效读取；失败后切换容错读取。"""
        try:
            chunk_reader = pd.read_csv(
                csv_path,
                usecols=["text", "label"],
                dtype={"text": "string", "label": "int8"},
                chunksize=self.chunk_size,
            )

            for chunk in chunk_reader:
                chunk = chunk.dropna(subset=["label"]).copy()
                chunk["label"] = chunk["label"].astype("int64")
                chunk["text"] = chunk["text"].fillna("").astype(str)
                yield chunk
            return
        except Exception as exc:
            print(f"Warning: standard CSV parse failed ({exc}); switching to tolerant parser.")

        buffer_texts = []
        buffer_labels = []
        for text, label in self._iter_fallback_csv_rows(csv_path):
            buffer_texts.append(text)
            buffer_labels.append(label)
            if len(buffer_texts) >= self.chunk_size:
                yield pd.DataFrame({"text": buffer_texts, "label": buffer_labels})
                buffer_texts = []
                buffer_labels = []

        if buffer_texts:
            yield pd.DataFrame({"text": buffer_texts, "label": buffer_labels})

    def _detect_csv_legacy_binary(self, csv_path: Path) -> bool:
        """整份 CSV 级别判断是否为旧 0/1 标签，避免按 chunk 误判。"""
        if self.labels_are_normalized or self.label_map:
            return False

        override = os.getenv("MIGRATE_LEGACY_BINARY_LABELS", "auto").strip().lower()
        if override in {"1", "true", "yes"}:
            return True
        if override in {"0", "false", "no"}:
            return False

        cache_key = str(csv_path.resolve())
        if cache_key in self._csv_legacy_binary_cache:
            return self._csv_legacy_binary_cache[cache_key]

        values: set[int] = set()
        chunk_reader = None
        try:
            chunk_reader = pd.read_csv(
                csv_path,
                usecols=["label"],
                dtype={"label": "int16"},
                chunksize=self.chunk_size,
            )
            for chunk in chunk_reader:
                for raw_label in chunk["label"].dropna().tolist():
                    label = int(raw_label)
                    values.add(label)
                    if label not in LEGACY_BINARY_TO_SCORE:
                        self._csv_legacy_binary_cache[cache_key] = False
                        return False
        except Exception:
            values.clear()
            for _, label in self._iter_fallback_csv_rows(csv_path):
                values.add(int(label))
                if int(label) not in LEGACY_BINARY_TO_SCORE:
                    self._csv_legacy_binary_cache[cache_key] = False
                    return False
        finally:
            if chunk_reader is not None:
                chunk_reader.close()

        result = bool(values) and values.issubset(set(LEGACY_BINARY_TO_SCORE))
        self._csv_legacy_binary_cache[cache_key] = result
        return result

    def __iter__(self):
        # 多进程环境下每个 worker 处理不同的 chunk，避免重复读取和竞争。
        # 用于 DataLoader 多进程
        worker_info = torch.utils.data.get_worker_info()
        
        # worker_id 从 0 开始，num_workers 是总 worker 数量。每个 worker 处理 chunk_index % num_workers == worker_id 的数据。
        worker_id = 0 if worker_info is None else worker_info.id
        # 单 worker 模式或无 worker_info 时，num_workers 设为 1，确保所有数据都被处理。
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # 支持两种数据源：CSV 文件路径或内存中的列表/数组。
        # CSV文件路径：按 chunk 流式读取，节省内存；内存数据集：按索引遍历。
        if isinstance(self.source, (str, Path)):
            csv_path = Path(self.source)
            legacy_binary_source = self._detect_csv_legacy_binary(csv_path)
            # 分块读取 CSV，确保每个 worker 处理不同的 chunk，避免重复读取和竞争。
            for chunk_index, chunk in enumerate(self._iter_csv_chunks(csv_path)):
                if chunk_index % num_workers != worker_id:
                    continue
                
                # 打乱 chunk 内部样本顺序，增加训练随机性。
                chunk = chunk.sample(frac=1).reset_index(drop=True)
                
                # 提取文本和标签列，进行必要的清洗和类型转换。
                texts = chunk["text"].fillna("").astype(str).tolist()
                labels = chunk["label"].to_numpy(dtype="int64")

                # 逐行处理文本和标签，应用标签映射规则，过滤掉不需要的样本，最后编码成张量形式。
                for text, label in zip(texts, labels):
                    # 应用标签映射（如果存在）
                    # 只映射在 label_map 中的标签，其他标签保持原值
                    # 如果映射后的标签是 -1，表示跳过该样本（常用于过滤掉某些类别）。
                    # 验证映射后的标签在有效范围内（0-5），否则抛出异常。
                    if self.label_map and label in self.label_map:
                        mapped_label = self.label_map[label]
                    elif legacy_binary_source and int(label) in LEGACY_BINARY_TO_SCORE:
                        mapped_label = LEGACY_BINARY_TO_SCORE[int(label)]
                    else:
                        mapped_label = label

                    # 跳过标签映射后被标记为 -1 的样本（通常用于过滤掉某些类别）
                    if mapped_label == -1:
                        continue
                    
                    mapped_label = validate_sentiment_score(mapped_label)
                        
                    # return 一条数据（但不会结束函数） yield 关键字使函数成为生成器，每次迭代返回一个样本，保持内存效率。
                    yield (
                        # 转换为tensor格式，文本编码成固定长度的整数序列，超过部分截断，不足部分 padding；使用哈希映射的方式将词汇映射到 [0, vocab_size) 的范围内。
                        # 例如 "hello world" -> [12345, 67890, 0, 0, 0, ...] (长度为 max_len)
                        torch.tensor(
                            # 编码文本成固定长度的整数序列，超过部分截断，不足部分 padding；使用哈希映射的方式将词汇映射到 [0, vocab_size) 的范围内。
                            # "hello world" -> [12345, 67890, 0, 0, 0, ...] (长度为 max_len)
                            encode_text(text, max_len=self.max_len, vocab_size=self.vocab_size),
                            dtype=torch.long,
                        ),
                        # 标签映射成 0-5 情感分，并转换为 tensor 格式。
                        torch.tensor(mapped_label, dtype=torch.long),
                    )
            return

        # 情况2：HuggingFace Dataset / list-like 数据集，支持按索引访问和长度查询。
        # 本质：dataset[i]
        if hasattr(self.source, "__len__") and hasattr(self.source, "__getitem__"):
            total = len(self.source)
            for idx in range(worker_id, total, num_workers):
                # 标签映射
                row = self.source[idx]
                # 某些拼接后的样本会保留字段名但值为 None，必须使用“值回退”而不是仅 key 回退。
                raw_text = row.get("text") or row.get("review") or row.get("content") or ""
                # 确保文本是字符串格式，避免 None 或其他类型导致编码失败。
                text = str(raw_text)
                label = int(row.get("label", 0))
                
                # 应用标签映射（如果存在）
                # 只映射在 label_map 中的标签，其他标签保持原值
                if self.label_map and label in self.label_map:
                    mapped_label = self.label_map[label]
                else:
                    mapped_label = label

                if mapped_label == -1:
                    continue
                
                mapped_label = validate_sentiment_score(mapped_label)
                
                yield (
                    torch.tensor(
                        encode_text(text, max_len=self.max_len, vocab_size=self.vocab_size),
                        dtype=torch.long,
                    ),
                    torch.tensor(mapped_label, dtype=torch.long),
                )
            return

        raise TypeError(f"Unsupported dataset source type: {type(self.source)!r}")
