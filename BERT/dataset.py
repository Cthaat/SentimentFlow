"""数据集模块。"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from sentiment_scale import LEGACY_BINARY_TO_SCORE, validate_sentiment_score


class CsvStreamDataset(IterableDataset):
    """流式加载文本样本，仅返回原始数据（不编码）。
    
    关键设计：Dataset 返回 (text, label)，让 DataLoader 的 collate_fn 负责批量 tokenize。
    这样可以一次性处理整个 batch 的数据，避免逐条 tokenizer 调用的性能灾难。
    """

    def __init__(
        self,
        source,
        chunk_size: int,
        label_map: dict | None = None,
        labels_are_normalized: bool = False,
    ):
        """初始化数据集。

        Args:
            source: 数据来源，可以是 CSV 文件路径（str/Path）或支持下标访问的数据集对象。
            chunk_size: 每次从 CSV 分块读取的行数，用于控制内存占用。
            label_map: 可选的标签映射字典，将原始标签整数映射到目标评分（0-5/-1）。
                       -1 表示跳过该样本；为 None 时不做映射。
        """
        super().__init__()
        self.source = source
        self.chunk_size = chunk_size
        self.label_map = label_map or {}
        self.labels_are_normalized = labels_are_normalized
        self._csv_legacy_binary_cache: dict[str, bool] = {}

    @staticmethod
    def _parse_label(value):
        """尝试将字段值解析为整数标签。

        Args:
            value: 待解析的原始字段值。

        Returns:
            解析成功时返回整数，解析失败时返回 None。
        """
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _iter_fallback_csv_rows(self, csv_path: Path) -> Iterable[tuple[str, int]]:
        """使用宽容解析器逐行读取 CSV，适用于非标准格式的文件。

        标准 pandas 解析失败后调用此方法作为备选方案。
        支持"标签在首列"和"标签在末列"两种布局。
        自动跳过包含标准 text/label 表头的首行。

        Args:
            csv_path: CSV 文件路径。

        Yields:
            (text, label) 二元组，text 为文本字符串，label 为整数标签。
        """
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                # 跳过空行
                if not row:
                    continue

                # 若首行是标准 text/label 表头则跳过
                if row_index == 0:
                    header = [str(item).strip().lower() for item in row]
                    if "label" in header and "text" in header:
                        continue

                # 只有一列时无法区分文本和标签，跳过
                if len(row) == 1:
                    continue

                # 优先尝试首列为标签的格式
                label_first = self._parse_label(row[0])
                if label_first is not None:
                    yield ",".join(row[1:]).strip(), label_first
                    continue

                # 再尝试末列为标签的格式
                label_last = self._parse_label(row[-1])
                if label_last is not None:
                    yield ",".join(row[:-1]).strip(), label_last

    def _iter_csv_chunks(self, csv_path: Path):
        """将 CSV 文件按 chunk_size 分块读取，逐块返回 DataFrame。

        首先尝试使用 pandas 高效读取（要求存在 text/label 列）；
        若解析失败则回退到宽容的逐行解析器，并将结果缓冲为等大小的 DataFrame。

        Args:
            csv_path: CSV 文件路径。

        Yields:
            包含 "text"（str）和 "label"（int64）两列的 pandas DataFrame。
        """
        try:
            # 标准路径：使用 pandas 分块读取，仅加载 text/label 两列以节省内存
            chunk_reader = pd.read_csv(
                csv_path,
                usecols=["text", "label"],
                dtype={"text": "string", "label": "int8"},
                chunksize=self.chunk_size,
            )

            for chunk in chunk_reader:
                # 删除 label 缺失的行，并统一类型
                chunk = chunk.dropna(subset=["label"]).copy()
                chunk["label"] = chunk["label"].astype("int64")
                chunk["text"] = chunk["text"].fillna("").astype(str)
                yield chunk
            return
        except Exception as exc:
            print(f"Warning: standard CSV parse failed ({exc}); switching to tolerant parser.")

        # 回退路径：使用宽容解析器，将结果按 chunk_size 缓冲后批量返回
        buffer_texts = []
        buffer_labels = []
        for text, label in self._iter_fallback_csv_rows(csv_path):
            buffer_texts.append(text)
            buffer_labels.append(label)
            if len(buffer_texts) >= self.chunk_size:
                yield pd.DataFrame({"text": buffer_texts, "label": buffer_labels})
                buffer_texts = []
                buffer_labels = []

        # 将最后不足一个 chunk 的剩余数据也一并返回
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
        """迭代数据集，按 worker 分片后逐样本 yield (text, label)。

        支持两种数据源类型：
        - CSV 文件路径：分块读取，每块内部随机打乱后按 worker 分片。
        - 支持 __len__ 和 __getitem__ 的对象（如 HuggingFace Dataset）：
          按 worker id 步长采样，兼容多进程 DataLoader。

        标签映射规则：
        - 若 label_map 中对应值为 -1，则跳过该样本。
        - 映射后的标签必须为 0-5，否则抛出 ValueError。

        Yields:
            (text, mapped_label) 二元组。
        """
        # 获取当前 DataLoader worker 信息，用于多进程数据分片
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if isinstance(self.source, (str, Path)):
            # ---- CSV 文件数据源 ----
            csv_path = Path(self.source)
            legacy_binary_source = self._detect_csv_legacy_binary(csv_path)
            for chunk_index, chunk in enumerate(self._iter_csv_chunks(csv_path)):
                # 按 worker id 对 chunk 编号取模，实现多 worker 分片
                if chunk_index % num_workers != worker_id:
                    continue

                # 在 chunk 内随机打乱，提升训练多样性
                chunk = chunk.sample(frac=1).reset_index(drop=True)
                texts = chunk["text"].fillna("").astype(str).tolist()
                labels = chunk["label"].to_numpy(dtype="int64")

                for text, label in zip(texts, labels):
                    # 应用标签映射（若存在）
                    if self.label_map and label in self.label_map:
                        mapped_label = self.label_map[label]
                    elif legacy_binary_source and int(label) in LEGACY_BINARY_TO_SCORE:
                        mapped_label = LEGACY_BINARY_TO_SCORE[int(label)]
                    else:
                        mapped_label = label

                    # -1 表示该样本应被跳过（中性/无效标签）
                    if mapped_label == -1:
                        continue

                    mapped_label = validate_sentiment_score(mapped_label)

                    yield {
                        "text": text,
                        "label": mapped_label,
                        "soft_labels": None,
                        "sample_weight": 1.0,
                        "label_source": "csv",
                    }

        elif hasattr(self.source, "__len__") and hasattr(self.source, "__getitem__"):
            # ---- 类 HuggingFace Dataset 数据源 ----
            total = len(self.source)
            # 通过步长 num_workers 实现多 worker 均匀分片
            for idx in range(worker_id, total, num_workers):
                row = self.source[idx]
                # 按优先级尝试多个常见文本列名
                raw_text = row.get("text") or row.get("review") or row.get("content") or ""
                text = str(raw_text)
                label = int(row.get("label", 0))

                # 应用标签映射（若存在）
                if self.label_map and label in self.label_map:
                    mapped_label = self.label_map[label]
                else:
                    mapped_label = label

                # -1 表示该样本应被跳过
                if mapped_label == -1:
                    continue

                mapped_label = validate_sentiment_score(mapped_label)

                sample_weight = row.get("sample_weight", 1.0)
                label_source = row.get("label_source") or row.get("_sf_label_source") or "real"
                soft_labels = None
                if str(label_source) in {"pseudo", "interpolated"}:
                    soft_labels = row.get("soft_labels") or row.get("probabilities")
                yield {
                    "text": text,
                    "label": mapped_label,
                    "soft_labels": soft_labels,
                    "sample_weight": float(sample_weight),
                    "label_source": str(label_source),
                }

        else:
            raise TypeError(f"Unsupported dataset source type: {type(self.source)!r}")
