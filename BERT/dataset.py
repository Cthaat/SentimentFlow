"""数据集模块。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .text_processing import encode_text


class CsvStreamDataset(IterableDataset):
    """把文本样本流式编码成 BERT 可消费张量。"""

    def __init__(self, source, chunk_size: int, max_len: int, label_map: dict | None = None):
        super().__init__()
        self.source = source
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.label_map = label_map or {}

    @staticmethod
    def _parse_label(value):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _iter_fallback_csv_rows(self, csv_path: Path) -> Iterable[tuple[str, int]]:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                if not row:
                    continue

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if isinstance(self.source, (str, Path)):
            csv_path = Path(self.source)
            for chunk_index, chunk in enumerate(self._iter_csv_chunks(csv_path)):
                if chunk_index % num_workers != worker_id:
                    continue

                chunk = chunk.sample(frac=1).reset_index(drop=True)
                texts = chunk["text"].fillna("").astype(str).tolist()
                labels = chunk["label"].to_numpy(dtype="int64")

                for text, label in zip(texts, labels):
                    if self.label_map and label in self.label_map:
                        mapped_label = self.label_map[label]
                    else:
                        mapped_label = label

                    if mapped_label == -1:
                        continue

                    if mapped_label < 0 or mapped_label > 1:
                        raise ValueError(
                            f"Invalid mapped label value: {mapped_label} (original: {label}). "
                            "Expected binary labels [0, 1]."
                        )

                    encoded = encode_text(text, max_len=self.max_len)
                    yield {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "labels": torch.tensor(mapped_label, dtype=torch.long),
                    }
            return

        if hasattr(self.source, "__len__") and hasattr(self.source, "__getitem__"):
            total = len(self.source)
            for idx in range(worker_id, total, num_workers):
                row = self.source[idx]
                raw_text = row.get("text") or row.get("review") or row.get("content") or ""
                text = str(raw_text)
                label = int(row.get("label", 0))

                if self.label_map and label in self.label_map:
                    mapped_label = self.label_map[label]
                else:
                    mapped_label = label

                if mapped_label == -1:
                    continue

                if mapped_label < 0 or mapped_label > 1:
                    raise ValueError(
                        f"Invalid mapped label value: {mapped_label} (original: {label}, index: {idx}). "
                        "Expected binary labels [0, 1]."
                    )

                encoded = encode_text(text, max_len=self.max_len)
                yield {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": torch.tensor(mapped_label, dtype=torch.long),
                }
            return

        raise TypeError(f"Unsupported dataset source type: {type(self.source)!r}")
