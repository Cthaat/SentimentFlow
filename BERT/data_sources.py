"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from typing import Dict, Tuple


DATASET_ALIASES = {
    "seamew/chnsenticorp_htl_all": "dirtycomputer/ChnSentiCorp_htl_all",
    "dmsc": "BerlinWang/DMSC",
    "jd_binary_sentiment": "dirtycomputer/JD_review",
    "jd_reviews": "dirtycomputer/JD_review",
    "nlpcc_sentiment": "ndiy/NLPCC14-SC",
    "hotel_reviews_sentiment": "dirtycomputer/ChnSentiCorp_htl_all",
    "simplified_weibo_sentiment": "dirtycomputer/weibo_senti_100k",
}


def _resolve_dataset_name(dataset_name: str) -> str:
    alias = DATASET_ALIASES.get(dataset_name.strip().lower())
    return alias or dataset_name


def _coerce_binary_label(raw_label, dataset_name: str, label_col: str) -> int:
    if dataset_name == "BerlinWang/DMSC" and label_col == "Star":
        try:
            star = int(raw_label)
        except (TypeError, ValueError):
            return -1
        if star <= 2:
            return 0
        if star >= 4:
            return 1
        return -1

    if dataset_name == "dirtycomputer/JD_review" and label_col == "rating":
        try:
            rating = int(float(raw_label))
        except (TypeError, ValueError):
            return -1
        if rating <= 2:
            return 0
        if rating >= 4:
            return 1
        return -1

    if isinstance(raw_label, bool):
        return int(raw_label)

    if isinstance(raw_label, (int, float)):
        value = int(raw_label)
        if value in (0, 1):
            return value
        if value == -1:
            return 0
        return -1

    text = str(raw_label).strip().lower()
    if text in {"1", "pos", "positive", "正面", "好评"}:
        return 1
    if text in {"0", "-1", "neg", "negative", "负面", "差评"}:
        return 0
    return -1


def _normalize_split_columns(split, dataset_name: str, preserve_label: bool = False):
    columns = set(split.column_names)

    if "text" in columns and "label" in columns:
        def cast_existing(row, ds_name=dataset_name, keep_raw=preserve_label):
            row_copy = dict(row)
            row_copy["text"] = str(row.get("text") or "")
            if keep_raw:
                try:
                    row_copy["label"] = int(row.get("label"))
                except (TypeError, ValueError):
                    row_copy["label"] = -1
            else:
                row_copy["label"] = _coerce_binary_label(row.get("label"), ds_name, "label")
            return row_copy

        split = split.map(cast_existing)
    else:
        text_candidates = ["text", "review", "content", "Comment", "comment", "sentence"]
        label_candidates = ["label", "sentiment", "Star", "score", "rating"]

        text_col = next((c for c in text_candidates if c in columns), None)
        label_col = next((c for c in label_candidates if c in columns), None)

        if text_col is None or label_col is None:
            raise ValueError(
                f"Dataset {dataset_name} has unsupported schema. "
                f"Missing text/label columns in {sorted(columns)}"
            )

        def normalize_row(row, ds_name=dataset_name, t_col=text_col, l_col=label_col, keep_raw=preserve_label):
            row_copy = dict(row)
            row_copy["text"] = str(row.get(t_col) or "")
            if keep_raw:
                try:
                    row_copy["label"] = int(row.get(l_col))
                except (TypeError, ValueError):
                    row_copy["label"] = -1
            else:
                row_copy["label"] = _coerce_binary_label(row.get(l_col), ds_name, l_col)
            return row_copy

        split = split.map(normalize_row)

    before = len(split)
    if preserve_label:
        split = split.filter(lambda row: int(row["label"]) >= 0)
    else:
        split = split.filter(lambda row: int(row["label"]) in (0, 1))
    after = len(split)
    if after < before:
        print(f"After schema/label cleanup for {dataset_name}: {after}/{before}")
    return split


def _force_balance_binary_split(split, dataset_name: str, split_name: str, seed: int = 42):
    neg_split = split.filter(lambda row: int(row["label"]) == 0)
    pos_split = split.filter(lambda row: int(row["label"]) == 1)

    neg_count = len(neg_split)
    pos_count = len(pos_split)
    min_count = min(neg_count, pos_count)

    if min_count == 0:
        print(
            f"Warning: Skip force-balance for {dataset_name} {split_name} "
            f"because one class is empty (neg={neg_count}, pos={pos_count})."
        )
        return split

    neg_split = neg_split.shuffle(seed=seed).select(range(min_count))
    pos_split = pos_split.shuffle(seed=seed).select(range(min_count))

    from datasets import concatenate_datasets

    balanced = concatenate_datasets([neg_split, pos_split]).shuffle(seed=seed)
    print(
        f"Force-balanced {dataset_name} {split_name}: "
        f"neg={min_count}, pos={min_count}, total={len(balanced)} "
        f"(from neg={neg_count}, pos={pos_count})"
    )
    return balanced


def _parse_label_map_from_env(raw: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for item in raw.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid label map item '{pair}'. Expected format like '0:1'."
            )
        src, dst = pair.split(":", 1)
        src_label = int(src.strip())
        dst_label = int(dst.strip())
        if dst_label not in (-1, 0, 1):
            raise ValueError(
                f"Invalid target label {dst_label} in '{pair}'. Expected -1, 0 or 1."
            )
        mapping[src_label] = dst_label
    if not mapping:
        raise ValueError("Parsed empty label map from environment variable.")
    return mapping


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    if dataset_name == "ttxy/online_shopping_10_cats":
        raw_map = os.getenv(
            "TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP",
            "1:0,2:0,3:0,4:0,5:-1,6:-1,7:1,8:1,9:1,10:1",
        )
        return _parse_label_map_from_env(raw_map)
    if dataset_name == "dirtycomputer/simplifyweibo_4_moods":
        raw_map = os.getenv("SIMPLIFYWEIBO_4_MOODS_LABEL_MAP", "0:1,1:0,2:-1,3:-1")
        return _parse_label_map_from_env(raw_map)
    return None


def build_train_split_and_val_split():
    from datasets import concatenate_datasets, load_dataset

    dataset_names = [
        item.strip()
        for item in os.getenv(
            "BERT_TRAIN_DATASETS",
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k",
        ).split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    label_map = None
    loaded_dataset_names = []
    val_ratio = float(os.getenv("TRAIN_VAL_RATIO", "0.1"))
    val_ratio = min(max(val_ratio, 0.01), 0.5)

    for raw_name in dataset_names:
        name = _resolve_dataset_name(raw_name)
        if name != raw_name:
            print(f"Dataset alias resolved: {raw_name} -> {name}")

        if name == "dirtycomputer/simplifyweibo_4_moods" and os.getenv("ALLOW_SIMPLIFYWEIBO_4_MOODS", "0") != "1":
            print(
                "Skip dataset dirtycomputer/simplifyweibo_4_moods: "
                "4-class mood dataset is disabled by default. "
                "Use dirtycomputer/weibo_senti_100k for binary sentiment, "
                "or set ALLOW_SIMPLIFYWEIBO_4_MOODS=1 to force-enable."
            )
            continue

        try:
            ds = load_dataset(name)
        except Exception as e:
            print(f"Warning: Failed to load dataset '{raw_name}' (resolved '{name}'): {e}")
            continue

        if "train" not in ds:
            raise ValueError(f"Dataset {name} has no train split.")

        current_label_map = get_label_map(name)
        if current_label_map:
            label_map = current_label_map
            print(f"Using label mapping for dataset: {name}")
            print(f"Label map: {current_label_map}")

        train_part = ds["train"]
        if "validation" in ds:
            val_part = ds["validation"]
        elif "test" in ds:
            val_part = ds["test"]
        else:
            if len(train_part) < 2:
                raise ValueError(f"Dataset {name} is too small to split train/val.")

            split_result = train_part.train_test_split(test_size=val_ratio, seed=42)
            train_part = split_result["train"]
            val_part = split_result["test"]
            print(
                f"Dataset {name} has no validation/test split; "
                f"auto-split train with TRAIN_VAL_RATIO={val_ratio:.2f}."
            )

        preserve_raw_label = current_label_map is not None
        train_part = _normalize_split_columns(train_part, name, preserve_label=preserve_raw_label)
        val_part = _normalize_split_columns(val_part, name, preserve_label=preserve_raw_label)

        if current_label_map:
            def apply_mapping(row, mapping=current_label_map):
                row_copy = dict(row)
                row_copy["label"] = mapping.get(int(row["label"]), -1)
                return row_copy

            train_part = train_part.map(apply_mapping)
            val_part = val_part.map(apply_mapping)

            train_before = len(train_part)
            val_before = len(val_part)
            train_part = train_part.filter(lambda row: int(row["label"]) in (0, 1))
            val_part = val_part.filter(lambda row: int(row["label"]) in (0, 1))
            print(
                f"After mapping filter for {name}: "
                f"train {len(train_part)}/{train_before}, val {len(val_part)}/{val_before}"
            )

        if name == "BerlinWang/DMSC":
            train_part = _force_balance_binary_split(train_part, name, "train", seed=42)
            val_part = _force_balance_binary_split(val_part, name, "val", seed=43)

        train_splits.append(train_part)
        val_splits.append(val_part)
        loaded_dataset_names.append(name)

    if not train_splits:
        raise ValueError(
            "No datasets were successfully loaded after preprocessing. "
            f"Requested datasets: {dataset_names}"
        )

    train_split = train_splits[0] if len(train_splits) == 1 else concatenate_datasets(train_splits)
    val_split = val_splits[0] if len(val_splits) == 1 else concatenate_datasets(val_splits)

    train_split = train_split.shuffle(seed=42)
    val_split = val_split.shuffle(seed=42)

    max_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    if max_samples > 0:
        max_samples = min(max_samples, len(train_split))
        train_split = train_split.select(range(max_samples))

    max_val_samples = int(os.getenv("TRAIN_MAX_VAL_SAMPLES", "0"))
    if max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_split))
        val_split = val_split.select(range(max_val_samples))

    if os.getenv("USE_EXTRACTED_SHORT_SENTENCES", "1") == "1":
        try:
            from pathlib import Path

            csv_path = Path(__file__).resolve().parent.parent / "extracted_short_sentences.csv"

            if csv_path.exists():
                print(f"Loading extracted short sentences from {csv_path}...")

                short_texts = []
                short_labels = []

                with open(csv_path, "r", encoding="utf-8") as f:
                    next(f)
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        last_comma = line.rfind(",")
                        if last_comma > 0:
                            text = line[:last_comma]
                            label_str = line[last_comma + 1 :]
                            try:
                                label = int(label_str)
                                if label in (0, 1):
                                    short_texts.append(text)
                                    short_labels.append(label)
                            except ValueError:
                                continue

                if short_texts:
                    from datasets import Dataset

                    short_sentences_ds = Dataset.from_dict({
                        "text": short_texts,
                        "label": short_labels,
                    })

                    max_short_samples = int(os.getenv("EXTRACTED_SHORT_SENTENCES_MAX", "0"))
                    if max_short_samples > 0 and len(short_sentences_ds) > max_short_samples:
                        short_sentences_ds = short_sentences_ds.select(range(max_short_samples))

                    short_ratio = float(os.getenv("SHORT_SENTENCES_RATIO", "0.3"))
                    short_samples_to_use = int(len(train_split) * short_ratio / (1 - short_ratio))
                    short_samples_to_use = min(short_samples_to_use, len(short_sentences_ds))

                    if short_samples_to_use > 0:
                        short_sentences_ds = short_sentences_ds.select(range(short_samples_to_use))
                        train_split = concatenate_datasets([train_split, short_sentences_ds])
                        neg_short = sum(1 for l in short_labels[:short_samples_to_use] if l == 0)
                        pos_short = sum(1 for l in short_labels[:short_samples_to_use] if l == 1)
                        print(f"Added {len(short_sentences_ds)} extracted short sentences (neg={neg_short}, pos={pos_short}).")
                        print(f"  New train size: {len(train_split)} (long:short = {100 * (1 - short_ratio):.0f}%:{100 * short_ratio:.0f}%)")
                else:
                    print("Warning: No valid short sentences found in CSV")
            else:
                print(f"Note: extracted_short_sentences.csv not found at {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to load short sentences: {e}")

    if os.getenv("USE_SYNTHETIC_DATA", "0") == "1":
        try:
            from .generate_synthetic_data import generate_short_sentence_dataset
            from datasets import concatenate_datasets

            synthetic_size = int(os.getenv("SYNTHETIC_DATA_SIZE", "5000"))
            print(f"Generating {synthetic_size} synthetic short sentences...")
            synthetic_ds = generate_short_sentence_dataset(size=synthetic_size)
            train_split = concatenate_datasets([train_split, synthetic_ds])
            print(f"Added {len(synthetic_ds)} synthetic samples. New train size: {len(train_split)}")
        except ImportError:
            print("Warning: Synthetic data generation not available (generate_synthetic_data module not found)")

    return loaded_dataset_names, train_split, val_split, None


def get_label_distribution(split, label_map: dict | None = None) -> Tuple[int, int]:
    labels = split["label"]
    neg = sum(1 for x in labels if int(x) == 0)
    pos = sum(1 for x in labels if int(x) == 1)
    return neg, pos
