"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from sentiment_scale import (
    LEGACY_BINARY_DATASETS,
    LEGACY_BINARY_TO_SCORE,
    NUM_SENTIMENT_CLASSES,
    SENTIMENT_SCORES,
    choose_score_from_binary_teacher_probabilities,
    coerce_sentiment_score,
    parse_label_map,
    validate_sentiment_score,
)


DATASET_ALIASES = {
    # 用户常写法 -> 可访问的数据集 ID
    "seamew/chnsenticorp_htl_all": "dirtycomputer/ChnSentiCorp_htl_all",
    "dmsc": "BerlinWang/DMSC",
    # 新增：常见别名
    "jd_binary_sentiment": "dirtycomputer/JD_review",
    "jd_reviews": "dirtycomputer/JD_review",
    "nlpcc_sentiment": "ndiy/NLPCC14-SC",
    "hotel_reviews_sentiment": "dirtycomputer/ChnSentiCorp_htl_all",
    "simplified_weibo_sentiment": "dirtycomputer/weibo_senti_100k",
}

META_COLUMNS = {
    "text",
    "label",
    "_sf_is_legacy_binary",
    "_sf_binary_label",
    "_sf_label_source",
    "_sf_teacher_confidence",
}


def _resolve_dataset_name(dataset_name: str) -> str:
    """将用户输入的数据集名解析为可访问的数据集 ID。"""
    alias = DATASET_ALIASES.get(dataset_name.strip().lower())
    return alias or dataset_name


def _should_migrate_legacy_binary(split, label_col: str, dataset_name: str) -> bool:
    """自动识别旧 0/1 数据并迁移为 0/5，避免破坏真实 0-5 数据。"""
    override = os.getenv("MIGRATE_LEGACY_BINARY_LABELS", "auto").strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    if override in {"0", "false", "no"}:
        return False
    if dataset_name in LEGACY_BINARY_DATASETS:
        return True

    try:
        raw_values = split.unique(label_col)
    except Exception:
        return False

    values: set[int] = set()
    for raw_value in raw_values:
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            return False
        if not number.is_integer():
            return False
        values.add(int(number))
    return bool(values) and values.issubset({0, 1})


def _parse_binary_label(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    if not number.is_integer():
        return -1
    int_value = int(number)
    return int_value if int_value in (0, 1) else -1


def _project_standard_columns(split):
    keep_columns = [column for column in split.column_names if column in META_COLUMNS]
    remove_columns = [column for column in split.column_names if column not in keep_columns]
    if remove_columns:
        split = split.remove_columns(remove_columns)
    return split


def _normalize_short_sentence_labels(raw_rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """兼容历史二分类短句 CSV 与新的 0-5 短句 CSV。"""
    raw_values = {label for _, label in raw_rows}
    legacy_binary = bool(raw_values) and raw_values.issubset(set(LEGACY_BINARY_TO_SCORE))
    normalized_rows: list[tuple[str, int]] = []
    for text, label in raw_rows:
        try:
            score = LEGACY_BINARY_TO_SCORE[label] if legacy_binary else validate_sentiment_score(label)
        except ValueError:
            continue
        normalized_rows.append((text, score))
    return normalized_rows


def _normalize_split_columns(split, dataset_name: str, preserve_label: bool = False):
    """统一数据集字段为 text/label，并按需做标签清洗。"""
    columns = set(split.column_names)

    # 已经是标准字段时，按模式处理标签。
    if "text" in columns and "label" in columns:
        legacy_binary = _should_migrate_legacy_binary(split, "label", dataset_name)

        def cast_existing(row, ds_name=dataset_name, keep_raw=preserve_label):
            raw_label = row.get("label")
            binary_label = _parse_binary_label(raw_label) if legacy_binary else -1
            row_copy = dict(row)
            row_copy["text"] = str(row.get("text") or "")
            row_copy["_sf_is_legacy_binary"] = legacy_binary and binary_label in (0, 1)
            row_copy["_sf_binary_label"] = binary_label
            row_copy["_sf_teacher_confidence"] = -1.0
            if keep_raw:
                try:
                    row_copy["label"] = int(raw_label)
                except (TypeError, ValueError):
                    row_copy["label"] = -1
                row_copy["_sf_label_source"] = "raw_preserved"
            else:
                row_copy["label"] = coerce_sentiment_score(
                    raw_label,
                    dataset_name=ds_name,
                    label_col="label",
                    legacy_binary=legacy_binary,
                )
                row_copy["_sf_label_source"] = (
                    "weak_binary_endpoint" if row_copy["_sf_is_legacy_binary"] else "normalized"
                )
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

        legacy_binary = _should_migrate_legacy_binary(split, label_col, dataset_name)

        def normalize_row(row, ds_name=dataset_name, t_col=text_col, l_col=label_col, keep_raw=preserve_label):
            raw_label = row.get(l_col)
            binary_label = _parse_binary_label(raw_label) if legacy_binary else -1
            row_copy = dict(row)
            row_copy["text"] = str(row.get(t_col) or "")
            row_copy["_sf_is_legacy_binary"] = legacy_binary and binary_label in (0, 1)
            row_copy["_sf_binary_label"] = binary_label
            row_copy["_sf_teacher_confidence"] = -1.0
            if keep_raw:
                try:
                    row_copy["label"] = int(raw_label)
                except (TypeError, ValueError):
                    row_copy["label"] = -1
                row_copy["_sf_label_source"] = "raw_preserved"
            else:
                row_copy["label"] = coerce_sentiment_score(
                    raw_label,
                    dataset_name=ds_name,
                    label_col=l_col,
                    legacy_binary=legacy_binary,
                )
                row_copy["_sf_label_source"] = (
                    "weak_binary_endpoint" if row_copy["_sf_is_legacy_binary"] else "normalized"
                )
            return row_copy

        split = split.map(normalize_row)

    before = len(split)
    if preserve_label:
        split = split.filter(lambda row: int(row["label"]) >= 0)
    else:
        split = split.filter(lambda row: int(row["label"]) in SENTIMENT_SCORES)
    after = len(split)
    if after < before:
        print(f"After schema/label cleanup for {dataset_name}: {after}/{before}")
    return _project_standard_columns(split)


def _semi_supervised_enabled() -> bool:
    raw = os.getenv("SEMI_SUPERVISED_01_TO_05", "auto").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_teacher_path() -> str:
    return (
        os.getenv("PSEUDO_LABEL_TEACHER_PATH", "").strip()
        or os.getenv("LSTM_PSEUDO_LABEL_TEACHER_PATH", "").strip()
        or os.getenv("MODEL_PATH", "").strip()
    )


def _has_legacy_binary_rows(split) -> bool:
    return (
        "_sf_is_legacy_binary" in split.column_names
        and any(bool(value) for value in split["_sf_is_legacy_binary"])
    )


def _maybe_apply_semi_supervised_binary_refinement(split, dataset_name: str, split_name: str):
    """Use an existing 6-class LSTM teacher to split weak 0/1 labels into 0-5."""
    if not _semi_supervised_enabled() or not _has_legacy_binary_rows(split):
        return split

    teacher_path = _get_teacher_path()
    if not teacher_path:
        print(
            f"Semi-supervised 0/1->0-5 skipped for {dataset_name} {split_name}: "
            "PSEUDO_LABEL_TEACHER_PATH is not set."
        )
        return split

    checkpoint_path = Path(teacher_path)
    if not checkpoint_path.exists():
        print(
            f"Semi-supervised 0/1->0-5 skipped for {dataset_name} {split_name}: "
            f"teacher checkpoint not found: {checkpoint_path}"
        )
        return split

    try:
        import torch

        from .checkpoint import load_checkpoint
        from .config import MAX_LEN, VOCAB_SIZE
        from .text_processing import encode_text

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        teacher_max_len = int(checkpoint.get("max_len", MAX_LEN)) if isinstance(checkpoint, dict) else MAX_LEN
        teacher_vocab_size = int(checkpoint.get("vocab_size", VOCAB_SIZE)) if isinstance(checkpoint, dict) else VOCAB_SIZE
        teacher = load_checkpoint(str(checkpoint_path), device=device, default_vocab_size=VOCAB_SIZE)
        if teacher is None:
            return split
        if int(getattr(teacher.fc, "out_features", 0)) != NUM_SENTIMENT_CLASSES:
            print(
                f"Semi-supervised 0/1->0-5 skipped for {dataset_name} {split_name}: "
                "teacher is not a 6-class checkpoint."
            )
            return split

        min_confidence = float(os.getenv("PSEUDO_LABEL_MIN_CONFIDENCE", "0.75"))
        fallback_to_endpoint = os.getenv("PSEUDO_LABEL_FALLBACK_TO_ENDPOINT", "0") == "1"
        batch_size = max(1, int(os.getenv("PSEUDO_LABEL_BATCH_SIZE", "128")))
        teacher.eval()

        def refine_batch(batch):
            labels = [int(value) for value in batch["label"]]
            sources = list(batch["_sf_label_source"])
            confidences = [float(value) for value in batch["_sf_teacher_confidence"]]

            legacy_indices = [
                index for index, flag in enumerate(batch["_sf_is_legacy_binary"])
                if bool(flag)
            ]
            if legacy_indices:
                encoded = [
                    encode_text(
                        str(batch["text"][index]),
                        max_len=teacher_max_len,
                        vocab_size=teacher_vocab_size,
                    )
                    for index in legacy_indices
                ]
                tensor = torch.tensor(encoded, dtype=torch.long, device=device)
                with torch.inference_mode():
                    probabilities = torch.softmax(teacher(tensor), dim=1).detach().cpu().tolist()

                for offset, row_index in enumerate(legacy_indices):
                    choice = choose_score_from_binary_teacher_probabilities(
                        probabilities[offset],
                        int(batch["_sf_binary_label"][row_index]),
                        min_confidence=min_confidence,
                        fallback_to_endpoint=fallback_to_endpoint,
                    )
                    labels[row_index] = int(choice["score"])
                    sources[row_index] = str(choice["source"])
                    confidences[row_index] = round(float(choice["confidence"]), 6)

            batch["label"] = labels
            batch["_sf_label_source"] = sources
            batch["_sf_teacher_confidence"] = confidences
            return batch

        before = len(split)
        split = split.map(refine_batch, batched=True, batch_size=batch_size)
        split = split.filter(lambda row: int(row["label"]) in SENTIMENT_SCORES)
        accepted = sum(1 for source in split["_sf_label_source"] if source == "pseudo_teacher")
        fallback = sum(1 for source in split["_sf_label_source"] if source == "weak_binary_endpoint")
        print(
            f"Semi-supervised 0/1->0-5 for {dataset_name} {split_name}: "
            f"rows={len(split)}/{before}, pseudo={accepted}, fallback={fallback}, "
            f"teacher={checkpoint_path}"
        )
        return split
    except Exception as exc:
        print(
            f"Warning: semi-supervised 0/1->0-5 failed for {dataset_name} {split_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return split


def _parse_label_map_from_env(raw: str) -> Dict[int, int]:
    """解析标签映射配置。

    形如 "0:1,1:0,2:-1,3:-1"：
    - 0-5 表示映射到情感评分
    - -1 表示丢弃该原始标签样本
    """
    return parse_label_map(raw)


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """获取数据集的标签映射规则。
    
    用于兼容多类数据集，将其转换为 0-5 情感分。
    返回 None 表示无需额外映射（已能通过通用规则映射到 0-5）。
    """
    if dataset_name == "ttxy/online_shopping_10_cats":
        # 10类评分转 0-5 情感分。
        # 可通过环境变量覆盖，例如：TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP=1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4,10:5
        raw_map = os.getenv(
            "TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP",
            "1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4,10:5",
        )
        return _parse_label_map_from_env(raw_map)
    if dataset_name == "dirtycomputer/simplifyweibo_4_moods":
        # 4类情绪转评分。默认规则：正面->5，负面->0，其余丢弃。
        # 可通过环境变量覆盖，例如：SIMPLIFYWEIBO_4_MOODS_LABEL_MAP=0:5,1:0,2:3,3:-1
        raw_map = os.getenv("SIMPLIFYWEIBO_4_MOODS_LABEL_MAP", "0:5,1:0,2:-1,3:-1")
        return _parse_label_map_from_env(raw_map)
    return None  # 其他数据集无需映射


def build_train_split_and_val_split():
    """加载并拼接多个训练/验证数据集。

    返回：(dataset_names, train_split, val_split, label_map)
    - label_map: 标签映射规则（用于多类数据集转 0-5 评分），None 表示无需映射
    
    行为说明：
    - 优先读取 validation split
    - 没有 validation 时退化到 test split
    - 两者都没有时，从 train 自动切分验证集
    - 如果数据集需要标签映射，在拼接前应用映射
    """
    from datasets import concatenate_datasets, load_dataset

    dataset_names = [
        item.strip()
        # TRAIN_DATASETS 环境变量示例：lansinuote/ChnSentiCorp,XiangPan/waimai_10k
        # 改为仅使用两个高质量数据集，避免多数据集混合导致的类别不平衡
        for item in os.getenv(
            "TRAIN_DATASETS",
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/JD_review,BerlinWang/DMSC"
        ).split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    label_map = None  # 标记是否有任何数据集需要映射
    loaded_dataset_names = []
    # TRAIN_VAL_RATIO 环境变量示例：0.1（表示 10% 的训练数据用于验证）
    val_ratio = float(os.getenv("TRAIN_VAL_RATIO", "0.1"))
    val_ratio = min(max(val_ratio, 0.01), 0.5)

    for raw_name in dataset_names:
        name = _resolve_dataset_name(raw_name)
        if name != raw_name:
            print(f"Dataset alias resolved: {raw_name} -> {name}")

        # 4 情绪微博数据默认禁用，避免不明确情绪类污染评分训练。
        if name == "dirtycomputer/simplifyweibo_4_moods" and os.getenv("ALLOW_SIMPLIFYWEIBO_4_MOODS", "0") != "1":
            print(
                "Skip dataset dirtycomputer/simplifyweibo_4_moods: "
                "4-class mood dataset is disabled by default. "
                "Use dirtycomputer/weibo_senti_100k for legacy binary sentiment, "
                "or set ALLOW_SIMPLIFYWEIBO_4_MOODS=1 to force-enable."
            )
            continue

        dataset_path = Path(raw_name).expanduser()
        try:
            if dataset_path.exists() and dataset_path.suffix.lower() == ".csv":
                name = str(dataset_path.resolve())
                ds = load_dataset("csv", data_files={"train": name})
            else:
                ds = load_dataset(name)
        except Exception as e:
            print(f"Warning: Failed to load dataset '{raw_name}' (resolved '{name}'): {e}")
            continue

        if "train" not in ds:
            raise ValueError(f"Dataset {name} has no train split.")

        # 检查是否需要标签映射
        current_label_map = get_label_map(name)
        if current_label_map:
            label_map = current_label_map  # 记录映射规则
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

        # 新增：字段标准化与标签清洗（统一为 text/label 且 label in {0,1}）。
        preserve_raw_label = current_label_map is not None
        train_part = _normalize_split_columns(train_part, name, preserve_label=preserve_raw_label)
        val_part = _normalize_split_columns(val_part, name, preserve_label=preserve_raw_label)

        # 如果需要标签映射，在拼接前应用。
        # 当映射目标为 -1 时，表示丢弃该标签样本（如噪声类）。
        if current_label_map:
            def apply_mapping(row, mapping=current_label_map):
                row_copy = dict(row)
                row_copy["label"] = mapping.get(int(row["label"]), -1)
                row_copy["_sf_label_source"] = "label_map" if row_copy["label"] != -1 else "mapped_skip"
                return row_copy
            
            train_part = train_part.map(apply_mapping)
            val_part = val_part.map(apply_mapping)

            train_before = len(train_part)
            val_before = len(val_part)
            train_part = train_part.filter(lambda row: int(row["label"]) in SENTIMENT_SCORES)
            val_part = val_part.filter(lambda row: int(row["label"]) in SENTIMENT_SCORES)
            print(
                f"After mapping filter for {name}: "
                f"train {len(train_part)}/{train_before}, val {len(val_part)}/{val_before}"
            )

        train_part = _maybe_apply_semi_supervised_binary_refinement(train_part, name, "train")
        if os.getenv("PSEUDO_LABEL_VALIDATION_SPLIT", "0") == "1":
            val_part = _maybe_apply_semi_supervised_binary_refinement(val_part, name, "val")

        print(
            f"Dataset ready: {name} "
            f"train_score_counts={get_label_distribution(train_part)}, "
            f"val_score_counts={get_label_distribution(val_part)}"
        )

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

    # TRAIN_MAX_SAMPLES 环境变量示例：10000（表示最多使用 10,000 个训练样本）
    max_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    if max_samples > 0:
        max_samples = min(max_samples, len(train_split))
        train_split = train_split.select(range(max_samples))

    # TRAIN_MAX_VAL_SAMPLES 环境变量示例：1000（表示最多使用 1,000 个验证样本）
    max_val_samples = int(os.getenv("TRAIN_MAX_VAL_SAMPLES", "0"))
    if max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_split))
        val_split = val_split.select(range(max_val_samples))

    # 可选：添加提取的真实短句数据来补充分布差异
    # USE_EXTRACTED_SHORT_SENTENCES 环境变量：1 表示启用（默认），0 表示禁用
    if os.getenv("USE_EXTRACTED_SHORT_SENTENCES", "1") == "1":
        try:
            csv_path = Path(__file__).resolve().parent / "extracted_short_sentences.csv"
            
            if csv_path.exists():
                print(f"Loading extracted short sentences from {csv_path}...")
                
                # 直接读取CSV，避免pandas转换引起的类型问题
                raw_short_rows: list[tuple[str, int]] = []
                
                with open(csv_path, 'r', encoding='utf-8') as f:
                    # 跳过header (text,label)
                    next(f)
                    for line in f:
                        line = line.rstrip('\n')
                        if not line:
                            continue
                        # 格式：text,label，找最后一个逗号来分割
                        last_comma = line.rfind(',')
                        if last_comma > 0:
                            text = line[:last_comma]
                            label_str = line[last_comma+1:]
                            try:
                                label = int(label_str)
                                raw_short_rows.append((text, label))
                            except ValueError:
                                continue
                
                normalized_rows = _normalize_short_sentence_labels(raw_short_rows)
                short_texts = [text for text, _ in normalized_rows]
                short_labels = [label for _, label in normalized_rows]
                if short_texts:
                    from datasets import Dataset
                    short_sentences_ds = Dataset.from_dict({
                        'text': short_texts,
                        'label': short_labels
                    })
                    
                    # 可选限制数量
                    max_short_samples = int(os.getenv("EXTRACTED_SHORT_SENTENCES_MAX", "0"))
                    if max_short_samples > 0 and len(short_sentences_ds) > max_short_samples:
                        short_sentences_ds = short_sentences_ds.select(range(max_short_samples))
                    
                    # 混合：长句 + 短句
                    short_ratio = float(os.getenv("SHORT_SENTENCES_RATIO", "0.3"))  # 默认30%短句
                    short_samples_to_use = int(len(train_split) * short_ratio / (1 - short_ratio))
                    short_samples_to_use = min(short_samples_to_use, len(short_sentences_ds))
                    
                    if short_samples_to_use > 0:
                        short_sentences_ds = short_sentences_ds.select(range(short_samples_to_use))
                        train_split = concatenate_datasets([train_split, short_sentences_ds])
                        short_counts = get_label_distribution(short_sentences_ds)
                        print(f"Added {len(short_sentences_ds)} extracted short sentences (score_counts={short_counts}).")
                        print(f"  New train size: {len(train_split)} (long:short = {100*(1-short_ratio):.0f}%:{100*short_ratio:.0f}%)")
                else:
                    print("Warning: No valid short sentences found in CSV")
            else:
                print(f"Note: extracted_short_sentences.csv not found at {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to load short sentences: {e}")

    # 可选：添加合成短句数据来补充分布差异（不推荐，已证明无效）
    # USE_SYNTHETIC_DATA 环境变量：1 表示启用，0 表示禁用
    if os.getenv("USE_SYNTHETIC_DATA", "0") == "1":
        try:
            from .generate_synthetic_data import generate_short_sentence_dataset
            
            synthetic_size = int(os.getenv("SYNTHETIC_DATA_SIZE", "5000"))
            print(f"Generating {synthetic_size} synthetic short sentences...")
            synthetic_ds = generate_short_sentence_dataset(size=synthetic_size)
            
            # 将合成数据追加到训练集
            train_split = concatenate_datasets([train_split, synthetic_ds])
            print(f"Added {len(synthetic_ds)} synthetic samples. New train size: {len(train_split)}")
        except ImportError:
            print("Warning: Synthetic data generation not available (generate_synthetic_data module not found)")

    # 返回 None 作为 label_map，因为映射已在拼接前应用
    return loaded_dataset_names, train_split, val_split, None


def get_label_distribution(split, label_map: dict | None = None) -> list[int]:
    """统计 0-5 情感评分标签分布。
    
    注意：如果标签映射在 build_train_split_and_val_split() 中应用过，label_map 参数会是 None。
    此函数仅统计最终的 0-5 标签。
    """
    counts = [0 for _ in range(NUM_SENTIMENT_CLASSES)]
    labels = split["label"]
    for label in labels:
        score = validate_sentiment_score(int(label))
        counts[score] += 1
    return counts
