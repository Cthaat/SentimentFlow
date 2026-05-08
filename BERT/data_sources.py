"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from sentiment_scale import (
    LEGACY_BINARY_DATASETS,
    NUM_SENTIMENT_CLASSES,
    SENTIMENT_SCORES,
    choose_score_from_binary_teacher_probabilities,
    coerce_sentiment_score,
    parse_label_map,
    validate_sentiment_score,
)


# 数据集别名映射表：将简短/易记的名称映射到 HuggingFace Hub 上的实际 repo 路径
DATASET_ALIASES = {
    "seamew/chnsenticorp_htl_all": "dirtycomputer/ChnSentiCorp_htl_all",
    "dmsc": "BerlinWang/DMSC",
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
    """将数据集别名解析为 HuggingFace Hub 的标准名称。

    若名称不在别名表中，则原样返回（即假定用户直接传入了标准名称）。

    Args:
        dataset_name: 原始数据集名称（可以是别名或标准名称）。

    Returns:
        解析后的标准数据集名称。
    """
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


def _normalize_split_columns(split, dataset_name: str, preserve_label: bool = False):
    """将数据集的列规范化为统一的 text/label 格式。

    若数据集已有 text 和 label 列，则仅对类型做强制转换；
    否则，从候选列名列表中自动匹配文本列和标签列，并进行重命名/转换。
    最后过滤掉无效标签行（标签值不合法的样本）。

    Args:
        split: HuggingFace Dataset 的一个分割（train/val/test）。
        dataset_name: 数据集标准名称，传递给通用标签转换逻辑使用。
        preserve_label: 若为 True，则保留原始整数标签（用于后续显式映射）；
                        否则直接通过通用标签转换逻辑将标签转为 0-5/-1。

    Returns:
        仅含 "text" 和 "label" 列、标签已过滤合法的 HuggingFace Dataset。
    """
    columns = set(split.column_names)

    if "text" in columns and "label" in columns:
        legacy_binary = _should_migrate_legacy_binary(split, "label", dataset_name)
        # 数据集已有标准列名，仅做类型强制转换
        def cast_existing(row, ds_name=dataset_name, keep_raw=preserve_label):
            raw_label = row.get("label")
            binary_label = _parse_binary_label(raw_label) if legacy_binary else -1
            row_copy = dict(row)
            row_copy["text"] = str(row.get("text") or "")
            row_copy["_sf_is_legacy_binary"] = legacy_binary and binary_label in (0, 1)
            row_copy["_sf_binary_label"] = binary_label
            row_copy["_sf_teacher_confidence"] = -1.0
            if keep_raw:
                # 保留原始整数标签，以便后续 label_map 映射
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
        # 自动从候选列名中匹配文本列和标签列
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

    # 过滤无效标签：preserve_label 模式下保留 >=0 的标签，否则仅保留 0-5
    before = len(split)
    if preserve_label:
        split = split.filter(lambda row: int(row["label"]) >= 0)
    else:
        split = split.filter(lambda row: int(row["label"]) in SENTIMENT_SCORES)
    after = len(split)
    if after < before:
        print(f"After schema/label cleanup for {dataset_name}: {after}/{before}")
    return _project_standard_columns(split)


def _force_balance_score_split(split, dataset_name: str, split_name: str, seed: int = 42):
    """对 0-5 评分数据集按存在的类别做下采样均衡。"""
    class_splits = {
        score: split.filter(lambda row, s=score: int(row["label"]) == s)
        for score in SENTIMENT_SCORES
    }
    class_counts = {score: len(class_split) for score, class_split in class_splits.items()}
    present_counts = [count for count in class_counts.values() if count > 0]

    if len(present_counts) < 2:
        print(
            f"Warning: Skip force-balance for {dataset_name} {split_name} "
            f"because fewer than two score classes are present: {class_counts}."
        )
        return split

    min_count = min(present_counts)
    from datasets import concatenate_datasets

    balanced_parts = [
        class_split.shuffle(seed=seed + score).select(range(min_count))
        for score, class_split in class_splits.items()
        if len(class_split) > 0
    ]
    balanced = concatenate_datasets(balanced_parts).shuffle(seed=seed)
    print(
        f"Force-balanced {dataset_name} {split_name}: "
        f"per_class={class_counts}, kept_per_present_class={min_count}, total={len(balanced)}"
    )
    return balanced


def _semi_supervised_enabled() -> bool:
    raw = os.getenv("SEMI_SUPERVISED_01_TO_05", "auto").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_teacher_path() -> str:
    return (
        os.getenv("BERT_PSEUDO_LABEL_TEACHER_PATH", "").strip()
        or os.getenv("PSEUDO_LABEL_TEACHER_PATH", "").strip()
        or os.getenv("BERT_CHECKPOINT_PATH", "").strip()
    )


def _has_legacy_binary_rows(split) -> bool:
    return (
        "_sf_is_legacy_binary" in split.column_names
        and any(bool(value) for value in split["_sf_is_legacy_binary"])
    )


def _infer_teacher_num_labels(teacher) -> int:
    backbone = getattr(teacher, "backbone", teacher)
    config_num_labels = int(getattr(getattr(backbone, "config", None), "num_labels", 0) or 0)
    if config_num_labels:
        return config_num_labels
    classifier = getattr(backbone, "classifier", None)
    if classifier is not None:
        out_features = getattr(classifier, "out_features", None)
        if out_features is not None:
            return int(out_features)
        out_proj = getattr(classifier, "out_proj", None)
        if out_proj is not None and getattr(out_proj, "out_features", None) is not None:
            return int(out_proj.out_features)
    return 0


def _maybe_apply_semi_supervised_binary_refinement(split, dataset_name: str, split_name: str):
    """Use an existing 6-class BERT teacher to split weak 0/1 labels into 0-5."""
    if not _semi_supervised_enabled() or not _has_legacy_binary_rows(split):
        return split

    teacher_path = _get_teacher_path()
    if not teacher_path:
        print(
            f"Semi-supervised 0/1->0-5 skipped for {dataset_name} {split_name}: "
            "BERT_PSEUDO_LABEL_TEACHER_PATH is not set."
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
        from transformers import AutoTokenizer

        from .checkpoint import load_checkpoint
        from .config import MAX_LEN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = load_checkpoint(str(checkpoint_path), device=device)
        if teacher is None:
            return split
        if _infer_teacher_num_labels(teacher) != NUM_SENTIMENT_CLASSES:
            print(
                f"Semi-supervised 0/1->0-5 skipped for {dataset_name} {split_name}: "
                "teacher is not a 6-class checkpoint."
            )
            return split

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        min_confidence = float(os.getenv("PSEUDO_LABEL_MIN_CONFIDENCE", "0.45"))
        fallback_to_endpoint = os.getenv("PSEUDO_LABEL_FALLBACK_TO_ENDPOINT", "1") == "1"
        batch_size = max(1, int(os.getenv("PSEUDO_LABEL_BATCH_SIZE", "64")))
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
                texts = [str(batch["text"][index]) for index in legacy_indices]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device, non_blocking=True)
                attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
                with torch.inference_mode():
                    probabilities = torch.softmax(
                        teacher(input_ids=input_ids, attention_mask=attention_mask),
                        dim=1,
                    ).detach().cpu().tolist()

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
    """将环境变量中的标签映射字符串解析为 {原始标签: 目标标签} 字典。

    期望格式为逗号分隔的 "src:dst" 对，例如 "1:0,2:0,3:-1,4:1"。
    目标标签只能为 -1（跳过）或 0-5 情感评分。

    Args:
        raw: 从环境变量读取的原始字符串。

    Returns:
        解析后的整数到整数映射字典。

    Raises:
        ValueError: 格式错误或目标标签超出 {-1, 0, 1, 2, 3, 4, 5} 范围。
    """
    return parse_label_map(raw)


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """根据数据集名称返回对应的标签映射字典，或 None（无需映射）。

    目前支持：
    - ttxy/online_shopping_10_cats：10 分类映射到 0-5，可通过环境变量覆盖默认映射。
    - dirtycomputer/simplifyweibo_4_moods：4 种情绪映射到 0-5，可通过环境变量覆盖。

    Args:
        dataset_name: 数据集标准名称。

    Returns:
        {原始整数标签: 目标标签} 字典，目标标签为 0-5/-1；
        若该数据集无需额外映射则返回 None。
    """
    if dataset_name == "ttxy/online_shopping_10_cats":
        # 默认：10 档评分线性映射到 0-5。
        raw_map = os.getenv(
            "TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP",
            "1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4,10:5",
        )
        return _parse_label_map_from_env(raw_map)
    if dataset_name == "dirtycomputer/simplifyweibo_4_moods":
        # 默认：0→极端正面(5)，1→极端负面(0)，2/3→跳过(-1)
        raw_map = os.getenv("SIMPLIFYWEIBO_4_MOODS_LABEL_MAP", "0:5,1:0,2:-1,3:-1")
        return _parse_label_map_from_env(raw_map)
    return None


def build_train_split_and_val_split():
    """从环境变量指定的数据集列表中构建训练集和验证集。

    流程概述：
    1. 读取 BERT_TRAIN_DATASETS 环境变量，解析数据集名称列表（支持别名）。
    2. 逐个加载数据集，自动处理 validation/test split 不存在的情况。
    3. 规范化列名与标签格式，应用数据集专属的 label_map。
    4. 对 DMSC 等数据集按存在的 0-5 评分档位进行强制均衡。
    5. 合并所有数据集，全局随机打乱，按环境变量限制样本上限。
    6. 可选：混入人工提取的短句数据（extracted_short_sentences.csv）。
    7. 可选：混入合成短句数据（USE_SYNTHETIC_DATA=1）。

    相关环境变量：
    - BERT_TRAIN_DATASETS: 逗号分隔的数据集名称列表（默认为 lansinuote/ChnSentiCorp,XiangPan/waimai_10k）
    - TRAIN_VAL_RATIO: 验证集比例（仅在无预设验证集时生效，默认 0.1）
    - TRAIN_MAX_SAMPLES: 训练集最大样本数（0 表示不限制）
    - TRAIN_MAX_VAL_SAMPLES: 验证集最大样本数（0 表示不限制）
    - USE_EXTRACTED_SHORT_SENTENCES: 是否混入短句数据（默认 1=开启）
    - SHORT_SENTENCES_RATIO: 短句数据在训练集中的目标占比（默认 0.3）
    - EXTRACTED_SHORT_SENTENCES_MAX: 短句数据集的上限数量（0 表示不限制）
    - USE_SYNTHETIC_DATA: 是否混入合成数据（默认 0=关闭）
    - SYNTHETIC_DATA_SIZE: 合成数据样本数（默认 5000）
    - ALLOW_SIMPLIFYWEIBO_4_MOODS: 是否允许加载简化微博4情绪数据集（默认 0=禁用）

    Returns:
        (loaded_dataset_names, train_split, val_split, None) 四元组。
        - loaded_dataset_names: 成功加载的数据集名称列表。
        - train_split: 处理后的训练集 HuggingFace Dataset。
        - val_split: 处理后的验证集 HuggingFace Dataset。
        - None: 预留字段，暂未使用。

    Raises:
        ValueError: 所有数据集均加载失败，或数据集缺少必要的 train split。
    """
    from datasets import concatenate_datasets, load_dataset

    # 从环境变量读取数据集名称列表，支持多个数据集用逗号分隔
    dataset_names = [
        item.strip()
        for item in os.getenv(
            "BERT_TRAIN_DATASETS",
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/JD_review,BerlinWang/DMSC",
        ).split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    label_map = None
    loaded_dataset_names = []
    # 验证集比例，限制在 [0.01, 0.5] 之间，避免极端值
    val_ratio = float(os.getenv("TRAIN_VAL_RATIO", "0.1"))
    val_ratio = min(max(val_ratio, 0.01), 0.5)

    for raw_name in dataset_names:
        # 将别名解析为 HuggingFace Hub 标准名称
        name = _resolve_dataset_name(raw_name)
        if name != raw_name:
            print(f"Dataset alias resolved: {raw_name} -> {name}")

        # simplifyweibo_4_moods 是 4 分类数据集，默认禁用，需显式开启
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

        # 获取该数据集的专属标签映射（如多分类转 0-5）
        current_label_map = get_label_map(name)
        if current_label_map:
            label_map = current_label_map
            print(f"Using label mapping for dataset: {name}")
            print(f"Label map: {current_label_map}")

        train_part = ds["train"]
        # 按优先级选取验证集：优先用 validation，其次用 test，否则从 train 中自动切分
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

        # 若存在 label_map，先保留原始标签以便后续显式映射；否则直接在归一化时转换
        preserve_raw_label = current_label_map is not None
        train_part = _normalize_split_columns(train_part, name, preserve_label=preserve_raw_label)
        val_part = _normalize_split_columns(val_part, name, preserve_label=preserve_raw_label)

        # 应用数据集专属的 label_map，并过滤掉映射为 -1（无效）的样本
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

        # DMSC 星级分布不均，对训练集和验证集分别做强制均衡
        if name == "BerlinWang/DMSC":
            train_part = _force_balance_score_split(train_part, name, "train", seed=42)
            val_part = _force_balance_score_split(val_part, name, "val", seed=43)

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

    # 合并所有数据集（单个数据集时直接使用，避免不必要的 concatenate）
    train_split = train_splits[0] if len(train_splits) == 1 else concatenate_datasets(train_splits)
    val_split = val_splits[0] if len(val_splits) == 1 else concatenate_datasets(val_splits)

    # 全局随机打乱，确保多数据集混合时样本分布均匀
    train_split = train_split.shuffle(seed=42)
    val_split = val_split.shuffle(seed=42)

    # 按环境变量限制训练集样本上限（用于快速调试或资源受限场景）
    max_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    if max_samples > 0:
        max_samples = min(max_samples, len(train_split))
        train_split = train_split.select(range(max_samples))

    # 按环境变量限制验证集样本上限
    max_val_samples = int(os.getenv("TRAIN_MAX_VAL_SAMPLES", "0"))
    if max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_split))
        val_split = val_split.select(range(max_val_samples))

    # 可选：混入人工提取的短句数据，以提升模型对短文本的泛化能力
    if os.getenv("USE_EXTRACTED_SHORT_SENTENCES", "1") == "1":
        try:
            csv_path = Path(__file__).resolve().parent / "extracted_short_sentences.csv"

            if csv_path.exists():
                print(f"Loading extracted short sentences from {csv_path}...")

                short_texts = []
                short_labels = []

                # 逐行解析 CSV，通过最后一个逗号分隔文本和标签
                with open(csv_path, "r", encoding="utf-8") as f:
                    next(f)  # 跳过表头行
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
                                score = validate_sentiment_score(label)
                                if score in SENTIMENT_SCORES:
                                    short_texts.append(text)
                                    short_labels.append(score)
                            except ValueError:
                                continue

                if short_texts:
                    from datasets import Dataset

                    short_sentences_ds = Dataset.from_dict({
                        "text": short_texts,
                        "label": short_labels,
                    })

                    # 限制短句数据集的最大样本数
                    max_short_samples = int(os.getenv("EXTRACTED_SHORT_SENTENCES_MAX", "0"))
                    if max_short_samples > 0 and len(short_sentences_ds) > max_short_samples:
                        short_sentences_ds = short_sentences_ds.select(range(max_short_samples))

                    # 根据目标短句占比计算实际使用数量：
                    # 若目标占比为 r，当前长句数量为 N，则短句数量 = N * r / (1 - r)
                    short_ratio = float(os.getenv("SHORT_SENTENCES_RATIO", "0.3"))
                    short_samples_to_use = int(len(train_split) * short_ratio / (1 - short_ratio))
                    short_samples_to_use = min(short_samples_to_use, len(short_sentences_ds))

                    if short_samples_to_use > 0:
                        short_sentences_ds = short_sentences_ds.select(range(short_samples_to_use))
                        train_split = concatenate_datasets([train_split, short_sentences_ds])
                        short_counts = get_label_distribution(short_sentences_ds)
                        print(f"Added {len(short_sentences_ds)} extracted short sentences (score_counts={short_counts}).")
                        print(f"  New train size: {len(train_split)} (long:short = {100 * (1 - short_ratio):.0f}%:{100 * short_ratio:.0f}%)")
                else:
                    print("Warning: No valid short sentences found in CSV")
            else:
                print(f"Note: extracted_short_sentences.csv not found at {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to load short sentences: {e}")

    # 可选：混入合成短句数据，进一步增强短文本样本多样性
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


def get_label_distribution(split, label_map: dict | None = None) -> list[int]:
    """统计数据集分割中 0-5 评分样本的数量分布。

    Args:
        split: HuggingFace Dataset 分割，须包含整数类型的 "label" 列。
        label_map: 预留参数，当前未使用（保留以兼容未来扩展）。

    Returns:
        长度为 6 的列表，索引即评分。
    """
    counts = [0 for _ in range(NUM_SENTIMENT_CLASSES)]
    labels = split["label"]
    for label in labels:
        score = validate_sentiment_score(int(label))
        counts[score] += 1
    return counts
