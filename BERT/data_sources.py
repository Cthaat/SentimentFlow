"""训练/验证数据集构建模块。"""

from __future__ import annotations

import os
from typing import Dict, Tuple


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


def _coerce_binary_label(raw_label, dataset_name: str, label_col: str) -> int:
    """将各数据集的原始标签统一转换为二分类标签 0/1。

    针对不同数据集的标签体系做专项处理：
    - BerlinWang/DMSC：星级评分（Star），1-2 → 负面(0)，4-5 → 正面(1)，3 → 跳过(-1)
    - dirtycomputer/JD_review：评分（rating），同上规则
    - 布尔值：True → 1，False → 0
    - 整数/浮点数：仅接受 0/1/-1（-1 映射为 0）
    - 字符串：支持 pos/positive/正面/好评 等关键词

    Args:
        raw_label: 原始标签值，类型不限。
        dataset_name: 数据集的标准名称，用于区分专项规则。
        label_col: 标签所在的列名，用于区分同一数据集下的不同列。

    Returns:
        0（负面）、1（正面）或 -1（无效/中性，应跳过）。
    """
    # 针对 DMSC 电影评分数据集：通过星级判断情感极性
    if dataset_name == "BerlinWang/DMSC" and label_col == "Star":
        try:
            star = int(raw_label)
        except (TypeError, ValueError):
            return -1
        if star <= 2:
            return 0  # 1-2 星：负面
        if star >= 4:
            return 1  # 4-5 星：正面
        return -1     # 3 星：中性，跳过

    # 针对京东评论数据集：通过评分判断情感极性
    if dataset_name == "dirtycomputer/JD_review" and label_col == "rating":
        try:
            rating = int(float(raw_label))
        except (TypeError, ValueError):
            return -1
        if rating <= 2:
            return 0  # 1-2 分：负面
        if rating >= 4:
            return 1  # 4-5 分：正面
        return -1     # 3 分：中性，跳过

    # 布尔类型直接转换
    if isinstance(raw_label, bool):
        return int(raw_label)

    # 数值类型：仅接受 0/1/-1，其余视为无效
    if isinstance(raw_label, (int, float)):
        value = int(raw_label)
        if value in (0, 1):
            return value
        if value == -1:
            return 0  # 部分数据集用 -1 表示负面
        return -1

    # 字符串类型：匹配常见的情感关键词
    text = str(raw_label).strip().lower()
    if text in {"1", "pos", "positive", "正面", "好评"}:
        return 1
    if text in {"0", "-1", "neg", "negative", "负面", "差评"}:
        return 0
    return -1


def _normalize_split_columns(split, dataset_name: str, preserve_label: bool = False):
    """将数据集的列规范化为统一的 text/label 格式。

    若数据集已有 text 和 label 列，则仅对类型做强制转换；
    否则，从候选列名列表中自动匹配文本列和标签列，并进行重命名/转换。
    最后过滤掉无效标签行（标签值不合法的样本）。

    Args:
        split: HuggingFace Dataset 的一个分割（train/val/test）。
        dataset_name: 数据集标准名称，传递给 _coerce_binary_label 使用。
        preserve_label: 若为 True，则保留原始整数标签（用于后续显式映射）；
                        否则直接通过 _coerce_binary_label 将标签转为 0/1/-1。

    Returns:
        仅含 "text" 和 "label" 列、标签已过滤合法的 HuggingFace Dataset。
    """
    columns = set(split.column_names)

    if "text" in columns and "label" in columns:
        # 数据集已有标准列名，仅做类型强制转换
        def cast_existing(row, ds_name=dataset_name, keep_raw=preserve_label):
            row_copy = dict(row)
            row_copy["text"] = str(row.get("text") or "")
            if keep_raw:
                # 保留原始整数标签，以便后续 label_map 映射
                try:
                    row_copy["label"] = int(row.get("label"))
                except (TypeError, ValueError):
                    row_copy["label"] = -1
            else:
                row_copy["label"] = _coerce_binary_label(row.get("label"), ds_name, "label")
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

    # 过滤无效标签：preserve_label 模式下保留 >=0 的标签，否则仅保留 0/1
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
    """对二分类数据集进行强制正负样本均衡处理（下采样至少数类数量）。

    若某一类别为空，则跳过均衡并打印警告，避免产生空数据集。
    均衡后数据集会重新随机打乱，确保正负样本均匀分布。

    Args:
        split: 需要均衡的 HuggingFace Dataset 分割。
        dataset_name: 数据集名称，仅用于日志输出。
        split_name: 分割名称（如 "train"/"val"），仅用于日志输出。
        seed: 随机种子，用于 shuffle 和 select，保证可复现性。

    Returns:
        正负样本数量相等的 HuggingFace Dataset。
    """
    neg_split = split.filter(lambda row: int(row["label"]) == 0)
    pos_split = split.filter(lambda row: int(row["label"]) == 1)

    neg_count = len(neg_split)
    pos_count = len(pos_split)
    # 取两类中数量较少的一方作为均衡目标数
    min_count = min(neg_count, pos_count)

    # 若有某一类为空则无法均衡，直接返回原数据并警告
    if min_count == 0:
        print(
            f"Warning: Skip force-balance for {dataset_name} {split_name} "
            f"because one class is empty (neg={neg_count}, pos={pos_count})."
        )
        return split

    # 对多数类进行下采样，使两类数量一致
    neg_split = neg_split.shuffle(seed=seed).select(range(min_count))
    pos_split = pos_split.shuffle(seed=seed).select(range(min_count))

    from datasets import concatenate_datasets

    # 合并后再次打乱，避免正负样本成块出现
    balanced = concatenate_datasets([neg_split, pos_split]).shuffle(seed=seed)
    print(
        f"Force-balanced {dataset_name} {split_name}: "
        f"neg={min_count}, pos={min_count}, total={len(balanced)} "
        f"(from neg={neg_count}, pos={pos_count})"
    )
    return balanced


def _parse_label_map_from_env(raw: str) -> Dict[int, int]:
    """将环境变量中的标签映射字符串解析为 {原始标签: 目标标签} 字典。

    期望格式为逗号分隔的 "src:dst" 对，例如 "1:0,2:0,3:-1,4:1"。
    目标标签只能为 -1（跳过）、0（负面）或 1（正面）。

    Args:
        raw: 从环境变量读取的原始字符串。

    Returns:
        解析后的整数到整数映射字典。

    Raises:
        ValueError: 格式错误或目标标签超出 {-1, 0, 1} 范围。
    """
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
        # 目标标签只允许 -1/0/1
        if dst_label not in (-1, 0, 1):
            raise ValueError(
                f"Invalid target label {dst_label} in '{pair}'. Expected -1, 0 or 1."
            )
        mapping[src_label] = dst_label
    if not mapping:
        raise ValueError("Parsed empty label map from environment variable.")
    return mapping


def get_label_map(dataset_name: str) -> Dict[int, int] | None:
    """根据数据集名称返回对应的标签映射字典，或 None（无需映射）。

    目前支持：
    - ttxy/online_shopping_10_cats：10 分类映射到二分类，可通过环境变量覆盖默认映射。
    - dirtycomputer/simplifyweibo_4_moods：4 种情绪映射到二分类，可通过环境变量覆盖。

    Args:
        dataset_name: 数据集标准名称。

    Returns:
        {原始整数标签: 目标标签} 字典，目标标签为 0/1/-1；
        若该数据集无需额外映射则返回 None。
    """
    if dataset_name == "ttxy/online_shopping_10_cats":
        # 默认：类别 1-4 映射为负面(0)，5-6 跳过，7-10 映射为正面(1)
        raw_map = os.getenv(
            "TTXY_ONLINE_SHOPPING_10_CATS_LABEL_MAP",
            "1:0,2:0,3:0,4:0,5:-1,6:-1,7:1,8:1,9:1,10:1",
        )
        return _parse_label_map_from_env(raw_map)
    if dataset_name == "dirtycomputer/simplifyweibo_4_moods":
        # 默认：0→正面(1)，1→负面(0)，2/3→跳过(-1)
        raw_map = os.getenv("SIMPLIFYWEIBO_4_MOODS_LABEL_MAP", "0:1,1:0,2:-1,3:-1")
        return _parse_label_map_from_env(raw_map)
    return None


def build_train_split_and_val_split():
    """从环境变量指定的数据集列表中构建训练集和验证集。

    流程概述：
    1. 读取 BERT_TRAIN_DATASETS 环境变量，解析数据集名称列表（支持别名）。
    2. 逐个加载数据集，自动处理 validation/test split 不存在的情况。
    3. 规范化列名与标签格式，应用数据集专属的 label_map。
    4. 对 DMSC 等数据集进行强制正负均衡。
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
            "lansinuote/ChnSentiCorp,XiangPan/waimai_10k",
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

        # 获取该数据集的专属标签映射（如多分类转二分类）
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

        # DMSC 星级分布不均，对训练集和验证集分别做强制均衡
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
            from pathlib import Path

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


def get_label_distribution(split, label_map: dict | None = None) -> Tuple[int, int]:
    """统计数据集分割中正负样本的数量分布。

    Args:
        split: HuggingFace Dataset 分割，须包含整数类型的 "label" 列。
        label_map: 预留参数，当前未使用（保留以兼容未来扩展）。

    Returns:
        (neg_count, pos_count) 二元组，分别为负面（label=0）和正面（label=1）样本数量。
    """
    labels = split["label"]
    neg = sum(1 for x in labels if int(x) == 0)
    pos = sum(1 for x in labels if int(x) == 1)
    return neg, pos
