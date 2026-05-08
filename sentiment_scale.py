"""Shared sentiment score contract for the 0-5 sentiment system."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


SENTIMENT_SCORE_MIN = 0
SENTIMENT_SCORE_MAX = 5
NUM_SENTIMENT_CLASSES = SENTIMENT_SCORE_MAX - SENTIMENT_SCORE_MIN + 1
SENTIMENT_SCORES = tuple(range(SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX + 1))

SENTIMENT_LABELS: dict[int, str] = {
    0: "extremely_negative",
    1: "clearly_negative",
    2: "slightly_negative",
    3: "neutral",
    4: "slightly_positive",
    5: "extremely_positive",
}

SENTIMENT_DISPLAY_LABELS: dict[int, str] = {
    0: "极端负面",
    1: "明显负面",
    2: "略微负面",
    3: "中性",
    4: "略微正面",
    5: "极端正面",
}

LABEL_TO_SCORE: dict[str, int] = {
    **{label: score for score, label in SENTIMENT_LABELS.items()},
    "very_negative": 0,
    "negative": 1,
    "somewhat_negative": 2,
    "neutral": 3,
    "somewhat_positive": 4,
    "positive": 5,
    "very_positive": 5,
    "neg": 0,
    "pos": 5,
    "差评": 0,
    "负面": 0,
    "消极": 0,
    "中性": 3,
    "一般": 3,
    "正面": 5,
    "积极": 5,
    "好评": 5,
}

LEGACY_BINARY_TO_SCORE = {0: 0, 1: 5}
BINARY_POLARITY_SCORE_BUCKETS = {
    0: (0, 1, 2),
    1: (4, 5),
}
STAR_RATING_TO_SCORE = {1: 0, 2: 1, 3: 3, 4: 4, 5: 5}
TEN_POINT_RATING_TO_SCORE = {
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 5,
}

LEGACY_BINARY_DATASETS = {
    "lansinuote/ChnSentiCorp",
    "XiangPan/waimai_10k",
    "dirtycomputer/weibo_senti_100k",
    "dirtycomputer/ChnSentiCorp_htl_all",
    "ndiy/NLPCC14-SC",
}


def is_valid_sentiment_score(value: int) -> bool:
    return SENTIMENT_SCORE_MIN <= int(value) <= SENTIMENT_SCORE_MAX


def validate_sentiment_score(value: int) -> int:
    score = int(value)
    if not is_valid_sentiment_score(score):
        raise ValueError(
            f"Invalid sentiment score {value!r}. "
            f"Expected an integer in [{SENTIMENT_SCORE_MIN}, {SENTIMENT_SCORE_MAX}]."
        )
    return score


def score_to_label(score: int) -> str:
    return SENTIMENT_LABELS[validate_sentiment_score(score)]


def score_to_display_label(score: int) -> str:
    return SENTIMENT_DISPLAY_LABELS[validate_sentiment_score(score)]


def score_to_reasoning(score: int) -> str:
    display = score_to_display_label(score)
    return f"模型将文本情感强度判定为 {score} 分（{display}）。"


def parse_label_map(raw: str) -> dict[int, int]:
    """Parse an env label map like ``0:0,1:5,2:-1``.

    Target ``-1`` means skip the sample. Other targets must be valid 0-5 scores.
    """
    mapping: dict[int, int] = {}
    for item in raw.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid label map item {pair!r}. Expected format like '0:5'.")
        src, dst = pair.split(":", 1)
        src_label = int(src.strip())
        dst_label = int(dst.strip())
        if dst_label != -1:
            validate_sentiment_score(dst_label)
        mapping[src_label] = dst_label
    if not mapping:
        raise ValueError("Parsed empty label map from environment variable.")
    return mapping


def coerce_sentiment_score(
    raw_label,
    dataset_name: str = "",
    label_col: str = "label",
    *,
    legacy_binary: bool = True,
) -> int:
    """Coerce a raw dataset label into the 0-5 sentiment score contract.

    Returns ``-1`` when the label is unusable and should be filtered.
    """
    dataset_name = str(dataset_name or "")
    label_col = str(label_col or "label")

    if dataset_name == "BerlinWang/DMSC" and label_col == "Star":
        return _map_int(raw_label, STAR_RATING_TO_SCORE)

    if dataset_name == "dirtycomputer/JD_review" and label_col == "rating":
        return _map_int(raw_label, STAR_RATING_TO_SCORE)

    if dataset_name == "ttxy/online_shopping_10_cats":
        return _map_int(raw_label, TEN_POINT_RATING_TO_SCORE)

    if isinstance(raw_label, bool):
        return LEGACY_BINARY_TO_SCORE[int(raw_label)]

    numeric_value = _try_parse_number(raw_label)
    if numeric_value is not None:
        value = int(numeric_value)
        if value == -1:
            return 0
        if legacy_binary and dataset_name in LEGACY_BINARY_DATASETS and value in LEGACY_BINARY_TO_SCORE:
            return LEGACY_BINARY_TO_SCORE[value]
        if legacy_binary and value in LEGACY_BINARY_TO_SCORE:
            return LEGACY_BINARY_TO_SCORE[value]
        if is_valid_sentiment_score(value):
            return value
        return -1

    text = str(raw_label).strip().lower()
    if text in LABEL_TO_SCORE:
        return LABEL_TO_SCORE[text]
    return -1


def _map_int(raw_label, mapping: dict[int, int]) -> int:
    numeric_value = _try_parse_number(raw_label)
    if numeric_value is None:
        return -1
    return mapping.get(int(numeric_value), -1)


def _try_parse_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if not float(number).is_integer():
        return None
    return number


def probabilities_to_prediction(probabilities: Sequence[float]) -> dict:
    """Convert model probabilities to the API/inference prediction contract.

    Legacy 2-class checkpoints are mapped as 0 -> score 0 and 1 -> score 5.
    """
    probs = [float(value) for value in probabilities]
    if len(probs) == 2:
        probs = [probs[0], 0.0, 0.0, 0.0, 0.0, probs[1]]
    if len(probs) != NUM_SENTIMENT_CLASSES:
        raise ValueError(
            f"Expected {NUM_SENTIMENT_CLASSES} sentiment probabilities, got {len(probs)}."
        )

    score = max(range(NUM_SENTIMENT_CLASSES), key=lambda index: probs[index])
    confidence = probs[score]
    return {
        "score": score,
        "label": score_to_label(score),
        "label_zh": score_to_display_label(score),
        "confidence": round(confidence, 6),
        "probabilities": [round(value, 6) for value in probs],
        "reasoning": score_to_reasoning(score),
    }


def choose_score_from_binary_teacher_probabilities(
    probabilities: Sequence[float],
    binary_label: int,
    *,
    min_confidence: float,
    fallback_to_endpoint: bool = False,
) -> dict:
    """Choose a 0-5 pseudo label from teacher probabilities and a weak 0/1 label.

    The weak binary label is treated as a polarity constraint:
    - 0 may only become 0/1/2
    - 1 may only become 4/5

    If the best constrained score is below ``min_confidence``, the default
    behavior is to reject the sample. ``fallback_to_endpoint=True`` is kept
    only for explicit legacy compatibility.
    """
    weak_label = int(binary_label)
    if weak_label not in LEGACY_BINARY_TO_SCORE:
        raise ValueError(f"Invalid weak binary label {binary_label!r}. Expected 0 or 1.")

    probs = [float(value) for value in probabilities]
    if len(probs) == 2:
        probs = [probs[0], 0.0, 0.0, 0.0, 0.0, probs[1]]
    if len(probs) != NUM_SENTIMENT_CLASSES:
        raise ValueError(
            f"Expected {NUM_SENTIMENT_CLASSES} teacher probabilities, got {len(probs)}."
        )

    candidates = BINARY_POLARITY_SCORE_BUCKETS[weak_label]
    score = max(candidates, key=lambda index: probs[index])
    confidence = probs[score]
    if confidence >= min_confidence:
        return {
            "score": score,
            "confidence": confidence,
            "accepted": True,
            "source": "pseudo_teacher",
        }

    if fallback_to_endpoint:
        endpoint = LEGACY_BINARY_TO_SCORE[weak_label]
        return {
            "score": endpoint,
            "confidence": confidence,
            "accepted": False,
            "source": "weak_binary_endpoint",
        }

    return {
        "score": -1,
        "confidence": confidence,
        "accepted": False,
        "source": "pseudo_rejected",
    }


def compute_classification_metrics(
    true_labels: Iterable[int],
    pred_labels: Iterable[int],
    *,
    num_classes: int = NUM_SENTIMENT_CLASSES,
) -> dict:
    """Compute multiclass and ordinal metrics for 0-5 sentiment scores."""
    true_list = [validate_sentiment_score(label) for label in true_labels]
    pred_list = [validate_sentiment_score(label) for label in pred_labels]
    if len(true_list) != len(pred_list):
        raise ValueError("true_labels and pred_labels must have the same length.")

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(true_list, pred_list):
        confusion_matrix[true_label][pred_label] += 1

    total = len(true_list)
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "spearman": 0.0,
            "confusion_matrix": confusion_matrix,
            "support": [0 for _ in range(num_classes)],
            "per_class_f1": [0.0 for _ in range(num_classes)],
        }

    support = [sum(confusion_matrix[row]) for row in range(num_classes)]
    predicted_support = [
        sum(confusion_matrix[row][col] for row in range(num_classes))
        for col in range(num_classes)
    ]

    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = confusion_matrix[class_index][class_index]
        false_positive = predicted_support[class_index] - true_positive
        false_negative = support[class_index] - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class_f1.append(f1)

    present_classes = [index for index, count in enumerate(support) if count > 0]
    macro_f1 = sum(per_class_f1[index] for index in present_classes) / max(1, len(present_classes))
    weighted_f1 = sum(per_class_f1[index] * support[index] for index in range(num_classes)) / total
    accuracy = sum(confusion_matrix[index][index] for index in range(num_classes)) / total
    mae = sum(abs(true - pred) for true, pred in zip(true_list, pred_list)) / total
    rmse = math.sqrt(sum((true - pred) ** 2 for true, pred in zip(true_list, pred_list)) / total)
    qwk = _quadratic_weighted_kappa(confusion_matrix, support, predicted_support, total)
    spearman = _spearman_correlation(true_list, pred_list)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "rmse": rmse,
        "quadratic_weighted_kappa": qwk,
        "spearman": spearman,
        "confusion_matrix": confusion_matrix,
        "support": support,
        "per_class_f1": per_class_f1,
    }


def _quadratic_weighted_kappa(
    confusion_matrix: list[list[int]],
    support: list[int],
    predicted_support: list[int],
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    max_distance = NUM_SENTIMENT_CLASSES - 1
    observed = 0.0
    expected = 0.0
    for row in range(NUM_SENTIMENT_CLASSES):
        for col in range(NUM_SENTIMENT_CLASSES):
            weight = ((row - col) ** 2) / (max_distance**2)
            observed += weight * confusion_matrix[row][col]
            expected += weight * support[row] * predicted_support[col] / total

    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def _spearman_correlation(true_labels: list[int], pred_labels: list[int]) -> float:
    if len(true_labels) < 2:
        return 0.0
    true_ranks = _average_ranks(true_labels)
    pred_ranks = _average_ranks(pred_labels)
    true_mean = sum(true_ranks) / len(true_ranks)
    pred_mean = sum(pred_ranks) / len(pred_ranks)
    numerator = sum((left - true_mean) * (right - pred_mean) for left, right in zip(true_ranks, pred_ranks))
    true_var = sum((value - true_mean) ** 2 for value in true_ranks)
    pred_var = sum((value - pred_mean) ** 2 for value in pred_ranks)
    denominator = math.sqrt(true_var * pred_var)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _average_ranks(values: list[int]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(indexed):
        next_cursor = cursor + 1
        while next_cursor < len(indexed) and indexed[next_cursor][1] == indexed[cursor][1]:
            next_cursor += 1
        average_rank = (cursor + 1 + next_cursor) / 2.0
        for index in range(cursor, next_cursor):
            original_index = indexed[index][0]
            ranks[original_index] = average_rank
        cursor = next_cursor
    return ranks
