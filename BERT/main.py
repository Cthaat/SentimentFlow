"""脚本入口模块。"""

from __future__ import annotations

import os
from pathlib import Path

from .config import BERT_MODEL_NAME, MAX_LEN
from .env_utils import load_env_file
from .inference import predict_text
from .pipeline import load_or_train
from .sample_texts import DEFAULT_SAMPLES
from .trainer import train_model


def run() -> None:
    """执行训练流程并打印样例预测。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_env_file(env_path, override=False)

    raw_datasets = os.getenv(
        "BERT_TRAIN_DATASETS",
        "lansinuote/ChnSentiCorp,XiangPan/waimai_10k,dirtycomputer/JD_review,BerlinWang/DMSC",
    )
    active_datasets = [item.strip() for item in raw_datasets.split(",") if item.strip()]
    print(f"Active BERT_TRAIN_DATASETS ({len(active_datasets)}): {', '.join(active_datasets)}")
    print(f"Active BERT_MODEL_NAME: {BERT_MODEL_NAME}")

    force_retrain = os.getenv("BERT_FORCE_RETRAIN", os.getenv("FORCE_RETRAIN", "1")) == "1"
    print(f"Effective BERT_FORCE_RETRAIN={os.getenv('BERT_FORCE_RETRAIN', '') or os.getenv('FORCE_RETRAIN', '0')} (from {env_path})")

    if force_retrain:
        print("BERT_FORCE_RETRAIN=1 -> Retraining BERT model and overwriting checkpoint.")
        model, device = train_model()
    else:
        model, device = load_or_train()

    print("\n" + "=" * 80)
    print("DEFAULT SAMPLE PREDICTIONS:")
    print("=" * 80)
    for text in DEFAULT_SAMPLES:
        result = predict_text(text, model, device, max_len=MAX_LEN)
        print(
            f"{result['text']} -> score={result['score']} {result['label_zh']} "
            f"(conf={result['confidence']:.6f}, probs={result['probabilities']})"
        )

    print("\n" + "=" * 80)
    print("CUSTOM TEST CASES (Quality Verification):")
    print("=" * 80)
    try:
        from BERT.custom_test_cases import CUSTOM_TEST_CASES

        correct = 0
        total = len(CUSTOM_TEST_CASES)
        print(f"Total custom test cases: {total}")

        for text, expected_label in CUSTOM_TEST_CASES:
            result = predict_text(text, model, device, max_len=MAX_LEN)
            predicted_score = int(result["score"])
            is_correct = predicted_score == expected_label
            correct += is_correct

            status = "OK" if is_correct else "NG"
            print(
                f"{status} '{text}' -> {predicted_score} (expected: {expected_label}, "
                f"conf={result['confidence']:.2%})"
            )

        accuracy = 100 * correct / total
        print(f"\n{'=' * 80}")
        print(f"Custom Test Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"{'=' * 80}\n")
    except ImportError:
        print("Custom test cases not found, skipping validation.")


if __name__ == "__main__":
    run()
