"""脚本入口模块。

用于命令行直接运行训练 + 样例推理。
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import MAX_LEN, VOCAB_SIZE
from .env_utils import load_env_file
from .inference import predict_text
from .pipeline import load_or_train
from .sample_texts import DEFAULT_SAMPLES
from .trainer import train_model


def run() -> None:
    """执行训练流程并打印样例预测。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_env_file(env_path, override=True)

    # FORCE_RETRAIN 环境变量示例：1（表示强制重新训练模型并覆盖现有 checkpoint）
    force_retrain = os.getenv("FORCE_RETRAIN", "1") == "1"
    # FORCE_RETRAIN 的默认值为 "1"（启用），表示默认情况下会强制重新训练模型并覆盖现有 checkpoint。这是为了确保在你第一次运行脚本时能够训练出一个新的模型。如果你希望在后续运行中使用已经训练好的模型，可以将 FORCE_RETRAIN 设置为 "0" 来跳过训练阶段，直接加载现有 checkpoint 进行推理。
    print(f"Effective FORCE_RETRAIN={os.getenv('FORCE_RETRAIN', '0')} (from {env_path})")

    if force_retrain:
        print("FORCE_RETRAIN=1 -> Retraining model and overwriting checkpoint.")
        model, device = train_model()
    else:
        model, device = load_or_train()

    print("\n" + "="*80)
    print("DEFAULT SAMPLE PREDICTIONS:")
    print("="*80)
    for text in DEFAULT_SAMPLES:
        result = predict_text(text, model, device, max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
        print(
            f"{result['text']} -> {result['label']} "
            f"(neg={result['negative_score']:.6f}, pos={result['positive_score']:.6f}, "
            f"conf={result['confidence']:.6f})"
        )
    
    # 新增：自定义测试集验证
    print("\n" + "="*80)
    print("CUSTOM TEST CASES (Quality Verification):")
    print("="*80)
    try:
        from custom_test_cases import CUSTOM_TEST_CASES
        correct = 0
        total = len(CUSTOM_TEST_CASES)
        
        for text, expected_label in CUSTOM_TEST_CASES:
            result = predict_text(text, model, device, max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
            predicted_label = 1 if result['label'] == '正面' else 0
            is_correct = predicted_label == expected_label
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            expected_text = '正面' if expected_label == 1 else '负面'
            print(
                f"{status} '{text}' -> {result['label']} (expected: {expected_text}, "
                f"conf={result['confidence']:.2%})"
            )
        
        accuracy = 100 * correct / total
        print(f"\n{'='*80}")
        print(f"Custom Test Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"{'='*80}\n")
    except ImportError:
        print("Custom test cases not found, skipping validation.")


if __name__ == "__main__":
    run()

