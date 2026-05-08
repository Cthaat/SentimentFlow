from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from datasets import Dataset

from ordinal_loss import DistanceAwareOrdinalLoss
from sentiment_scale import (
    NUM_SENTIMENT_CLASSES,
    choose_score_from_binary_teacher_probabilities,
    coerce_sentiment_score,
    compute_classification_metrics,
    probabilities_to_prediction,
)
from training.data_sources import (
    _maybe_apply_semi_supervised_binary_refinement,
    _normalize_split_columns,
    _normalize_short_sentence_labels,
)
from BERT.data_sources import (
    _binary_pseudo_dataset_names,
    _load_pseudo_label_split,
    _normalize_short_sentence_labels as _normalize_bert_short_sentence_labels,
    _infer_teacher_num_labels,
    _teacher_dataset_names,
    _with_training_columns,
)
from training.dataset import CsvStreamDataset as LstmCsvStreamDataset
from training.evaluate import evaluate
from training.inference import predict_text
from training.model import SentimentLSTMModel
from BERT.dataset import CsvStreamDataset as BertCsvStreamDataset
from BERT.inference import outputs_to_predictions, predicted_classes_from_outputs


class SentimentScaleContractTest(unittest.TestCase):
    def test_legacy_binary_labels_migrate_to_score_endpoints(self) -> None:
        self.assertEqual(coerce_sentiment_score(0, dataset_name="lansinuote/ChnSentiCorp"), 0)
        self.assertEqual(coerce_sentiment_score(1, dataset_name="lansinuote/ChnSentiCorp"), 5)
        self.assertEqual(coerce_sentiment_score(1, legacy_binary=False), 1)

    def test_star_ratings_map_to_score_scale(self) -> None:
        self.assertEqual(coerce_sentiment_score(1, dataset_name="BerlinWang/DMSC", label_col="Star"), 0)
        self.assertEqual(coerce_sentiment_score(2, dataset_name="BerlinWang/DMSC", label_col="Star"), 1)
        self.assertEqual(coerce_sentiment_score(3, dataset_name="BerlinWang/DMSC", label_col="Star"), 3)
        self.assertEqual(coerce_sentiment_score(5, dataset_name="BerlinWang/DMSC", label_col="Star"), 5)

    def test_probabilities_require_six_score_classes(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_to_prediction([0.2, 0.8])

        result = probabilities_to_prediction([0.05, 0.05, 0.05, 0.05, 0.1, 0.7])
        self.assertEqual(result["score"], 5)
        self.assertEqual(len(result["probabilities"]), NUM_SENTIMENT_CLASSES)
        self.assertAlmostEqual(result["probabilities"][5], 0.7)

    def test_metrics_include_multiclass_and_ordinal_values(self) -> None:
        metrics = compute_classification_metrics([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), NUM_SENTIMENT_CLASSES)

    def test_binary_constrained_teacher_choice(self) -> None:
        negative_choice = choose_score_from_binary_teacher_probabilities(
            [0.05, 0.2, 0.7, 0.01, 0.02, 0.02],
            0,
            min_confidence=0.5,
        )
        positive_choice = choose_score_from_binary_teacher_probabilities(
            [0.05, 0.2, 0.7, 0.01, 0.55, 0.04],
            1,
            min_confidence=0.5,
        )
        fallback_choice = choose_score_from_binary_teacher_probabilities(
            [0.1, 0.2, 0.1, 0.1, 0.3, 0.2],
            1,
            min_confidence=0.5,
        )
        self.assertEqual(negative_choice["score"], 2)
        self.assertEqual(positive_choice["score"], 4)
        self.assertEqual(fallback_choice["score"], -1)
        self.assertFalse(fallback_choice["accepted"])


class TrainingSmokeTest(unittest.TestCase):
    def test_lstm_train_eval_and_inference_shapes(self) -> None:
        device = torch.device("cpu")
        model = SentimentLSTMModel(vocab_size=256).to(device)
        inputs = torch.randint(0, 255, (6, 8), dtype=torch.long)
        labels = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (6, NUM_SENTIMENT_CLASSES))

        loss = DistanceAwareOrdinalLoss()(logits, labels)
        loss.backward()

        split = [
            {"text": "非常糟糕", "label": 0},
            {"text": "有点失望", "label": 2},
            {"text": "一般", "label": 3},
            {"text": "非常满意", "label": 5},
        ]
        metrics = evaluate(model, split, device, batch_size=2, max_len=8, vocab_size=256)
        self.assertEqual(len(metrics.confusion_matrix), NUM_SENTIMENT_CLASSES)

        result = predict_text("体验还不错", model, device, max_len=8, vocab_size=256)
        self.assertIn(result["score"], range(NUM_SENTIMENT_CLASSES))
        self.assertEqual(len(result["probabilities"]), NUM_SENTIMENT_CLASSES)

    def test_lstm_teacher_refines_legacy_binary_dataset(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "teacher.pt"
            model = SentimentLSTMModel(vocab_size=128)
            with torch.no_grad():
                model.fc.weight.zero_()
                model.fc.bias[:] = torch.tensor([0.0, 5.0, 0.0, 0.0, 5.0, 0.0])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "max_len": 8,
                    "vocab_size": 128,
                    "num_classes": NUM_SENTIMENT_CLASSES,
                },
                checkpoint_path,
            )

            previous_env = {
                "PSEUDO_LABEL_TEACHER_PATH": os.environ.get("PSEUDO_LABEL_TEACHER_PATH"),
                "SEMI_SUPERVISED_01_TO_05": os.environ.get("SEMI_SUPERVISED_01_TO_05"),
                "PSEUDO_LABEL_MIN_CONFIDENCE": os.environ.get("PSEUDO_LABEL_MIN_CONFIDENCE"),
            }
            try:
                os.environ["PSEUDO_LABEL_TEACHER_PATH"] = str(checkpoint_path)
                os.environ["SEMI_SUPERVISED_01_TO_05"] = "1"
                os.environ["PSEUDO_LABEL_MIN_CONFIDENCE"] = "0.3"

                split = Dataset.from_dict({"text": ["很差", "很好"], "label": [0, 1]})
                normalized = _normalize_split_columns(split, "custom-binary")
                refined = _maybe_apply_semi_supervised_binary_refinement(
                    normalized,
                    "custom-binary",
                    "train",
                )
                self.assertEqual(list(refined["label"]), [1, 4])
                self.assertEqual(list(refined["_sf_label_source"]), ["pseudo_teacher", "pseudo_teacher"])
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_csv_stream_dataset_detects_legacy_binary_at_file_level(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            import os

            previous_override = os.environ.get("MIGRATE_LEGACY_BINARY_LABELS")
            os.environ["MIGRATE_LEGACY_BINARY_LABELS"] = "auto"
            try:
                six_score_path = Path(tmp_dir) / "six_score.csv"
                six_score_path.write_text(
                    "text,label\n"
                    "score0,0\n"
                    "score1,1\n"
                    "score2,2\n"
                    "score3,3\n"
                    "score4,4\n"
                    "score5,5\n",
                    encoding="utf-8",
                )
                binary_path = Path(tmp_dir) / "binary.csv"
                binary_path.write_text(
                    "text,label\n"
                    "bad,0\n"
                    "good,1\n",
                    encoding="utf-8",
                )

                lstm_six_labels = sorted(
                    int(label.item())
                    for _, label in LstmCsvStreamDataset(
                        six_score_path,
                        chunk_size=2,
                        max_len=8,
                        vocab_size=128,
                    )
                )
                bert_six_labels = sorted(
                    int(record["label"])
                    for record in BertCsvStreamDataset(
                        six_score_path,
                        chunk_size=2,
                    )
                )
                lstm_binary_labels = sorted(
                    int(label.item())
                    for _, label in LstmCsvStreamDataset(
                        binary_path,
                        chunk_size=2,
                        max_len=8,
                        vocab_size=128,
                    )
                )
                bert_binary_labels = sorted(
                    int(record["label"])
                    for record in BertCsvStreamDataset(
                        binary_path,
                        chunk_size=2,
                    )
                )

                self.assertEqual(lstm_six_labels, [0, 1, 2, 3, 4, 5])
                self.assertEqual(bert_six_labels, [0, 1, 2, 3, 4, 5])
                self.assertEqual(lstm_binary_labels, [0, 5])
                self.assertEqual(bert_binary_labels, [0, 5])
            finally:
                if previous_override is None:
                    os.environ.pop("MIGRATE_LEGACY_BINARY_LABELS", None)
                else:
                    os.environ["MIGRATE_LEGACY_BINARY_LABELS"] = previous_override

    def test_legacy_binary_short_sentence_csv_labels_map_to_endpoints(self) -> None:
        raw_rows = [("很好用", 1), ("很糟糕", 0)]
        self.assertEqual(_normalize_short_sentence_labels(raw_rows), [("很好用", 5), ("很糟糕", 0)])
        self.assertEqual(
            _normalize_bert_short_sentence_labels(raw_rows),
            [("很好用", 5), ("很糟糕", 0)],
        )

    def test_score_short_sentence_csv_labels_stay_on_score_scale(self) -> None:
        raw_rows = [("明显负面", 1), ("极端正面", 5), ("中性", 3)]
        self.assertEqual(_normalize_short_sentence_labels(raw_rows), raw_rows)
        self.assertEqual(_normalize_bert_short_sentence_labels(raw_rows), raw_rows)

    def test_distance_aware_ordinal_loss_supports_soft_labels_and_weights(self) -> None:
        logits = torch.randn(3, NUM_SENTIMENT_CLASSES, requires_grad=True)
        labels = torch.tensor([0, 3, 5], dtype=torch.long)
        soft_labels = torch.tensor(
            [
                [0.70, 0.20, 0.10, 0.00, 0.00, 0.00],
                [0.00, 0.05, 0.15, 0.60, 0.15, 0.05],
                [0.00, 0.00, 0.00, 0.05, 0.20, 0.75],
            ],
            dtype=torch.float32,
        )
        sample_weights = torch.tensor([1.0, 0.3, 0.3], dtype=torch.float32)

        loss = DistanceAwareOrdinalLoss()(
            {"logits": logits},
            labels,
            soft_labels=soft_labels,
            sample_weights=sample_weights,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_multiclass_outputs_use_softmax_argmax_for_prediction(self) -> None:
        outputs = {
            "logits": torch.tensor([[0.1, 0.1, 0.1, 5.0, 0.2, 0.1]], dtype=torch.float32),
        }
        pred = predicted_classes_from_outputs(outputs)
        self.assertEqual(pred.tolist(), [3])
        prediction = outputs_to_predictions(outputs)[0]
        self.assertEqual(prediction["score"], 3)
        self.assertEqual(len(prediction["probabilities"]), NUM_SENTIMENT_CLASSES)

    def test_multiclass_teacher_num_labels_detects_custom_classifier_head(self) -> None:
        class DummyTeacher(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.classifier = torch.nn.Linear(4, NUM_SENTIMENT_CLASSES)

        self.assertEqual(_infer_teacher_num_labels(DummyTeacher()), NUM_SENTIMENT_CLASSES)

    def test_bert_pseudo_label_split_loads_soft_training_records(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pseudo_labels.jsonl"
            payload = {
                "id": "row-1",
                "text": "体验还不错",
                "score": 4,
                "confidence": 0.87,
                "probabilities": [0.01, 0.02, 0.03, 0.07, 0.82, 0.05],
                "source_dataset": "lansinuote/ChnSentiCorp",
                "sample_weight": 0.3,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            split = _load_pseudo_label_split(path)
            self.assertEqual(len(split), 1)
            row = split[0]
            self.assertEqual(row["label"], 4)
            self.assertEqual(row["label_source"], "pseudo")
            self.assertEqual(len(row["soft_labels"]), NUM_SENTIMENT_CLASSES)
            self.assertAlmostEqual(row["sample_weight"], 0.3)

    def test_teacher_stage_rejects_binary_datasets(self) -> None:
        previous = os.environ.get("BERT_TEACHER_DATASETS")
        try:
            os.environ["BERT_TEACHER_DATASETS"] = "lansinuote/ChnSentiCorp"
            with self.assertRaises(ValueError):
                _teacher_dataset_names()
        finally:
            if previous is None:
                os.environ.pop("BERT_TEACHER_DATASETS", None)
            else:
                os.environ["BERT_TEACHER_DATASETS"] = previous

    def test_bert_selected_datasets_filter_stage_specific_sources(self) -> None:
        previous = {
            "BERT_TRAIN_DATASETS": os.environ.get("BERT_TRAIN_DATASETS"),
            "BERT_SELECTED_DATASETS": os.environ.get("BERT_SELECTED_DATASETS"),
            "BERT_TEACHER_DATASETS": os.environ.get("BERT_TEACHER_DATASETS"),
            "BERT_BINARY_PSEUDO_DATASETS": os.environ.get("BERT_BINARY_PSEUDO_DATASETS"),
        }
        try:
            os.environ["BERT_TRAIN_DATASETS"] = "BerlinWang/DMSC,XiangPan/waimai_10k,ttxy/online_shopping_10_cats"
            os.environ.pop("BERT_SELECTED_DATASETS", None)
            os.environ.pop("BERT_TEACHER_DATASETS", None)
            os.environ.pop("BERT_BINARY_PSEUDO_DATASETS", None)

            self.assertEqual(_teacher_dataset_names(), ["BerlinWang/DMSC"])
            self.assertEqual(_binary_pseudo_dataset_names(), ["XiangPan/waimai_10k"])
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_real_and_pseudo_splits_have_compatible_bert_training_schema(self) -> None:
        real = Dataset.from_dict({"text": ["很差"], "label": [1], "_sf_label_source": ["normalized"]})
        real = _with_training_columns(real, label_source="real", sample_weight=1.0, source_dataset="BerlinWang/DMSC")

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pseudo_labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "row-1",
                        "text": "一般",
                        "score": 3,
                        "confidence": 0.8,
                        "probabilities": [0.02, 0.03, 0.10, 0.76, 0.07, 0.02],
                        "source_dataset": "XiangPan/waimai_10k",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            pseudo = _load_pseudo_label_split(path)

            from datasets import concatenate_datasets

            combined = concatenate_datasets([real, pseudo])
            records = list(BertCsvStreamDataset(combined, chunk_size=2))
            self.assertEqual(len(records), 2)
            by_source = {record["label_source"]: record for record in records}
            self.assertIsNone(by_source["real"]["soft_labels"])
            self.assertEqual(len(by_source["pseudo"]["soft_labels"]), NUM_SENTIMENT_CLASSES)


if __name__ == "__main__":
    unittest.main()
