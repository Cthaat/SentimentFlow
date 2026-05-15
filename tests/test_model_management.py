from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.api.models import delete_model  # noqa: E402


class ModelDeleteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_project_root = os.environ.get("SENTIMENTFLOW_PROJECT_ROOT")
        self._temp_dir = TemporaryDirectory()
        os.environ["SENTIMENTFLOW_PROJECT_ROOT"] = self._temp_dir.name
        self.models_dir = Path(self._temp_dir.name) / "models"
        self.models_dir.mkdir()

    def tearDown(self) -> None:
        if self._old_project_root is None:
            os.environ.pop("SENTIMENTFLOW_PROJECT_ROOT", None)
        else:
            os.environ["SENTIMENTFLOW_PROJECT_ROOT"] = self._old_project_root
        self._temp_dir.cleanup()

    def test_delete_model_removes_directory_and_empty_leftovers(self) -> None:
        model_dir = self.models_dir / "lstm_20260515_120000"
        model_dir.mkdir()
        (model_dir / "model.pt").write_bytes(b"checkpoint")

        stale_empty_dir = self.models_dir / "lstm_20260515_121000"
        stale_empty_dir.mkdir()

        result = delete_model("lstm_20260515_120000")

        self.assertEqual(result, {"ok": True, "model_id": "lstm_20260515_120000"})
        self.assertFalse(model_dir.exists())
        self.assertFalse(stale_empty_dir.exists())
        self.assertTrue(self.models_dir.exists())

    def test_delete_model_accepts_empty_residual_directory(self) -> None:
        empty_model_dir = self.models_dir / "bert_20260515_130000"
        empty_model_dir.mkdir()

        result = delete_model("bert_20260515_130000")

        self.assertEqual(result, {"ok": True, "model_id": "bert_20260515_130000"})
        self.assertFalse(empty_model_dir.exists())

    def test_delete_model_rejects_path_traversal(self) -> None:
        with self.assertRaises(HTTPException) as exc:
            delete_model("../outside")

        self.assertEqual(exc.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
