"""BERT deployment export helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import MAX_LEN
from .inference import export_onnx, prepare_inference_model


def export_onnx_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    *,
    device: torch.device | None = None,
    max_len: int = MAX_LEN,
    opset_version: int = 17,
) -> Path:
    """Load a fine-tuned BERT checkpoint and export ONNX logits."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, device=resolved_device)
    if model is None:
        raise FileNotFoundError(f"BERT checkpoint is not loadable: {checkpoint_path}")
    model = prepare_inference_model(model, resolved_device, compile_model=False, quantize=False)
    return export_onnx(
        model,
        output_path,
        resolved_device,
        max_len=max_len,
        opset_version=opset_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SentimentFlow BERT checkpoint to ONNX.")
    parser.add_argument("checkpoint_path", help="Fine-tuned BERT checkpoint directory.")
    parser.add_argument("output_path", help="Output .onnx path.")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cpu", action="store_true", help="Force CPU export.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    output = export_onnx_from_checkpoint(
        args.checkpoint_path,
        args.output_path,
        device=device,
        max_len=args.max_len,
        opset_version=args.opset,
    )
    print(f"ONNX exported: {output}")


if __name__ == "__main__":
    main()
