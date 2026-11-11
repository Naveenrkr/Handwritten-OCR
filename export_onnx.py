"""Script to export trained PyTorch CRNN model to ONNX format.

Configures dynamic axes for the width dimension so inference handles varying text lengths,
drastically reducing memory overhead when deployed with ONNX Runtime.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import torch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vision_project.models.architecture import CRNN
from vision_project.data.dataset import DEFAULT_VOCABULARY


def export_crnn_to_onnx(
    checkpoint_path: Optional[str] = None,
    output_onnx_path: str = "crnn.onnx",
    num_classes: int = len(DEFAULT_VOCABULARY) + 1,
    opset_version: int = 14,
) -> Path:
    """Load PyTorch CRNN weights and export to ONNX format.

    Args:
        checkpoint_path: Path to trained PyTorch weights (.pt or .pth). If None or file
                         does not exist, exports initialized model architecture.
        output_onnx_path: Destination path for the exported .onnx file (default: 'crnn.onnx').
        num_classes: Total vocabulary size including the CTC blank token (default: 37).
        opset_version: ONNX operator set version (default: 14).

    Returns:
        Path to the exported ONNX model file.
    """
    out_path = Path(output_onnx_path)

    # 1. Initialize CRNN model architecture
    model = CRNN(img_h=32, in_channels=1, num_classes=num_classes)

    # 2. Load trained weights if checkpoint exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[ONNX Export] Loading weights from '{checkpoint_path}'...")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("[ONNX Export] No checkpoint found or provided; exporting model architecture with initial weights.")

    model.eval()

    # 3. Define dummy input tensor of shape (1, 1, 32, 128) representing a single grayscale padded line
    dummy_input = torch.randn(1, 1, 32, 128, dtype=torch.float32)

    # 4. Set dynamic axes for the width dimension so CLI handles varying word lengths during inference
    # Input shape: (batch_size, 1, 32, width)
    # Output shape: (sequence_length, batch_size, num_classes)
    dynamic_axes = {
        "input": {0: "batch_size", 3: "width"},
        "output": {0: "sequence_length", 1: "batch_size"},
    }

    print(f"[ONNX Export] Exporting to '{out_path}' with dynamic axes...")

    # 5. Export using torch.onnx.export
    try:
        # Use TorchScript exporter (dynamo=False) for maximum backward compatibility
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print(f"[ONNX Export] Success! Saved ONNX model to '{out_path}'.")
    except Exception as exc:
        print(f"[ONNX Export] Note: Export encountered an error: {exc}")
        print("Ensure 'onnx' is installed in your uv environment (`uv sync` or `uv pip install onnx`).")
        raise

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch CRNN to ONNX format")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="checkpoints/crnn_iam.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="crnn.onnx",
        help="Path for exported ONNX model (default: crnn.onnx)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=len(DEFAULT_VOCABULARY) + 1,
        help="Number of classes including blank token (default: 37)",
    )

    args = parser.parse_args()
    export_crnn_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        num_classes=args.classes,
    )
