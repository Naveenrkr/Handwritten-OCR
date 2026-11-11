"""CLI Entry Point for Vision Project.

Built with Typer.
Uses ONNX Runtime for lightweight, instantaneous full-page text recognition inference
without loading the heavy PyTorch runtime into memory.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import typer

from vision_project.data.dataset import DEFAULT_VOCABULARY
from vision_project.data.preprocessing import (
    preprocess_image_crnn,
    prepare_input_for_onnx,
)
from vision_project.data.segmentation import segment_document_lines

app = typer.Typer(
    name="vision-cli",
    help="Fast OCR CLI using ONNX Runtime and OpenCV line segmentation (optimized for low memory).",
)


def ctc_greedy_decode(
    logits: np.ndarray,
    alphabet: str = DEFAULT_VOCABULARY,
    blank_idx: int = 0,
) -> str:
    """Perform greedy CTC decoding on ONNX model output sequence using NumPy.

    Collapses contiguous duplicate tokens and removes the blank token.

    Args:
        logits: Output array from CRNN of shape (T, B, num_classes) or (T, num_classes).
        alphabet: Target alphabet string (blank token is at index 0).
        blank_idx: Index representing CTC blank symbol (default: 0).

    Returns:
        Decoded text string.
    """
    # Argmax over the vocabulary / class dimension
    best_path = np.argmax(logits, axis=-1)  # shape (T, B) or (T,)
    if best_path.ndim == 2:
        best_path = best_path[:, 0]  # Take first element in batch

    decoded_chars = []
    prev_idx = None

    for idx in best_path.tolist():
        # Collapse repeat characters and drop blank token
        if idx != blank_idx and idx != prev_idx:
            char_idx = idx - 1 if blank_idx == 0 else idx
            if 0 <= char_idx < len(alphabet):
                decoded_chars.append(alphabet[char_idx])
        prev_idx = idx

    return "".join(decoded_chars)


@app.command()
def predict(
    image: str = typer.Option(
        ..., "--image", "-i", help="Path to the document image file for OCR prediction"
    ),
    model: str = typer.Option(
        "crnn.onnx", "--model", "-m", help="Path to exported ONNX model file (.onnx)"
    ),
    max_width: int = typer.Option(
        128, "--max-width", "-w", help="Fixed maximum tensor width for preprocessing"
    ),
    alphabet: str = typer.Option(
        DEFAULT_VOCABULARY, "--alphabet", help="Target character alphabet"
    ),
) -> None:
    """Predict text from a document image using OpenCV line segmentation and ONNX Runtime."""
    image_path = Path(image)
    if not image_path.exists():
        typer.secho(f"Error: Image file does not exist at '{image}'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    model_path = Path(model)
    if not model_path.exists():
        typer.secho(
            f"Error: ONNX model not found at '{model}'.\n"
            f"Please export your CRNN model first using: python export_onnx.py --output {model}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # 1. Initialize ONNX Runtime InferenceSession
    try:
        import onnxruntime as ort
    except ImportError as exc:
        typer.secho(
            "Error: 'onnxruntime' is not installed. Install it with `uv sync` or `pip install onnxruntime`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    # Prefer CPUExecutionProvider for instantaneous startup and minimal memory footprint
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    typer.echo(f"Loaded ONNX model: {model_path}")
    typer.echo(f"Processing document: {image_path}")

    # 2. Segment document image into individual text lines using OpenCV Horizontal Projection Profile
    try:
        lines: List[np.ndarray] = segment_document_lines(image_path)
    except Exception as exc:
        typer.secho(f"Segmentation error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Detected {len(lines)} text line(s). Running ONNX recognition...\n", fg=typer.colors.CYAN)

    decoded_lines: List[str] = []

    # 3. Iterate over each cropped line and run inference
    for line_idx, line_crop in enumerate(lines, 1):
        # Preprocess line: grayscale, adaptive threshold, aspect-ratio resize to height 32, white padding
        preprocessed = preprocess_image_crnn(
            image_input=line_crop,
            max_width=max_width,
            target_height=32,
        )

        # Prepare normalized float32 tensor of shape (1, 1, 32, max_width)
        input_tensor = prepare_input_for_onnx(preprocessed)

        # Run ONNX Runtime session
        outputs = session.run([output_name], {input_name: input_tensor})
        logits = outputs[0]  # shape: (sequence_length, 1, num_classes)

        # Decode logits with greedy CTC decoder
        text = ctc_greedy_decode(logits, alphabet=alphabet, blank_idx=0)
        decoded_lines.append(text)
        typer.echo(f"  Line {line_idx:02d}: {text}")

    # 4. Print combined full document text
    typer.secho("\n--- Full Document Transcription ---", fg=typer.colors.BRIGHT_BLUE, bold=True)
    full_text = "\n".join(decoded_lines)
    typer.secho(full_text if full_text.strip() else "(No text detected)", fg=typer.colors.GREEN, bold=True)


@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir", "-d", help="Path to training dataset"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Micro-batch size (16 recommended for 6GB VRAM)"),
    grad_accum_steps: int = typer.Option(2, "--accum-steps", help="Gradient accumulation steps to simulate larger batch"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Enable torch.amp mixed precision (saves ~50% VRAM)"),
) -> None:
    """Train the model with 6GB VRAM hardware-conscious settings."""
    typer.echo(f"Initializing training with batch_size={batch_size}, accum_steps={grad_accum_steps}, amp={amp}...")


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model weights checkpoint"),
    data_dir: str = typer.Option(..., "--data-dir", "-d", help="Path to evaluation dataset"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Evaluation batch size"),
) -> None:
    """Evaluate model checkpoint."""
    typer.echo(f"Evaluating checkpoint {checkpoint}...")


if __name__ == "__main__":
    app()
