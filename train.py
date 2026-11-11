"""Training script for CRNN on IAM Handwriting Database.

Optimized for 6GB VRAM GPU memory envelopes:
  - Small micro-batch size (default 16)
  - Automatic mixed precision (AMP)
  - Gradient clipping to prevent exploding gradients in bidirectional LSTM
  - Target sequence padding via custom collate_fn
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vision_project.models.architecture import CRNN
from vision_project.data.dataset import IAMDataset, iam_collate_fn, DEFAULT_VOCABULARY


def train(
    words_file: str,
    img_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_clip_norm: float = 5.0,
    save_path: str = "checkpoints/crnn_iam.pt",
    use_amp: bool = True,
    device_name: str = "auto",
) -> None:
    """Run PyTorch CRNN training loop on IAM Handwriting Dataset."""
    # 1. Device selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[Device] Using compute device: {device}")

    # 2. Dataset & Collate Function
    dataset = IAMDataset(
        words_file=words_file,
        img_dir=img_dir,
        vocabulary=DEFAULT_VOCABULARY,
        max_width=128,
        target_height=32,
    )

    if len(dataset) == 0:
        print(f"[Warning] No samples found in '{words_file}'.")
        print("To run with real data, download the IAM Handwriting Database and extract words.txt & word images.")
        return

    print(f"[Dataset] Loaded {len(dataset)} word samples. Vocabulary size: {len(dataset.vocab)} (+1 blank = {dataset.num_classes} classes).")

    # 3. DataLoader with custom collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=iam_collate_fn,
        num_workers=2 if os.name != "nt" else 0,  # 0 on Windows avoids spawn overhead for quick testing
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # 4. Model Initialization
    model = CRNN(
        img_h=32,
        in_channels=1,
        num_classes=dataset.num_classes,
        hidden_size=256,
    ).to(device)

    # 5. Loss & Optimizer
    # zero_infinity=True zeroes infinite losses to prevent NaN gradients when targets are too long
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Enable mixed precision scaler for CUDA (saves ~50% VRAM on 6GB GPUs)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # Create checkpoint directory
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Training] Starting {epochs} epochs | Batch Size: {batch_size} | Gradient Clip: {max_clip_norm}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(dataloader):
            # Move all tensors to target device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                # log_probs shape: (T, B, num_classes)
                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Backward pass with gradient clipping
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # Unscale gradients prior to clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Prevent exploding gradients in BiLSTM
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch}/{epochs}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(1, num_batches)
        print(f"==> Epoch [{epoch}/{epochs}] Average CTC Loss: {epoch_loss:.4f}")

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), save_path)

    print("-" * 65)
    print(f"[Done] Training complete. Saved model checkpoint to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN on IAM Handwriting Database (6GB VRAM safe)")
    parser.add_argument("--words-file", type=str, default="data/words.txt", help="Path to IAM words.txt index")
    parser.add_argument("--img-dir", type=str, default="data/words", help="Path to IAM word images directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (small 16 for 6GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip-grad", type=float, default=5.0, help="Max gradient norm for LSTM clipping")
    parser.add_argument("--save-path", type=str, default="checkpoints/crnn_iam.pt", help="Path to save model weights")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    args = parser.parse_args()

    train(
        words_file=args.words_file,
        img_dir=args.img_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_clip_norm=args.clip_grad,
        save_path=args.save_path,
        use_amp=not args.no_amp,
        device_name=args.device,
    )
