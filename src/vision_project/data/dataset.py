"""IAM Handwriting Dataset and DataLoader utilities.

Implements custom PyTorch Dataset for IAM words.txt parsing,
character-to-integer mapping, and sequence preparation for CTC loss.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from vision_project.data.preprocessing import (
    preprocess_image_crnn,
    image_to_crnn_tensor,
)

# Default vocabulary for alphanumeric scene and handwriting text
DEFAULT_VOCABULARY = "0123456789abcdefghijklmnopqrstuvwxyz"


class IAMDataset(Dataset):
    """Custom PyTorch Dataset for the IAM Handwriting Database.

    Parses 'words.txt', skips comment lines, resolves image file paths,
    and returns preprocessed image tensors, integer-encoded text sequences,
    and sequence lengths required for CTC loss.
    """

    def __init__(
        self,
        words_file: Union[str, Path],
        img_dir: Union[str, Path],
        vocabulary: Optional[Union[str, Sequence[str]]] = None,
        max_width: int = 128,
        target_height: int = 32,
        case_sensitive: bool = False,
        filter_errors: bool = True,
    ) -> None:
        """Initialize IAMDataset.

        Args:
            words_file: Path to 'words.txt' transcription index.
            img_dir: Base directory containing IAM word images.
            vocabulary: Character vocabulary (default: alphanumeric 0-9a-z).
            max_width: Target padded width for preprocessed image tensors.
            target_height: Target height for CRNN input (must be 32).
            case_sensitive: Whether to maintain letter casing.
            filter_errors: If True, only keep entries where segmentation status is 'ok'.
        """
        super().__init__()
        self.words_file = Path(words_file)
        self.img_dir = Path(img_dir)
        self.max_width = max_width
        self.target_height = target_height
        self.case_sensitive = case_sensitive
        self.filter_errors = filter_errors

        # ---------------------------------------------------------------------
        # Vocabulary mapping (Blank token index is 0 for CTC Loss)
        # ---------------------------------------------------------------------
        self.vocab: str = (
            "".join(vocabulary) if vocabulary is not None else DEFAULT_VOCABULARY
        )
        self.char_to_idx: Dict[str, int] = {
            char: idx + 1 for idx, char in enumerate(self.vocab)
        }
        self.idx_to_char: Dict[int, str] = {
            idx + 1: char for idx, char in enumerate(self.vocab)
        }
        self.blank_idx: int = 0
        self.num_classes: int = len(self.vocab) + 1  # Includes CTC blank token

        self.samples: List[Tuple[Path, str]] = []
        if self.words_file.exists():
            self._parse_words_file()

    def _resolve_image_path(self, word_id: str) -> Path:
        """Resolve IAM hierarchical or flat directory structure for a word_id.

        Hierarchy in IAM:
            word_id: 'a01-000u-00-00'
            path: <img_dir>/words/a01/a01-000u/a01-000u-00-00.png
                  or <img_dir>/a01/a01-000u/a01-000u-00-00.png
        """
        parts = word_id.split("-")
        if len(parts) >= 2:
            part0 = parts[0]
            part1 = f"{parts[0]}-{parts[1]}"
            filename = f"{word_id}.png"

            candidate1 = self.img_dir / part0 / part1 / filename
            if candidate1.exists():
                return candidate1

            candidate2 = self.img_dir / "words" / part0 / part1 / filename
            if candidate2.exists():
                return candidate2

        # Flat fallback
        flat_candidate = self.img_dir / f"{word_id}.png"
        if flat_candidate.exists():
            return flat_candidate

        # Default canonical hierarchical path
        return self.img_dir / parts[0] / f"{parts[0]}-{parts[1]}" / f"{word_id}.png"

    def _parse_words_file(self) -> None:
        """Parse words.txt line by line, skipping comments and error segments."""
        with open(self.words_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                # Expected format: word_id status graylevel x y w h tag transcription...
                if len(parts) < 9:
                    continue

                word_id = parts[0]
                status = parts[1]

                # Filter segmentation errors if requested
                if self.filter_errors and status != "ok":
                    continue

                # Ground truth transcription (join remainder in case of spaces)
                gt_text = " ".join(parts[8:])
                if not self.case_sensitive:
                    gt_text = gt_text.lower()

                # Filter characters to target vocabulary
                valid_text = "".join(c for c in gt_text if c in self.char_to_idx)
                if not valid_text:
                    continue

                img_path = self._resolve_image_path(word_id)
                self.samples.append((img_path, valid_text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and return sample tensors for CTC training.

        Returns:
            image_tensor: Normalized float tensor of shape (1, 32, max_width).
            target: Long tensor of character indices of shape (seq_len,).
            input_length: Scalar tensor of input sequence length T after CNN downsampling.
            target_length: Scalar tensor of target text length.
        """
        img_path, text = self.samples[idx]

        # 1. Load and binarize/resize using OpenCV preprocessor
        preprocessed = preprocess_image_crnn(
            image_path=img_path,
            max_width=self.max_width,
            target_height=self.target_height,
        )

        # Convert to float tensor of shape (1, 32, max_width)
        image_tensor = image_to_crnn_tensor(preprocessed).squeeze(0)

        # 2. Integer-encoded text sequence
        target = torch.tensor(
            [self.char_to_idx[c] for c in text],
            dtype=torch.long,
        )

        # 3. Sequence lengths for CTC Loss:
        # With 7 CNN layers and input width max_width, output sequence length T is:
        # T = max_width // 4 + 1  (e.g., 128 // 4 + 1 = 33)
        input_length = torch.tensor(self.max_width // 4 + 1, dtype=torch.long)
        target_length = torch.tensor(len(target), dtype=torch.long)

        return image_tensor, target, input_length, target_length


def iam_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that pads target sequences to the maximum length in the batch.

    Args:
        batch: List of (image_tensor, target, input_length, target_length) tuples.

    Returns:
        images: Stacked tensor of shape (B, 1, 32, max_width).
        padded_targets: Padded tensor of shape (B, max_target_length).
        input_lengths: Tensor of shape (B,) containing input sequence lengths.
        target_lengths: Tensor of shape (B,) containing target sequence lengths.
    """
    images, targets, input_lengths, target_lengths = zip(*batch)

    # Stack images into batch tensor: (B, 1, 32, max_width)
    images = torch.stack(images, dim=0)
    input_lengths = torch.stack(input_lengths, dim=0)
    target_lengths = torch.stack(target_lengths, dim=0)

    # Pad target sequences to the maximum length in the current batch
    max_target_len = max(len(t) for t in targets)
    batch_size = len(targets)
    padded_targets = torch.zeros((batch_size, max_target_len), dtype=torch.long)

    for i, target in enumerate(targets):
        padded_targets[i, : len(target)] = target

    return images, padded_targets, input_lengths, target_lengths
