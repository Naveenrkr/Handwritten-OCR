"""Root data module forwarding to vision_project.data."""

import sys
from pathlib import Path

# Add src to sys.path if not present
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from vision_project.data.dataset import (
    IAMDataset,
    iam_collate_fn,
    DEFAULT_VOCABULARY,
)
from vision_project.data.preprocessing import (
    preprocess_image_crnn,
    image_to_crnn_tensor,
)

__all__ = [
    "IAMDataset",
    "iam_collate_fn",
    "DEFAULT_VOCABULARY",
    "preprocess_image_crnn",
    "image_to_crnn_tensor",
]
