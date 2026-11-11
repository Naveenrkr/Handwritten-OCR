"""Image Preprocessing and Augmentation Pipeline.

Includes OpenCV preprocessing for CRNN text recognition:
  - Grayscale conversion
  - Adaptive thresholding (binarization)
  - Aspect-ratio preserving resize to height 32
  - Right-padding with white pixels (value 255) to fixed maximum tensor width
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch


def preprocess_image_crnn(
    image_input: Union[str, Path, np.ndarray],
    max_width: int = 128,
    target_height: int = 32,
    block_size: int = 11,
    c: int = 2,
    pad_value: int = 255,
) -> np.ndarray:
    """Preprocess an image for CRNN text recognition.

    Steps:
      1. Load image (or accept np.ndarray) and convert to grayscale.
      2. Apply Gaussian adaptive thresholding.
      3. Resize height to exactly target_height (32) while preserving aspect ratio.
      4. Pad width with white pixels (pad_value=255) up to max_width.

    Args:
        image_input: File path to input image, or an existing numpy array (e.g., cropped line).
        max_width: Fixed maximum tensor width (default: 128).
        target_height: Fixed height for the CRNN input (must be 32).
        block_size: Neighborhood block size for adaptive thresholding (odd int).
        c: Constant subtracted from mean in adaptive thresholding.
        pad_value: Pixel value used for padding (255 for white).

    Returns:
        A preprocessed numpy array of shape (target_height, max_width) with dtype uint8.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV is required for image preprocessing. Install it with `uv sync` or `pip install opencv-python`."
        ) from e

    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at '{path_str}'. Verify file exists and format is valid.")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError(f"Expected file path or np.ndarray, got {type(image_input)}")

    # Apply adaptive Gaussian thresholding
    thresh = cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )

    h, w = thresh.shape[:2]
    # Calculate new width maintaining aspect ratio with height 32
    ratio = float(target_height) / float(h)
    new_w = max(1, int(round(w * ratio)))

    # Cap at max_width if image aspect ratio exceeds tensor width
    if new_w > max_width:
        new_w = max_width

    # Resize with INTER_AREA if downsampling, INTER_LINEAR if upsampling
    interpolation = cv2.INTER_AREA if target_height < h else cv2.INTER_LINEAR
    resized = cv2.resize(thresh, (new_w, target_height), interpolation=interpolation)

    # Pad width with white pixels (255) to reach fixed max_width
    padded = np.full((target_height, max_width), pad_value, dtype=np.uint8)
    padded[:, :new_w] = resized

    return padded


def prepare_input_for_onnx(
    image: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert preprocessed uint8 array (32, W) to float32 NumPy tensor (1, 1, 32, W) for ONNX Runtime.

    Args:
        image: Preprocessed numpy array of shape (32, W).
        normalize: If True, normalizes pixel values to [-1.0, 1.0].

    Returns:
        np.ndarray of shape (1, 1, 32, W) with dtype np.float32.
    """
    tensor = np.expand_dims(image.astype(np.float32), axis=(0, 1))
    if normalize:
        tensor = (tensor / 255.0 - 0.5) / 0.5
    return tensor


def image_to_crnn_tensor(
    image: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert preprocessed uint8 array (32, W) to PyTorch tensor (1, 1, 32, W).

    Args:
        image: Preprocessed numpy array of shape (32, W).
        normalize: If True, normalizes pixel values to [-1.0, 1.0] range.

    Returns:
        torch.Tensor of shape (1, 1, 32, W) ready for CRNN forward pass.
    """
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 32, W)
    if normalize:
        # Standard normalization for grayscale document images
        tensor = (tensor / 255.0 - 0.5) / 0.5
    return tensor
