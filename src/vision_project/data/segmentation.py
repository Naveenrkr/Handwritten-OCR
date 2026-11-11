"""Document Line Segmentation using OpenCV Horizontal Projection Profile (HPP).

Extracts individual text lines from full-page document images by:
  1. Converting to grayscale
  2. Applying Gaussian blur
  3. Applying Otsu's binarization (inverted)
  4. Computing horizontal projection profile (row-wise foreground pixel sums)
  5. Detecting contiguous line bands and cropping the original image
"""

from pathlib import Path
from typing import List, Union
import numpy as np


def segment_document_lines(
    image_input: Union[str, Path, np.ndarray],
    blur_ksize: int = 5,
    min_line_height: int = 8,
    min_profile_threshold: float = 0.05,
    margin: int = 2,
) -> List[np.ndarray]:
    """Segment a full document image into individual text line crops.

    Args:
        image_input: Path to document image file, or an already loaded numpy array.
        blur_ksize: Kernel size for Gaussian blur (must be odd, default: 5).
        min_line_height: Minimum pixel height for a detected band to be considered a valid text line.
        min_profile_threshold: Relative threshold (fraction of peak row density) to separate text from background.
        margin: Vertical padding (in pixels) added to each cropped text line.

    Returns:
        List of cropped single-line image arrays (numpy ndarrays).
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    # 1. Load image if a file path was provided
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if has_cv2:
            original_img = cv2.imread(path_str)
        else:
            raise ImportError(
                "OpenCV is required to read image files. Install it via `uv sync` or `pip install opencv-python`."
            )
        if original_img is None:
            raise FileNotFoundError(f"Failed to read image at '{path_str}'. Verify file exists.")
    elif isinstance(image_input, np.ndarray):
        original_img = image_input.copy()
    else:
        raise TypeError(f"Expected image path or np.ndarray, got {type(image_input)}")

    # 2. Convert to grayscale
    if original_img.ndim == 3:
        if has_cv2:
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.dot(original_img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    else:
        gray = original_img.copy()

    img_h, img_w = gray.shape[:2]

    # 3. Apply Gaussian blur & Otsu thresholding
    if has_cv2:
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # Pure NumPy fallback for Otsu thresholding if cv2 is not yet installed
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size
        curr_max, threshold_otsu = 0.0, 128
        sum_total = np.dot(np.arange(256), hist)
        sum_b, w_b = 0.0, 0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = float(w_b) * float(w_f) * ((m_b - m_f) ** 2)
            if var_between > curr_max:
                curr_max = var_between
                threshold_otsu = t
        binary = np.where(gray <= threshold_otsu, 255, 0).astype(np.uint8)

    # 5. Compute horizontal projection profile (count of foreground pixels along each row)
    hpp = np.sum(binary > 0, axis=1)

    # 6. Determine dynamic threshold for text row separation
    max_count = np.max(hpp) if len(hpp) > 0 else 0
    if max_count == 0:
        # Blank document
        return [original_img]

    threshold = max(3, int(max_count * min_profile_threshold))

    # 7. Identify line boundaries (Y-coordinates where profile exceeds threshold)
    lines: List[np.ndarray] = []
    in_line = False
    start_y = 0

    for y in range(img_h):
        if hpp[y] >= threshold and not in_line:
            in_line = True
            start_y = y
        elif hpp[y] < threshold and in_line:
            in_line = False
            end_y = y
            # Verify minimum height constraint to discard speckles/noise
            if (end_y - start_y) >= min_line_height:
                y1 = max(0, start_y - margin)
                y2 = min(img_h, end_y + margin)
                cropped_line = original_img[y1:y2, :]
                lines.append(cropped_line)

    # Handle case where the last line extends to bottom of the page
    if in_line and (img_h - start_y) >= min_line_height:
        y1 = max(0, start_y - margin)
        y2 = img_h
        lines.append(original_img[y1:y2, :])

    # Fallback to full image if no distinct lines were segmented
    if not lines:
        lines = [original_img]

    return lines
