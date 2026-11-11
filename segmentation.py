"""Root segmentation interface forwarding to vision_project.data.segmentation."""

import sys
from pathlib import Path

# Add src to sys.path if not present
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from vision_project.data.segmentation import segment_document_lines

__all__ = ["segment_document_lines"]
