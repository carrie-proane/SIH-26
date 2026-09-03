"""Frame extraction, quality scoring, selection, and review artifacts."""

from .extractor import ExtractedFrame, ExtractionResult, extract_frames
from .selector import SelectionWeights, select_keyframes

__all__ = [
    "ExtractedFrame",
    "ExtractionResult",
    "SelectionWeights",
    "extract_frames",
    "select_keyframes",
]
