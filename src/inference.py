"""Composable, QC-oriented inference orchestration."""
from dataclasses import dataclass
import numpy as np
from .segmentation import CCSegmenter, validate_binary_mask
from .morphology import Morphology, measure

@dataclass(frozen=True)
class InferenceResult:
    mask: np.ndarray
    morphology: Morphology
    disclaimer: str = "AI-assisted research output; requires expert quality control and is not diagnostic."

def run_cc_inference(sagittal_image: np.ndarray, spacing_mm: tuple[float, float], segmenter: CCSegmenter) -> InferenceResult:
    """Segment one selected plane and derive measurements only after validation."""
    mask = validate_binary_mask(segmenter.predict(sagittal_image), sagittal_image.shape)
    return InferenceResult(mask, measure(mask, spacing_mm))
