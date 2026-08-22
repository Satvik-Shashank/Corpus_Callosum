"""Model inference contracts for binary corpus callosum segmentation."""
from typing import Protocol
import numpy as np

class CCSegmenter(Protocol):
    """An implementation must return a binary mask aligned to its input image."""
    def predict(self, sagittal_image: np.ndarray) -> np.ndarray: ...

def validate_binary_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Validate a prediction before morphology is calculated."""
    if mask.shape != image_shape: raise ValueError("Mask and image shape differ.")
    if not np.isin(mask, [0, 1, False, True]).all(): raise ValueError("CC mask must be binary.")
    return mask.astype(bool)
