"""Physical-unit measurements from a validated mid-sagittal CC mask."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Morphology:
    area_mm2: float
    anterior_posterior_length_mm: float

def measure(mask: np.ndarray, pixel_spacing_mm: tuple[float, float]) -> Morphology:
    """Compute basic measurements. Mask validity and plane selection remain caller responsibilities."""
    if mask.ndim != 2 or not mask.any(): raise ValueError("A non-empty 2-D CC mask is required.")
    row_spacing, col_spacing = pixel_spacing_mm
    cols = np.where(mask)[1]
    return Morphology(float(mask.sum() * row_spacing * col_spacing), float((cols.max()-cols.min()+1)*col_spacing))
