"""CC localization interfaces; no heuristic output is presented as a validated mask."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class BoundingBox:
    y0: int; x0: int; y1: int; x1: int

class Localizer(Protocol):
    def predict(self, sagittal_image: np.ndarray) -> BoundingBox: ...

def crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop a 2-D image after bounds validation."""
    if not (0 <= box.y0 < box.y1 <= image.shape[0] and 0 <= box.x0 < box.x1 <= image.shape[1]):
        raise ValueError("Bounding box lies outside image bounds.")
    return image[box.y0:box.y1, box.x0:box.x1]
