"""Non-destructive QC overlay creation."""
import numpy as np

def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an RGB overlay; input intensities are only used for display."""
    if image.shape != mask.shape: raise ValueError("Image and mask shapes differ.")
    scaled = (255*(image-image.min())/(image.max()-image.min()+1e-8)).astype(np.uint8)
    result = np.repeat(scaled[...,None],3,axis=2); result[mask.astype(bool)] = (255,0,0)
    return result
