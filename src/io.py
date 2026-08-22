"""NIfTI I/O with explicit spatial metadata preservation."""
from pathlib import Path
from typing import Union
import nibabel as nib
import numpy as np

PathLike = Union[str, Path]

def load_nifti(path: PathLike, canonical: bool = True) -> nib.Nifti1Image:
    """Load an image, optionally reorienting its voxel array to RAS+."""
    image = nib.load(str(path))
    return nib.as_closest_canonical(image) if canonical else image

def save_nifti(data: np.ndarray, reference: nib.spatialimages.SpatialImage, path: PathLike) -> None:
    """Save data using the reference affine and header; never silently lose geometry."""
    output = nib.Nifti1Image(data, reference.affine, reference.header.copy())
    nib.save(output, str(path))
