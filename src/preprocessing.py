"""Configurable, geometry-preserving MRI preprocessing for research QC."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json, logging
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class PreprocessingConfig:
    """Intensity-only options. Analytical volumes are never resized."""
    canonical_orientation: bool = True
    apply_n4_bias_correction: bool = False
    brain_mask_percentile: float = 20.0
    minimum_foreground_voxels: int = 100
    normalization: str = "zscore"
    version: str = "checkpoint-1"

def validate_volume(data: np.ndarray, minimum_foreground_voxels: int = 100) -> None:
    """Fail clearly for malformed, empty, constant, or non-finite 3-D images."""
    if data.ndim != 3: raise ValueError(f"Expected 3-D MRI, received {data.shape}.")
    if not np.isfinite(data).all(): raise ValueError("MRI contains NaN or infinite intensities.")
    if np.count_nonzero(data) < minimum_foreground_voxels: raise ValueError("MRI contains too few non-zero voxels.")
    if float(data.max()) == float(data.min()): raise ValueError("MRI has constant intensity.")

def load_and_canonicalize(path: str | Path, canonical: bool = True) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load NIfTI and optionally reorient to RAS+ without changing sampling."""
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    image = nib.load(str(path))
    if len(image.shape) != 3: raise ValueError(f"Expected one 3-D NIfTI volume, found {image.shape}.")
    image = nib.as_closest_canonical(image) if canonical else image
    data = image.get_fdata(dtype=np.float32); validate_volume(data)
    LOGGER.info("Loaded %s: shape=%s spacing=%s", path, data.shape, image.header.get_zooms()[:3])
    return image, data

def n4_bias_correct(data: np.ndarray) -> np.ndarray:
    """Apply N4 to intensities only; NIfTI geometry remains with the caller."""
    try: import SimpleITK as sitk
    except ImportError as error: raise RuntimeError("N4 requested but SimpleITK is not installed.") from error
    image = sitk.GetImageFromArray(data.astype(np.float32)); mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter(); corrector.SetMaximumNumberOfIterations([50]*4)
    LOGGER.info("Applied N4 bias correction.")
    return sitk.GetArrayFromImage(corrector.Execute(image, mask)).astype(np.float32)

def candidate_brain_mask(data: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """Conservative intensity-derived foreground mask, not validated skull stripping."""
    nonzero = data[data != 0]
    if nonzero.size == 0: raise ValueError("Cannot derive brain mask from empty volume.")
    mask = data > np.percentile(nonzero, percentile)
    if not mask.any(): raise ValueError("Candidate brain mask is empty.")
    LOGGER.info("Created candidate brain mask (percentile=%.1f, voxels=%d).", percentile, mask.sum())
    return mask

def normalize_intensity(data: np.ndarray, mask: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize inside mask while preserving shape, affine and voxel spacing."""
    values = data[mask]
    if values.size == 0: raise ValueError("Normalization mask contains no voxels.")
    result = np.zeros_like(data, dtype=np.float32)
    if method == "zscore": result[mask] = (values-values.mean())/(values.std()+1e-8)
    elif method == "minmax": result[mask] = (values-values.min())/(values.max()-values.min()+1e-8)
    else: raise ValueError(f"Unsupported normalization method: {method}")
    LOGGER.info("Applied %s normalization.", method); return result

def save_processed_volume(data: np.ndarray, reference: nib.Nifti1Image, output_path: str | Path) -> Path:
    """Save processed data with reference affine/header untouched."""
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, reference.affine, reference.header.copy()), str(output_path)); return output_path

def preprocessing_metadata(config: PreprocessingConfig, image: nib.Nifti1Image, source_path: str | Path) -> dict[str, Any]:
    """Create reproducible processing provenance."""
    return {"source_mri":str(source_path),"original_shape":list(image.shape),"voxel_spacing_mm":list(image.header.get_zooms()[:3]),"affine":image.affine.tolist(),"config":asdict(config)}

def write_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write provenance JSON."""
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(metadata,indent=2),encoding="utf-8"); return path
