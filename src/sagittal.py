"""Robust mid-sagittal candidate selection and display-only quality control."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json, logging
from pathlib import Path
from typing import Optional
import numpy as np
LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class SagittalConfig:
    search_radius: int = 5
    minimum_mask_pixels: int = 50
    display_max_dimension: int = 768
@dataclass(frozen=True)
class SagittalSelection:
    index: int
    anatomical_center_index: int
    score: float
    search_indices: tuple[int,...]

def _score(plane: np.ndarray, mask: Optional[np.ndarray], minimum_pixels: int) -> float:
    width = plane.shape[1]//2; left,right=plane[:,:width],np.fliplr(plane[:,-width:]); valid=np.isfinite(left)&np.isfinite(right)
    if mask is not None: valid &= mask[:,:width]&np.fliplr(mask[:,-width:])
    if valid.sum() < minimum_pixels: return np.inf
    return float(np.median(np.abs(left[valid]-right[valid]))/(np.median(np.abs(plane[:,:width][valid]))+1e-6))

def select_mid_sagittal(volume: np.ndarray, brain_mask: Optional[np.ndarray] = None, config: SagittalConfig = SagittalConfig()) -> tuple[np.ndarray,SagittalSelection]:
    """Select most symmetric nearby RAS sagittal plane at native resolution."""
    if volume.ndim != 3: raise ValueError("Expected a 3-D RAS volume.")
    if brain_mask is not None and brain_mask.shape != volume.shape: raise ValueError("Brain mask and volume shapes differ.")
    centre=volume.shape[0]//2; indices=tuple(range(max(0,centre-config.search_radius),min(volume.shape[0],centre+config.search_radius+1)))
    scores=[_score(volume[i],None if brain_mask is None else brain_mask[i],config.minimum_mask_pixels) for i in indices]
    if not np.isfinite(scores).any(): raise ValueError("No candidate had enough paired foreground pixels.")
    local=int(np.nanargmin(scores)); selection=SagittalSelection(indices[local],centre,float(scores[local]),indices)
    LOGGER.info("Selected sagittal index %d (centre=%d score=%.5f).",selection.index,centre,selection.score)
    return volume[selection.index],selection

def symmetry_mid_sagittal(volume: np.ndarray, search_radius: int = 5) -> tuple[np.ndarray,int]:
    """Backward-compatible unmasked symmetry selection."""
    plane,selection=select_mid_sagittal(volume,config=SagittalConfig(search_radius=search_radius)); return plane,selection.index

def display_image(plane: np.ndarray) -> np.ndarray:
    """Contrast-scale only for display; this never changes analytical data."""
    finite=plane[np.isfinite(plane)]; low,high=np.percentile(finite,(2,98)); return (255*np.clip((plane-low)/(high-low+1e-8),0,1)).astype(np.uint8)

def save_sagittal_qc(original: np.ndarray, processed: np.ndarray, output_dir: str|Path, subject_id: str, selection: SagittalSelection, config: SagittalConfig) -> dict[str,Path]:
    """Save original, processed and side-by-side QC PNGs plus selection metadata."""
    try: import matplotlib.pyplot as plt
    except ImportError as error: raise RuntimeError("QC rendering requires matplotlib.") from error
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); original_png,processed_png=display_image(original),display_image(processed)
    paths={"original":output/f"{subject_id}_sagittal_original.png","processed":output/f"{subject_id}_sagittal_processed.png","comparison":output/f"{subject_id}_sagittal_comparison.png","metadata":output/f"{subject_id}_sagittal_selection.json"}
    plt.imsave(paths["original"],original_png,cmap="gray"); plt.imsave(paths["processed"],processed_png,cmap="gray")
    figure,axes=plt.subplots(1,2,figsize=(10,5)); axes[0].imshow(original_png,cmap="gray"); axes[1].imshow(processed_png,cmap="gray")
    axes[0].set_title("Original sagittal"); axes[1].set_title("Processed sagittal")
    for axis in axes: axis.axis("off")
    figure.tight_layout(); figure.savefig(paths["comparison"],dpi=150); plt.close(figure)
    paths["metadata"].write_text(json.dumps({"selection":asdict(selection),"config":asdict(config)},indent=2),encoding="utf-8"); return paths
