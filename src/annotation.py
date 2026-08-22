"""Unvalidated CC candidate generation and human-review provenance.

Candidates are localization aids only. This module never names a candidate a ground
truth label and requires an explicit contrast polarity from a reviewer/protocol.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal
import numpy as np
from evaluation.visualization import mask_overlay

@dataclass(frozen=True)
class CandidateConfig:
    """Spatial prior expressed as central plane fractions; tune only after QC."""
    polarity: Literal["bright", "dark"]
    threshold_percentile: float = 85.0
    y_range: tuple[float,float] = (0.25,0.75)
    x_range: tuple[float,float] = (0.25,0.75)

def generate_candidate_mask(sagittal: np.ndarray, config: CandidateConfig) -> np.ndarray:
    """Make a prior-bounded intensity candidate, explicitly unsuitable as ground truth."""
    if sagittal.ndim != 2 or not np.isfinite(sagittal).all(): raise ValueError("Expected finite 2-D sagittal image.")
    height,width=sagittal.shape; y0,y1=int(height*config.y_range[0]),int(height*config.y_range[1]); x0,x1=int(width*config.x_range[0]),int(width*config.x_range[1])
    roi=sagittal[y0:y1,x0:x1]
    if roi.size == 0: raise ValueError("Candidate ROI is empty.")
    threshold=np.percentile(roi,config.threshold_percentile if config.polarity=="bright" else 100-config.threshold_percentile)
    selected=roi >= threshold if config.polarity=="bright" else roi <= threshold
    result=np.zeros_like(sagittal,dtype=bool); result[y0:y1,x0:x1]=selected
    return result

def candidate_metadata(source_mri: str, subject_id: str, slice_index: int, preprocessing_version: str, config: CandidateConfig) -> dict[str,object]:
    """Create required provenance, preserving the unvalidated status explicitly."""
    return {"source_mri":source_mri,"subject_id":subject_id,"slice_index":slice_index,"label_source":"candidate_spatial_intensity_prior","manually_validated":False,"preprocessing_version":preprocessing_version,"candidate_config":asdict(config),"warning":"Candidate only; not ground truth and not for model training."}

def save_candidate_qc(sagittal: np.ndarray, candidate: np.ndarray, metadata: dict[str,object], output_dir: str|Path) -> dict[str,Path]:
    """Save candidate, overlay and JSON under candidates; never validated_masks."""
    try: import matplotlib.pyplot as plt
    except ImportError as error: raise RuntimeError("Candidate QC rendering requires matplotlib.") from error
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); subject=str(metadata["subject_id"])
    paths={"mask":output/f"{subject}_candidate_mask.npy","mri":output/f"{subject}_mri.png","overlay":output/f"{subject}_candidate_overlay.png","metadata":output/f"{subject}_candidate.json"}
    np.save(paths["mask"],candidate.astype(np.uint8)); plt.imsave(paths["mri"],sagittal,cmap="gray"); plt.imsave(paths["overlay"],mask_overlay(sagittal,candidate))
    paths["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf-8"); return paths

def initialize_review_record(candidate_metadata_path: str|Path, validated_directory: str|Path) -> Path:
    """Create a review template; a human must add the corrected mask and sign-off."""
    source=Path(candidate_metadata_path); target=Path(validated_directory); target.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(source.read_text(encoding="utf-8")); metadata.update({"manually_validated":False,"reviewer_id":None,"review_date":None,"validated_mask_path":None,"review_instructions":"Inspect candidate overlay, correct in an annotation tool, save binary mask separately, then set manually_validated true with reviewer and date."})
    path=target/f"{metadata['subject_id']}_review.json"; path.write_text(json.dumps(metadata,indent=2),encoding="utf-8"); return path
