"""Run preprocessing/QC for exactly one MRI; no dataset-wide loop is provided."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
from src.preprocessing import PreprocessingConfig, candidate_brain_mask, load_and_canonicalize, n4_bias_correct, normalize_intensity, preprocessing_metadata, save_processed_volume, write_metadata
from src.sagittal import SagittalConfig, save_sagittal_qc, select_mid_sagittal

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("mri",type=Path); parser.add_argument("--subject-id",required=True); parser.add_argument("--output",type=Path,default=Path("outputs")); parser.add_argument("--n4",action="store_true"); parser.add_argument("--search-radius",type=int,default=5); args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config=PreprocessingConfig(apply_n4_bias_correction=args.n4); image,original=load_and_canonicalize(args.mri,config.canonical_orientation)
    corrected=n4_bias_correct(original) if config.apply_n4_bias_correction else original; mask=candidate_brain_mask(corrected,config.brain_mask_percentile); processed=normalize_intensity(corrected,mask,config.normalization)
    original_plane,selection=select_mid_sagittal(original,mask,SagittalConfig(search_radius=args.search_radius)); processed_plane=processed[selection.index]
    subject_output=args.output/args.subject_id; save_processed_volume(processed,image,subject_output/f"{args.subject_id}_processed.nii.gz"); save_sagittal_qc(original_plane,processed_plane,subject_output,args.subject_id,selection,SagittalConfig(search_radius=args.search_radius))
    metadata=preprocessing_metadata(config,image,args.mri); metadata.update({"subject_id":args.subject_id,"selected_sagittal_index":selection.index,"sagittal_selection_score":selection.score,"operations":["RAS canonicalization" if config.canonical_orientation else "native orientation","candidate intensity brain mask",config.normalization+" normalization"]}); write_metadata(subject_output/f"{args.subject_id}_processing.json",metadata)
if __name__ == "__main__": main()
