# Corpus Callosum

AI-assisted research prototype for early characterization of corpus callosum (CC) development from infant brain MRI. It supports image preparation, mid-sagittal candidate selection, future validated CC segmentation, morphology, and age-aware research analysis.

## Scientific scope

This repository is not a diagnostic system. MRI alone does not diagnose autism or any developmental disorder. Outputs require expert quality control and are intended only for research/decision-support workflows. Automatically generated masks, including the archived K-means tissue labels and threshold masks, are not ground truth.

## Layout

- `src/`: data, preprocessing, sagittal, localization, segmentation, morphology, developmental and inference interfaces.
- `models/`: reusable U-Net architecture only; no checkpoint is supplied.
- `training/`: guarded future training components that require expert-validated CC labels.
- `evaluation/`: metrics and QC visualization.
- `experiments/archive/`: preserved historical scripts, not production code.

MRI datasets and model artifacts are excluded from Git. Use `configs/default.yaml` as an explicit research-only starting point.

## Requirements for legitimate CC model training

Paired MRI and expert/manual (or independently validated) binary CC masks on the selected mid-sagittal plane; participant-level train/validation/test splits; provenance and QC records; and held-out evaluation against the validated labels are required before enabling training.

## Status

The architecture and safety boundaries are implemented. Training, clinical claims, and automatic-label promotion are intentionally disabled.

## Checkpoint 1: processing and annotation QC

Run exactly one image with `python scripts/process_one_mri.py PATH_TO_IMAGE --subject-id SUBJECT_ID`. The command writes a geometry-preserving processed NIfTI, the native-resolution selected mid-sagittal plane metadata, and three display-only QC images. It does not resize the analytical volume.

Candidate CC masks may be made only through an explicit spatial/intensity-prior protocol and are stored in `outputs/candidates/`. They are not labels. Human review records and genuinely corrected masks belong in `outputs/validated_masks/`; only masks with verified provenance and manual validation may later be considered for a training manifest. Use `src.labels.inspect_label_source` with an authoritative label lookup before accepting any `dseg.nii.gz` semantics.
