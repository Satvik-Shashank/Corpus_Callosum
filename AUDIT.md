# Repository audit (2026-08-22)

## KEEP

- N4 bias-field correction is a reasonable reusable operation when its mask and visual QC are retained.
- RAS canonicalization in `preprocessing8.py`/`preprocessing9.py` is the soundest orientation step. The explicit SimpleITK array convention in `preprocessing14.py` onward is also useful.
- The U-Net topology in archived `unet_tissue_segmentation.py` is a standard 2-D model and has been refactored into `models/unet.py`; it is not a CC model.
- NIfTI affine/header preservation is necessary and is now centralized in `src/io.py`.

## REFACTOR

- The repeated bias correction, slice-wise Otsu masking, z-score normalization, symmetry search, display conversion and hard-coded paths across `preprocessing*.py` belong in the new modules, with configuration and QC.
- `unet.py` mixed architecture, data ingestion, training and plotting in one import-time script. These concerns are now separated.

## REPLACE

- Hard-coded desktop locations, eager scripts with side effects, PNG-only intermediate workflows and implicit axis assumptions must be replaced by configurable, affine-aware calls.
- Any use of the generated `seg.nii.gz` labels to train a CC model must be replaced by validated binary CC annotations.

## EXPERIMENTAL / BASELINE ONLY

- `preprocessing1`–`preprocessing20`: iterative display-focused changes (ROI/full-slice experiments; denoise removal; percentile clipping; 1 mm/0.5 mm resampling; RAS and MONAI variants; CLAHE/sharpen/upscale tuning).
- `preprocessing21.py` is the only script that actually attempts a CC mask: threshold + morphology + central-component filtering on an enhanced PNG. It is an unvalidated heuristic baseline.
- `generate_tissue_kmeans.py` produces intensity clusters labelled CSF/GM/WM, not CC segmentation. The archived U-Net consequently learns those pseudo tissue labels (four output classes including background), not CC anatomy.
- OpenCV ROI, edge and denoising scripts are visual exploration only.

## BROKEN / INCORRECT

- `preprocessing21.py` and `preprocessing22.py` register an image to itself: `fixed` is derived from the moving image, so no template exists and the registration is not valid inter-subject registration.
- Slice-wise Otsu "skull stripping" can discard brain, change boundaries and create discontinuities; it is not a validated infant brain extraction method.
- Several early variants select different axes before canonical orientation, so a claimed sagittal slice is not guaranteed.
- Resizing, CLAHE, sharpening, diffusion/NLM, thresholding and super-resolution alter boundaries/intensities and must never feed physical morphology without validation. 0.5 mm or 2048 px upsampling adds no anatomical resolution.
- The original split is subject-level, which is good, but no empty-loader safeguards, test protocol, label validation or test reporting beyond pseudo-label Dice exists.

## MISSING

- Expert-delineated/validated binary CC masks, annotation protocol, annotator agreement, dataset manifest, consent/governance and image/label provenance.
- A real infant template for registration, transform/QC persistence, and a justified registration protocol; none currently exists.
- Mid-sagittal ground truth/QC, localization labels, lesion/artifact handling, physical spacing checks, calibration, held-out evaluation and error analysis.
- Age metadata and a sufficiently large age-matched normative reference cohort. No diagnostic labels should be invented.
- Reproducible configuration, tests, model cards, uncertainty/QC outputs, and a non-diagnostic web/API layer.

## Trustworthy outputs today

Only raw-image I/O and the fact that scripts write their stated files are mechanically trustworthy. N4 correction and canonical orientation are plausible processing steps pending visual QC. No CC mask, K-means tissue map, threshold result, Dice score, registration result or developmental conclusion is scientifically validated.

## Target architecture

`src` owns affine-aware processing, plane selection, inference contracts and measurements; `models` owns network definitions; `training` only consumes validated manifests; `evaluation` owns metrics/QC; `experiments/archive` preserves history; `configs` makes choices explicit; web layers consume reviewed research results only. The intended flow is MRI → conservative preprocessing → RAS mid-sagittal candidate + QC → validated localization/segmentation model → physical morphology → age-matched research comparison → explainable, non-diagnostic report.
