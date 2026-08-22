# ============================================================
#  AUTO SEGMENTATION — Generate seg.nii.gz for all subjects
#  Uses intensity thresholding (K-Means clustering)
#  CSF=1, Grey Matter=2, White Matter=3
# ============================================================

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from sklearn.cluster import KMeans

# ============================================================
#  CONFIG
# ============================================================
NORM_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
RAW_PATH  = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset\raw_mri"
SEG_PATH  = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset\raw_mri"  # saves seg.nii.gz inside each subject folder

print("✅ Config ready!")
print(f"📂 Normalized files: {len(os.listdir(NORM_PATH))}")

# ============================================================
#  SEGMENTATION FUNCTION
# ============================================================
def generate_seg(normalized_path, output_path):
    """
    Generate segmentation using K-Means clustering.
    Labels: 0=background, 1=CSF, 2=Grey Matter, 3=White Matter
    """
    # load normalized volume
    img = nib.load(normalized_path)
    data = img.get_fdata().astype(np.float32)

    # get brain mask (non-zero voxels)
    brain_mask = data != 0

    # extract brain voxels only
    brain_voxels = data[brain_mask].reshape(-1, 1)

    # K-Means with 3 clusters (CSF, GM, WM)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(brain_voxels)

    # get cluster centers and sort by intensity
    # darkest = CSF(1), medium = GM(2), brightest = WM(3)
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)  # sort from darkest to brightest

    # remap labels so: darkest→1, medium→2, brightest→3
    label_map = np.zeros(3, dtype=np.uint8)
    for new_label, old_label in enumerate(sorted_idx):
        label_map[old_label] = new_label + 1  # 1-indexed

    # create segmentation volume
    seg = np.zeros_like(data, dtype=np.uint8)
    raw_labels = kmeans.labels_
    remapped = label_map[raw_labels]
    seg[brain_mask] = remapped

    # save as nii.gz
    seg_img = nib.Nifti1Image(seg, img.affine, img.header)
    nib.save(seg_img, output_path)

# ============================================================
#  MAIN LOOP — process all subjects
# ============================================================
subjects_done = []
subjects_failed = []

norm_files = sorted([f for f in os.listdir(NORM_PATH) if f.endswith("_normalized.nii.gz")])

print(f"📂 Found {len(norm_files)} normalized files\n")

for norm_file in norm_files:
    # extract subject name e.g. s0024 from s0024_normalized.nii.gz
    subject = norm_file.replace("_normalized.nii.gz", "")
    norm_file_path = os.path.join(NORM_PATH, norm_file)
    output_path    = os.path.join(SEG_PATH, subject, "seg.nii.gz")

    print(f"🔄 Generating seg for: {subject}")

    try:
        generate_seg(norm_file_path, output_path)
        print(f"   ✅ Saved: {output_path}")
        subjects_done.append(subject)

    except Exception as e:
        import traceback
        print(f"   ❌ ERROR on {subject}: {e}")
        traceback.print_exc()
        subjects_failed.append(subject)

print(f"\n🎉 Done!")
print(f"✅ Generated: {len(subjects_done)} segmentations")
print(f"❌ Failed:    {len(subjects_failed)}")
if subjects_failed:
    print(f"   Failed: {subjects_failed}")