import os
import numpy as np
import nibabel as nib
from bm3d import bm3d

INPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\4_sagittal_slices"
OUTPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_denoised"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def normalize(slice_):
    return (slice_ - slice_.min()) / (slice_.max() - slice_.min() + 1e-8)

def denormalize(slice_, original):
    return slice_ * (original.max() - original.min()) + original.min()

print("Starting BM3D Denoising...")

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        
        path = os.path.join(INPUT_FOLDER, file)
        nii = nib.load(path)
        data = nii.get_fdata()

        denoised_volume = []

        for i in range(data.shape[0]):
            slice_ = data[i]

            norm = normalize(slice_)
            d = bm3d(norm, sigma_psd=0.02)
            d = denormalize(d, slice_)

            denoised_volume.append(d)

        denoised_volume = np.array(denoised_volume)

        out_path = os.path.join(OUTPUT_FOLDER, file)
        nib.save(nib.Nifti1Image(denoised_volume, nii.affine), out_path)

        print("Saved:", file)

print("Done!")