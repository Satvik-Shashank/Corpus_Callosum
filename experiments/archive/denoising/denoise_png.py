import os
import cv2
import numpy as np
from bm3d import bm3d

INPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\4_sagittal_slices"
OUTPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_denoised"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Starting BM3D Denoising...")

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".png"):
        
        path = os.path.join(INPUT_FOLDER, file)

        # read image (grayscale)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # normalize to [0,1]
        norm = img / 255.0

        # BM3D (LIGHT denoising)
        denoised = bm3d(norm, sigma_psd=0.02)

        # back to [0,255]
        denoised = (denoised * 255).astype(np.uint8)

        # save
        out_path = os.path.join(OUTPUT_FOLDER, file)
        cv2.imwrite(out_path, denoised)

        print("Saved:", file)

print("Done!")