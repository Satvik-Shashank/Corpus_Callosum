import os
import cv2
import numpy as np

INPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_denoised"
OUTPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\6_roi_opencv_2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Starting Optimized ROI Extraction...")

for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".png"):

        path = os.path.join(INPUT_FOLDER, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed:", file)
            continue

        h, w = img.shape

        # 🎯 Balanced ROI (not too tight, not too large)
        x1 = int(0.25 * w)
        x2 = int(0.75 * w)

        y1 = int(0.20 * h)
        y2 = int(0.60 * h)

        roi = img[y1:y2, x1:x2]

        # 🔥 LIGHT contrast enhancement (not aggressive)
        roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)

        # 🔥 VERY mild smoothing (avoid texture destruction)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        # 🔥 Gentle sharpening (controlled)
        sharp = cv2.addWeighted(roi, 1.3, cv2.GaussianBlur(roi, (3,3), 0), -0.3, 0)

        # Save
        out_path = os.path.join(OUTPUT_FOLDER, file)
        cv2.imwrite(out_path, sharp)

        print("Saved:", file)

print("Done!")