import os
import cv2
import numpy as np

INPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_denoised"
OUTPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\6_roi_opencv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Starting Improved ROI Extraction...")

for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".png"):

        path = os.path.join(INPUT_FOLDER, file)

        # Load image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed to load:", file)
            continue

        h, w = img.shape

        # 🔥 SAFE LARGE ROI (won’t miss corpus callosum)
        x1 = int(0.20 * w)
        x2 = int(0.80 * w)

        y1 = int(0.15 * h)
        y2 = int(0.65 * h)

        roi = img[y1:y2, x1:x2]

        # 🔥 Contrast Enhancement (VERY IMPORTANT)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(roi)

        # 🔥 Optional smoothing (removes noise)
        roi_enhanced = cv2.GaussianBlur(roi_enhanced, (3, 3), 0)

        # 🔥 Optional edge overlay (to see CC boundary)
        edges = cv2.Canny(roi_enhanced, 50, 150)
        overlay = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)
        overlay[edges > 0] = [0, 0, 255]  # red edges

        # Save outputs
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, "roi_" + file), roi_enhanced)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, "overlay_" + file), overlay)

        print("Saved:", file)

print("Done!")