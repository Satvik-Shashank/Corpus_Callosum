import os
import cv2
import numpy as np

INPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_denoised"
OUTPUT_FOLDER = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\6_roi_opencv_1"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Starting CLEAN ROI Extraction...")

for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".png"):

        path = os.path.join(INPUT_FOLDER, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed:", file)
            continue

        h, w = img.shape

        # 🎯 BETTER CENTER-FOCUSED ROI (captures CC reliably)
        x1 = int(0.30 * w)
        x2 = int(0.75 * w)

        y1 = int(0.25 * h)
        y2 = int(0.60 * h)

        roi = img[y1:y2, x1:x2]

        # 🔥 CLAHE (BEST for MRI contrast)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        roi = clahe.apply(roi)

        # 🔥 Mild sharpening (makes CC clearer)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        roi = cv2.filter2D(roi, -1, kernel)

        # Save clean ROI
        out_path = os.path.join(OUTPUT_FOLDER, file)
        cv2.imwrite(out_path, roi)

        print("Saved:", file)

print("Done!")