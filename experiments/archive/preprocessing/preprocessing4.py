import os
import numpy as np
import nibabel as nib
import cv2

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\8_ultra_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)


# ================= LOAD =================
def load_nifti(file_path):
    return nib.load(file_path).get_fdata()


# ================= SAGITTAL EXTRACTION =================
def get_sagittal_slice(data):

    center = data.shape[0] // 2
    best_slice = None
    best_score = float("inf")

    for i in range(center - 5, center + 6):

        if i < 0 or i >= data.shape[0]:
            continue

        s = data[i, :, :]   # sagittal slice

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)

        if score < best_score:
            best_score = score
            best_slice = s

    if best_slice is None:
        best_slice = data[data.shape[0] // 2, :, :]

    # orientation fix
    best_slice = np.rot90(best_slice, k=1)
    best_slice = np.fliplr(best_slice)

    return best_slice


# ================= ENHANCEMENT =================
def enhance_image(img):

    non_zero = img[img != 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # contrast clipping
    p2, p98 = np.percentile(non_zero, (2, 98))
    img = np.clip(img, p2, p98)

    # normalize to 0–255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # CLAHE (balanced)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
    img = clahe.apply(img)

    # sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.8, blur, -0.8, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):

    # pre-sharpen before upscale
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    data = load_nifti(file_path)

    # sagittal slice (correct orientation)
    sagittal = get_sagittal_slice(data)

    # enhance (NO denoising → keeps details sharp)
    sagittal = enhance_image(sagittal)

    # upscale
    final_img = upscale_image(sagittal)

    # save
    save_path = os.path.join(SAVE_PATH, f"{subject_name}.png")
    cv2.imwrite(save_path, final_img)

    print("   Saved →", save_path)


# ================= RUN =================
files = sorted(os.listdir(NORMALIZED_PATH))

for file in files:

    if not file.endswith(".nii.gz"):
        continue

    subject_name = file.replace("_normalized.nii.gz", "")
    file_path = os.path.join(NORMALIZED_PATH, file)

    try:
        process_subject(file_path, subject_name)
    except Exception as e:
        print(f"Error in {subject_name}: {e}")

print("\n✅ DONE — HIGH-QUALITY SAGITTAL IMAGES GENERATED")