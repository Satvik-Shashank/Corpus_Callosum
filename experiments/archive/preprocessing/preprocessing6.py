import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import zoom

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\10_ultra_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)


# ================= LOAD =================
def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]  # voxel spacing
    return data, spacing


# ================= RESAMPLE (CRITICAL FIX) =================
def resample_isotropic(data, spacing):

    # target = 1mm x 1mm x 1mm
    new_spacing = (1.0, 1.0, 1.0)

    zoom_factors = [
        spacing[0] / new_spacing[0],
        spacing[1] / new_spacing[1],
        spacing[2] / new_spacing[2]
    ]

    # high-quality resampling
    resampled = zoom(data, zoom_factors, order=3)  # cubic

    return resampled


# ================= TRUE MID SAGITTAL =================
def get_mid_sagittal(data):

    # EXACT center slice (after proper resampling)
    mid = data.shape[0] // 2

    slice_ = data[mid, :, :]

    # orientation fix
    slice_ = np.rot90(slice_, k=1)
    slice_ = np.fliplr(slice_)

    return slice_


# ================= CONTRAST =================
def enhance_image(img):

    non_zero = img[img > 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    p1, p99 = np.percentile(non_zero, (1, 99))
    img = np.clip(img, p1, p99)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # VERY LIGHT CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


# ================= UPSCALE =================
def upscale(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= SHARPEN (HIGH QUALITY) =================
def sharpen(img):

    # high-frequency enhancement (better than simple unsharp)
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    high_freq = cv2.subtract(img, gaussian)

    sharp = cv2.add(img, high_freq)

    return sharp


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    # 1. load
    data, spacing = load_nifti(file_path)

    # 2. RESAMPLE (THIS FIXES BLUR ISSUE)
    data = resample_isotropic(data, spacing)

    # 3. TRUE MID SAGITTAL
    sagittal = get_mid_sagittal(data)

    # 4. enhance
    sagittal = enhance_image(sagittal)

    # 5. upscale
    sagittal = upscale(sagittal)

    # 6. sharpen AFTER upscale
    final_img = sharpen(sagittal)

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

print("\n✅ DONE — TRUE HIGH-QUALITY SAGITTAL IMAGES")