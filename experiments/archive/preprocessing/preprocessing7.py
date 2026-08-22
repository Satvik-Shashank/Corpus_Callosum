import os
import numpy as np
import nibabel as nib
import cv2

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\11_ultra_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)


# ================= LOAD =================
def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    return data


# ================= TRUE SAGITTAL =================
def get_sagittal_slice(data):

    mid = data.shape[0] // 2
    slice_ = data[mid, :, :]

    # orientation (keep minimal transforms)
    slice_ = np.rot90(slice_, k=1)

    return slice_


# ================= SAFE NORMALIZATION =================
def normalize_image(img):
    # NO percentile clipping
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    return img


# ================= HIGH-FIDELITY SHARPEN =================
def sharpen_float(img):

    # operate in float domain (VERY IMPORTANT)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.7)
    sharp = img + (img - blur) * 1.5   # controlled high-frequency boost

    return np.clip(sharp, 0, 1)


# ================= UPSCALE =================
def upscale(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= FINAL CONVERT =================
def to_uint8(img):
    return (img * 255).astype(np.uint8)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    # 1. Load (float precision)
    data = load_nifti(file_path)

    # 2. True sagittal slice
    sagittal = get_sagittal_slice(data)

    # 3. Normalize (safe)
    sagittal = normalize_image(sagittal)

    # 4. Sharpen (float domain → preserves detail)
    sagittal = sharpen_float(sagittal)

    # 5. Upscale (no prior blur)
    sagittal = upscale(sagittal)

    # 6. Final convert
    final_img = to_uint8(sagittal)

    # Save
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

print("\n✅ DONE — HIGH-FIDELITY SAGITTAL OUTPUT")