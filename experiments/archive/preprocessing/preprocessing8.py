import os
import numpy as np
import nibabel as nib
import cv2

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\12_ras_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)


# ================= LOAD + RAS =================
def load_ras(file_path):
    nii = nib.load(file_path)
    nii = nib.as_closest_canonical(nii)   # 🔥 KEY STEP
    data = nii.get_fdata().astype(np.float32)
    return data


# ================= TRUE SAGITTAL =================
def get_sagittal(data):

    # after RAS → axis 0 is sagittal
    mid = data.shape[0] // 2
    slice_ = data[mid, :, :]

    # minimal orientation fix
    slice_ = np.rot90(slice_, k=1)

    return slice_


# ================= ENHANCE (NO DAMAGE) =================
def enhance(img):

    # normalize safely
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)

    # convert late
    img = (img * 255).astype(np.uint8)

    # CLAHE (mild)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # strong sharpening (post)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.8, blur, -0.8, 0)

    return img


# ================= UPSCALE =================
def upscale(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    data = load_ras(file_path)   # 🔥 FIXES ORIENTATION PROPERLY

    sagittal = get_sagittal(data)

    sagittal = enhance(sagittal)

    final_img = upscale(sagittal)

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

print("\n✅ DONE — RAS-ALIGNED SAGITTAL OUTPUT")