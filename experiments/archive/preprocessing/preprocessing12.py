import os
import numpy as np
import nibabel as nib
import cv2

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_ULTRA = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\16_ultra_sagittal"

os.makedirs(SAVE_ULTRA, exist_ok=True)


# ================= LOAD (RAS FIX) =================
def load_nifti(file_path):
    nii = nib.load(file_path)
    nii = nib.as_closest_canonical(nii)   # ✅ FORCE CORRECT ORIENTATION
    return nii.get_fdata()


# ================= TRUE SAGITTAL =================
def get_best_sagittal(data):

    # 🔥 Downsample for speed
    small = data[::2, ::2, ::2]

    mask = small > np.percentile(small, 20)
    coords = np.argwhere(mask)

    if len(coords) == 0:
        center = data.shape[0] // 2
    else:
        com = coords.mean(axis=0)
        center = int(com[0] * 2)

    best_slice, best_score = None, float("inf")

    # 🔥 Wider search range (better midline)
    for i in range(center - 8, center + 9):

        if i < 0 or i >= data.shape[0]:
            continue

        s = data[i, :, :]   # ✅ TRUE SAGITTAL

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid + 1:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)

        if score < best_score:
            best_score = score
            best_slice = s

    if best_slice is None:
        best_slice = data[data.shape[0] // 2, :, :]

    # Orientation fix
    best_slice = np.rot90(best_slice, k=1)
    best_slice = np.fliplr(best_slice)

    return best_slice


# ================= LIGHT DENOISE =================
def denoise_image(img):
    # ✅ gentle smoothing (keeps anatomy)
    return cv2.GaussianBlur(img, (3, 3), 0.5)


# ================= BALANCED ENHANCEMENT =================
def enhance_image(img):

    non_zero = img[img > 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Gentle intensity clipping
    p2, p98 = np.percentile(non_zero, (2, 98))
    img = np.clip(img, p2, p98)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Softer CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Mild sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    img = cv2.addWeighted(img, 1.3, blur, -0.3, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):
    return cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    data = load_nifti(file_path)

    sagittal = get_best_sagittal(data)
    sagittal = denoise_image(sagittal)
    sagittal = enhance_image(sagittal)

    final_img = upscale_image(sagittal)

    save_path = os.path.join(SAVE_ULTRA, f"{subject_name}.png")
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

print("\n✅ DONE — CLEAN SAGITTAL MRI GENERATED")