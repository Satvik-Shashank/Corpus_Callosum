import os
import numpy as np
import nibabel as nib
import cv2

from skimage.restoration import denoise_nl_means, estimate_sigma

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_ULTRA = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\14_ultra_fullbrain"

os.makedirs(SAVE_ULTRA, exist_ok=True)


# ================= LOAD (WITH RAS FIX) =================
def load_nifti(file_path):
    nii = nib.load(file_path)

    # 🔥 FORCE STANDARD ORIENTATION (CRITICAL)
    nii = nib.as_closest_canonical(nii)

    data = nii.get_fdata()
    return data


# ================= TRUE SAGITTAL EXTRACTION =================
def get_best_sagittal(data):

    # 🔥 Downsample for speed
    small = data[::2, ::2, ::2]

    mask = small > np.percentile(small, 20)
    coords = np.argwhere(mask)

    if len(coords) == 0:
        center = data.shape[0] // 2
    else:
        com = coords.mean(axis=0)
        center = int(com[0] * 2)  # scale back

    best_slice, best_score = None, float("inf")

    for i in range(center - 5, center + 6):

        if i < 0 or i >= data.shape[0]:
            continue

        # ✅ TRUE SAGITTAL AXIS
        s = data[i, :, :]

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid + 1:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)

        if score < best_score:
            best_score = score
            best_slice = s

    # 🔥 Safety fallback
    if best_slice is None:
        best_slice = data[data.shape[0] // 2, :, :]

    # Orientation for display
    best_slice = np.rot90(best_slice, k=1)
    best_slice = np.fliplr(best_slice)

    return best_slice


# ================= DENOISE =================
def denoise_image(img):
    sigma = np.mean(estimate_sigma(img))
    return denoise_nl_means(
        img,
        h=1.15 * sigma,
        fast_mode=True,
        patch_size=3,
        patch_distance=5
    )


# ================= ENHANCE =================
def enhance_image(img):

    non_zero = img[img != 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    p2, p98 = np.percentile(non_zero, (2, 98))
    img = np.clip(img, p2, p98)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    img = clahe.apply(img)

    # Sharpen
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    data = load_nifti(file_path)

    # ✅ TRUE SAGITTAL SLICE
    sagittal = get_best_sagittal(data)

    # Enhance
    sagittal = denoise_image(sagittal)
    sagittal = enhance_image(sagittal)

    # Ultra-HD
    final_img = upscale_image(sagittal)

    # Save
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

print("\n✅ DONE — TRUE SAGITTAL ULTRA-HD IMAGES GENERATED")