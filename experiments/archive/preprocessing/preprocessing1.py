import os
import numpy as np
import nibabel as nib
import cv2

from skimage.restoration import denoise_nl_means, estimate_sigma

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_ULTRA = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\5_ultra_sagittal"

os.makedirs(SAVE_ULTRA, exist_ok=True)


# ================= LOAD =================
def load_nifti(file_path):
    return nib.load(file_path).get_fdata()


# ================= FAST MID-SAGITTAL =================
def get_true_mid_sagittal(data):

    # 🔥 Downsample for speed
    small = data[::2, ::2, ::2]

    mask = small > np.percentile(small, 20)
    coords = np.argwhere(mask)

    if len(coords) == 0:
        mid_x = data.shape[2] // 2
    else:
        com = coords.mean(axis=0)
        mid_x = int(com[2] * 2)

    best_slice = None
    best_score = float("inf")

    for i in range(mid_x - 3, mid_x + 4):

        if i < 0 or i >= data.shape[2]:
            continue

        s = data[:, :, i]

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)

        if score < best_score:
            best_score = score
            best_slice = s

    if best_slice is None:
        best_slice = data[:, :, data.shape[2] // 2]

    best_slice = np.rot90(best_slice)
    best_slice = np.fliplr(best_slice)

    return best_slice


# ================= DENOISE =================
def denoise_image(img):
    sigma = np.mean(estimate_sigma(img))
    return denoise_nl_means(
        img,
        h=1.15 * sigma,
        fast_mode=True,
        patch_size=3,        # faster
        patch_distance=5
    )


# ================= ROI =================
def extract_cc_roi(img):
    h, w = img.shape
    return img[
        int(0.35*h):int(0.65*h),
        int(0.3*w):int(0.7*w)
    ]


# ================= EDGE =================
def overlay_edges(img):
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.85, edges, 0.15, 0)


# ================= ENHANCE =================
def enhance_image(img):

    non_zero = img[img != 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    p2, p98 = np.percentile(non_zero, (2, 98))
    img = np.clip(img, p2, p98)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    img = clahe.apply(img)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):
    return cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_CUBIC)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    # Load normalized MRI
    data = load_nifti(file_path)

    # Mid-sagittal slice
    sagittal = get_true_mid_sagittal(data)

    # Denoise
    sagittal = denoise_image(sagittal)

    # Enhance
    sagittal = enhance_image(sagittal)

    # ROI (corpus callosum region)
    roi = extract_cc_roi(sagittal)

    # Edge overlay
    roi = overlay_edges(roi)

    # Ultra-HD upscale
    final_img = upscale_image(roi)

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

print("\n✅ DONE — ULTRA-HD DATASET CREATED SUCCESSFULLY")