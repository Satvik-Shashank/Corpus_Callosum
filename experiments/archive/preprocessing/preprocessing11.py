import os
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\15_final_ai_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)

# ================= LOAD =================
def load_nifti(file_path):
    return nib.load(file_path).get_fdata().astype(np.float32)

# ================= SAGITTAL =================
def get_best_sagittal(data):

    center = data.shape[0] // 2
    best_slice, best_score = None, float("inf")

    for i in range(center - 5, center + 6):

        if i < 0 or i >= data.shape[0]:
            continue

        s = data[i, :, :]

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)

        if score < best_score:
            best_score = score
            best_slice = s

    best_slice = np.rot90(best_slice, k=1)
    best_slice = np.fliplr(best_slice)

    return best_slice

# ================= ENHANCE =================
def enhance_image(img):

    non_zero = img[img != 0]
    if len(non_zero) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    p2, p98 = np.percentile(non_zero, (2, 98))
    img = np.clip(img, p2, p98)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img

# ================= GPU UPSCALE =================
def gpu_upscale(img):

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    # bicubic upscale on GPU
    upscaled = F.interpolate(
        tensor,
        size=(2048, 2048),
        mode='bicubic',
        align_corners=False
    )

    return upscaled.squeeze().cpu().numpy()

# ================= AI-LIKE SHARPEN =================
def sharp_enhance(img):

    img = (img * 255).astype(np.uint8)

    # CLAHE (your best component)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
    img = clahe.apply(img)

    # strong edge-preserving sharpen
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
    sharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)

    return sharp

# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    data = load_nifti(file_path)

    # sagittal slice
    sagittal = get_best_sagittal(data)

    # normalize (keep details)
    sagittal = enhance_image(sagittal)

    # 🔥 GPU UPSCALE FIRST
    sagittal = gpu_upscale(sagittal)

    # 🔥 AI-like sharpening AFTER upscale
    final_img = sharp_enhance(sagittal)

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

print("\n✅ DONE — HIGH-QUALITY GPU SAGITTAL OUTPUT")