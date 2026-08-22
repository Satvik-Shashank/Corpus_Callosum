import os
import torch
import nibabel as nib
import numpy as np
import cv2

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensity,
    Resize
)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= PATHS =================
NORMALIZED_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\13_gpu_sagittal"

os.makedirs(SAVE_PATH, exist_ok=True)

# ================= MONAI PIPELINE =================
transforms = Compose([
    LoadImage(image_only=True),             # load NIfTI
    EnsureChannelFirst(),                  # (C, H, W, D)
    Orientation(axcodes="RAS"),            # 🔥 fix orientation globally
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  # isotropic resample
    ScaleIntensity(),                      # normalize intensity
])

# ================= SAGITTAL EXTRACTION =================
def get_sagittal(volume):

    # volume shape: (1, H, W, D)
    volume = volume[0]   # remove channel

    # sagittal = along X-axis
    mid = volume.shape[0] // 2
    slice_ = volume[mid, :, :]

    # orientation fix
    slice_ = np.rot90(slice_, k=1)

    return slice_


# ================= ENHANCEMENT =================
def enhance(img):

    img = (img * 255).astype(np.uint8)

    # CLAHE (balanced)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # strong sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 2.0, blur, -1.0, 0)

    return img


# ================= UPSCALE =================
def upscale(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    # MONAI pipeline
    volume = transforms(file_path)

    # move to GPU (optional but ready)
    volume = torch.tensor(volume).to(device)

    # back to numpy for slicing
    volume = volume.cpu().numpy()

    # sagittal slice
    sagittal = get_sagittal(volume)

    # enhance
    sagittal = enhance(sagittal)

    # upscale
    final_img = upscale(sagittal)

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

print("\n✅ DONE — GPU SAGITTAL PIPELINE COMPLETE")