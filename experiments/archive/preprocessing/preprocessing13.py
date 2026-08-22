import os
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

# ================= PATHS =================
DATASET_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset"
SAVE_SLICES  = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\final_sagittal"

os.makedirs(SAVE_SLICES, exist_ok=True)

print(f"Subjects found: {len(os.listdir(DATASET_PATH))}")

# ================= CORE FUNCTIONS =================

def load_and_preprocess(file_path):
    """
    Full geometric + intensity preprocessing:
    - Reorient to RAS
    - Bias correction
    - Isotropic resampling
    - 3D skull stripping
    """

    # ---- Load ----
    image = sitk.ReadImage(file_path, sitk.sitkFloat32)

    # ---- Fix orientation (CRITICAL) ----
    image = sitk.DICOMOrient(image, "RAS")

    # ---- Bias field correction ----
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    image = corrector.Execute(image, mask)

    # ---- Resample to isotropic (CRITICAL FIX) ----
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [1.0, 1.0, 1.0]
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    image = resampler.Execute(image)

    # ---- 3D Skull stripping (SAFE) ----
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    image = sitk.Mask(image, mask)

    return image


def normalize_volume(data):
    brain = data[data > 0]
    if len(brain) == 0:
        return data
    mean = np.mean(brain)
    std = np.std(brain)
    out = np.zeros_like(data, dtype=np.float32)
    out[data > 0] = (data[data > 0] - mean) / (std + 1e-8)
    return out


def get_mid_sagittal(data):
    """
    Robust mid-sagittal detection using symmetry
    """
    scores = []

    for i in range(data.shape[2]):
        s = data[:, :, i]

        if np.sum(s) == 0:
            scores.append(np.inf)
            continue

        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid:])

        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)
        scores.append(score)

    best_idx = np.argmin(scores)
    sag = data[:, :, best_idx]

    # Standard orientation
    sag = np.rot90(sag, k=1)
    sag = np.fliplr(sag)

    return sag


def enhance_and_save(slice_img, save_path):
    """
    Clean enhancement pipeline (kept from your version, but correctly applied)
    """

    # ---- Clip outliers ----
    nonzero = slice_img[slice_img != 0]
    if len(nonzero) == 0:
        return

    p2 = np.percentile(nonzero, 2)
    p98 = np.percentile(nonzero, 98)
    slice_img = np.clip(slice_img, p2, p98)

    # ---- Resize ----
    slice_img = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # ---- Normalize to 8-bit ----
    slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # ---- CLAHE ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    slice_img = clahe.apply(slice_img)

    # ---- Sharpen ----
    blur = cv2.GaussianBlur(slice_img, (0, 0), sigmaX=1.0)
    slice_img = cv2.addWeighted(slice_img, 1.5, blur, -0.5, 0)

    cv2.imwrite(save_path, slice_img)


# ================= MAIN LOOP =================

processed = 0
failed = 0

for subject in sorted(os.listdir(DATASET_PATH)):
    subject_path = os.path.join(DATASET_PATH, subject)
    file_path = os.path.join(subject_path, "t1.nii.gz")

    if not (os.path.isdir(subject_path) and os.path.exists(file_path)):
        continue

    print(f"\nProcessing: {subject}")

    try:
        # ---- Full preprocessing ----
        sitk_img = load_and_preprocess(file_path)

        # ---- Convert to numpy ----
        data = sitk.GetArrayFromImage(sitk_img)

        # ---- Normalize ----
        data = normalize_volume(data)

        # ---- Extract sagittal ----
        sag = get_mid_sagittal(data)

        # ---- Save ----
        save_file = os.path.join(SAVE_SLICES, f"{subject}.png")
        enhance_and_save(sag, save_file)

        print(" Done")
        processed += 1

    except Exception as e:
        print(f" Failed: {e}")
        failed += 1


print("\n==========================")
print(f"Processed: {processed}")
print(f"Failed:    {failed}")
print("==========================")


# ================= QC DISPLAY =================

png_files = sorted(os.listdir(SAVE_SLICES))[:6]

if png_files:
    fig, axes = plt.subplots(1, len(png_files), figsize=(18, 4))
    if len(png_files) == 1:
        axes = [axes]

    for ax, fname in zip(axes, png_files):
        img = cv2.imread(os.path.join(SAVE_SLICES, fname), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title(fname)
        ax.axis("off")

    plt.suptitle("QC: Mid-Sagittal Slices", fontsize=14)
    plt.tight_layout()
    plt.show()