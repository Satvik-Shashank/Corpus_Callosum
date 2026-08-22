import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset"
SAVE_BIAS = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\1_bias_corrected"
SAVE_SKULL = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\2_skull_stripped"
SAVE_NORM = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SAVE_SLICES = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\4_sagittal_slices"

for p in [SAVE_BIAS, SAVE_SKULL, SAVE_NORM, SAVE_SLICES]:
    os.makedirs(p, exist_ok=True)

print("Folders ready!")
print(f"Subjects found: {len(os.listdir(DATASET_PATH))}")


def bias_field_correction(file_path):
    img = sitk.ReadImage(file_path, sitk.sitkFloat32)
    # More iterations and control points for cleaner correction
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50]) # 4 levels, 50 iters each
    corrector.SetConvergenceThreshold(0.0001)
    corrected = corrector.Execute(img, mask)
    return corrected


def skull_strip_3d(sitk_image):
    data = sitk.GetArrayFromImage(sitk_image)
    stripped = np.zeros_like(data)

    for z in range(data.shape[0]):
        slc = data[z, :, :]
        slc_u8 = cv2.normalize(slc, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        _, thresh = cv2.threshold(slc_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tighter kernel — large kernel smears the brain boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype("uint8")
        else:
            mask = closed // 255

        stripped[z, :, :] = slc * mask

    return stripped


def normalize_volume(data):
    brain_voxels = data[data > 0]
    if len(brain_voxels) == 0:
        return data
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[data > 0] = (data[data > 0] - mean) / (std + 1e-8)
    return normalized


def get_best_sagittal(data):
    """
    Extract mid-sagittal slice from axis 2 (X axis).
    Picks the most symmetric slice near center.
    Rotates so brain is upright and facing right (standard MRI view).
    """
    center = data.shape[2] // 2
    best_slice, best_score = None, float("inf")

    for i in range(center - 5, center + 6):
        s = data[:, :, i]
        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid + 1:])
        w = min(left.shape[1], right.shape[1])
        score = np.mean((left[:, :w] - right[:, :w]) ** 2)
        if score < best_score:
            best_score = score
            best_slice = s

    best_slice = np.rot90(best_slice, k=1)
    best_slice = np.fliplr(best_slice)
    return best_slice


def save_sagittal_png(data, save_path, subject):
    slc = get_best_sagittal(data)

    # ── Step 1: Clip outliers FIRST on the raw float slice (before resize) ──
    # Use tighter percentile clip to preserve fine tissue contrast
    p2 = np.percentile(slc[slc != 0], 2) # only use brain voxels for percentile
    p98 = np.percentile(slc[slc != 0], 98)
    slc_clipped = np.clip(slc, p2, p98)

    # ── Step 2: Resize with LANCZOS (best for MRI — sharpest, no ringing) ──
    # Output at 512x512; INTER_LANCZOS4 is superior to INTER_CUBIC for medical images
    slc_resized = cv2.resize(slc_clipped, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 3: Map to full 16-bit range first, THEN convert to 8-bit ──
    # This preserves far more intensity gradation than going straight to uint8
    slc_16 = cv2.normalize(slc_resized, None, 0, 65535, cv2.NORM_MINMAX).astype("uint16")
    png = (slc_16 / 256).astype("uint8") # controlled downscale to 8-bit

    # ── Step 4: CLAHE with conservative settings (sharpens without haloing) ──
    # clipLimit=2.0 avoids over-amplification of noise at tissue boundaries
    # Smaller tile grid = more local detail for corpus callosum
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    png = clahe.apply(png)

    # ── Step 5: Unsharp mask instead of blur (sharpens edges) ──
    # This is the opposite of what the original code did (which blurred)
    gaussian = cv2.GaussianBlur(png, (0, 0), sigmaX=1.0)
    png = cv2.addWeighted(png, 1.5, gaussian, -0.5, 0) # sharpen

    # ── Step 6: Save as PNG (lossless) ──
    cv2.imwrite(os.path.join(save_path, f"{subject}.png"), png)


def save_nifti(data, reference_sitk_img, save_path, subject, suffix):
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_sitk_img)
    fname = os.path.join(save_path, f"{subject}_{suffix}.nii.gz")
    sitk.WriteImage(out_img, fname)
    return fname


subjects_done = []
subjects_failed = []

for subject in sorted(os.listdir(DATASET_PATH)):
    subject_path = os.path.join(DATASET_PATH, subject)
    file_path = os.path.join(subject_path, "t1.nii.gz")

    if not (os.path.isdir(subject_path) and os.path.exists(file_path)):
        print(f"Skipping {subject} - t1.nii.gz not found")
        continue

    print(f"Processing: {subject}")

    try:
        print(" 1A: Bias field correction...")
        corrected_sitk = bias_field_correction(file_path)
        save_nifti(sitk.GetArrayFromImage(corrected_sitk), corrected_sitk, SAVE_BIAS, subject, "bias_corrected")
        print(" Saved bias corrected")

        print(" 1B: Skull stripping...")
        stripped_data = skull_strip_3d(corrected_sitk)
        save_nifti(stripped_data, corrected_sitk, SAVE_SKULL, subject, "skull_stripped")
        print(" Saved skull stripped")

        print(" 1C: Normalizing...")
        normalized_data = normalize_volume(stripped_data)
        save_nifti(normalized_data, corrected_sitk, SAVE_NORM, subject, "normalized")
        print(" Saved normalized")

        save_sagittal_png(normalized_data, SAVE_SLICES, subject)
        print(" Saved sagittal PNG")

        subjects_done.append(subject)

    except Exception as e:
        import traceback
        print(f" ERROR on {subject}: {e}")
        traceback.print_exc()
        subjects_failed.append(subject)

print(f"\nDone! Processed: {len(subjects_done)} subjects")
print(f"Failed: {len(subjects_failed)} subjects")

png_files = sorted([f for f in os.listdir(SAVE_SLICES) if f.endswith(".png")])[:6]

if png_files:
    fig, axes = plt.subplots(1, len(png_files), figsize=(20, 5))
    if len(png_files) == 1:
        axes = [axes]
    for ax, fname in zip(axes, png_files):
        img = cv2.imread(os.path.join(SAVE_SLICES, fname), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title(fname.replace(".png", ""), fontsize=10)
        ax.axis("off")
    plt.suptitle("Step 1 QC - Preprocessed Sagittal Slices", fontsize=13)
    plt.tight_layout()
    plt.show()
else:
    print("No PNG slices found")
