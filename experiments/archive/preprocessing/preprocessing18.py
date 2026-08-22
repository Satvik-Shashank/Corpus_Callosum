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
SAVE_SLICES = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\21_sagittal_slices"

for p in [SAVE_BIAS, SAVE_SKULL, SAVE_NORM, SAVE_SLICES]:
    os.makedirs(p, exist_ok=True)

print("Folders ready!")
print(f"Subjects found: {len(os.listdir(DATASET_PATH))}")


# ---------------- BIAS CORRECTION ----------------
def bias_field_correction(file_path):
    img = sitk.ReadImage(file_path, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(0.0001)
    corrected = corrector.Execute(img, mask)
    return corrected


# ---------------- SKULL STRIPPING ----------------
def skull_strip_3d(sitk_image):
    data = sitk.GetArrayFromImage(sitk_image)
    stripped = np.zeros_like(data)

    for z in range(data.shape[0]):
        slc = data[z, :, :]
        slc_u8 = cv2.normalize(slc, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        _, thresh = cv2.threshold(slc_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


# ---------------- NORMALIZATION ----------------
def normalize_volume(data):
    brain_voxels = data[data > 0]
    if len(brain_voxels) == 0:
        return data
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[data > 0] = (data[data > 0] - mean) / (std + 1e-8)
    return normalized


# ---------------- MID SAGITTAL ----------------
def get_best_sagittal(data):
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


# ---------------- ANISOTROPIC DIFFUSION (NEW) ----------------
def anisotropic_diffusion(img, iterations=10, kappa=30, gamma=0.1):
    img = img.astype(np.float32)

    for _ in range(iterations):
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east  = np.roll(img, -1, axis=1) - img
        west  = np.roll(img, 1, axis=1) - img

        cN = np.exp(-(north/kappa)**2)
        cS = np.exp(-(south/kappa)**2)
        cE = np.exp(-(east/kappa)**2)
        cW = np.exp(-(west/kappa)**2)

        img += gamma * (cN*north + cS*south + cE*east + cW*west)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- SAVE SAGITTAL ----------------
def save_sagittal_png(data, save_path, subject):
    slc = get_best_sagittal(data)

    # Step 1: Clip outliers
    p2 = np.percentile(slc[slc != 0], 2)
    p98 = np.percentile(slc[slc != 0], 98)
    slc_clipped = np.clip(slc, p2, p98)

    # Step 2: Resize
    slc_resized = cv2.resize(slc_clipped, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Step 3: 16-bit → 8-bit
    slc_16 = cv2.normalize(slc_resized, None, 0, 65535, cv2.NORM_MINMAX).astype("uint16")
    png = (slc_16 / 256).astype("uint8")

    # Step 4: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    png = clahe.apply(png)

    # 🔥 Step 5A: Anisotropic Diffusion
    png = anisotropic_diffusion(png, iterations=10, kappa=30, gamma=0.1)

    # 🔥 Step 5B: Non-Local Means
    png = cv2.fastNlMeansDenoising(png, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 🔥 Step 5C: Edge Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    png = cv2.filter2D(png, -1, kernel)

    # Step 6: Save
    cv2.imwrite(os.path.join(save_path, f"{subject}.png"), png)


# ---------------- SAVE NIFTI ----------------
def save_nifti(data, reference_sitk_img, save_path, subject, suffix):
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_sitk_img)
    fname = os.path.join(save_path, f"{subject}_{suffix}.nii.gz")
    sitk.WriteImage(out_img, fname)
    return fname


# ---------------- MAIN LOOP ----------------
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
        print(" 1A: Bias correction...")
        corrected_sitk = bias_field_correction(file_path)
        save_nifti(sitk.GetArrayFromImage(corrected_sitk), corrected_sitk, SAVE_BIAS, subject, "bias_corrected")

        print(" 1B: Skull stripping...")
        stripped_data = skull_strip_3d(corrected_sitk)
        save_nifti(stripped_data, corrected_sitk, SAVE_SKULL, subject, "skull_stripped")

        print(" 1C: Normalizing...")
        normalized_data = normalize_volume(stripped_data)
        save_nifti(normalized_data, corrected_sitk, SAVE_NORM, subject, "normalized")

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


# ---------------- VISUALIZATION ----------------
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
    plt.suptitle("Enhanced Medical-Grade Sagittal Slices", fontsize=13)
    plt.tight_layout()
    plt.show()
else:
    print("No PNG slices found")
