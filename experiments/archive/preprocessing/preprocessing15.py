import os
import SimpleITK as sitk
import numpy as np
import cv2

# ================= PATHS =================
INPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
OUTPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\18_sagittal"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ================= RESAMPLE (ISOTROPIC) =================
def resample_to_isotropic(image, new_spacing=(1.0, 1.0, 1.0)):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)  # high-quality
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)


# ================= SAGITTAL EXTRACTION =================
def get_sagittal_slice(array):

    # IMPORTANT: SimpleITK → NumPy gives (Z, Y, X)
    # sagittal = X-axis = last dimension

    mid = array.shape[2] // 2
    slice_ = array[:, :, mid]

    # orientation correction (matches your best output style)
    slice_ = np.rot90(slice_, k=1)
    slice_ = np.fliplr(slice_)

    return slice_


# ================= FIXED ENHANCEMENT =================
def enhance_image(img):

    # normalize safely (NO percentile clipping)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)

    img = (img * 255).astype(np.uint8)

    # CLAHE (your best-performing step)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
    img = clahe.apply(img)

    # sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    # 1. Load
    image = sitk.ReadImage(file_path)

    # 2. Isotropic resampling (KEY STEP)
    iso_image = resample_to_isotropic(image)

    # 3. Convert to numpy
    array = sitk.GetArrayFromImage(iso_image)

    # 4. Extract sagittal
    sagittal = get_sagittal_slice(array)

    # 5. Enhance (fixed)
    sagittal = enhance_image(sagittal)

    # 6. Upscale
    final_img = upscale_image(sagittal)

    # 7. Save
    save_path = os.path.join(OUTPUT_PATH, f"{subject_name}_sagittal.png")
    cv2.imwrite(save_path, final_img)

    print("   Saved →", save_path)


# ================= RUN =================
files = sorted(os.listdir(INPUT_PATH))

for file in files:

    if not file.endswith(".nii.gz"):
        continue

    subject_name = file.replace("_normalized.nii.gz", "")
    file_path = os.path.join(INPUT_PATH, file)

    try:
        process_subject(file_path, subject_name)
    except Exception as e:
        print(f"Error in {subject_name}: {e}")

print("\n✅ DONE — FINAL ISOTROPIC SAGITTAL OUTPUT READY")