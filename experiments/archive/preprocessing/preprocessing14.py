import os
import SimpleITK as sitk
import numpy as np
import cv2

# ================= PATHS =================
INPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
OUTPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\17_sagittal"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ================= RESAMPLE =================
def resample_to_isotropic(image, new_spacing=(1.0, 1.0, 1.0)):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)  # keep high quality
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)


# ================= SAGITTAL =================
def get_sagittal_slice(array):

    # (Z, Y, X)
    mid = array.shape[2] // 2
    slice_ = array[:, :, mid]

    # orientation fix
    slice_ = np.rot90(slice_, k=1)
    slice_ = np.fliplr(slice_)

    return slice_


# ================= FIXED ENHANCEMENT =================
def enhance_image(img):

    # ✅ NO percentile clipping (THIS FIXES YOUR ISSUE)

    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)

    img = (img * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    img = clahe.apply(img)

    # sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


# ================= UPSCALE =================
def upscale_image(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    image = sitk.ReadImage(file_path)

    # isotropic
    iso_image = resample_to_isotropic(image)

    # to numpy
    array = sitk.GetArrayFromImage(iso_image)

    # sagittal
    sagittal = get_sagittal_slice(array)

    # FIXED enhancement
    sagittal = enhance_image(sagittal)

    # upscale
    final_img = upscale_image(sagittal)

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

print("\n✅ DONE — FIXED ISOTROPIC SAGITTAL OUTPUT")