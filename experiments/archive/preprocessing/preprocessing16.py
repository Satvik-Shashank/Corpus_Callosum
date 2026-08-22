import os
import SimpleITK as sitk
import numpy as np
import cv2

# ================= PATHS =================
INPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
OUTPUT_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\19_sagittal"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ================= RESAMPLE (HIGH-RES FIX) =================
def resample_to_highres(image, new_spacing=(0.5, 0.5, 0.5)):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)  # best quality
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)


# ================= SAGITTAL =================
def get_sagittal(array):

    # (Z, Y, X)
    mid = array.shape[2] // 2
    slice_ = array[:, :, mid]

    slice_ = np.rot90(slice_, k=1)
    slice_ = np.fliplr(slice_)

    return slice_


# ================= ENHANCE =================
def enhance(img):

    # normalize properly
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)

    img = (img * 255).astype(np.uint8)

    # CLAHE (balanced)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


# ================= SHARPEN =================
def sharpen(img):

    # edge-preserving sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(img, 2.2, blur, -1.2, 0)

    return sharp


# ================= UPSCALE =================
def upscale(img):
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC)


# ================= PROCESS =================
def process_subject(file_path, subject_name):

    print(f"\nProcessing: {subject_name}")

    image = sitk.ReadImage(file_path)

    # 🔥 HIGH-RES RESAMPLING (MAIN FIX)
    image = resample_to_highres(image)

    array = sitk.GetArrayFromImage(image)

    sagittal = get_sagittal(array)

    sagittal = enhance(sagittal)

    # 🔥 sharpen BEFORE final upscale
    sagittal = sharpen(sagittal)

    final_img = upscale(sagittal)

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

print("\n✅ DONE — HIGH-RES SHARP SAGITTAL OUTPUT")