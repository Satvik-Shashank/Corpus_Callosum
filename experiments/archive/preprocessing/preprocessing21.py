import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset"
SAVE_SLICES = r"C:\Users\Admin\Desktop\infant_mri_project\final_output"

os.makedirs(SAVE_SLICES, exist_ok=True)

print("Folders ready!")
print(f"Subjects found: {len(os.listdir(DATASET_PATH))}")


# ---------------- BIAS CORRECTION ----------------
def bias_field_correction(file_path):
    img = sitk.ReadImage(file_path, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50]*4)
    return corrector.Execute(img, mask)


# ---------------- SKULL STRIPPING ----------------
def skull_strip_3d(sitk_image):
    data = sitk.GetArrayFromImage(sitk_image)
    stripped = np.zeros_like(data)

    for z in range(data.shape[0]):
        slc = data[z]
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

        stripped[z] = slc * mask

    return stripped


# ---------------- NORMALIZATION ----------------
def normalize_volume(data):
    brain = data[data > 0]
    if len(brain) == 0:
        return data
    data[data > 0] = (brain - np.mean(brain)) / (np.std(brain) + 1e-8)
    return data


# ---------------- REGISTRATION ----------------
def register_to_template(img):
    fixed = sitk.Cast(img, sitk.sitkFloat32)

    transform = sitk.CenteredTransformInitializer(
        fixed, img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 100)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(transform, False)

    final_transform = reg.Execute(fixed, img)

    return sitk.Resample(img, fixed, final_transform,
                         sitk.sitkLinear, 0.0, img.GetPixelID())


# ---------------- SAGITTAL ----------------
def get_best_sagittal(data):
    center = data.shape[2] // 2
    best, score = None, float("inf")

    for i in range(center - 5, center + 6):
        s = data[:, :, i]
        mid = s.shape[1] // 2
        left = s[:, :mid]
        right = np.fliplr(s[:, mid+1:])
        w = min(left.shape[1], right.shape[1])
        diff = np.mean((left[:, :w] - right[:, :w])**2)
        if diff < score:
            score = diff
            best = s

    return np.fliplr(np.rot90(best))


# ---------------- DIFFUSION ----------------
def anisotropic_diffusion(img, it=10, k=30, g=0.1):
    img = img.astype(np.float32)
    for _ in range(it):
        n = np.roll(img, -1, 0) - img
        s = np.roll(img, 1, 0) - img
        e = np.roll(img, -1, 1) - img
        w = np.roll(img, 1, 1) - img

        cN = np.exp(-(n/k)**2)
        cS = np.exp(-(s/k)**2)
        cE = np.exp(-(e/k)**2)
        cW = np.exp(-(w/k)**2)

        img += g*(cN*n + cS*s + cE*e + cW*w)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- SUPER RES ----------------
def super_res(img):
    h, w = img.shape

    up = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

    # Better for MRI (no fake textures)
    blur = cv2.GaussianBlur(up, (0,0), 1.0)
    sharp = cv2.addWeighted(up, 1.6, blur, -0.6, 0)

    return cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)

# ---------------- SEGMENTATION ----------------
def segment_cc(img):
    _, th = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(clean)

    h, w = img.shape
    cx = w // 2
    mask = np.zeros_like(img)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if abs((x+bw//2)-cx) < w*0.2 and area > 500:
            mask[labels == i] = 255

    return mask


# ---------------- MAIN PROCESS ----------------
for subject in sorted(os.listdir(DATASET_PATH)):
    path = os.path.join(DATASET_PATH, subject, "t1.nii.gz")
    if not os.path.exists(path):
        continue

    print("Processing:", subject)

    img = bias_field_correction(path)
    data = skull_strip_3d(img)
    data = normalize_volume(data)

    sitk_img = sitk.GetImageFromArray(data)
    sitk_img.CopyInformation(img)

    reg = register_to_template(sitk_img)
    data = sitk.GetArrayFromImage(reg)

    slc = get_best_sagittal(data)

    p2, p98 = np.percentile(slc[slc!=0], [2, 98])
    slc = np.clip(slc, p2, p98)

    slc = cv2.resize(slc, (512,512))
    slc = cv2.normalize(slc, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    clahe = cv2.createCLAHE(2.0, (4,4))
    slc = clahe.apply(slc)

    slc = anisotropic_diffusion(slc)
    slc = cv2.fastNlMeansDenoising(slc, None, 10)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    slc = cv2.filter2D(slc, -1, kernel)

    slc = super_res(slc)

    # 🔥 segmentation
    mask = segment_cc(slc)

    overlay = cv2.cvtColor(slc, cv2.COLOR_GRAY2BGR)
    overlay[mask == 255] = [0,0,255]

    cv2.imwrite(os.path.join(SAVE_SLICES, f"{subject}.png"), slc)
    cv2.imwrite(os.path.join(SAVE_SLICES, f"{subject}_mask.png"), mask)
    cv2.imwrite(os.path.join(SAVE_SLICES, f"{subject}_overlay.png"), overlay)

print("DONE 🚀")