import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

NORM_PATH   = r"C:\Users\Admin\Desktop\infant_mri_project\preprocessed\3_normalized"
SEG_PATH    = r"C:\Users\Admin\Desktop\infant_mri_project\infant_mri_dataset\raw_mri"
SAVE_MODEL  = r"C:\Users\Admin\Desktop\infant_mri_project\models"
SAVE_RESULT = r"C:\Users\Admin\Desktop\infant_mri_project\results\unet"

for p in [SAVE_MODEL, SAVE_RESULT]:
    os.makedirs(p, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 10
BATCH_SIZE  = 4
LR          = 1e-4
IMG_SIZE    = 128
NUM_CLASSES = 4

print(f"Using device: {DEVICE}")

class MRIDataset(Dataset):
    def __init__(self, subject_list):
        self.samples = []
        for subject in subject_list:
            norm_file = os.path.join(NORM_PATH, f"{subject}_normalized.nii.gz")
            seg_file  = os.path.join(SEG_PATH, subject, "seg.nii.gz")
            if not (os.path.exists(norm_file) and os.path.exists(seg_file)):
                continue
            img = nib.load(norm_file).get_fdata().astype(np.float32)
            seg = nib.load(seg_file).get_fdata().astype(np.int64)
            for z in range(img.shape[0]):
                img_slice = img[z, :, :]
                seg_slice = seg[z, :, :]
                if np.sum(img_slice) == 0:
                    continue
                img_slice = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
                seg_slice = cv2.resize(seg_slice.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.int64)
                self.samples.append((img_slice, seg_slice))
        print(f"Total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, seg = self.samples[idx]
        img = torch.tensor(img).unsqueeze(0).float()
        seg = torch.tensor(seg).long()
        return img, seg

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(UNet, self).__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

def dice_score(pred, target, num_classes=4):
    dice = 0.0
    pred = torch.argmax(pred, dim=1)
    for c in range(1, num_classes):
        pred_c   = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            continue
        dice += (2 * intersection + 1e-8) / (union + 1e-8)
    return dice / (num_classes - 1)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_dice = 0, 0
    for imgs, segs in tqdm(loader, desc="Training"):
        imgs, segs = imgs.to(DEVICE), segs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, segs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(outputs, segs).item()
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for imgs, segs in tqdm(loader, desc="Validating"):
            imgs, segs = imgs.to(DEVICE), segs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, segs)
            total_loss += loss.item()
            total_dice += dice_score(outputs, segs).item()
    return total_loss / len(loader), total_dice / len(loader)

all_subjects = sorted([s for s in os.listdir(SEG_PATH) if os.path.isdir(os.path.join(SEG_PATH, s)) and s.startswith("s")])
print(f"Total subjects: {len(all_subjects)}")

train_sub, test_sub = train_test_split(all_subjects, test_size=0.3, random_state=42)
val_sub, test_sub   = train_test_split(test_sub, test_size=0.5, random_state=42)
print(f"Train: {len(train_sub)} | Val: {len(val_sub)} | Test: {len(test_sub)}")

print("Loading training data...")
train_dataset = MRIDataset(train_sub)
print("Loading validation data...")
val_dataset   = MRIDataset(val_sub)
print("Loading test data...")
test_dataset  = MRIDataset(test_sub)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model     = UNet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

best_dice = 0
train_losses, val_losses = [], []
train_dices, val_dices   = [], []

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_dice     = validate(model, val_loader, criterion)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)
    print(f"   Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
    print(f"   Val   Loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f}")
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), os.path.join(SAVE_MODEL, "unet_best.pth"))
        print(f"   Best model saved! Dice: {best_dice:.4f}")

print(f"Training done! Best Val Dice: {best_dice:.4f}")

print("Testing...")
model.load_state_dict(torch.load(os.path.join(SAVE_MODEL, "unet_best.pth")))
model.eval()
test_dice_scores = []
with torch.no_grad():
    for imgs, segs in tqdm(test_loader, desc="Testing"):
        imgs, segs = imgs.to(DEVICE), segs.to(DEVICE)
        outputs = model(imgs)
        test_dice_scores.append(dice_score(outputs, segs).item())

mean_test_dice = np.mean(test_dice_scores)
print(f"Test Dice Score: {mean_test_dice:.4f}")

with open(os.path.join(SAVE_RESULT, "metrics.txt"), "w") as f:
    f.write(f"Best Val Dice: {best_dice:.4f}\n")
    f.write(f"Test Dice:     {mean_test_dice:.4f}\n")
print("Metrics saved!")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses,   label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(train_dices, label="Train Dice")
axes[1].plot(val_dices,   label="Val Dice")
axes[1].set_title("Dice Score Curve")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_RESULT, "training_curves.png"))
plt.show()
print("Training curves saved!")
