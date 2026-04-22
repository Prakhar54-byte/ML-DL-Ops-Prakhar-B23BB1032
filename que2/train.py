import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

image_paths = sorted(glob.glob('data/CameraRGB/*.png'))
mask_paths = sorted(glob.glob('data/CameraMask/*.png'))

train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask

train_dataset = CityscapesDataset(train_imgs, train_masks)
test_dataset = CityscapesDataset(test_imgs, test_masks)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=23):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        
        return self.outc(x)

os.makedirs('Question2', exist_ok=True)

def compute_metrics(pred, target, num_classes=23):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    dice_list = []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()
        union = pred_sum + target_sum - intersection
        
        if target_sum == 0 and pred_sum == 0:
            continue
            
        iou = intersection / union if union > 0 else 0.0
        dice = (2. * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 0.0
        
        iou_list.append(iou)
        dice_list.append(dice)
        
    return np.mean(iou_list) if iou_list else 0.0, np.mean(dice_list) if dice_list else 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=23).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses, train_mious, train_mdices = [], [], []
num_epochs = 15

print("Starting Training...")

for epoch in range(num_epochs):
    model.train()
    running_loss, epoch_miou, epoch_mdice = 0.0, 0.0, 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            miou, mdice = compute_metrics(outputs, masks)
            epoch_miou += miou
            epoch_mdice += mdice
            
    avg_loss = running_loss / len(train_loader)
    avg_miou = epoch_miou / len(train_loader)
    avg_mdice = epoch_mdice / len(train_loader)
    
    train_losses.append(avg_loss)
    train_mious.append(avg_miou)
    train_mdices.append(avg_mdice)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} | mIOU: {avg_miou:.4f} | mDice: {avg_mdice:.4f}")

epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, label='Training Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('Question2/loss_curve.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_mious, label='mIOU', color='blue')
plt.plot(epochs_range, train_mdices, label='mDice', color='green')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Training mIOU and mDice Scores')
plt.legend()
plt.grid(True)
plt.savefig('Question2/metrics_curve.png')
plt.close()

print("Plots saved in 'Question2/' directory.")

print("\nEvaluating on Test Set...")
model.eval()
test_miou_total, test_mdice_total = 0.0, 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        miou, mdice = compute_metrics(outputs, masks)
        
        test_miou_total += miou
        test_mdice_total += mdice

final_test_miou = test_miou_total / len(test_loader)
final_test_mdice = test_mdice_total / len(test_loader)

print(f"\nFINAL TEST METRICS:")
print(f"mIOU: {final_test_miou:.4f}")
print(f"mDICE: {final_test_mdice:.4f}")