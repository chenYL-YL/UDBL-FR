import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model.fusionSR_model import FusionSR
import random


# ==========================================
# 1. 边缘损失 (CELoss)
# ==========================================
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        e0 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        e1 = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        e2 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernels = nn.Parameter(torch.cat([e0, e1, e2], dim=0), requires_grad=False)

    def forward(self, sr, gt):
        sr_gray = 0.299 * sr[:, 0:1, :, :] + 0.587 * sr[:, 1:2, :, :] + 0.114 * sr[:, 2:3, :, :]
        gt_gray = 0.299 * gt[:, 0:1, :, :] + 0.587 * gt[:, 1:2, :, :] + 0.114 * gt[:, 2:3, :, :]
        sr_feat = F.conv2d(sr_gray, self.kernels, padding=1)
        gt_feat = F.conv2d(gt_gray, self.kernels, padding=1)
        return F.l1_loss(sr_feat, gt_feat)


# ==========================================
# 2. 数据集
# ==========================================
class MedicalDataset(Dataset):
    """
    数据目录结构:
        root/
          hr/
          lr_1/   (T1, 灰度)
          lr_2/   (T2, 灰度)
          lr_3/   (PET, 彩色)
    """
    def __init__(self, root_dir, train=True, scale=2, lr_patch_size=128, augment=True):
        self.root_dir = root_dir
        self.hr_dir   = os.path.join(root_dir, 'hr')
        self.lr1_dir  = os.path.join(root_dir, 'lr_1')
        self.lr2_dir  = os.path.join(root_dir, 'lr_2')
        self.lr3_dir  = os.path.join(root_dir, 'lr_3')

        self.filenames = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        self.train     = train
        self.scale     = scale
        self.lr_patch_size  = lr_patch_size
        self.hr_patch_size  = lr_patch_size * scale
        self.augment   = augment

    def __len__(self):
        return len(self.filenames)

    def _paired_random_crop(self, lr1, lr2, lr3, hr):
        w_lr, h_lr = lr1.size
        w_hr, h_hr  = hr.size

        if w_hr == w_lr * self.scale and h_hr == h_lr * self.scale:
            x_lr = random.randint(0, w_lr - self.lr_patch_size)
            y_lr = random.randint(0, h_lr - self.lr_patch_size)
            x_hr = x_lr * self.scale
            y_hr = y_lr * self.scale
            lr1 = TF.crop(lr1, y_lr, x_lr, self.lr_patch_size, self.lr_patch_size)
            lr2 = TF.crop(lr2, y_lr, x_lr, self.lr_patch_size, self.lr_patch_size)
            lr3 = TF.crop(lr3, y_lr, x_lr, self.lr_patch_size, self.lr_patch_size)
            hr  = TF.crop(hr,  y_hr, x_hr, self.hr_patch_size, self.hr_patch_size)
        else:
            patch = self.lr_patch_size
            x = random.randint(0, min(w_lr, w_hr) - patch)
            y = random.randint(0, min(h_lr, h_hr) - patch)
            lr1 = TF.crop(lr1, y, x, patch, patch)
            lr2 = TF.crop(lr2, y, x, patch, patch)
            lr3 = TF.crop(lr3, y, x, patch, patch)
            hr  = TF.crop(hr,  y, x, patch, patch)
        return lr1, lr2, lr3, hr

    def _paired_augment(self, lr1, lr2, lr3, hr):
        if random.random() < 0.5:
            lr1 = TF.hflip(lr1); lr2 = TF.hflip(lr2); lr3 = TF.hflip(lr3); hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr1 = TF.vflip(lr1); lr2 = TF.vflip(lr2); lr3 = TF.vflip(lr3); hr = TF.vflip(hr)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            lr1 = TF.rotate(lr1, angle); lr2 = TF.rotate(lr2, angle)
            lr3 = TF.rotate(lr3, angle); hr  = TF.rotate(hr,  angle)
        return lr1, lr2, lr3, hr

    def _degrade_lr_only(self, lr1, lr2, lr3):
        if random.random() < 0.2:
            radius = random.uniform(0.1, 0.8)
            lr1 = lr1.filter(ImageFilter.GaussianBlur(radius))
            lr2 = lr2.filter(ImageFilter.GaussianBlur(radius))
            lr3 = lr3.filter(ImageFilter.GaussianBlur(radius))
        return lr1, lr2, lr3

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        hr_img  = Image.open(os.path.join(self.hr_dir,  fname)).convert('RGB')
        lr1_img = Image.open(os.path.join(self.lr1_dir, fname)).convert('L')
        lr2_img = Image.open(os.path.join(self.lr2_dir, fname)).convert('L')
        lr3_img = Image.open(os.path.join(self.lr3_dir, fname)).convert('RGB')

        if self.train:
            lr1_img, lr2_img, lr3_img, hr_img = self._paired_random_crop(
                lr1_img, lr2_img, lr3_img, hr_img)
            if self.augment:
                lr1_img, lr2_img, lr3_img, hr_img = self._paired_augment(
                    lr1_img, lr2_img, lr3_img, hr_img)

        lr1 = TF.to_tensor(lr1_img)   # [1, H, W]
        lr2 = TF.to_tensor(lr2_img)   # [1, H, W]
        lr3 = TF.to_tensor(lr3_img)   # [3, H, W]
        hr  = TF.to_tensor(hr_img)    # [3, H, W]

        return lr1, lr2, lr3, hr, fname


# ==========================================
# 3. 训练
# ==========================================
def main():
    # --- 路径配置 ---
    train_dir       = '/root/autodl-tmp/CDDFuse/datasets_fusion/train'
    val_dir         = '/root/autodl-tmp/CDDFuse/datasets_fusion/test'
    save_base_dir   = 'twoscales'
    best_img_dir    = os.path.join(save_base_dir, 'best_epoch_images')

    os.makedirs(save_base_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 超参数 ---
    epochs      = 200
    batch_size  = 4
    lr_rate     = 1e-4
    scale       = 2
    beta        = 0.5  # CELoss 权重

    # --- 数据加载 ---
    train_loader = DataLoader(
        MedicalDataset(train_dir, train=True, scale=scale),
        batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        MedicalDataset(val_dir, train=False, scale=scale),
        batch_size=1, shuffle=False)

    # --- 模型 ---
    model = FusionSR(dim=128, num_heads=8, scale=scale).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_l1 = nn.L1Loss()
    criterion_ce  = CELoss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # --- 日志文件 ---
    csv_file = os.path.join(save_base_dir, 'train_log.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Train_PSNR', 'Train_SSIM', 'Val_PSNR', 'Val_SSIM'])

    # --- 断点续训---
    ckpt_path = os.path.join(save_base_dir, 'latest.pth')
    start_epoch = 0
    best_psnr = 0.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_psnr   = ckpt['best_psnr']
        print(f"从 Epoch {start_epoch+1} 继续训练，最佳 PSNR: {best_psnr:.2f}")

    # --- 训练循环 ---
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0

        for i, (lr1, lr2, lr3, hr, _) in enumerate(train_loader):
            lr1, lr2, lr3, hr = lr1.to(device), lr2.to(device), lr3.to(device), hr.to(device)

            optimizer.zero_grad()

            sr = model(lr1, lr2, lr3)

            loss = criterion_l1(sr, hr) + beta * criterion_ce(sr, hr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sr_clamp = torch.clamp(sr, 0, 1)
            train_psnr += psnr_metric(sr_clamp, hr).item()
            train_ssim += ssim_metric(sr_clamp, hr).item()

        scheduler.step()

        avg_loss     = total_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        avg_train_ssim = train_ssim / len(train_loader)

        # --- 验证 ---
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for lr1, lr2, lr3, hr, _ in val_loader:
                lr1, lr2, lr3, hr = lr1.to(device), lr2.to(device), lr3.to(device), hr.to(device)
                sr = model(lr1, lr2, lr3)
                sr = torch.clamp(sr, 0, 1)
                val_psnr += psnr_metric(sr, hr).item()
                val_ssim += ssim_metric(sr, hr).item()

        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)

        # --- 记录日志 ---
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_loss:.6f}",
                f"{avg_train_psnr:.2f}",
                f"{avg_train_ssim:.4f}",
                f"{avg_val_psnr:.2f}",
                f"{avg_val_ssim:.4f}",
            ])

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Train PSNR: {avg_train_psnr:.2f} | SSIM: {avg_train_ssim:.4f} | "
            f"Val PSNR: {avg_val_psnr:.2f} | SSIM: {avg_val_ssim:.4f}"
        )

        # --- 保存最佳模型 ---
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), os.path.join(save_base_dir, 'best_model.pth'))

            print(f"===> New Best PSNR: {best_psnr:.2f} dB，保存最佳图片...")
            if os.path.exists(best_img_dir):
                import shutil
                shutil.rmtree(best_img_dir)
            os.makedirs(best_img_dir, exist_ok=True)

            with torch.no_grad():
                for lr1, lr2, lr3, hr, fname in val_loader:
                    lr1, lr2, lr3 = lr1.to(device), lr2.to(device), lr3.to(device)
                    sr_save = torch.clamp(model(lr1, lr2, lr3), 0, 1)
                    vutils.save_image(sr_save, os.path.join(best_img_dir, fname[0]))

        # --- 保存最新模型---
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, ckpt_path)

    print("Training Finished.")


if __name__ == "__main__":
    main()