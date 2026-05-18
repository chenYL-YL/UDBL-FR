import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fusionSR_model import FusionSR

# ==========================================
# 数据集加载 (与训练时相同)
# ==========================================
class MedicalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lr1_dir = os.path.join(root_dir, 'lr_1')
        self.lr2_dir = os.path.join(root_dir, 'lr_2')
        self.lr3_dir = os.path.join(root_dir, 'lr_3')
        # 确保只读取图像文件
        self.filenames = sorted([f for f in os.listdir(self.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr_img = Image.open(os.path.join(self.hr_dir, fname)).convert('RGB')
        lr1_img = Image.open(os.path.join(self.lr1_dir, fname)).convert('L')
        lr2_img = Image.open(os.path.join(self.lr2_dir, fname)).convert('L')
        lr3_img = Image.open(os.path.join(self.lr3_dir, fname)).convert('RGB')
        return self.transform(lr1_img), self.transform(lr2_img), self.transform(lr3_img), self.transform(hr_img), fname

# ==========================================
# 测试函数
# ==========================================
def test_model():
    # --- 配置参数 ---
    test_dir = '/root/autodl-tmp/CDDFuse/datasets_fusion/lrdown8/test'  # 测试集路径（也可以改为单独的test目录）
    model_path = 'checkpoints/double8.pth'  # 模型权重路径
    save_dir = 'test_results_8'  # 测试结果保存目录
    result_img_dir = os.path.join(save_dir, 'test_images_8')  # 测试图片保存目录
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_img_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # --- 加载模型 ---
    print("正在加载模型...")
    model = FusionSR(scale=8).to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    model.eval()
    
    # --- 初始化指标计算器 ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # --- 加载测试数据 ---
    test_dataset = MedicalDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"测试集包含 {len(test_dataset)} 张图片")
    print("开始测试...")
    
    # --- 测试并保存结果 ---
    results = []
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for idx, (lr1, lr2, lr3, hr, fname) in enumerate(test_loader):
            # 移动到设备
            lr1 = lr1.to(device)
            lr2 = lr2.to(device)
            lr3 = lr3.to(device)
            hr = hr.to(device)
            
            # 模型推理
            sr = model(lr1, lr2, lr3)
            sr = torch.clamp(sr, 0, 1)  # 限制在[0,1]范围
            
            # 计算指标
            psnr = psnr_metric(sr, hr).item()
            ssim = ssim_metric(sr, hr).item()
            
            total_psnr += psnr
            total_ssim += ssim
            
            # 保存结果
            filename = fname[0]  # batch_size=1
            results.append({
                'filename': filename,
                'psnr': psnr,
                'ssim': ssim
            })
            
            # 保存测试结果图片
            save_path = os.path.join(result_img_dir, filename)
            vutils.save_image(sr, save_path)
            
            print(f"[{idx+1}/{len(test_loader)}] {filename} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # --- 计算平均指标 ---
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print("\n" + "="*50)
    print("测试完成!")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print("="*50)
    
    # --- 保存指标到CSV ---
    csv_file = os.path.join(save_dir, 'test_metrics.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名', 'PSNR (dB)', 'SSIM'])
        for result in results:
            writer.writerow([result['filename'], f"{result['psnr']:.4f}", f"{result['ssim']:.6f}"])
        writer.writerow([])  # 空行
        writer.writerow(['平均值', f"{avg_psnr:.4f}", f"{avg_ssim:.6f}"])
    
    print(f"\n测试结果已保存:")
    print(f"  - 测试图片: {result_img_dir}")
    print(f"  - 指标文件: {csv_file}")

if __name__ == "__main__":
    test_model()