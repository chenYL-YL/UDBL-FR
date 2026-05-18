import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F



class MambaLite(nn.Module):
    """
    轻量版 Mamba：不依赖 mamba_ssm 和 selective_scan_cuda，
    只用 PyTorch 实现一个带门控 + depthwise conv 的长序列模块。
    输入形状: (B, L, C)
    """
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        # depthwise conv 沿着序列维度做局部建模
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x_in = self.in_proj(x)           # (B, L, 2C)
        x1, x2 = x_in.chunk(2, dim=-1)   # 两条分支

        # depthwise conv 在 seq 维度上建模
        x2 = x2.permute(0, 2, 1)         # (B, C, L)
        x2 = self.conv(x2)
        x2 = x2.permute(0, 2, 1)         # (B, L, C)

        # 门控
        y = x1 * self.act(x2)

        # 输出投影
        y = self.out_proj(y)
        return y


# =========================================================
#                      Deform Block
# =========================================================
class ModulatedDeformBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # offset：每个 3x3 kernel 有 18 个 offset（2 * 9）
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)
        # mask：每个点 9 个权重
        self.mask_conv = nn.Conv2d(channels, 9, 3, padding=1)

        # torchvision 的 DeformConv2d 支持 offset & mask
        self.deform = DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        offset = self.offset_conv(x)             # (B, 18, H, W)
        mask = torch.sigmoid(self.mask_conv(x))  # (B, 9, H, W)
        out = self.deform(x, offset, mask)
        return out


class DWConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return self.dw(x)


# =========================================================
#                        SS2D
# =========================================================
class SS2D(nn.Module):
    """
    2D-Selective-Scan 的简化实现：
    - 沿四个方向展开成序列
    - 用 MambaLite 做长序列建模
    - 再 reshape 回 2D
    """
    def __init__(self, channels):
        super().__init__()
        self.mamba = MambaLite(d_model=channels)

    def forward(self, x):
        B, C, H, W = x.shape

        seqs = []
        # 左→右
        seqs.append(x.permute(0, 2, 3, 1).reshape(B, -1, C))
        # 右→左
        x_rl = torch.flip(x, dims=[3])
        seqs.append(x_rl.permute(0, 2, 3, 1).reshape(B, -1, C))
        # 上→下
        x_ud = x.permute(0, 3, 2, 1)     # (B, W, H, C)
        seqs.append(x_ud.reshape(B, -1, C))
        # 下→上
        x_du = torch.flip(x, dims=[2]).permute(0, 3, 2, 1)
        seqs.append(x_du.reshape(B, -1, C))

        outs = []
        for seq in seqs:
            out = self.mamba(seq)        # (B, L, C)
            outs.append(out)

        # 四个方向平均
        out = sum(outs) / 4.0
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# =========================================================
#                    Vision Mamba Block
# =========================================================
class VisionMambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        # Branch A: Linear → Act
        self.fc_a = nn.Linear(channels, channels)

        # Branch B: Linear → DWConv → Act → SS2D → LN
        self.fc_b = nn.Linear(channels, channels)
        self.dwconv = DWConv(channels)
        self.ss2d = SS2D(channels)
        self.norm_b = nn.LayerNorm(channels)
        self.fc_out = nn.Linear(channels, channels)

        # SE 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H, W, C)
        x_norm = self.norm(x.permute(0, 2, 3, 1))

        # Branch A
        A = self.fc_a(x_norm)
        A = torch.relu(A)
        A = A.permute(0, 3, 1, 2)

        # Branch B
        B1 = self.fc_b(x_norm)
        B1 = B1.permute(0, 3, 1, 2)
        B1 = self.dwconv(B1)
        B1 = torch.relu(B1)
        B1 = self.ss2d(B1)
        B1 = self.norm_b(B1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        out = A * B1

        out = out.permute(0, 2, 3, 1)
        out = self.fc_out(out)
        out = out.permute(0, 3, 1, 2)

        # 通道注意力
        out = out * self.se(out)
        return out


# =========================================================
#                 Multi-View Context Module
# =========================================================
class MultiViewContext(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.d6  = nn.Conv2d(channels, channels, 3, padding=6,  dilation=6)
        self.d12 = nn.Conv2d(channels, channels, 3, padding=12, dilation=12)
        self.d18 = nn.Conv2d(channels, channels, 3, padding=18, dilation=18)

        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.id_conv = nn.Conv2d(channels, channels, 1)

        self.conv1_d6   = nn.Conv2d(channels, channels, 1)
        self.conv1_d12  = nn.Conv2d(channels, channels, 1)
        self.conv1_d18  = nn.Conv2d(channels, channels, 1)
        self.conv1_pool = nn.Conv2d(channels, channels, 1)
        self.conv1_id   = nn.Conv2d(channels, channels, 1)

        self.fuse = nn.Conv2d(channels * 5, channels, 1)

    def forward(self, x):
        f1 = self.conv1_d6(self.d6(x))
        f2 = self.conv1_d12(self.d12(x))
        f3 = self.conv1_d18(self.d18(x))
        f4 = self.conv1_pool(self.pool(x))
        f5 = self.conv1_id(self.id_conv(x))

        out = torch.cat([f1, f2, f3, f4, f5], dim=1)
        out = self.fuse(out)
        return out + x


# =========================================================
#              Patch Embedding / Merging / Expanding
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, x):
        return self.proj(x)


class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class PatchExpanding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


# =========================================================
#                  Deform-Mamba Block
# =========================================================
class DeformMambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.deform = ModulatedDeformBlock(channels)
        self.mamba  = VisionMambaBlock(channels)

    def forward(self, x):
        return self.deform(x) + self.mamba(x)

   

# =========================================================
#                  Deform-Mamba Net (2× / 4×   /  8x)
# =========================================================
class DeformMambaNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, scale=2):
        """
        scale: 2 或 4，对应 2× / 4× 超分
        - 输入:  (B, in_ch, H, W)    (LR)
        - 输出:  (B, out_ch, H*scale, W*scale)  (SR)
        """
        super().__init__()
        assert scale in [2, 4, 8], "scale 必须为 2 或 4或8"

        self.scale = scale

        # 1. PixelShuffle 上采样
        self.up_conv = nn.Conv2d(in_ch, in_ch * (scale ** 2), 3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(scale)

        # 2. Patch Embedding
        # feat = [96, 128, 384, 768]
        feat = [48, 64, 128, 256]
        self.patch_embed = PatchEmbed(in_ch, feat[0])

        # 3. Encoder
        self.e1  = DeformMambaBlock(feat[0])
        self.pm1 = PatchMerging(feat[0], feat[1])

        self.e2  = DeformMambaBlock(feat[1])
        self.pm2 = PatchMerging(feat[1], feat[2])

        self.e3  = DeformMambaBlock(feat[2])
        self.pm3 = PatchMerging(feat[2], feat[3])

        self.e4 = DeformMambaBlock(feat[3])

        # 4. Bottleneck MVC
        self.mvc = MultiViewContext(feat[3])

        # 5. Decoder
        self.v3  = VisionMambaBlock(feat[3])
        self.up3 = PatchExpanding(feat[3], feat[2])

        self.v2  = VisionMambaBlock(feat[2])
        self.up2 = PatchExpanding(feat[2], feat[1])

        self.v1  = VisionMambaBlock(feat[1])
        self.up1 = PatchExpanding(feat[1], feat[0])

        self.v0 = VisionMambaBlock(feat[0])
        self.final = nn.Conv2d(feat[0], out_ch, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        # x: LR
        # PixelShuffle 放大 scale 倍
        x = self.pixelshuffle(self.up_conv(x))   # (B, in_ch, H*scale, W*scale)

        # Patch Embedding
        x0 = self.patch_embed(x)

        # Encoder
        e1 = self.e1(x0)
        p1 = self.pm1(e1)

        e2 = self.e2(p1)
        p2 = self.pm2(e2)

        e3 = self.e3(p2)
        p3 = self.pm3(e3)

        e4 = self.e4(p3)

        # Bottleneck
        b = self.mvc(e4)

        # Decoder
        d3_in = b + p3
        d3    = self.v3(d3_in)
        up3   = self.up3(d3)

        d2_in = up3 + p2
        d2    = self.v2(d2_in)
        up2   = self.up2(d2)

        d1_in = up2 + p1
        d1    = self.v1(d1_in)
        up1   = self.up1(d1)

        d0_in = up1 + x0
        d0    = self.v0(d0_in)

        out = self.final(d0)   
        return out
    
    
    
