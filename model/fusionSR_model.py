import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from torchinfo import summary
from einops import rearrange
import clip
from deform_mamba import DeformMambaNet 
import torch.fft as fft


# =========================================================
#              结构分支编码器
# =========================================================
class StructureBranchEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=128, num_blocks=2):
        super().__init__()
        self.modal1_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        self.modal2_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        self.structure_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, modal1, modal2, return_parts=False):
        """
        modal1: (B, C, H, W) - 结构模态1
        modal2: (B, C, H, W) - 结构模态2
        """
        feat1 = self.modal1_embed(modal1)
        feat2 = self.modal2_embed(modal2)

        feat_concat = torch.cat([feat1, feat2], dim=1)         # (B, 2*dim, H, W)
        feat_context = self.structure_conv(feat_concat)        # (B, dim, H, W)
        return feat_context
    


# =========================================================
#              功能分支编码器
# =========================================================
class FunctionBranchEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=128, num_blocks=2):
        super().__init__()
        self.modal3_embed = nn.Conv2d(in_channels, dim, kernel_size=7, padding=3, bias=False)

        self.function_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.modal3_lf_embed = nn.Conv2d(in_channels, dim, kernel_size=7, padding=3, bias=False)
    def forward(self, modal3, return_parts=False):
        """
        modal3: (B, C, H, W) - 功能模态
        """

        # =====================================================
        # 1) 原始上下文路径：提取功能全局上下文
        # =====================================================
        feat = self.modal3_embed(modal3)
        feat_context = self.function_conv(feat)   # (B, dim, H, W)
        return feat_context

    

    

class ResBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))
    

#=========================================================
# 交叉注意力融合模块 (双分支)
#=========================================================
class CrossAttentionFusion(nn.Module):
    """
    双分支交叉注意力融合:
    1. 结构分支作为Q，功能分支作为K,V -> 得到 struct2func_feat
    2. 功能分支作为Q，结构分支作为K,V -> 得到 func2struct_feat
    3. 两者在通道维度拼接
    """
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # ===== 分支1: 结构 -> 功能 =====
        self.q1_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k1_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v1_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q1_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k1_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v1_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        # ===== 分支2: 功能 -> 结构 =====
        self.q2_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k2_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v2_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q2_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k2_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v2_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        
        self.project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=False)

    def forward(self, struct_feat, func_feat):
        """
        struct_feat: 结构分支特征 (B, C, H, W)
        func_feat: 功能分支特征 (B, C, H, W)
        返回: 融合后的特征 (B, C, H, W)
        """
        b, c, h, w = struct_feat.shape

        # ==================== 分支1: 结构 -> 功能 ====================
        # Q来自结构, K,V来自功能
        q1 = self.q1_dw(self.q1_proj(struct_feat))
        k1 = self.k1_dw(self.k1_proj(func_feat))
        v1 = self.v1_dw(self.v1_proj(func_feat))

        q1 = q1.view(b, self.num_heads, c // self.num_heads, -1)
        k1 = k1.view(b, self.num_heads, c // self.num_heads, -1)
        v1 = v1.view(b, self.num_heads, c // self.num_heads, -1)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1)
        out1 = out1.view(b, c, h, w)

        # 残差连接
        struct2func = struct_feat + out1

        # ==================== 分支2: 功能 -> 结构 ====================
        # Q来自功能, K,V来自结构
        q2 = self.q2_dw(self.q2_proj(func_feat))
        k2 = self.k2_dw(self.k2_proj(struct_feat))
        v2 = self.v2_dw(self.v2_proj(struct_feat))

        q2 = q2.view(b, self.num_heads, c // self.num_heads, -1)
        k2 = k2.view(b, self.num_heads, c // self.num_heads, -1)
        v2 = v2.view(b, self.num_heads, c // self.num_heads, -1)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2)
        out2 = out2.view(b, c, h, w)

        # 残差连接
        func2struct = func_feat + out2

        # ==================== 通道拼接 ====================
        fused = torch.cat([struct2func, func2struct], dim=1)  # (B, 2*C, H, W)

        return fused



class FusionSR(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        out_channels=3,
        scale=2,
        num_refine_blocks=2,
    ):
        super().__init__()
        assert scale in [2, 4, 8]
        # 结构分支
        self.struct_encoder = StructureBranchEncoder(in_channels=1, dim=dim)

        # 功能分支
        self.func_encoder = FunctionBranchEncoder(in_channels=3, dim=dim)

        # 融合模块,交叉注意力机制
        self.fusion = CrossAttentionFusion(dim=dim, num_heads=num_heads)

        # refine
        refine = [ResBlock(2*dim) for _ in range(num_refine_blocks)]
        self.refine = nn.Sequential(*refine) if len(refine) > 0 else nn.Identity()
        

        # SR head
        self.sr_net = DeformMambaNet(
            in_ch=2*dim,
            out_ch=out_channels,
            scale=scale,
        )

    def forward(self, t1, t2, pet):
        # =====================================================
        # 1) 编码阶段：提取结构分支和功能分支特征
        # =====================================================
        struct_feat= self.struct_encoder(t1, t2, return_parts=True)
        func_feat = self.func_encoder(pet, return_parts=True)
        #fused_feat = torch.cat([struct_feat, func_feat], dim=1)
        fused_feat = self.fusion(struct_feat, func_feat)

        # =====================================================
        # 3) refine + SR
        # =====================================================
        fused_feat = self.refine(fused_feat)
        sr = self.sr_net(fused_feat)

        return sr


if __name__ == "__main__":
    # 统计参数量
    model = FusionSR(dim=128, num_heads=8, scale=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")










