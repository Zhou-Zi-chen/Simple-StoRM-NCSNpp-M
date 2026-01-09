# file: ncsnpp_m.py (修正时间嵌入)
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_components import ResnetBlock, AttentionBlock, Downsample, Upsample

class NCSNppM(nn.Module):
    """
    NCSN++M网络架构 - 修正时间嵌入维度
    """
    
    def __init__(self, 
                in_channels=2,
                out_channels=2,
                base_channels=32,
                condition_dim=0,  # 注意：这里的condition_dim应该是时间嵌入的维度
                num_res_blocks=1):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.condition_dim = condition_dim  # 这应该是时间嵌入的输出维度
        
        # ========== 输入层 ==========
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # ========== 编码器路径 ==========
        self.down1_blocks = nn.ModuleList([
            ResnetBlock(base_channels, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        self.downsample1 = Downsample(base_channels, base_channels * 2)
        
        self.down2_blocks = nn.ModuleList([
            ResnetBlock(base_channels * 2, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        self.downsample2 = Downsample(base_channels * 2, base_channels * 4)
        
        self.down3_blocks = nn.ModuleList([
            ResnetBlock(base_channels * 4, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        self.downsample3 = Downsample(base_channels * 4, base_channels * 8)
        
        # ========== 瓶颈层 ==========
        self.bottleneck1 = ResnetBlock(base_channels * 8, condition_dim=condition_dim)
        self.bottleneck_attention = AttentionBlock(base_channels * 8)
        self.bottleneck2 = ResnetBlock(base_channels * 8, condition_dim=condition_dim)
        
        # ========== 解码器路径 ==========
        self.upsample3 = Upsample(base_channels * 8, base_channels * 4)
        self.up3_blocks = nn.ModuleList([
            ResnetBlock(base_channels * 4, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        
        self.upsample2 = Upsample(base_channels * 4, base_channels * 2)
        self.up2_blocks = nn.ModuleList([
            ResnetBlock(base_channels * 2, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        
        self.upsample1 = Upsample(base_channels * 2, base_channels)
        self.up1_blocks = nn.ModuleList([
            ResnetBlock(base_channels, condition_dim=condition_dim)
            for _ in range(num_res_blocks)
        ])
        
        # ========== 输出层 ==========
        self.output_norm = nn.GroupNorm(32, base_channels)
        self.output_activation = nn.SiLU()
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # 时间/噪声嵌入 - 修正：输入应该是时间嵌入的维度
        if condition_dim > 0:
            # 注意：这里不需要time_embed，因为条件已经由外部TimeEmbedding处理好了
            # ResnetBlock会直接使用传入的condition
            self.time_embed = nn.Identity()  # 改为恒等映射
            print(f"  NCSN++M: condition_dim={condition_dim}, 使用恒等映射")
    
    def forward(self, x, condition=None):
        """
        前向传播
        """
        # 时间/噪声嵌入 - 简单传递，不进行额外处理
        t_emb = condition  # 直接使用传入的条件
        
        # 输入卷积
        x = self.input_conv(x)
        
        # ========== 编码器路径 ==========
        # 第一层
        for block in self.down1_blocks:
            x = block(x, t_emb)
        h1 = x.clone()
        
        x = self.downsample1(x)
        
        # 第二层
        for block in self.down2_blocks:
            x = block(x, t_emb)
        h2 = x.clone()
        
        x = self.downsample2(x)
        
        # 第三层
        for block in self.down3_blocks:
            x = block(x, t_emb)
        h3 = x.clone()
        
        x = self.downsample3(x)
        
        # ========== 瓶颈层 ==========
        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck_attention(x)
        x = self.bottleneck2(x, t_emb)
        
        # ========== 解码器路径 ==========
        # 上采样第三层
        x = self.upsample3(x)
        
        # 调整h3尺寸
        if h3.shape[2:] != x.shape[2:]:
            h3 = F.interpolate(h3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = x + h3
        for block in self.up3_blocks:
            x = block(x, t_emb)
        
        # 上采样第二层
        x = self.upsample2(x)
        
        # 调整h2尺寸
        if h2.shape[2:] != x.shape[2:]:
            h2 = F.interpolate(h2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = x + h2
        for block in self.up2_blocks:
            x = block(x, t_emb)
        
        # 上采样第一层
        x = self.upsample1(x)
        
        # 调整h1尺寸
        if h1.shape[2:] != x.shape[2:]:
            h1 = F.interpolate(h1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = x + h1
        for block in self.up1_blocks:
            x = block(x, t_emb)
        
        # ========== 输出层 ==========
        x = self.output_norm(x)
        x = self.output_activation(x)
        x = self.output_conv(x)
        
        return x
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable