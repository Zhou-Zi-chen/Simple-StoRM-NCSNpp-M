# file: predictive_model.py
import torch
import torch.nn as nn
from ncsnpp_m import NCSNppM
from base_components import STFTProcessor

class PredictiveModel(nn.Module):
    """
    判别模型（第一阶段）
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        # NCSN++M架构，无条件
        self.network = NCSNppM(
            in_channels=2,
            out_channels=2,
            base_channels=base_channels,
            condition_dim=0,  # 无条件
            num_res_blocks=1
        )
        
        # STFT处理器
        self.stft = STFTProcessor()
        
        print("判别模型初始化:")
        total_params, _ = self.network.count_parameters()
        print(f"总参数: {total_params:,}")
        print(f"预计文件大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def forward(self, y_stft):
        """
        前向传播
        Args:
            y_stft: 噪声谱图 [B, 2, F, T]
        Returns:
            初步去噪谱图 [B, 2, F, T]
        """
        # 应用平方根幅度压缩
        y_stft_warped = self.stft.sqrt_magnitude_warping(y_stft)
        
        # 通过网络
        output_warped = self.network(y_stft_warped)
        
        return output_warped
    
    def enhance_waveform(self, noisy_waveform):
        """
        增强波形信号
        """
        if noisy_waveform.dim() == 1:
            noisy_waveform = noisy_waveform.unsqueeze(0)
        
        with torch.no_grad():
            noisy_stft = self.stft.stft(noisy_waveform)
            denoised_stft = self(noisy_stft)
            enhanced_waveform = self.stft.istft(denoised_stft)
        
        return enhanced_waveform.squeeze()