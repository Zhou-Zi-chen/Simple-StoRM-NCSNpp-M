# file: diffusion_model.py (修正维度)
import torch
import torch.nn as nn
import numpy as np
from ncsnpp_m import NCSNppM
from base_components import STFTProcessor

class TimeEmbedding(nn.Module):
    """时间/噪声水平嵌入 - 修正输出维度"""
    
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.embed_dim = embed_dim
        # 修正：输出维度应该与ResnetBlock期望的condition_dim一致
        self.output_dim = 128  # 硬编码为128，与ResnetBlock匹配
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, self.output_dim)  # 输出128维
        )
        
        print(f"TimeEmbedding: 输入维度={embed_dim}, 输出维度={self.output_dim}")
    
    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # 正弦位置编码
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        return self.proj(emb)


class StoRMDiffusionModel(nn.Module):
    """
    StoRM扩散模型 - 修正维度配置
    """
    
    def __init__(self, base_channels=32, time_embed_dim=256):
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        
        # 时间嵌入 - 输出128维
        self.time_embed = TimeEmbedding(time_embed_dim)
        condition_dim = 128  # TimeEmbedding的输出维度
        
        # NCSN++M架构 - 使用正确的condition_dim
        self.network = NCSNppM(
            in_channels=6,
            out_channels=2,
            base_channels=base_channels,
            condition_dim=condition_dim,  # 使用128
            num_res_blocks=1
        )
        
        # STFT处理器
        self.stft = STFTProcessor()
        
        print("扩散模型初始化:")
        total_params, _ = self.network.count_parameters()
        print(f"总参数: {total_params:,}")
        print(f"预计文件大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"输入通道: 6 (xτ + y + Dθ(y))")
        print(f"时间嵌入维度: 输入={time_embed_dim}, 输出={condition_dim}")
    
    def forward(self, x_tau, noisy_stft, denoised_stft, time):
        # 确保time是正确形状
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=x_tau.device).expand(x_tau.shape[0])
        elif time.dim() == 0:
            time = time.unsqueeze(0).expand(x_tau.shape[0])
        
        # 时间嵌入 - 输出[B, 128]
        t_emb = self.time_embed(time)
        
        # 应用平方根幅度压缩
        x_tau_warped = self.stft.sqrt_magnitude_warping(x_tau)
        noisy_stft_warped = self.stft.sqrt_magnitude_warping(noisy_stft)
        denoised_stft_warped = self.stft.sqrt_magnitude_warping(denoised_stft)
        
        # 拼接输入
        network_input = torch.cat([
            x_tau_warped,
            noisy_stft_warped,
            denoised_stft_warped
        ], dim=1)
        
        # 通过网络
        score_estimate = self.network(network_input, t_emb)
        
        return score_estimate


# OrnsteinUhlenbeckSDE保持不变
class OrnsteinUhlenbeckSDE:
    def __init__(self, gamma=1.5, sigma_min=0.05, sigma_max=0.5):
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        self.log_sigma_ratio = np.log(sigma_max / sigma_min)
        self.const_denom = gamma + self.log_sigma_ratio
    
    def g(self, tau):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau * \
               np.sqrt(2 * self.log_sigma_ratio)
    
    def drift(self, x, y):
        return self.gamma * (y - x)
    
    def marginal_mean(self, x0, y, tau):
        return torch.exp(-self.gamma * tau) * x0 + (1 - torch.exp(-self.gamma * tau)) * y
    
    def marginal_std(self, tau):
        tau_tensor = torch.tensor(tau) if not isinstance(tau, torch.Tensor) else tau
        sigma_sq = (self.sigma_min ** 2 * 
                   ((self.sigma_max / self.sigma_min) ** (2 * tau_tensor) - 
                    torch.exp(-2 * self.gamma * tau_tensor)) * 
                    self.log_sigma_ratio / self.const_denom)
        # 当tau=0时，公式中的两项抵消，sigma_sq可能接近0
        
        # 数值稳定性：确保平方根有效
        return torch.sqrt(torch.clamp(sigma_sq, min=1e-6))
    
    def sample_perturbed_state(self, x0, y, tau, noise=None):
        if isinstance(tau, (int, float)):
            tau = torch.tensor(tau).to(x0.device).expand(x0.shape[0])
        
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if tau.dim() == 1:
            tau = tau.view(-1, 1, 1, 1)
        
        mean = self.marginal_mean(x0, y, tau)
        std = self.marginal_std(tau).view(-1, 1, 1, 1)
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        x_tau = mean + std * noise
        
        return x_tau, noise