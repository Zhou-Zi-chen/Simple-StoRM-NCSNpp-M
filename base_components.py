# file: base_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# file: base_components.py (修正ResnetBlock)
class ResnetBlock(nn.Module):
    """NCSN++中的残差块 - 修正条件维度处理"""
    
    def __init__(self, channels, dropout=0.1, condition_dim=0):
        super().__init__()
        
        self.channels = channels
        self.condition_dim = condition_dim
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        
        # 时间/噪声条件嵌入
        if condition_dim > 0:
            # 修正：将condition_dim投影到channels*2
            self.cond_proj = nn.Linear(condition_dim, channels * 2)
            print(f"  ResnetBlock: channels={channels}, condition_dim={condition_dim}")
        
        # 激活函数和dropout
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, condition=None):
        residual = x
        
        # 第一个卷积 + 归一化
        x = self.conv1(x)
        x = self.norm1(x)
        
        # 添加条件信息
        if condition is not None and self.condition_dim > 0:
            # 投影条件到正确维度
            condition = self.cond_proj(condition)  # [B, condition_dim] -> [B, channels*2]
            scale, shift = torch.chunk(condition, 2, dim=1)
            scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
            shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)
            x = x * (1 + scale)
            x = x + shift
        
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二个卷积 + 归一化
        x = self.conv2(x)
        x = self.norm2(x)
        
        # 残差连接
        return self.activation(x + residual)


class AttentionBlock(nn.Module):
    """注意力块（仅在瓶颈使用）"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(channels // 4, 1), channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 空间注意力
        residual = x
        x_norm = self.norm(x)
        
        # 计算Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 多头注意力（简化为单头）
        q = q.reshape(B, C, -1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1)
        
        attention = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (C ** 0.5), dim=-1)
        x_attn = torch.bmm(v, attention.transpose(1, 2))
        x_attn = x_attn.reshape(B, C, H, W)
        x_attn = self.proj(x_attn)
        
        # 通道注意力
        channel_weights = self.channel_attention(x)
        
        return residual + x_attn * channel_weights


class Downsample(nn.Module):
    """下采样层"""
    
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """上采样层"""
    
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class STFTProcessor:
    """STFT处理类"""
    
    def __init__(self, n_fft=510, hop_length=128, win_length=510):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        window = torch.hann_window(win_length)
        self.window = torch.sqrt(window)
    
    def stft(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        stft = torch.stft(waveform, 
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        window=self.window,
                        return_complex=True)
        
        real = stft.real.unsqueeze(1)
        imag = stft.imag.unsqueeze(1)
        
        return torch.cat([real, imag], dim=1)
    
    def istft(self, complex_spec):
        real = complex_spec[:, 0:1, :, :]
        imag = complex_spec[:, 1:2, :, :]
        
        stft = torch.complex(real.squeeze(1), imag.squeeze(1))
        
        waveform = torch.istft(stft,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window)
        
        return waveform
    
    def sqrt_magnitude_warping(self, complex_spec):
        real = complex_spec[:, 0:1, :, :]
        imag = complex_spec[:, 1:2, :, :]
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        magnitude_warped = torch.sqrt(magnitude)
        
        phase = torch.atan2(imag, real)
        real_warped = magnitude_warped * torch.cos(phase)
        imag_warped = magnitude_warped * torch.sin(phase)
        
        return torch.cat([real_warped, imag_warped], dim=1)
    
class PCSampler:
    """预测器-校正器采样器"""
    
    def __init__(self, num_steps=50, corrector_steps=1, r=0.5, method='euler'):
        """
        初始化采样器
        Args:
            num_steps: 时间步数 (N=50)
            corrector_steps: 校正器步数 (1)
            r: 步长参数 (0.5)
            method: 预测器方法 ('euler', 'heun')
        """
        self.num_steps = num_steps
        self.corrector_steps = corrector_steps
        self.r = r
        self.method = method
        self.dt = 1.0 / num_steps
    
    def predictor_step(self, model, x, denoised_stft, noisy_stft, t):
        """
        预测器步骤（欧拉-丸山）
        """
        # 获取分数估计
        t_tensor = torch.ones(x.shape[0], device=x.device) * t
        score = model.diffusion_model(x, noisy_stft, denoised_stft, t_tensor)
        
        # 计算漂移项
        drift = model.sde.drift(x, denoised_stft)
        g_t = model.sde.g(t)
        
        # 欧拉-丸山更新
        dx = (-drift + g_t**2 * score) * self.dt
        x_pred = x + dx
        
        return x_pred, score, g_t
    
    def corrector_step(self, model, x, denoised_stft, noisy_stft, t, score, g_t):
        """
        校正器步骤（退火朗之万动力学）
        """
        x_corr = x
        
        for _ in range(self.corrector_steps):
            # 重新估计分数
            t_corr = torch.ones(x_corr.shape[0], device=x_corr.device) * (t - self.dt/2)
            score_corr = model.diffusion_model(x_corr, noisy_stft, denoised_stft, t_corr)
            
            # 朗之万动力学更新
            step_size = self.r * (g_t * torch.sqrt(torch.tensor(2 * self.dt, device=g_t.device)))**2
            noise = torch.randn_like(x_corr)
            
            x_corr = x_corr + step_size * score_corr + torch.sqrt(2 * step_size) * noise
        
        return x_corr
    
    def sample(self, model, x_init, denoised_stft, noisy_stft):
        """
        执行完整的预测器-校正器采样
        """
        x = x_init
        
        for i in range(self.num_steps):
            t = 1.0 - i * self.dt  # 从1到0
            
            # ===== 预测器步骤 =====
            x_pred, score, g_t = self.predictor_step(model, x, denoised_stft, noisy_stft, t)
            
            # 添加噪声（除了最后一步）
            if i < self.num_steps - 1:
                noise = torch.randn_like(x_pred)
                x_pred = x_pred + g_t * torch.sqrt(torch.tensor(2 * self.dt, device=g_t.device)) * noise
            
            # ===== 校正器步骤 =====
            if self.corrector_steps > 0:
                x_corr = self.corrector_step(model, x_pred, denoised_stft, noisy_stft, t, score, g_t)
                x = x_corr
            else:
                x = x_pred
        
        return x