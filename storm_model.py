# file: storm_model.py (修正版本)
import torch
import torch.nn as nn
from predictive_model import PredictiveModel
from diffusion_model import StoRMDiffusionModel, OrnsteinUhlenbeckSDE
from base_components import PCSampler

class StoRMModel(nn.Module):
    """
    完整的StoRM模型 - 修正时间传递问题
    """
    
    def __init__(self, base_channels=32, condition_dim=256):
        super().__init__()
        
        # 判别模型
        self.predictive_model = PredictiveModel(base_channels)
        
        # 扩散模型
        self.diffusion_model = StoRMDiffusionModel(base_channels, condition_dim)
        
        # SDE
        self.sde = OrnsteinUhlenbeckSDE()
        
        # 参数
        self.sigma_min = 0.05
        self.sigma_max = 0.5
        self.gamma = 1.5
        self.T = 1.0
        self.condition_dim = condition_dim

        # 添加采样器配置
        self.num_steps = 50  # 反向扩散步数 N=50
        self.corrector_steps = 1  # 校正器步数
        self.r = 0.5  # 步长参数
        
        # 创建采样器
        self.sampler = PCSampler(
            num_steps=self.num_steps,
            corrector_steps=self.corrector_steps,
            r=self.r,
            method='euler'
        )
        
        print(f"采样配置: N={self.num_steps}, 校正器步数={self.corrector_steps}, r={self.r}")
        
    def forward(self, noisy_stft, time=None, x0=None, return_intermediate=False):
        batch_size = noisy_stft.shape[0]
        device = noisy_stft.device
        
        # 如果没有提供时间，随机采样
        if time is None:
            time = torch.rand(batch_size, device=device)
        else:
            # 确保时间是一维张量
            if isinstance(time, (int, float)):
                time = torch.tensor([time], device=device).expand(batch_size)
            elif time.dim() == 0:
                time = time.unsqueeze(0).expand(batch_size)
            elif time.dim() == 2:
                time = time.squeeze(-1) if time.shape[1] == 1 else time.squeeze(0)
        
        # 第一阶段：判别模型
        denoised_stft = self.predictive_model(noisy_stft)
        
        # 准备扩散模型的输入
        if self.training and x0 is not None:
            # 训练模式：使用真实干净样本
            # 确保time是正确形状用于SDE
            time_for_sde = time.view(-1, 1, 1, 1) if time.dim() == 1 else time
            x_tau, noise = self.sde.sample_perturbed_state(x0, denoised_stft, time_for_sde)
        else:
            # 推理模式或没有x0：从先验采样
            std = self.sde.marginal_std(time).view(-1, 1, 1, 1).to(device)
            noise = torch.randn_like(denoised_stft)
            x_tau = denoised_stft + std * noise
        
        # 第二阶段：扩散模型 - 分数估计
        score = self.diffusion_model(x_tau, noisy_stft, denoised_stft, time)
        
        if return_intermediate:
            return denoised_stft, score, x_tau
        else:
            return score
    
    def enhance(self, noisy_stft, num_steps=50, denoise_only=False, use_pc_sampler=True):
        """
        增强推理函数
        Args:
            noisy_stft: 噪声STFT
            num_steps: 时间步数（如果为None，使用默认配置）
            denoise_only: 是否只使用判别模型
            use_pc_sampler: 是否使用预测器-校正器采样
        """

        self.eval()
        
        with torch.no_grad():
            # 第一阶段：判别模型
            denoised_stft = self.predictive_model(noisy_stft)
            
            if denoise_only:
                return denoised_stft
            
            # 第二阶段：扩散模型 - 反向扩散
            batch_size = noisy_stft.shape[0]
            device = noisy_stft.device
            
            # 从先验采样初始状态
            time_T = torch.ones(batch_size, device=device) * self.T
            std_T = self.sde.marginal_std(time_T).view(-1, 1, 1, 1)
            x_init = denoised_stft + std_T * torch.randn_like(denoised_stft)
            
            if use_pc_sampler and hasattr(self, 'sampler'):
                # ===== 使用预测器-校正器采样 =====
                print(f"使用预测器-校正器采样: N={self.sampler.num_steps}")
                
                # 如果需要覆盖默认步数
                if num_steps is not None:
                    temp_sampler = PCSampler(
                        num_steps=num_steps,
                        corrector_steps=self.corrector_steps,
                        r=self.r,
                        method='euler'
                    )
                    x = temp_sampler.sample(self, x_init, denoised_stft, noisy_stft)
                else:
                    x = self.sampler.sample(self, x_init, denoised_stft, noisy_stft)
            else:
                # ===== 使用原始的欧拉-丸山方法 =====
                print("使用欧拉-丸山采样")
                
                num_steps = num_steps or self.num_steps
                dt = self.T / num_steps
                
                x = x_init
                
                for i in range(num_steps):
                    t = self.T - i * dt
                    t_tensor = torch.ones(batch_size, device=device) * t
                    
                    # 预测分数
                    score = self.diffusion_model(x, noisy_stft, denoised_stft, t_tensor)
                    
                    # 更新步骤
                    drift = self.sde.drift(x, denoised_stft)
                    g_t = self.sde.g(t)
                    
                    # 反向SDE
                    dx = (-drift + g_t**2 * score) * dt
                    x = x + dx
                    
                    # 添加噪声（除了最后一步）
                    if i < num_steps - 1:
                        noise = torch.randn_like(x)
                        x = x + g_t * torch.sqrt(torch.tensor(2 * dt, device=g_t.device)) * noise
            
            return x
    
    def count_parameters(self):
        """计算总参数"""
        pred_total = sum(p.numel() for p in self.predictive_model.parameters())
        diff_total = sum(p.numel() for p in self.diffusion_model.parameters())
        total = pred_total + diff_total
        
        print(f"判别模型参数: {pred_total:,}")
        print(f"扩散模型参数: {diff_total:,}")
        print(f"总参数: {total:,}")
        print(f"预计总文件大小: {total * 4 / 1024 / 1024:.2f} MB")
        return total


def test_storm_model_fixed():
    """修正后的测试函数 - 包含采样器测试"""
    
    print("=" * 60)
    print("StoRM模型测试 - 包含预测器-校正器采样")
    print("=" * 60)
    
    # 创建模型
    model = StoRMModel(base_channels=32, condition_dim=256)
    
    # 测试不同采样方法
    test_cases = [
        {'use_pc_sampler': True, 'num_steps': 50, 'name': 'PC采样器 (N=50)'},
        {'use_pc_sampler': False, 'num_steps': 50, 'name': '欧拉-丸山 (N=50)'},
        {'use_pc_sampler': True, 'num_steps': 20, 'name': 'PC采样器 (N=20)'},
        {'use_pc_sampler': False, 'num_steps': 20, 'name': '欧拉-丸山 (N=20)'},
    ]
    
    # 测试输入
    batch_size = 1
    freq_bins = 128
    time_frames = 128
    noisy_stft = torch.randn(batch_size, 2, freq_bins, time_frames)
    
    for config in test_cases:
        print(f"\n测试配置: {config['name']}")
        
        try:
            model.eval()
            with torch.no_grad():
                enhanced = model.enhance(
                    noisy_stft, 
                    num_steps=config['num_steps'],
                    denoise_only=False,
                    use_pc_sampler=config['use_pc_sampler']
                )
                print(f"  输入形状: {noisy_stft.shape}")
                print(f"  输出形状: {enhanced.shape}")
                print(f"  ✓ 测试通过")
                
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 测试不同尺寸 - 确保尺寸能被8整除
    test_cases = [
        (1, 128, 128),   # 小尺寸
        (2, 256, 256),   # 中等尺寸
        (1, 64, 128),    # 非对称尺寸
    ]
    
    for batch_size, freq_bins, time_frames in test_cases:
        print(f"\n测试尺寸: batch={batch_size}, freq={freq_bins}, time={time_frames}")
        
        try:
            # 检查并调整尺寸
            if freq_bins % 8 != 0 or time_frames % 8 != 0:
                print(f"  调整尺寸: {freq_bins}x{time_frames} -> ", end="")
                freq_bins = ((freq_bins + 7) // 8) * 8
                time_frames = ((time_frames + 7) // 8) * 8
                print(f"{freq_bins}x{time_frames}")
            
            # 模拟输入
            noisy_stft = torch.randn(batch_size, 2, freq_bins, time_frames)
            
            # 测试训练模式
            print("  训练模式测试...")
            model.train()
            clean_stft = torch.randn_like(noisy_stft)
            time = torch.rand(batch_size)  # [B]
            
            print(f"    时间形状: {time.shape}")
            
            denoised, score, x_tau = model(noisy_stft, time, clean_stft, return_intermediate=True)
            
            print(f"    输入形状: {noisy_stft.shape}")
            print(f"    判别输出形状: {denoised.shape}")
            print(f"    分数估计形状: {score.shape}")
            print(f"    扰动状态形状: {x_tau.shape}")
            
            # 验证所有输出形状相同
            if not (noisy_stft.shape == denoised.shape == score.shape == x_tau.shape):
                print(f"    ✗ 形状不匹配!")
                print(f"      输入: {noisy_stft.shape}")
                print(f"      判别输出: {denoised.shape}")
                print(f"      分数估计: {score.shape}")
                print(f"      扰动状态: {x_tau.shape}")
            else:
                print("    ✓ 所有形状匹配")
            
            # 验证数值范围
            print(f"    判别输出范围: [{denoised.min():.3f}, {denoised.max():.3f}]")
            print(f"    分数估计范围: [{score.min():.3f}, {score.max():.3f}]")
            
            print("  ✓ 训练模式测试完成")
            
            # 测试推理模式
            print("  推理模式测试...")
            model.eval()
            
            with torch.no_grad():
                # 仅判别模型
                denoised_only = model.predictive_model(noisy_stft)
                print(f"    仅判别输出形状: {denoised_only.shape}")
                
                # 完整增强（使用较少的步数以加快测试）
                enhanced = model.enhance(noisy_stft, num_steps=5, denoise_only=False)
                print(f"    完整增强输出形状: {enhanced.shape}")
                
                # 测试denoise_only模式
                denoise_only_output = model.enhance(noisy_stft, num_steps=5, denoise_only=True)
                print(f"    仅去噪输出形状: {denoise_only_output.shape}")
            
            print("  ✓ 推理模式测试完成")
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 额外的维度测试
    print("\n" + "=" * 60)
    print("额外维度测试:")
    print("=" * 60)
    
    # 测试不同批次大小的时间处理
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        try:
            freq, time = 128, 128
            noisy_stft = torch.randn(batch_size, 2, freq, time)
            
            # 测试标量时间
            model.eval()
            with torch.no_grad():
                # 标量时间
                score_scalar = model(noisy_stft, time=0.5, return_intermediate=False)
                print(f"  标量时间 (0.5): {score_scalar.shape}")
                
                # 向量时间
                time_vector = torch.rand(batch_size)
                score_vector = model(noisy_stft, time=time_vector, return_intermediate=False)
                print(f"  向量时间: {score_vector.shape}")
                
                # 无时间（随机）
                score_random = model(noisy_stft, return_intermediate=False)
                print(f"  随机时间: {score_random.shape}")
                
            print(f"  ✓ 批次大小{batch_size}测试通过")
        except Exception as e:
            print(f"  ✗ 批次大小{batch_size}测试失败: {e}")
    
    # 计算参数
    print("\n" + "=" * 60)
    print("模型参数统计:")
    print("=" * 60)
    model.count_parameters()


if __name__ == "__main__":
    test_storm_model_fixed()