# file: train_storm.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from pathlib import Path

# 导入模型
from storm_model import StoRMModel
from data_loader import create_dataloaders


class StoRMTrainer:
    """
    StoRM训练器
    实现论文中的联合训练策略
    """
    
    def __init__(self, 
                model: StoRMModel,
                train_loader: DataLoader,
                test_loader: DataLoader,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                lr: float = 1e-4,
                alpha: float = 1.0,  # 监督损失权重
                ema_decay: float = 0.999,
                log_dir: str = './logs',
                checkpoint_dir: str = './checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 训练参数
        self.lr = lr
        self.alpha = alpha
        self.ema_decay = ema_decay
        
        # 目录设置
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.model.predictive_model.parameters()) + 
            list(self.model.diffusion_model.parameters()),
            lr=lr,
            betas=(0.9, 0.999)
        )
        
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,      # 当损失不再下降时，学习率减半
            patience=3,      # 容忍3个epoch没有改善
            threshold=0.01,  # 改善阈值
            threshold_mode='rel',  # 相对改善模式
            cooldown=0,      # 冷却时间
            min_lr=1e-6      # 最小学习率
        )

        # EMA（指数移动平均）
        self.ema_model = self._create_ema_model()
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 创建实验ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f'storm_{timestamp}'
        
        print(f"\nStoRM训练器初始化:")
        print(f"  设备: {device}")
        print(f"  学习率: {lr}")
        print(f"  alpha(监督权重): {alpha}")
        print(f"  EMA衰减: {ema_decay}")
        print(f"  实验ID: {self.experiment_id}")
        print(f"  日志目录: {self.log_dir}")
        print(f"  检查点目录: {self.checkpoint_dir}")
    
    def _create_ema_model(self):
        """创建EMA模型"""
        ema_model = StoRMModel(base_channels=32).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model
    
    def update_ema(self):
        """更新EMA模型权重"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def compute_loss(self, clean_stft, noisy_stft):
        """
        计算StoRM联合损失（公式17）
        
        Loss = α * ||x - Dθ(y)||² + ||sφ(xτ, [y, Dθ(y)], τ) + z/σ(τ)||²
        """
        batch_size = clean_stft.shape[0]
        
        # 随机采样扩散时间
        time = torch.rand(batch_size, device=self.device)
        time = torch.clamp(time, min=0.01, max=0.99)  # 避免接近0或1的极端值
        
        # 用于DeBug
        # print(f"\n[DEBUG] Batch开始")
        # print(f"  time范围: [{time.min():.4f}, {time.max():.4f}]")

        # 获取判别模型输出
        denoised_stft = self.model.predictive_model(noisy_stft)
        
        # 计算监督损失（第一阶段）
        supervised_loss = self.mse_loss(denoised_stft, clean_stft)
        
        # 采样扰动状态
        time_for_sde = time.view(-1, 1, 1, 1)
        x_tau, noise = self.model.sde.sample_perturbed_state(clean_stft, denoised_stft, time_for_sde)
        
        # 获取分数估计
        score = self.model.diffusion_model(x_tau, noisy_stft, denoised_stft, time)
        
        # 计算噪声水平σ(τ)
        std = self.model.sde.marginal_std(time).view(-1, 1, 1, 1)

        # 用于DeBug
        # print(f"  std范围: [{std.min():.6f}, {std.max():.6f}]")
        # print(f"  score范围: [{score.min():.3f}, {score.max():.3f}]")
        
        # 计算分数匹配损失（第二阶段）
        target = -noise / (std + 1e-6)  # 添加epsilon防止除零  # 根据公式(7)：∇log p = -(xτ - μ)/σ²，而μ = E[xτ]，noise = (xτ - μ)/σ
        
        # 用于DeBug
        # print(f"  target范围: [{target.min():.3f}, {target.max():.3f}]")  ## 实测发现target范围很大
        # print(f"  noise范围: [{noise.min():.3f}, {noise.max():.3f}]")
        
        # 权重函数 λ(t) = std² 或 1/std²，根据具体推导
        # 对于去噪分数匹配，常用 λ(t) = std²
        weight = std**2
        
        # 加权损失
        weighted_score = score * weight
        weighted_target = target * weight
        
        dsm_loss = self.mse_loss(weighted_score, weighted_target) / (weight.mean() + 1e-8)
        
        # 总损失
        total_loss = self.alpha * supervised_loss + dsm_loss
        
        return {
            'total': total_loss,
            'supervised': supervised_loss,
            'dsm': dsm_loss,
            'denoised': denoised_stft.detach()
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_supervised = 0
        total_dsm = 0
        
        pbar = tqdm(self.train_loader, desc=f'训练 Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            clean_stft = batch['clean_stft'].to(self.device)
            noisy_stft = batch['noisy_stft'].to(self.device)
            
            # 前向传播和损失计算
            self.optimizer.zero_grad()
            loss_dict = self.compute_loss(clean_stft, noisy_stft)
            
            # 反向传播
            loss_dict['total'].backward()
            
            # 梯度裁剪
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
            
            # 如果梯度范数过大，额外处理
            if total_norm > 10.0:  # 梯度范数超过10
                print(f"警告: 梯度范数过大: {total_norm:.2f}")
                # 进一步裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # 优化器步骤
            self.optimizer.step()
            
            # 更新EMA
            self.update_ema()
            
            # 记录损失
            total_loss += loss_dict['total'].item()
            total_supervised += loss_dict['supervised'].item()
            total_dsm += loss_dict['dsm'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'total': f'{loss_dict["total"].item():.4f}',
                'sup': f'{loss_dict["supervised"].item():.4f}',
                'dsm': f'{loss_dict["dsm"].item():.4f}',
                'grad_norm': f'{total_norm:.2f}'  # 显示梯度范数
            })
            
            # 定期保存检查点
            if (batch_idx + 1) % 100 == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}_batch_{batch_idx}.pt')
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_supervised = total_supervised / len(self.train_loader)
        avg_dsm = total_dsm / len(self.train_loader)
        
        return {
            'train_loss': avg_loss,
            'train_supervised': avg_supervised,
            'train_dsm': avg_dsm
        }
    
    def evaluate(self):
        """在测试集上评估模型"""
        self.ema_model.eval()  # 使用EMA模型进行评估
        
        total_loss = 0
        total_supervised = 0
        total_dsm = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='评估')
            
            for batch in pbar:
                clean_stft = batch['clean_stft'].to(self.device)
                noisy_stft = batch['noisy_stft'].to(self.device)
                
                # 使用EMA模型计算损失
                loss_dict = self.compute_loss(clean_stft, noisy_stft)
                
                total_loss += loss_dict['total'].item()
                total_supervised += loss_dict['supervised'].item()
                total_dsm += loss_dict['dsm'].item()
                
                pbar.set_postfix({
                    'loss': f'{loss_dict["total"].item():.4f}'
                })
        
        avg_loss = total_loss / len(self.test_loader)
        avg_supervised = total_supervised / len(self.test_loader)
        avg_dsm = total_dsm / len(self.test_loader)
        
        return {
            'val_loss': avg_loss,
            'val_supervised': avg_supervised,
            'val_dsm': avg_dsm
        }
    
    def train(self, num_epochs: int, save_every: int = 5):
        """主训练循环"""
        
        print(f"\n开始训练，共 {num_epochs} 个epoch")
        print("=" * 60)
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'train_supervised': [],
            'train_dsm': [],
            'val_loss': [],
            'val_supervised': [],
            'val_dsm': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print("-" * 40)
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 评估
            val_metrics = self.evaluate()
            
            # 记录历史
            for key in train_metrics:
                history[key].append(train_metrics[key])
            for key in val_metrics:
                history[key].append(val_metrics[key])
            
            # 打印结果
            print(f"训练损失: {train_metrics['train_loss']:.4f} "
                f"(监督: {train_metrics['train_supervised']:.4f}, "
                f"DSM: {train_metrics['train_dsm']:.4f})")
            print(f"验证损失: {val_metrics['val_loss']:.4f} "
                f"(监督: {val_metrics['val_supervised']:.4f}, "
                f"DSM: {val_metrics['val_dsm']:.4f})")
            
            # 保存最佳模型
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pt')
                print(f"✓ 保存最佳模型，损失: {self.best_loss:.4f}")
            
            # 定期保存
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}.pt')
                self.save_history(history)
            
            # 保存最后一个epoch
            if epoch == num_epochs - 1:
                self.save_checkpoint('final_model.pt')
                self.save_history(history)
        
        print(f"\n训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / self.experiment_id / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': {
                'lr': self.lr,
                'alpha': self.alpha,
                'ema_decay': self.ema_decay
            }
        }, checkpoint_path)
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def save_history(self, history):
        """保存训练历史"""
        history_path = self.log_dir / self.experiment_id / 'history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"检查点已加载: {checkpoint_path}")
        print(f"从epoch {self.current_epoch}继续训练")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练StoRM模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./speech_data',
                        help='语音数据根目录')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='监督损失权重')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA衰减率')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=32,
                        help='基础通道数')
    
    # 音频参数
    parser.add_argument('--sr', type=int, default=16000,
                        help='采样率')
    parser.add_argument('--segment_length', type=int, default=32000,
                        help='音频片段长度（样本数）')
    parser.add_argument('--n_fft', type=int, default=510,
                        help='STFT的FFT点数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备（cpu/mps/cuda）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--test_only', action='store_true',
                        help='仅测试模式')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sr=args.sr,
        segment_length=args.segment_length,
        n_fft=args.n_fft
    )
    
    # 创建模型
    print("\n创建模型...")
    model = StoRMModel(base_channels=args.base_channels)
    
    # 创建训练器
    trainer = StoRMTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        alpha=args.alpha,
        ema_decay=args.ema_decay
    )
    
    # 恢复检查点（如果有）
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 测试模式
    if args.test_only:
        print("\n测试模式...")
        metrics = trainer.evaluate()
        print(f"测试结果: {metrics}")
        return
    
    # 训练
    print(f"\n开始训练，共 {args.num_epochs} 个epoch")
    print("=" * 60)
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_every=5
    )
    
    # 保存最终结果
    final_checkpoint = trainer.checkpoint_dir / trainer.experiment_id / 'final_results.json'
    with open(final_checkpoint, 'w') as f:
        json.dump({
            'config': vars(args),
            'final_metrics': {
                'best_val_loss': trainer.best_loss,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            }
        }, f, indent=2)
    
    print(f"\n训练完成!")
    print(f"最终结果已保存到: {final_checkpoint}")


def test_inference():
    """测试推理流程"""
    print("=" * 60)
    print("测试推理流程")
    print("=" * 60)
    
    # 创建模型
    model = StoRMModel(base_channels=32)
    
    # 测试输入
    batch_size = 1
    freq = 128  # 调整后的频率维度
    time = 128  # 调整后的时间维度
    
    print(f"创建测试输入: [{batch_size}, 2, {freq}, {time}]")
    noisy_stft = torch.randn(batch_size, 2, freq, time)
    
    # 测试不同模式
    print("\n1. 测试判别模型单独使用:")
    model.eval()
    with torch.no_grad():
        denoised = model.enhance(noisy_stft, denoise_only=True)
        print(f"   输入形状: {noisy_stft.shape}")
        print(f"   输出形状: {denoised.shape}")
        print(f"   ✓ 判别模型测试通过")
    
    print("\n2. 测试完整增强:")
    with torch.no_grad():
        enhanced = model.enhance(noisy_stft, num_steps=10, denoise_only=False)
        print(f"   输入形状: {noisy_stft.shape}")
        print(f"   输出形状: {enhanced.shape}")
        print(f"   ✓ 完整增强测试通过")
    
    print("\n3. 测试波形增强（需要STFT处理器）:")
    try:
        from data_loader import STFTProcessor
        
        # 创建STFT处理器
        stft_processor = STFTProcessor()
        
        # 创建测试波形
        sr = 16000
        duration = 2.0
        t = torch.linspace(0, duration, int(sr * duration))
        test_waveform = 0.5 * torch.sin(2 * np.pi * 440 * t)
        
        # 转换为STFT
        test_stft = stft_processor.stft(test_waveform.unsqueeze(0))
        
        # 增强
        enhanced_stft = model.enhance(test_stft, denoise_only=True)
        
        # 转换回波形
        enhanced_waveform = stft_processor.istft(enhanced_stft)
        
        print(f"   输入波形长度: {len(test_waveform)}")
        print(f"   输出波形长度: {len(enhanced_waveform)}")
        print(f"   ✓ 波形增强测试通过")
        
    except Exception as e:
        print(f"   ✗ 波形增强测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("推理测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 测试推理
    test_inference()
    
    # 如果需要训练，取消注释下面的行
    main()