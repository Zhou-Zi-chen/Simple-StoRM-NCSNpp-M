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
        
        # 添加训练阶段标志
        self.training_phase = 'joint'  # 'pretrain_predictor', 'joint'
        
        # 创建单独的预测模型优化器
        self.predictive_optimizer = optim.Adam(
            self.model.predictive_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999)
        )

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
        ema_model = StoRMModel(base_channels=32, condition_dim=256, verbose=False).to(self.device)
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
        
        # 采样扰动状态
        time_for_sde = time.view(-1, 1, 1, 1)
        x_tau, noise = self.model.sde.sample_perturbed_state(clean_stft, denoised_stft, time_for_sde)
        
        # 获取分数估计
        score = self.model.diffusion_model(x_tau, noisy_stft, denoised_stft, time)
        
        # 计算噪声水平σ(τ)
        std = self.model.sde.marginal_std(time).view(-1, 1, 1, 1)

        # 计算目标：根据公式(7)：∇log p = -(xτ - μ)/σ²
        # 由于 noise = (xτ - μ)/σ，所以 target = -noise/σ
        target = -noise / (std + 1e-6)
        
        # ========== 实现DSM损失 ==========
        # 使用权重函数 λ(t) = σ(τ)²
        weight = std**2  # λ(t)
        
        # 计算加权平方误差
        squared_error = (score - target)**2
        weighted_squared_error = squared_error * weight
        
        # DSM损失
        dsm_loss = weighted_squared_error.mean()

        # ========== 监督损失 ==========
        supervised_loss = self.mse_loss(denoised_stft, clean_stft)
    
        # 用于DeBug
        # print(f"    std范围: [{std.min():.6f}, {std.max():.6f}]")
        # print(f"    weight范围: [{weight.min():.6f}, {weight.max():.6f}]")
        # print(f"    score范围: [{score.min():.3f}, {score.max():.3f}]")
        # print(f"    target范围: [{target.min():.3f}, {target.max():.3f}]")
        # print(f"    DSM损失: {dsm_loss.item():.6f}")
        # print(f"    监督损失: {supervised_loss.item():.6f}")

        # 总损失
        total_loss = self.alpha * supervised_loss + dsm_loss
        
        return {
            'total': total_loss,
            'supervised': supervised_loss,
            'dsm': dsm_loss,
            'denoised': denoised_stft.detach()
        }
    
    # 训练predictive model
    def _evaluate_predictive_model(self):
        """评估预测模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='评估预测模型')
            
            for batch in pbar:
                clean_stft = batch['clean_stft'].to(self.device)
                noisy_stft = batch['noisy_stft'].to(self.device)
                
                denoised = self.model.predictive_model(noisy_stft)
                loss = self.mse_loss(denoised, clean_stft)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.test_loader)
        return {'val_loss':avg_loss, 'loss': avg_loss}

    def pretrain_predictive_model(self, max_epochs=50, patience=5, min_delta=0.001, start_epoch=0):
        """预训练预测模型，带早停机制"""
        print(f"\n开始预训练预测模型，最多 {max_epochs} 个epoch")

        if start_epoch > 0:
            print(f"从epoch {start_epoch}继续训练")

        self.training_phase = 'pretrain_predictor'
        
        # 创建预测模型专用的优化器
        self.predictive_optimizer = optim.Adam(
            self.model.predictive_model.parameters(),
            lr=self.lr,  # 可以使用相同的学习率
            betas=(0.9, 0.999)
        )
        
        # 可选：为预测模型使用学习率调度器
        predictive_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.predictive_optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(start_epoch, max_epochs):  # 从start_epoch开始
            # 训练一个epoch
            train_result = self._train_predictive_epoch()
            
            # 在验证集上评估
            val_result = self._evaluate_predictive_model()
            
            # 从字典中提取损失值
            train_loss = train_result.get('train_loss', train_result.get('loss', 0.0))
            val_loss = val_result.get('val_loss', val_result.get('loss', 0.0))

            # 更新学习率
            predictive_scheduler.step(val_loss)
            
            print(f"预训练 Epoch {epoch+1}/{max_epochs}: "
                f"训练损失 = {train_loss:.4f}, "
                f"验证损失 = {val_loss:.4f}, "
                f"学习率 = {self.predictive_optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停检查
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {
                    'model': self.model.predictive_model.state_dict(),
                    'optimizer': self.predictive_optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss
                }
                
                # 保存检查点
                checkpoint_path = self.checkpoint_dir / self.experiment_id / 'best_pretrained_predictor.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, checkpoint_path)
                print(f"  ✓ 保存最佳预测模型: {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"  早停计数器: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"  ⚠️ 早停触发！在epoch {epoch+1}停止预训练")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.predictive_model.load_state_dict(best_model_state['model'])
            print(f"  加载最佳预测模型 (epoch {best_model_state['epoch']+1}, 损失={best_model_state['loss']:.4f})")
        
        print(f"预测模型预训练完成! 最佳验证损失: {best_loss:.4f}")
        self.training_phase = 'joint'
        
        # 清理不再需要的优化器
        del self.predictive_optimizer
        
        return best_loss
    
    def train_epoch(self):
        """修改后的训练epoch"""
        self.model.train()
        
        if self.training_phase == 'pretrain_predictor':
            return self._train_predictive_epoch()
        else:
            return self._train_joint_epoch()
    
    def _train_predictive_epoch(self):
        """只训练预测模型的epoch"""
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'预测模型训练')
        
        for batch_idx, batch in enumerate(pbar):
            clean_stft = batch['clean_stft'].to(self.device)
            noisy_stft = batch['noisy_stft'].to(self.device)
            
            self.predictive_optimizer.zero_grad()
            denoised = self.model.predictive_model(noisy_stft)
            loss = self.mse_loss(denoised, clean_stft)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.predictive_model.parameters(), max_norm=1.0)
            self.predictive_optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return {'train_loss': avg_loss, 'loss': avg_loss}

    def _train_joint_epoch(self):
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
            # if (batch_idx + 1) % 100 == 0:
            #     self.save_checkpoint(f'epoch_{self.current_epoch}_batch_{batch_idx}.pt')
        
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
    
    def train(self, num_epochs: int, save_every: int = 10, pretrain_epochs: int = 10, ):
        """主训练循环"""
        
        # 第一阶段：预训练预测模型
        if pretrain_epochs > 0:
            self.pretrain_predictive_model(pretrain_epochs)
            # 保存预训练好的预测模型
            self.save_checkpoint('pretrained_predictor.pt')


        # 第二阶段：联合训练
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

        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print("-" * 40)
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 评估
            val_metrics = self.evaluate()
            
            # 记录历史 - 确保 train_metrics 和 val_metrics 是字典
            if isinstance(train_metrics, dict):
                for key in train_metrics:
                    if key in history:
                        history[key].append(train_metrics[key])
            else:
                # 如果是标量，假设它是总损失
                history['train_loss'].append(train_metrics)
                print(f"警告: train_metrics 是标量，不是字典: {train_metrics}")
            
            if isinstance(val_metrics, dict):
                for key in val_metrics:
                    if key in history:
                        history[key].append(val_metrics[key])
            else:
                # 如果是标量，假设它是总损失
                history['val_loss'].append(val_metrics)
                print(f"警告: val_metrics 是标量，不是字典: {val_metrics}")
            
            # 打印结果 - 安全地访问字典键
            if isinstance(train_metrics, dict):
                print(f"训练损失: {train_metrics.get('train_loss', 'N/A'):.4f} "
                    f"(监督: {train_metrics.get('train_supervised', 'N/A'):.4f}, "
                    f"DSM: {train_metrics.get('train_dsm', 'N/A'):.4f})")
            else:
                print(f"训练损失: {train_metrics:.4f}")
                
            if isinstance(val_metrics, dict):
                print(f"验证损失: {val_metrics.get('val_loss', 'N/A'):.4f} "
                    f"(监督: {val_metrics.get('val_supervised', 'N/A'):.4f}, "
                    f"DSM: {val_metrics.get('val_dsm', 'N/A'):.4f})")
            else:
                print(f"验证损失: {val_metrics:.4f}")
                
            
            # 保存最佳模型 - 需要提取损失值
            if isinstance(val_metrics, dict):
                current_val_loss = val_metrics.get('val_loss', float('inf'))
            else:
                current_val_loss = val_metrics
                
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                self.best_loss = best_loss  # 更新类属性
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
                print(f"✓ 保存最佳模型，损失: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  早停计数器: {patience_counter}/{pretrain_epochs}")
            
            # 早停检查
            if patience_counter >= pretrain_epochs:
                print(f"  ⚠️ 早停触发！在epoch {self.current_epoch}停止训练")
                break
            
            # 定期保存
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}.pt')
                self.save_history(history)
            
            # 保存最后一个epoch
            if epoch == num_epochs - 1:
                self.save_checkpoint('final_model.pt')
                self.save_history(history)
        
        print(f"\n训练完成!")
        print(f"最佳验证损失: {best_loss:.4f}")
        print(f"总训练epoch数: {self.current_epoch}")
        
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
    
    # 训练阶段参数
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='预测模型预训练最大轮数')
    parser.add_argument('--joint_epochs', type=int, default=200,
                        help='联合训练最大轮数')
    parser.add_argument('--no_pretrain', action='store_true',
                        help='跳过预训练阶段')
    parser.add_argument('--pretrain_patience', type=int, default=10,
                        help='预训练早停耐心值')
    parser.add_argument('--joint_patience', type=int, default=15,
                        help='联合训练早停耐心值')

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
    
    # ===== 恢复训练参数 =====
    parser.add_argument('--resume_pretrained', type=str, default=None,
                        help='加载预训练的预测模型')
    parser.add_argument('--resume_joint', type=str, default=None,
                        help='加载联合训练模型继续训练')

    # 其他参数
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备（cpu/mps/cuda）')
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
    model = StoRMModel(base_channels=args.base_channels, verbose=True)
    
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
    
    # 测试模式
    if args.test_only:
        print("\n测试模式...")
        metrics = trainer.evaluate()
        print(f"测试结果: {metrics}")
        return
    
    # ===== 阶段1: 预训练预测模型 =====
    if not args.no_pretrain:
        # 检查是否有 resume_pretrained 属性
        if hasattr(args, 'resume_pretrained') and args.resume_pretrained:
            # 加载预训练模型
            checkpoint = torch.load(args.resume_pretrained, map_location=args.device)
            model.predictive_model.load_state_dict(checkpoint['model'])
            print(f"加载预训练模型: {args.resume_pretrained}")
                    
            # 获取之前的训练信息（如果有）
            previous_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('loss', float('inf'))
            
            print(f"  之前训练到epoch: {previous_epoch}")
            print(f"  最佳验证损失: {best_loss:.4f}")
            
            # 继续预训练
            trainer.pretrain_predictive_model(
                max_epochs=args.pretrain_epochs,  # 新的总epoch数
                patience=args.pretrain_patience,
                start_epoch=previous_epoch  # 从之前的epoch继续
            )
        else:
            # 从头开始预训练
            trainer.pretrain_predictive_model(
                max_epochs=args.pretrain_epochs,
                patience=args.pretrain_patience
            )

    else:
        print("✓ 跳过预训练（因为 no_pretrain=True）")
    
    # ===== 阶段2: 联合训练 =====
    if args.resume_joint:
        # 继续联合训练
        trainer.load_checkpoint(args.resume_joint)
    
    # 开始联合训练
    if args.resume_joint or args.no_pretrain:
        # 情况1: 恢复联合训练 或 跳过预训练
        # 设置 pretrain_epochs=0 确保不进行预训练
        history = trainer.train(
            num_epochs=args.joint_epochs,
            pretrain_epochs=0,
            save_every=10
        )
    else:
        # 情况2: 完整训练（包含预训练）
        history = trainer.train(
            num_epochs=args.joint_epochs,
            pretrain_epochs=args.pretrain_epochs,
            save_every=10
        )
    
    print(f"\n训练完成!")
    
    # 保存最终历史
    final_path = trainer.log_dir / trainer.experiment_id / 'training_history.json'
    with open(final_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'history': history,
            'best_loss': trainer.best_loss
        }, f, indent=2)
    
    print(f"训练历史已保存到: {final_path}")


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
    # test_inference()
    
    # 如果需要训练，取消注释下面的行
    main()