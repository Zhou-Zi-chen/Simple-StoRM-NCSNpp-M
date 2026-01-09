# file: data_loader.py
import os
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import random
import json
from pathlib import Path


class VoicebankDataset(Dataset):
    """
    语音数据集加载器，支持Voicebank/DEMAND格式
    目录结构:
    speech_data/
    ├── clean_testset_wav/
    ├── noisy_testset_wav/
    ├── clean_trainset_wav/
    ├── noisy_trainset_wav/
    ├── testset_txt/
    └── trainset_txt/
    """
    
    def __init__(self, 
                data_root: str,
                mode: str = 'train',
                sr: int = 16000,
                segment_length: int = 32000,  # 2秒，16000*2
                n_fft: int = 510,
                hop_length: int = 128,
                win_length: int = 510,
                shuffle: bool = True,
                augment: bool = False):
        """
        Args:
            data_root: 数据根目录
            mode: 'train' 或 'test'
            sr: 采样率
            segment_length: 音频片段长度（样本数）
            n_fft: STFT的FFT点数
            hop_length: STFT的hop长度
            win_length: STFT的窗口长度
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.sr = sr
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.shuffle = shuffle
        self.augment = augment and (mode == 'train')
        
        # 设置路径
        if mode == 'train':
            self.clean_dir = self.data_root / 'clean_trainset_wav'
            self.noisy_dir = self.data_root / 'noisy_trainset_wav'
            self.txt_dir = self.data_root / 'trainset_txt'
        else:
            self.clean_dir = self.data_root / 'clean_testset_wav'
            self.noisy_dir = self.data_root / 'noisy_testset_wav'
            self.txt_dir = self.data_root / 'testset_txt'
        
        # 验证目录存在
        for dir_path in [self.clean_dir, self.noisy_dir, self.txt_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        # 获取文件列表
        self.clean_files = sorted(list(self.clean_dir.glob('*.wav')))
        self.noisy_files = sorted(list(self.noisy_dir.glob('*.wav')))
        
        # 确保clean和noisy文件对应
        self._validate_file_pairs()
        
        # 加载文本文件（如果有）
        self.transcriptions = self._load_transcriptions()
        
        print(f"数据集初始化: mode={mode}")
        print(f"  干净文件数: {len(self.clean_files)}")
        print(f"  噪声文件数: {len(self.noisy_files)}")
        print(f"  采样率: {sr} Hz")
        print(f"  片段长度: {segment_length} 样本 ({segment_length/sr:.2f}秒)")
        print(f"  STFT参数: n_fft={n_fft}, hop={hop_length}, win={win_length}")
    
    def _validate_file_pairs(self):
        """验证clean和noisy文件一一对应"""
        clean_names = {f.stem for f in self.clean_files}
        noisy_names = {f.stem for f in self.noisy_files}
        
        common_names = clean_names & noisy_names
        if len(common_names) != len(self.clean_files) or len(common_names) != len(self.noisy_files):
            print(f"警告: clean和noisy文件不完全匹配")
            print(f"  clean文件数: {len(self.clean_files)}")
            print(f"  noisy文件数: {len(self.noisy_files)}")
            print(f"  共同文件数: {len(common_names)}")
            
            # 只保留共同的文件
            self.clean_files = [f for f in self.clean_files if f.stem in common_names]
            self.noisy_files = [f for f in self.noisy_files if f.stem in common_names]
            
            # 按文件名排序确保对应
            self.clean_files.sort(key=lambda x: x.stem)
            self.noisy_files.sort(key=lambda x: x.stem)
    
    def _load_transcriptions(self):
        """加载文本转录（如果有）"""
        transcriptions = {}
        
        # 查找txt文件
        txt_files = list(self.txt_dir.glob('*.txt'))
        if not txt_files:
            return transcriptions
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                file_id, text = parts
                                transcriptions[file_id] = text
            except Exception as e:
                print(f"警告: 无法加载转录文件 {txt_file}: {e}")
        
        print(f"  加载转录数: {len(transcriptions)}")
        return transcriptions
    
    def __len__(self):
        return len(self.clean_files)
    
    def _load_audio(self, file_path: Path) -> torch.Tensor:
        """加载音频文件并预处理"""
        try:
            # 加载音频
            waveform, sr = torchaudio.load(str(file_path))
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 重采样到目标采样率
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            
            return waveform.squeeze(0)  # 去除通道维度
            
        except Exception as e:
            print(f"错误: 无法加载音频 {file_path}: {e}")
            # 返回静音作为占位符
            return torch.zeros(self.segment_length)
    
    def _random_segment(self, clean_wav: torch.Tensor, noisy_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机截取音频片段"""
        length = clean_wav.shape[-1]
        
        if length <= self.segment_length:
            # 如果音频太短，填充
            pad_len = self.segment_length - length
            clean_wav = F.pad(clean_wav, (0, pad_len))
            noisy_wav = F.pad(noisy_wav, (0, pad_len))
            start = 0
        else:
            # 随机选择起始点
            start = random.randint(0, length - self.segment_length)
        
        # 截取片段
        clean_segment = clean_wav[start:start + self.segment_length]
        noisy_segment = noisy_wav[start:start + self.segment_length]
        
        return clean_segment, noisy_segment
    
    def _augment_audio(self, clean_wav: torch.Tensor, noisy_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """数据增强（仅在训练时使用）"""
        if not self.augment:
            return clean_wav, noisy_wav
        
        # 随机增益
        gain = random.uniform(0.8, 1.2)
        clean_wav = clean_wav * gain
        noisy_wav = noisy_wav * gain
        
        # 随机反转（概率0.5）
        if random.random() < 0.5:
            clean_wav = clean_wav.flip(0)
            noisy_wav = noisy_wav.flip(0)
        
        return clean_wav, noisy_wav
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个数据样本
        
        Returns:
            Dict包含:
            - 'clean': 干净音频 [T]
            - 'noisy': 噪声音频 [T]
            - 'clean_stft': 干净STFT [2, F, T_frames]
            - 'noisy_stft': 噪声STFT [2, F, T_frames]
            - 'file_id': 文件名
            - 'transcription': 转录文本（如果有）
        """
        # 获取文件路径
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]
        file_id = clean_path.stem
        
        # 加载音频
        clean_wav = self._load_audio(clean_path)
        noisy_wav = self._load_audio(noisy_path)
        
        # 确保长度一致
        min_len = min(len(clean_wav), len(noisy_wav))
        clean_wav = clean_wav[:min_len]
        noisy_wav = noisy_wav[:min_len]
        
        # 随机截取片段（训练时）或使用整个音频（测试时）
        if self.mode == 'train':
            clean_wav, noisy_wav = self._random_segment(clean_wav, noisy_wav)
        else:
            # 测试时，如果太长则截取开头
            if len(clean_wav) > self.segment_length:
                clean_wav = clean_wav[:self.segment_length]
                noisy_wav = noisy_wav[:self.segment_length]
        
        # 数据增强
        clean_wav, noisy_wav = self._augment_audio(clean_wav, noisy_wav)
        
        # 获取转录文本
        transcription = self.transcriptions.get(file_id, "")
        
        # 准备返回数据
        sample = {
            'clean': clean_wav,
            'noisy': noisy_wav,
            'file_id': file_id,
            'transcription': transcription
        }
        
        return sample


class STFTProcessor:
    """STFT处理器，与网络中的处理器保持一致"""
    
    def __init__(self, n_fft=510, hop_length=128, win_length=510):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # 平方根Hann窗
        window = torch.hann_window(win_length)
        self.window = torch.sqrt(window)
    
    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形转STFT"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        
        # 应用STFT
        stft = torch.stft(waveform,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        window=self.window,
                        return_complex=True)
        
        # 分离实部和虚部
        real = stft.real.unsqueeze(1)  # [B, 1, F, T]
        imag = stft.imag.unsqueeze(1)  # [B, 1, F, T]
        
        # 拼接为2通道张量
        return torch.cat([real, imag], dim=1)
    
    def istft(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """STFT转波形"""
        # 分离实部和虚部
        real = complex_spec[:, 0:1, :, :]
        imag = complex_spec[:, 1:2, :, :]
        
        # 重建复数张量
        stft = torch.complex(real.squeeze(1), imag.squeeze(1))
        
        # 应用逆STFT
        waveform = torch.istft(stft,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window)
        
        return waveform
    
    def sqrt_magnitude_warping(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """平方根幅度压缩"""
        real = complex_spec[:, 0:1, :, :]
        imag = complex_spec[:, 1:2, :, :]
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        magnitude_warped = torch.sqrt(magnitude)
        
        phase = torch.atan2(imag, real)
        real_warped = magnitude_warped * torch.cos(phase)
        imag_warped = magnitude_warped * torch.sin(phase)
        
        return torch.cat([real_warped, imag_warped], dim=1)


# file: data_loader.py (修正StoRMCollator部分)

import torch
import torch.nn.functional as F  # 确保正确导入F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union
import random


class StoRMCollator:
    """StoRM数据整理器 - 修正版"""
    
    def __init__(self, 
                stft_processor: 'STFTProcessor',
                segment_frames: int = 256,
                mode: str = 'train'):
        
        self.stft_processor = stft_processor
        self.segment_frames = segment_frames
        self.mode = mode
        
        # 计算频率维度
        self.n_freq = stft_processor.n_fft // 2 + 1
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理一批数据 - 修正版"""
        batch_size = len(batch)
        
        # 提取原始数据
        clean_wavs = [item['clean'] for item in batch]
        noisy_wavs = [item['noisy'] for item in batch]
        file_ids = [item['file_id'] for item in batch]
        
        # 转换为张量
        clean_wavs_tensor = torch.stack(clean_wavs)  # [B, T]
        noisy_wavs_tensor = torch.stack(noisy_wavs)  # [B, T]
        
        # 转换为STFT
        clean_stfts = self.stft_processor.stft(clean_wavs_tensor)  # [B, 2, F, T_frames]
        noisy_stfts = self.stft_processor.stft(noisy_wavs_tensor)  # [B, 2, F, T_frames]
        
        # 获取当前STFT尺寸
        B, C, F, T = clean_stfts.shape
        
        # 调整尺寸以匹配网络要求（8的倍数）
        target_freq = ((F + 7) // 8) * 8
        target_time = ((self.segment_frames + 7) // 8) * 8
        
        if F != target_freq or T != target_time:
            # 调整频率维度
            if F != target_freq:
                clean_stfts = torch.nn.functional.interpolate(  # 使用完整路径
                    clean_stfts, 
                    size=(target_freq, T), 
                    mode='bilinear', 
                    align_corners=False
                )
                noisy_stfts = torch.nn.functional.interpolate(
                    noisy_stfts, 
                    size=(target_freq, T), 
                    mode='bilinear', 
                    align_corners=False
                )
                F = target_freq
            
            # 调整时间维度
            if self.mode == 'train' and T > target_time:
                # 训练时随机截取
                start = random.randint(0, T - target_time)
                clean_stfts = clean_stfts[:, :, :, start:start + target_time]
                noisy_stfts = noisy_stfts[:, :, :, start:start + target_time]
            elif T != target_time:
                # 测试时居中截取或填充
                if T > target_time:
                    start = (T - target_time) // 2
                    clean_stfts = clean_stfts[:, :, :, start:start + target_time]
                    noisy_stfts = noisy_stfts[:, :, :, start:start + target_time]
                else:
                    pad = target_time - T
                    # 修正：使用正确的填充方式
                    # F.pad的padding格式是 (左, 右, 上, 下, 前, 后)
                    # 对于2D张量[B, C, H, W]，格式是(W左, W右, H上, H下)
                    # 我们需要在时间维度(最后一个维度)上填充
                    clean_stfts = torch.nn.functional.pad(clean_stfts, (0, pad))
                    noisy_stfts = torch.nn.functional.pad(noisy_stfts, (0, pad))
        
        return {
            'clean_wav': clean_wavs_tensor,
            'noisy_wav': noisy_wavs_tensor,
            'clean_stft': clean_stfts,
            'noisy_stft': noisy_stfts,
            'file_ids': file_ids
        }


def create_dataloaders(data_root: str,
                        batch_size: int = 8,
                        num_workers: int = 4,
                        sr: int = 16000,
                        segment_length: int = 32000,
                        segment_frames: int = 256,
                        n_fft: int = 510,
                        hop_length: int = 128,
                        win_length: int = 510) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Returns:
        train_loader, test_loader
    """
    
    # 创建STFT处理器
    stft_processor = STFTProcessor(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    # 创建数据集
    train_dataset = VoicebankDataset(
        data_root=data_root,
        mode='train',
        sr=sr,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        shuffle=True,
        augment=True
    )
    
    test_dataset = VoicebankDataset(
        data_root=data_root,
        mode='test',
        sr=sr,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        shuffle=False,
        augment=False
    )
    
    # 创建整理器
    train_collator = StoRMCollator(
        stft_processor=stft_processor,
        segment_frames=segment_frames,
        mode='train'
    )
    
    test_collator = StoRMCollator(
        stft_processor=stft_processor,
        segment_frames=segment_frames,
        mode='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        pin_memory=torch.cuda.is_available(),  # 只在有GPU时启用
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_collator,
        pin_memory=True,
        drop_last=False  # 测试时不丢弃
    )
    
    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  测试集: {len(test_dataset)} 个样本")
    print(f"  批次大小: {batch_size}")
    print(f"  STFT尺寸: {stft_processor.n_fft // 2 + 1}频点")
    
    return train_loader, test_loader


def test_data_loader():
    """测试数据加载器"""
    
    # 假设数据目录结构
    data_root = "./speech_data"  # 修改为实际路径
    
    if not os.path.exists(data_root):
        print(f"警告: 数据目录 {data_root} 不存在，创建虚拟数据用于测试...")
        create_dummy_data(data_root)
    
    try:
        print("=" * 60)
        print("测试数据加载器")
        print("=" * 60)
        
        # 创建数据加载器
        train_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=2,
            num_workers=0,  # 测试时设为0避免多进程问题
            segment_length=16000,  # 1秒，加快测试
            segment_frames=128  # 对应1秒
        )
        
        # 测试训练数据
        print("\n测试训练数据...")
        for i, batch in enumerate(train_loader):
            print(f"批次 {i + 1}:")
            print(f"  干净波形: {batch['clean_wav'].shape}")
            print(f"  噪声波形: {batch['noisy_wav'].shape}")
            print(f"  干净STFT: {batch['clean_stft'].shape}")
            print(f"  噪声STFT: {batch['noisy_stft'].shape}")
            print(f"  文件IDs: {batch['file_ids']}")
            
            # 验证尺寸
            clean_stft = batch['clean_stft']
            noisy_stft = batch['noisy_stft']
            
            assert clean_stft.shape == noisy_stft.shape, "STFT尺寸不匹配"
            assert clean_stft.shape[1] == 2, "STFT应为2通道"
            
            # 验证频率维度是8的倍数
            freq = clean_stft.shape[2]
            time = clean_stft.shape[3]
            assert freq % 8 == 0, f"频率维度{freq}不是8的倍数"
            assert time % 8 == 0, f"时间维度{time}不是8的倍数"
            
            print(f"  验证通过: STFT尺寸 {clean_stft.shape}")
            
            if i >= 1:  # 只测试2个批次
                break
        
        print("\n✓ 数据加载器测试通过!")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def create_dummy_data(data_root: str):
    """创建虚拟数据用于测试"""
    import warnings
    warnings.warn("创建虚拟数据，仅用于测试!")
    
    # 创建目录结构
    dirs = [
        'clean_trainset_wav',
        'noisy_trainset_wav',
        'clean_testset_wav',
        'noisy_testset_wav',
        'trainset_txt',
        'testset_txt'
    ]
    
    for dir_name in dirs:
        dir_path = os.path.join(data_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建一些虚拟wav文件
    sr = 16000
    duration = 2.0  # 2秒
    t = torch.linspace(0, duration, int(sr * duration))
    
    # 训练数据
    for i in range(5):
        # 干净音频：正弦波
        clean_wav = 0.5 * torch.sin(2 * np.pi * 440 * t)
        
        # 噪声音频：加性噪声
        noise = 0.1 * torch.randn_like(clean_wav)
        noisy_wav = clean_wav + noise
        
        # 保存文件
        for mode in ['train', 'test']:
            if mode == 'train' and i < 4:  # 前4个为训练
                clean_dir = 'clean_trainset_wav'
                noisy_dir = 'noisy_trainset_wav'
                txt_dir = 'trainset_txt'
            elif mode == 'test' and i >= 4:  # 第5个为测试
                clean_dir = 'clean_testset_wav'
                noisy_dir = 'noisy_testset_wav'
                txt_dir = 'testset_txt'
            else:
                continue
            
            # 保存wav文件
            clean_path = os.path.join(data_root, clean_dir, f'sample_{i:03d}.wav')
            noisy_path = os.path.join(data_root, noisy_dir, f'sample_{i:03d}.wav')
            
            torchaudio.save(clean_path, clean_wav.unsqueeze(0), sr)
            torchaudio.save(noisy_path, noisy_wav.unsqueeze(0), sr)
            
            # 保存文本文件
            txt_path = os.path.join(data_root, txt_dir, f'sample_{i:03d}.txt')
            with open(txt_path, 'w') as f:
                f.write(f"sample_{i:03d} This is a dummy transcription for sample {i}\n")
    
    print(f"虚拟数据已创建在: {data_root}")


if __name__ == "__main__":
    test_data_loader()