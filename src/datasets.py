"""
数据集模块
包含最优传输问题的数据生成和数据集类
"""

import torch
import numpy as np
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def generate_ot_pair(size=64):
    """
    生成一个(密度图像, OT图)数据对
    
    警告: 这是一个伪数据生成器，仅用于演示。
    在真实应用中，您应该用高精度的C++ OT求解器替换它。
    
    Args:
        size (int): 图像尺寸 (size x size)
        
    Returns:
        tuple: (density, ot_map)
            - density: torch.Tensor, shape (1, size, size)
            - ot_map: torch.Tensor, shape (2, size, size)
    """
    # 1. 生成随机的密度图像 (输入)
    # 使用多个高斯函数的叠加来模拟复杂的密度
    density = np.zeros((size, size))
    num_blobs = np.random.randint(3, 8)
    
    for _ in range(num_blobs):
        x0, y0 = np.random.rand(2) * size
        sx, sy = np.random.rand(2) * (size / 4) + (size / 10)
        amp = np.random.rand() + 0.5
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        density += amp * np.exp(-(((x-x0)/sx)**2 + ((y-y0)/sy)**2))
    
    # 归一化
    density += 1e-6
    density /= np.sum(density)

    # 2. 生成伪 OT 图 (输出)
    # 这是一个非常粗糙的近似，仅用于演示目的。
    # 真实场景中，这里应该调用您的高精度C++求解器。
    # 思想：粒子会从低密度区域流向高密度区域，可以用密度的梯度来近似。
    grad_y, grad_x = np.gradient(density)
    ot_map = np.stack([-grad_x, -grad_y], axis=0)
    # 缩放梯度，使其在合理范围内
    ot_map *= size * 2.0
    
    return torch.from_numpy(density).float().unsqueeze(0), \
           torch.from_numpy(ot_map).float()


class OTDataset(Dataset):
    """
    最优传输数据集
    
    每个样本包含：
        - 输入：密度函数 ρ_T，shape (1, H, W)
        - 输出：最优传输映射 T，shape (2, H, W)
    """
    def __init__(self, num_samples, size=64, cache=True, verbose=True):
        """
        Args:
            num_samples (int): 数据集样本数量
            size (int): 图像尺寸
            cache (bool): 是否预先生成并缓存所有数据
            verbose (bool): 是否打印进度信息
        """
        self.num_samples = num_samples
        self.size = size
        self.cache = cache
        self.data = []
        
        if cache:
            if verbose:
                print(f"正在生成 {num_samples} 个数据样本...")
            for i in range(num_samples):
                self.data.append(generate_ot_pair(self.size))
                if verbose and (i+1) % 100 == 0:
                    print(f"  已生成 {i+1}/{num_samples}")
            if verbose:
                print("数据生成完成！")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        Returns:
            tuple: (input, target)
                - input: 密度图，shape (1, H, W)
                - target: OT映射，shape (2, H, W)
        """
        if self.cache:
            return self.data[idx]
        else:
            # 不缓存则动态生成
            return generate_ot_pair(self.size)


def read_image(image_path, target_size=None):
    """
    读取灰度图像并转换为张量
    
    Args:
        image_path (str): 图像路径
        target_size (tuple): 目标尺寸 (H, W)，None则保持原尺寸
        
    Returns:
        torch.Tensor: 形状为 (1, H, W) 的图像张量，已归一化到 [0, 1]
    """
    # 读取图像并转换为灰度
    img = Image.open(image_path).convert('L')
    
    # 调整大小（如果需要）
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # 转换为numpy数组并归一化
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 转换为张量 (H, W) -> (1, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    
    return img_tensor


def read_ot_mesh_npy(mesh_path, target_size=None):
    """
    读取OT网格npy文件
    
    Args:
        mesh_path (str): .npy文件路径
        target_size (tuple): 目标尺寸 (H, W)，None则保持原尺寸
        
    Returns:
        torch.Tensor: 形状为 (2, H, W) 的OT映射张量
    """
    # 读取npy文件
    ot_map = np.load(mesh_path)  # shape: (2, H, W)
    
    # 检查形状
    if ot_map.ndim != 3 or ot_map.shape[0] != 2:
        raise ValueError(f"OT mesh应为形状 (2, H, W)，但得到 {ot_map.shape}")
    
    # 调整大小（如果需要）
    if target_size is not None:
        import torch.nn.functional as F
        # 将numpy转为torch tensor用于resize
        ot_tensor = torch.from_numpy(ot_map).unsqueeze(0)  # (1, 2, H, W)
        # 使用双线性插值调整大小
        ot_tensor = F.interpolate(
            ot_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        ot_tensor = ot_tensor.squeeze(0)  # (2, H, W)
    else:
        ot_tensor = torch.from_numpy(ot_map)
    
    return ot_tensor.float()


def to_target_measure(img, eps=1e-6):
    """
    将灰度图转换为目标测度 (概率密度)，形状保持 (1, H, W)
    """
    rho = img + eps
    rho = rho / rho.sum()
    return rho


def normalize_coords(ot_map, bounds):
    """
    将绝对坐标 (2, H, W) 归一化到 [0,1]^2。
    bounds: (x_min, x_max, y_min, y_max)
    """
    x_min, x_max, y_min, y_max = bounds
    norm = ot_map.clone()
    norm[0] = (norm[0] - x_min) / (x_max - x_min)
    norm[1] = (norm[1] - y_min) / (y_max - y_min)
    return norm.clamp(0.0, 1.0)


def maybe_to_measure_and_norm(image, ot_map, cfg):
    """
    按配置将输入转为测度并将坐标归一化到 [0,1]^2。
    期望配置位于 cfg['data_params']。
    """
    dp = cfg.get('data_params', {}) if isinstance(cfg, dict) else {}
    if dp.get('to_target_measure', True):
        image = to_target_measure(image, eps=dp.get('measure_eps', 1e-6))
    if dp.get('normalize_coords', True):
        bounds = dp.get('domain_bounds', [0.0, 1.0, 0.0, 1.0])
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            ot_map = normalize_coords(ot_map, tuple(bounds))
    return image, ot_map


class RealOTDataset(Dataset):
    """
    真实OT数据集 - 从图像和npy网格文件读取
    
    数据组织：
    - 图像目录：包含 1_xxx.jpg, 2_xxx.png 等
    - 网格目录：包含 1_xxx_mesh.npy, 2_xxx_mesh.npy 等
    - 前缀数字用于匹配图像和网格文件
    """
    def __init__(self, image_dir, mesh_dir, indices=None, target_size=None, verbose=True, config=None):
        """
        Args:
            image_dir (str): 图像目录路径
            mesh_dir (str): 网格文件目录路径（.npy）
            indices (list): 要使用的数据索引列表，None表示使用所有数据
            target_size (tuple): 目标图像尺寸 (H, W)，None则保持原尺寸
            verbose (bool): 是否打印详细信息
        """
        self.image_dir = image_dir
        self.mesh_dir = mesh_dir
        self.target_size = target_size
        self.verbose = verbose
        self.config = config or {}
        
        # 查找所有匹配的图像和网格文件对
        self.data_pairs = self._find_data_pairs()
        
        # 如果指定了索引，则筛选
        if indices is not None:
            self.data_pairs = [self.data_pairs[i] for i in indices]
        
        if verbose:
            logger.info(f"找到 {len(self.data_pairs)} 对数据")
    
    def _find_data_pairs(self):
        """查找所有匹配的图像和网格文件对"""
        pairs = []
        
        # 获取图像目录中的所有文件
        image_files = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in Path(self.image_dir).glob(ext):
                # 提取前缀数字，例如 9985_cifar_airplane.jpg -> 9985
                match = re.match(r'(\d+)_.*', img_path.name)
                if match:
                    prefix = int(match.group(1))
                    image_files[prefix] = str(img_path)
        
        # 获取网格目录中的所有.npy文件
        mesh_files = {}
        for mesh_path in Path(self.mesh_dir).glob('*.npy'):
            # 提取前缀数字，例如 9985_cifar_airplane_mesh.npy -> 9985
            match = re.match(r'(\d+)_.*\.npy', mesh_path.name)
            if match:
                prefix = int(match.group(1))
                mesh_files[prefix] = str(mesh_path)
        
        # 匹配图像和网格文件
        common_prefixes = sorted(set(image_files.keys()) & set(mesh_files.keys()))
        
        for prefix in common_prefixes:
            pairs.append({
                'prefix': prefix,
                'image_path': image_files[prefix],
                'mesh_path': mesh_files[prefix]
            })
        
        if self.verbose:
            logger.info(f"图像目录: {self.image_dir}")
            logger.info(f"网格目录: {self.mesh_dir}")
            logger.info(f"找到 {len(image_files)} 个图像文件")
            logger.info(f"找到 {len(mesh_files)} 个网格文件")
            logger.info(f"匹配到 {len(pairs)} 对数据")
            if len(pairs) > 0:
                logger.info(f"示例文件对: {pairs[0]['image_path']} <-> {pairs[0]['mesh_path']}")
        
        return pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        Returns:
            tuple: (image, ot_map)
                - image: 灰度图，shape (1, H, W)
                - ot_map: OT映射，shape (2, H, W)
        """
        pair = self.data_pairs[idx]
        
        # 读取图像
        image = read_image(pair['image_path'], self.target_size)
        
        # 读取OT网格（npy格式）
        ot_map = read_ot_mesh_npy(pair['mesh_path'], self.target_size)
        
        # 转测度 + 坐标归一化
        image, ot_map = maybe_to_measure_and_norm(image, ot_map, self.config)
        
        return image, ot_map


def split_dataset_indices(total_size, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    """
    划分数据集索引
    
    Args:
        total_size (int): 数据集总大小
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        random_seed (int): 随机种子
        
    Returns:
        tuple: (train_indices, val_indices)
    """
    indices = list(range(total_size))
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        indices, 
        train_size=train_ratio,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    return train_indices, val_indices


def get_data_loaders(config):
    """
    根据配置创建训练和验证数据加载器
    
    Args:
        config (dict): 配置字典
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    data_params = config['data_params']
    train_params = config['train_params']
    
    # 检查是否使用真实数据
    use_real_data = data_params.get('use_real_data', False)
    
    if use_real_data:
        # 使用真实数据
        print("\n使用真实OT数据集...")
        
        image_dir = data_params['image_dir']
        mesh_dir = data_params['mesh_dir']
        target_size = tuple(data_params.get('target_size', None)) if data_params.get('target_size') else None
        
        # 首先创建一个临时数据集来获取总数据量
        temp_dataset = RealOTDataset(
            image_dir=image_dir,
            mesh_dir=mesh_dir,
            target_size=target_size,
            verbose=True,
            config=config
        )
        
        total_size = len(temp_dataset)
        
        # 划分训练集和验证集索引
        train_ratio = data_params.get('train_ratio', 0.8)
        val_ratio = data_params.get('val_ratio', 0.2)
        train_indices, val_indices = split_dataset_indices(
            total_size, 
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=config['seed']
        )
        
        print(f"训练集样本数: {len(train_indices)}")
        print(f"验证集样本数: {len(val_indices)}")
        
        # 创建训练集和验证集
        train_dataset = RealOTDataset(
            image_dir=image_dir,
            mesh_dir=mesh_dir,
            indices=train_indices,
            target_size=target_size,
            verbose=False,
            config=config
        )
        
        val_dataset = RealOTDataset(
            image_dir=image_dir,
            mesh_dir=mesh_dir,
            indices=val_indices,
            target_size=target_size,
            verbose=False,
            config=config
        )
        
    else:
        # 使用伪数据（原来的方式）
        print("\n使用伪数据生成器...")
        train_dataset = OTDataset(
            num_samples=data_params['num_train_samples'],
            size=data_params['image_size'],
            cache=True,
            verbose=True
        )
        
        val_dataset = OTDataset(
            num_samples=data_params['num_val_samples'],
            size=data_params['image_size'],
            cache=True,
            verbose=True
        )
    
    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=data_params['num_workers'],
        pin_memory=data_params['pin_memory']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_params['batch_size'],
        shuffle=False,
        num_workers=data_params['num_workers'],
        pin_memory=data_params['pin_memory']
    )
    
    return train_loader, val_loader

