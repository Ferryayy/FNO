"""
工具函数模块
包含随机种子设置、配置保存等辅助功能
"""

import torch
import numpy as np
import random
import yaml
import os
from pathlib import Path


def setup_seed(seed):
    """
    设置随机种子以确保实验可复现
    
    Args:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config, save_path):
    """
    保存配置文件的快照
    
    Args:
        config (dict): 配置字典
        save_path (str): 保存路径
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path):
    """
    加载 YAML 配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_dir(base_dir, experiment_name):
    """
    创建带时间戳的实验目录
    
    Args:
        base_dir (str): 基础目录路径
        experiment_name (str): 实验名称
        
    Returns:
        str: 创建的实验目录路径
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}")
    
    # 创建必要的子目录
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(exp_dir, 'checkpoints')).mkdir(exist_ok=True)
    Path(os.path.join(exp_dir, 'tensorboard')).mkdir(exist_ok=True)
    
    return exp_dir


def count_parameters(model):
    """
    统计模型的可训练参数数量
    
    Args:
        model (nn.Module): PyTorch 模型
        
    Returns:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """
    用于跟踪和计算平均值的辅助类
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_mesh_comparison(gt_mesh, pred_mesh, fig_size=(10, 10), dpi=100):
    """
    可视化GT网格和预测网格的对比
    
    Args:
        gt_mesh (torch.Tensor or np.ndarray): Ground Truth网格坐标，shape (2, H, W)
        pred_mesh (torch.Tensor or np.ndarray): 预测网格坐标，shape (2, H, W)
        fig_size (tuple): 图像大小
        dpi (int): 图像分辨率
        
    Returns:
        np.ndarray: RGB图像数组，shape (H, W, 3)，范围[0, 255]
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    
    # 转换为numpy数组
    if torch.is_tensor(gt_mesh):
        gt_mesh = gt_mesh.cpu().numpy()
    if torch.is_tensor(pred_mesh):
        pred_mesh = pred_mesh.cpu().numpy()
    
    # 确保形状是 (2, H, W)
    assert gt_mesh.shape[0] == 2 and pred_mesh.shape[0] == 2
    H, W = gt_mesh.shape[1], gt_mesh.shape[2]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xlim(gt_mesh[0].min() - 0.1, gt_mesh[0].max() + 0.1)
    ax.set_ylim(gt_mesh[1].min() - 0.1, gt_mesh[1].max() + 0.1)
    
    # 绘制函数：绘制网格线
    def draw_mesh(mesh, color, alpha=0.6, linewidth=0.5):
        x_coords = mesh[0]  # (H, W)
        y_coords = mesh[1]  # (H, W)
        
        # 1. 横向连接（每一行）
        for i in range(H):
            for j in range(W - 1):
                ax.plot([x_coords[i, j], x_coords[i, j+1]], 
                       [y_coords[i, j], y_coords[i, j+1]], 
                       color=color, alpha=alpha, linewidth=linewidth)
        
        # 2. 纵向连接（每一列）
        for j in range(W):
            for i in range(H - 1):
                ax.plot([x_coords[i, j], x_coords[i+1, j]], 
                       [y_coords[i, j], y_coords[i+1, j]], 
                       color=color, alpha=alpha, linewidth=linewidth)
        
        # 3. 对角连接（每个方格的右上角连接左下角）
        for i in range(H - 1):
            for j in range(W - 1):
                ax.plot([x_coords[i, j+1], x_coords[i+1, j]], 
                       [y_coords[i, j+1], y_coords[i+1, j]], 
                       color=color, alpha=alpha, linewidth=linewidth)
    
    # 绘制GT网格（蓝色）和预测网格（红色）
    draw_mesh(gt_mesh, color='blue', alpha=0.6, linewidth=0.5)
    draw_mesh(pred_mesh, color='red', alpha=0.6, linewidth=0.5)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 移除坐标轴
    ax.axis('off')
    
    # 转换为图像数组
    fig.canvas.draw()
    # 使用 buffer_rgba() 替代已弃用的 tostring_rgb()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # 转换 RGBA 到 RGB (去掉 alpha 通道)
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    return img_array


def visualize_mesh_multichannel(gt_meshes, pred_meshes, fig_size=(10, 10), dpi=100):
    """
    可视化多通道网格对比（例如RGB三通道）
    
    Args:
        gt_meshes (torch.Tensor or np.ndarray): GT网格坐标，shape (C, 2, H, W)，C为通道数
        pred_meshes (torch.Tensor or np.ndarray): 预测网格坐标，shape (C, 2, H, W)
        fig_size (tuple): 每个子图的大小
        dpi (int): 图像分辨率
        
    Returns:
        list: 包含C个RGB图像数组的列表，每个shape (H, W, 3)
    """
    # 转换为numpy数组
    if torch.is_tensor(gt_meshes):
        gt_meshes = gt_meshes.cpu().numpy()
    if torch.is_tensor(pred_meshes):
        pred_meshes = pred_meshes.cpu().numpy()
    
    # 确保形状是 (C, 2, H, W)
    assert gt_meshes.ndim == 4 and pred_meshes.ndim == 4
    assert gt_meshes.shape[1] == 2 and pred_meshes.shape[1] == 2
    
    num_channels = gt_meshes.shape[0]
    images = []
    
    # 为每个通道生成一张可视化图像
    for c in range(num_channels):
        img = visualize_mesh_comparison(
            gt_meshes[c],  # (2, H, W)
            pred_meshes[c],  # (2, H, W)
            fig_size=fig_size,
            dpi=dpi
        )
        images.append(img)
    
    return images

