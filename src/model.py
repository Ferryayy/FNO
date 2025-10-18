"""
FNO (Fourier Neural Operator) 模型定义
包含 SpectralConv2d 和 FNO2d 主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np


class SpectralConv2d(nn.Module):
    """
    2D 傅里叶谱卷积层
    
    这是 FNO 的核心组件，在傅里叶域中对低频信息进行学习和变换。
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            modes1 (int): 傅里叶域中x方向要保留的低频模式数量
            modes2 (int): 傅里叶域中y方向要保留的低频模式数量
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 学习的权重，用于在傅里叶域中对低频模式进行线性变换
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，shape: (batch, in_channels, height, width)
            
        Returns:
            torch.Tensor: 输出张量，shape: (batch, out_channels, height, width)
        """
        batchsize = x.shape[0]
        
        # 1. 应用傅里叶变换
        # (batch, in_channel, x, y) -> (batch, in_channel, x, y//2 + 1)
        x_ft = torch.fft.rfft2(x)

        # 2. 在傅里叶域中进行滤波和线性变换
        # 初始化输出频谱
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device
        )
        
        # 对正频率部分进行操作
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], 
            self.weights1
        )
        
        # 对负频率部分进行操作
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], 
            self.weights2
        )

        # 3. 应用逆傅里叶变换
        # (batch, out_channel, x, y//2 + 1) -> (batch, out_channel, x, y)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator
    
    将谱卷积层与常规的卷积/线性层结合，构建完整的FNO模型。
    用于学习从密度函数到最优传输映射的算子。
    """
    def __init__(self, modes1, modes2, width):
        """
        Args:
            modes1 (int): 傅里叶模式数 (x方向)
            modes2 (int): 傅里叶模式数 (y方向)
            width (int): 隐层特征的通道数
        """
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # P: 将输入从3个通道(x,y坐标 + 密度)提升到高维隐空间
        self.fc0 = nn.Linear(3, self.width)

        # 4个傅里叶层
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # 傅里叶层后的 pointwise 线性变换，使用1x1卷积实现
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Q: 将高维隐空间特征投影回输出空间 (2个通道，代表OT map的x,y分量)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入密度，shape: (batch, height, width, 1)
            
        Returns:
            torch.Tensor: 预测的OT映射，shape: (batch, 2, height, width)
        """
        # 获取网格坐标并与输入拼接
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # (batch, h, w, 3)

        # 应用 P (Lifting)
        x = self.fc0(x)  # (batch, h, w, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, h, w)

        # 傅里叶层块 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # 傅里叶层块 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # 傅里叶层块 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # 傅里叶层块 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 应用 Q (Projection)
        x = x.permute(0, 2, 3, 1)  # (batch, h, w, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (batch, h, w, 2)
        x = x.permute(0, 3, 1, 2)  # (batch, 2, h, w)
        return x

    def get_grid(self, shape, device):
        """
        生成归一化的网格坐标
        
        Args:
            shape (tuple): 输入张量的形状 (batch, height, width, channels)
            device (torch.device): 设备
            
        Returns:
            torch.Tensor: 网格坐标，shape: (batch, height, width, 2)
        """
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

