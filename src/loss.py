"""
损失函数模块
包含各种损失项：数据拟合损失、物理约束损失（curl, Laplacian）、Beltrami系数约束等
"""

import torch
import torch.nn.functional as F
from .math_tool import compute_beltrami_from_grids, compute_constraint_penalty


class CompositeLoss:
    """
    复合损失函数类
    
    支持的损失项：
    1. 数据拟合损失（SmoothL1Loss）
    2. Curl 正则化（旋度约束）
    3. Laplacian 正则化（平滑约束）
    4. 范围约束（absolute模式下，坐标在[0,1]范围内）
    5. Beltrami系数约束（保证映射的双射性，mu的模长小于1）
    
    Args:
        config (dict): 配置字典，包含loss_params等参数
    """
    
    def __init__(self, config=None):
        """
        初始化复合损失函数
        
        Args:
            config (dict): 配置字典
        """
        self.config = config or {}
        cfg = self.config.get('loss_params', {})
        
        # 各项损失权重
        self.w_data = cfg.get('w_data', 1.0)
        self.w_curl = cfg.get('w_curl', 0.0)
        self.w_lap = cfg.get('w_lap', 0.0)
        self.w_range = cfg.get('w_range', 0.0)
        self.w_beltrami = cfg.get('w_beltrami', 0.0)  # Beltrami系数约束权重
        
        # 数据损失参数
        self.huber_beta = cfg.get('huber_beta', 0.5)
        
        # Beltrami约束参数
        self.beltrami_penalty_type = cfg.get('beltrami_penalty_type', 'relu')  # 'relu' 或 'log'
        self.beltrami_wc = cfg.get('beltrami_wc', 100.0)  # relu模式下的权重
        self.beltrami_wb = cfg.get('beltrami_wb', 1.0)    # log模式下的权重
        self.beltrami_delta = cfg.get('beltrami_delta', 1e-6)  # relu模式下的margin
        self.beltrami_epsilon = cfg.get('beltrami_epsilon', 1e-9)  # log模式下的数值稳定项
        
        # 预测模式
        train_params = self.config.get('train_params', {})
        self.predict_mode = train_params.get('predict_mode', 'offset')
        
        # 基础损失函数
        self.smooth_l1 = torch.nn.SmoothL1Loss(beta=self.huber_beta)
        self.mse_loss = torch.nn.MSELoss()
    
    def __call__(self, pred, target, base_grid=None):
        """
        计算总损失
        
        Args:
            pred (torch.Tensor): 预测值，shape (B, 2, H, W)
            target (torch.Tensor): 目标值，shape (B, 2, H, W)
            base_grid (torch.Tensor, optional): 基础网格，shape (B, 2, H, W)。
                                                 在计算Beltrami约束时需要。
        
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (torch.Tensor): 总损失（标量）
                - loss_dict (dict): 各项损失的字典 {'data': float, 'curl': float, ...}
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 数据拟合损失
        if self.w_data > 0.0:
            data_loss = self.smooth_l1(pred, target)
            total_loss = total_loss + self.w_data * data_loss
            loss_dict['data'] = data_loss.item()
        
        # 2. Curl 正则化
        if self.w_curl > 0.0:
            curl_loss = self._compute_curl_loss(pred)
            total_loss = total_loss + self.w_curl * curl_loss
            loss_dict['curl'] = curl_loss.item()
        
        # 3. Laplacian 正则化
        if self.w_lap > 0.0:
            lap_loss = self._compute_laplacian_loss(pred)
            total_loss = total_loss + self.w_lap * lap_loss
            loss_dict['laplacian'] = lap_loss.item()
        
        # 4. 范围约束（仅在absolute模式下）
        if self.w_range > 0.0 and self.predict_mode == 'absolute':
            range_loss = self._compute_range_loss(pred)
            total_loss = total_loss + self.w_range * range_loss
            loss_dict['range'] = range_loss.item()
        
        # 5. Beltrami系数约束
        if self.w_beltrami > 0.0 and base_grid is not None:
            beltrami_loss = self._compute_beltrami_loss(pred, base_grid)
            total_loss = total_loss + self.w_beltrami * beltrami_loss
            loss_dict['beltrami'] = beltrami_loss.item()
        
        # 记录总损失
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_curl_loss(self, u):
        """
        计算curl（旋度）的L2范数
        
        对于2D向量场 u = (u_x, u_y)，旋度为：
        curl(u) = ∂u_y/∂x - ∂u_x/∂y
        
        Args:
            u (torch.Tensor): 向量场，shape (B, 2, H, W)
        
        Returns:
            torch.Tensor: curl损失（标量）
        """
        # 计算梯度
        ux = u[:, 0:1]  # (B, 1, H, W)
        uy = u[:, 1:2]  # (B, 1, H, W)
        
        # ∂u_x/∂x 和 ∂u_x/∂y
        ux_x, ux_y = self._grad_xy(ux)
        
        # ∂u_y/∂x 和 ∂u_y/∂y
        uy_x, uy_y = self._grad_xy(uy)
        
        # curl = ∂u_y/∂x - ∂u_x/∂y
        # 需要对齐维度（取交集）
        curl = uy_x[..., :, :-1] - ux_y[..., :-1, :]
        
        return curl.pow(2).mean()
    
    def _compute_laplacian_loss(self, u):
        """
        计算Laplacian的L2范数
        
        Laplacian算子：Δu = ∂²u/∂x² + ∂²u/∂y²
        
        Args:
            u (torch.Tensor): 向量场，shape (B, 2, H, W)
        
        Returns:
            torch.Tensor: Laplacian损失（标量）
        """
        # 使用3x3卷积核近似Laplacian
        # [ 0  1  0 ]
        # [ 1 -4  1 ]
        # [ 0  1  0 ]
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
            dtype=u.dtype, 
            device=u.device
        ).view(1, 1, 3, 3)
        
        # 为每个通道复制kernel
        kernel = kernel.repeat(u.size(1), 1, 1, 1)
        
        # 分组卷积
        laplacian = F.conv2d(u, kernel, padding=1, groups=u.size(1))
        
        return laplacian.pow(2).mean()
    
    def _compute_range_loss(self, abs_coords):
        """
        计算范围约束损失（坐标超出[0,1]范围的惩罚）
        
        Args:
            abs_coords (torch.Tensor): 绝对坐标，shape (B, 2, H, W)
        
        Returns:
            torch.Tensor: 范围损失（标量）
        """
        # 低于0的部分
        oob_low = F.relu(0.0 - abs_coords)
        
        # 高于1的部分
        oob_high = F.relu(abs_coords - 1.0)
        
        return (oob_low + oob_high).abs().mean()
    
    def _compute_beltrami_loss(self, pred, base_grid):
        """
        计算Beltrami系数约束损失
        
        根据predict_mode计算源网格和目标网格：
        - offset模式：base_grid -> base_grid + pred
        - absolute模式：base_grid -> pred
        
        然后计算Beltrami系数mu，并约束|mu| < 1
        
        Args:
            pred (torch.Tensor): 预测值，shape (B, 2, H, W)
            base_grid (torch.Tensor): 基础网格，shape (B, 2, H, W)
        
        Returns:
            torch.Tensor: Beltrami约束损失（标量）
        """
        B = pred.size(0)
        total_penalty = 0.0
        
        # 根据predict_mode确定目标网格
        if self.predict_mode == 'offset':
            target_grid = base_grid + pred
        elif self.predict_mode == 'absolute':
            target_grid = pred
        else:
            raise ValueError(f"Unknown predict_mode: {self.predict_mode}")
        
        # 对batch中的每个样本分别计算
        for i in range(B):
            src_grid_2hw = base_grid[i]      # (2, H, W)
            tgt_grid_2hw = target_grid[i]    # (2, H, W)
            
            # 计算Beltrami系数
            mu = compute_beltrami_from_grids(
                src_grid_2hw, 
                tgt_grid_2hw,
                faces=None,  # 自动生成标准网格拓扑
                epsilon=1e-12
            )  # 返回 (M,) 复数张量，M为三角形数量
            
            # 计算约束惩罚
            penalty = compute_constraint_penalty(
                mu,
                penalty_type=self.beltrami_penalty_type,
                wc=self.beltrami_wc,
                wb=self.beltrami_wb,
                delta=self.beltrami_delta,
                epsilon=self.beltrami_epsilon
            )
            
            total_penalty = total_penalty + penalty
        
        # 对batch求平均
        return total_penalty / B
    
    def _grad_xy(self, f):
        """
        计算梯度（使用有限差分）
        
        Args:
            f (torch.Tensor): 标量场，shape (B, 1, H, W)
        
        Returns:
            tuple: (fx, fy)
                - fx: ∂f/∂x，shape (B, 1, H-1, W)
                - fy: ∂f/∂y，shape (B, 1, H, W-1)
        """
        fx = f[..., 1:, :] - f[..., :-1, :]  # (B, 1, H-1, W)
        fy = f[..., :, 1:] - f[..., :, :-1]  # (B, 1, H, W-1)
        return fx, fy


def create_loss_fn(config=None):
    """
    工厂函数：创建损失函数
    
    Args:
        config (dict): 配置字典
    
    Returns:
        CompositeLoss: 复合损失函数实例
    """
    return CompositeLoss(config)

