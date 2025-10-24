"""
训练器模块
包含完整的训练、验证、checkpoint管理逻辑
"""

import os
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
from .utils import AverageMeter, visualize_mesh_comparison, visualize_mesh_multichannel


class Trainer:
    """
    FNO 训练器类
    
    负责模型训练、验证、checkpoint 保存与加载、TensorBoard 日志记录等。
    """
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler,
        loss_fn, 
        train_loader, 
        val_loader, 
        device, 
        config, 
        exp_dir
    ):
        """
        Args:
            model (nn.Module): 模型
            optimizer (torch.optim.Optimizer): 优化器
            scheduler: 学习率调度器
            loss_fn (callable): 损失函数
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            device (torch.device): 设备
            config (dict): 配置字典
            exp_dir (str): 实验目录路径
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.exp_dir = exp_dir
        
        # 提取常用配置
        self.num_epochs = config['train_params']['num_epochs']
        self.save_freq = config['log_params']['save_freq']
        self.val_freq = config['log_params']['val_freq']
        self.gradient_clip = config['train_params'].get('gradient_clip', None)
        
        # checkpoint 目录
        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        
        # 初始化 TensorBoard
        self._setup_logging()
        
        # 记录最佳性能
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # 日志记录器
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """初始化 TensorBoard SummaryWriter"""
        tensorboard_dir = os.path.join(self.exp_dir, 'tensorboard')
        self.writer = SummaryWriter(tensorboard_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TensorBoard 日志保存至: {tensorboard_dir}")
    
    def _prep_batch(self, inputs, targets):
        """
        - 支持 (B,C,H,W) -> (B*C,1,H,W) 展开
        - 支持 'offset' 训练：目标 = GT - base_grid
        - 返回: inputs_bhw1 (B*,H,W,1), targets_b2hw (B*,2,H,W), base_grid (B*,2,H,W)
        """
        # 展开多通道
        if inputs.dim() == 4 and inputs.size(1) > 1:
            B, C, H, W = inputs.shape
            inputs = inputs.view(B * C, 1, H, W)
            targets = targets.view(B, 2 * C, H, W).view(B * C, 2, H, W)

        # 设备
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 标准网格 (与模型 get_grid 的 [0,1] 一致)
        with torch.no_grad():
            H, W = inputs.size(2), inputs.size(3)
            # gridx = torch.linspace(0, 1, H, device=inputs.device).view(1, -1, 1).expand(inputs.size(0), -1, W)
            # gridy = torch.linspace(0, 1, W, device=inputs.device).view(1, 1, -1).expand(inputs.size(0), H, -1)
            # base_grid = torch.stack([gridx, gridy], dim=1)  # (B*,2,H,W)
            gridx = torch.linspace(0, 1, W, device=inputs.device).view(1, 1, -1).expand(inputs.size(0), H, -1)
            gridy = torch.linspace(1, 0, H, device=inputs.device).view(1, -1, 1).expand(inputs.size(0), -1, W)
            base_grid = torch.stack([gridx, gridy], dim=1)  # shape: (B,2,H,W)

        predict_mode = self.config['train_params'].get('predict_mode', 'offset')
        if predict_mode == 'offset':
            targets = targets - base_grid
        elif predict_mode == 'absolute':
            pass
        else:
            raise ValueError(f"Unknown predict_mode: {predict_mode}")

        # FNO 输入需要 (B*,H,W,1)
        inputs = inputs.permute(0, 2, 3, 1)

        return inputs, targets, base_grid
    
    def train(self):
        """主训练循环"""
        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info("=" * 60)
        
        # 如果需要断点续训
        if self.config['checkpoint_params']['resume_from_checkpoint']:
            self._load_checkpoint()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.logger.info("-" * 60)
            
            # 训练一个 epoch
            train_loss = self._train_epoch(epoch)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.logger.info(f"学习率: {current_lr:.6f}")
            
            # 验证
            if (epoch + 1) % self.val_freq == 0 or epoch == self.num_epochs - 1:
                val_loss = self._validate_epoch(epoch)
                
                # 检查是否是最优模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.logger.info(f"🎉 发现更优模型! 验证损失: {val_loss:.6f}")
            else:
                is_best = False
            
            # 保存 checkpoint
            if (epoch + 1) % self.save_freq == 0 or epoch == self.num_epochs - 1:
                self._save_checkpoint(epoch, is_best)
            
            # 更新学习率（OneCycleLR 在训练循环中每个batch更新，这里跳过）
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau 需要验证损失
                    self.scheduler.step(val_loss if 'val_loss' in locals() else train_loss)
                else:
                    # 其他调度器按epoch更新
                    self.scheduler.step()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        self.logger.info("=" * 60)
        self.writer.close()
    
    def _train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch (int): 当前 epoch 编号
            
        Returns:
            float: 该 epoch 的平均损失
        """
        self.model.train()
        loss_meter = AverageMeter()
        # 为各项损失创建meter
        loss_components_meters = {}
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets, base_grid = self._prep_batch(inputs, targets)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # (B, 2, H, W)
            
            # 计算损失（传入base_grid用于Beltrami约束）
            loss, loss_dict = self.loss_fn(outputs, targets, base_grid=base_grid)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # OneCycleLR 在每个batch后更新学习率
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # 更新总损失统计
            loss_meter.update(loss.item(), inputs.size(0))
            
            # 更新各项损失统计
            for key, value in loss_dict.items():
                if key not in loss_components_meters:
                    loss_components_meters[key] = AverageMeter()
                loss_components_meters[key].update(value, inputs.size(0))
            
            # 更新进度条（只显示总损失）
            pbar.set_postfix({'loss': f'{loss_meter.avg:.6f}'})
            
            # 记录到 TensorBoard (每个batch)
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
        
        # 记录 epoch 级别的损失
        self.writer.add_scalar('Train/Loss_Epoch', loss_meter.avg, epoch)
        
        # 打印总损失和各项损失
        loss_info = f"训练损失: {loss_meter.avg:.6f}"
        if loss_components_meters:
            # 构建分项损失字符串
            components_str = " | ".join([
                f"{key}: {meter.avg:.6f}" 
                for key, meter in loss_components_meters.items() 
                if key != 'total'  # 避免重复显示total
            ])
            loss_info += f" ({components_str})"
        
        self.logger.info(loss_info)
        
        return loss_meter.avg
    
    def _validate_epoch(self, epoch):
        """
        验证一个 epoch
        
        Args:
            epoch (int): 当前 epoch 编号
            
        Returns:
            float: 验证集的平均损失
        """
        self.model.eval()
        loss_meter = AverageMeter()
        # 为各项损失创建meter
        loss_components_meters = {}
        
        # 用于存储第一个batch的可视化数据
        first_batch_raw_inputs = None
        first_batch_raw_targets = None
        first_batch_outputs = None
        first_batch_base_grid = None
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch+1}")
            
            for batch_idx, (raw_inputs, raw_targets) in enumerate(pbar):
                # 保存原始数据用于可视化（第一个batch）
                if batch_idx == 0:
                    first_batch_raw_inputs = raw_inputs
                    first_batch_raw_targets = raw_targets
                
                inputs, targets, base_grid = self._prep_batch(raw_inputs, raw_targets)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失（传入base_grid用于Beltrami约束）
                loss, loss_dict = self.loss_fn(outputs, targets, base_grid=base_grid)
                
                # 更新总损失统计
                loss_meter.update(loss.item(), inputs.size(0))
                
                # 更新各项损失统计
                for key, value in loss_dict.items():
                    if key not in loss_components_meters:
                        loss_components_meters[key] = AverageMeter()
                    loss_components_meters[key].update(value, inputs.size(0))
                
                # 更新进度条（只显示总损失）
                pbar.set_postfix({'val_loss': f'{loss_meter.avg:.6f}'})
                
                # 保存第一个batch的处理后数据用于可视化
                if batch_idx == 0:
                    first_batch_outputs = outputs
                    first_batch_base_grid = base_grid
        
        # 记录到 TensorBoard
        self.writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        
        # 打印总损失和各项损失
        loss_info = f"验证损失: {loss_meter.avg:.6f}"
        if loss_components_meters:
            # 构建分项损失字符串
            components_str = " | ".join([
                f"{key}: {meter.avg:.6f}" 
                for key, meter in loss_components_meters.items() 
                if key != 'total'  # 避免重复显示total
            ])
            loss_info += f" ({components_str})"
        
        self.logger.info(loss_info)
        
        # 网格可视化
        vis_params = self.config.get('visualization_params', {})
        enable_mesh_vis = vis_params.get('enable_mesh_vis', True)
        
        if first_batch_outputs is not None and enable_mesh_vis:
            num_samples = vis_params.get('vis_num_samples', 1)
            self._visualize_meshes(
                first_batch_raw_inputs,
                first_batch_raw_targets,
                first_batch_outputs, 
                first_batch_base_grid,
                epoch,
                num_samples=num_samples
            )
        
        return loss_meter.avg
    
    def _visualize_meshes(self, raw_inputs, raw_targets, outputs, base_grid, epoch, num_samples=1):
        """
        可视化网格对比并保存到TensorBoard
        
        Args:
            raw_inputs (torch.Tensor): 原始输入，shape (B, C, H, W)
            raw_targets (torch.Tensor): 原始目标，shape (B, C*2, H, W)
            outputs (torch.Tensor): 模型输出，shape (B*C, 2, H, W)
            base_grid (torch.Tensor): 基础网格，shape (B*C, 2, H, W)
            epoch (int): 当前epoch
            num_samples (int): 要可视化的样本数量
        """
        predict_mode = self.config['train_params'].get('predict_mode', 'offset')
        
        # 检测是单通道还是多通道
        if raw_inputs.dim() == 4 and raw_inputs.size(1) > 1:
            # 多通道情况 (B, C, H, W)
            B, C, H, W = raw_inputs.shape
            is_multichannel = True
            num_channels = C
        else:
            # 单通道情况 (B, 1, H, W)
            is_multichannel = False
            num_channels = 1
        
        # 将预测转换为absolute坐标
        if predict_mode == 'offset':
            pred_absolute = outputs + base_grid
        else:
            pred_absolute = outputs
        
        # 处理目标坐标（raw_targets 已经是绝对坐标）
        target_absolute = raw_targets.to(self.device)
        if is_multichannel:
            # 多通道: (B, C*2, H, W) -> (B*C, 2, H, W)
            B, C2, H, W = target_absolute.shape
            target_absolute = target_absolute.view(B, C2 // 2, 2, H, W).reshape(B * (C2 // 2), 2, H, W)
        
        # 可视化第一个样本
        sample_idx = 0
        
        if is_multichannel and num_channels > 1:
            # 多通道情况：为每个通道生成一张图
            self.logger.info(f"可视化多通道网格 ({num_channels} 通道)")
            
            for c in range(num_channels):
                idx = sample_idx * num_channels + c
                if idx >= pred_absolute.size(0):
                    break
                
                pred_mesh = pred_absolute[idx]  # (2, H, W)
                gt_mesh = target_absolute[idx]  # (2, H, W)
                
                try:
                    # 生成可视化图像
                    img_array = visualize_mesh_comparison(
                        gt_mesh, 
                        pred_mesh, 
                        fig_size=(8, 8), 
                        dpi=100
                    )
                    
                    # 转换为TensorBoard格式 (C, H, W)，范围[0, 1]
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    
                    # 保存到TensorBoard
                    self.writer.add_image(
                        f'Val/Mesh_Comparison_Channel_{c}', 
                        img_tensor, 
                        epoch
                    )
                    
                    self.logger.info(f"已保存网格可视化 (通道 {c}) 到 TensorBoard")
                    
                except Exception as e:
                    self.logger.warning(f"网格可视化失败 (通道 {c}): {str(e)}")
        else:
            # 单通道情况
            self.logger.info("可视化单通道网格")
            
            for idx in range(min(num_samples, pred_absolute.size(0))):
                pred_mesh = pred_absolute[idx]  # (2, H, W)
                gt_mesh = target_absolute[idx]  # (2, H, W)
                
                try:
                    # 生成可视化图像
                    img_array = visualize_mesh_comparison(
                        gt_mesh, 
                        pred_mesh, 
                        fig_size=(8, 8), 
                        dpi=100
                    )
                    
                    # 转换为TensorBoard格式 (C, H, W)，范围[0, 1]
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    
                    # 保存到TensorBoard
                    self.writer.add_image(
                        f'Val/Mesh_Comparison_Sample_{idx}', 
                        img_tensor, 
                        epoch
                    )
                    
                    self.logger.info(f"已保存网格可视化 (样本 {idx}) 到 TensorBoard")
                    
                except Exception as e:
                    self.logger.warning(f"网格可视化失败 (样本 {idx}): {str(e)}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """
        保存 checkpoint
        
        Args:
            epoch (int): 当前 epoch
            is_best (bool): 是否为最优模型
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新的 checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint, last_path)
        self.logger.info(f"已保存 checkpoint: {last_path}")
        
        # 如果是最优模型，额外保存一份
        if is_best and self.config['checkpoint_params']['save_best']:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"已保存最优模型: {best_path}")
    
    def _load_checkpoint(self):
        """从 checkpoint 恢复训练"""
        checkpoint_path = self.config['checkpoint_params']['resume_from_checkpoint']
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint 路径不存在: {checkpoint_path}")
            return
        
        self.logger.info(f"从 checkpoint 恢复: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练进度
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"已恢复到 Epoch {self.start_epoch}")
        self.logger.info(f"当前最佳验证损失: {self.best_val_loss:.6f}")

