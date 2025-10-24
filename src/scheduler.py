"""
学习率调度器模块
提供多种学习率调度策略和预热功能
"""

import logging
import torch
from torch.optim.lr_scheduler import (
    StepLR, 
    MultiStepLR, 
    ExponentialLR,
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR,
    SequentialLR
)


logger = logging.getLogger(__name__)


def create_scheduler(optimizer, config, train_loader=None):
    """
    根据配置创建学习率调度器
    
    支持的调度器类型:
        - StepLR: 固定步长降低学习率
        - MultiStepLR: 在指定的epoch降低学习率
        - ExponentialLR: 指数衰减
        - CosineAnnealingLR: 余弦退火（推荐）
        - CosineAnnealingWarmRestarts: 带周期性重启的余弦退火
        - ReduceLROnPlateau: 基于验证损失自适应调整
        - OneCycleLR: 一周期策略，快速收敛
        - None: 不使用调度器
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器
        config (dict): 配置字典
        train_loader (DataLoader, optional): 训练数据加载器，用于OneCycleLR计算总步数
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: 学习率调度器
    """
    scheduler_params = config.get('scheduler_params', {})
    scheduler_type = scheduler_params.get('type', None)
    
    # 不使用调度器
    if scheduler_type is None or scheduler_type == 'None':
        logger.info("不使用学习率调度器")
        return None
    
    # 获取总训练轮数
    num_epochs = config['train_params']['num_epochs']
    
    # 创建主调度器
    scheduler = _create_main_scheduler(
        optimizer, 
        scheduler_type, 
        scheduler_params, 
        num_epochs,
        train_loader,
        config
    )
    
    if scheduler is None:
        return None
    
    # 添加学习率预热（如果需要）
    scheduler = _add_warmup_if_needed(
        optimizer, 
        scheduler, 
        scheduler_type, 
        scheduler_params
    )
    
    return scheduler


def _create_main_scheduler(optimizer, scheduler_type, scheduler_params, num_epochs, train_loader, config):
    """
    创建主学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type (str): 调度器类型
        scheduler_params (dict): 调度器参数
        num_epochs (int): 总训练轮数
        train_loader: 训练数据加载器
        config (dict): 完整配置
        
    Returns:
        学习率调度器实例或None
    """
    
    if scheduler_type == 'StepLR':
        step_size = scheduler_params.get('step_size', 30)
        gamma = scheduler_params.get('gamma', 0.5)
        
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        logger.info(f"使用 StepLR: step_size={step_size}, gamma={gamma}")
        
    elif scheduler_type == 'MultiStepLR':
        milestones = scheduler_params.get('milestones', [60, 120, 160])
        gamma = scheduler_params.get('gamma', 0.2)
        
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        logger.info(f"使用 MultiStepLR: milestones={milestones}, gamma={gamma}")
        
    elif scheduler_type == 'ExponentialLR':
        gamma = scheduler_params.get('gamma', 0.95)
        
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
        logger.info(f"使用 ExponentialLR: gamma={gamma}")
        
    elif scheduler_type == 'CosineAnnealingLR':
        # 默认 T_max 为总训练轮数，避免学习率反弹
        T_max = scheduler_params.get('T_max', num_epochs)
        eta_min = scheduler_params.get('eta_min', 0)
        
        if T_max < num_epochs:
            logger.warning(
                f"警告: T_max ({T_max}) < num_epochs ({num_epochs}), "
                f"学习率可能会在训练后期反弹！建议设置 T_max={num_epochs}"
            )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        logger.info(f"使用 CosineAnnealingLR: T_max={T_max}, eta_min={eta_min}")
        
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler_params.get('T_0', 10)
        T_mult = scheduler_params.get('T_mult', 2)
        eta_min = scheduler_params.get('eta_min', 0)
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        logger.info(
            f"使用 CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}"
        )
        
    elif scheduler_type == 'ReduceLROnPlateau':
        patience = scheduler_params.get('patience', 10)
        factor = scheduler_params.get('gamma', 0.5)
        min_lr = scheduler_params.get('min_lr', 1e-7)
        threshold = scheduler_params.get('threshold', 1e-4)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            threshold=threshold,
            verbose=True
        )
        logger.info(
            f"使用 ReduceLROnPlateau: patience={patience}, factor={factor}, "
            f"min_lr={min_lr}, threshold={threshold}"
        )
        
    elif scheduler_type == 'OneCycleLR':
        # OneCycleLR 需要知道总步数
        if train_loader is None:
            logger.error("OneCycleLR 需要 train_loader 参数来计算总步数")
            return None
        
        steps_per_epoch = len(train_loader)
        max_lr = scheduler_params.get('max_lr', config['train_params']['learning_rate'] * 10)
        pct_start = scheduler_params.get('pct_start', 0.3)
        anneal_strategy = scheduler_params.get('anneal_strategy', 'cos')
        div_factor = scheduler_params.get('div_factor', 25.0)
        final_div_factor = scheduler_params.get('final_div_factor', 10000.0)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        logger.info(
            f"使用 OneCycleLR: max_lr={max_lr}, steps_per_epoch={steps_per_epoch}, "
            f"pct_start={pct_start}, div_factor={div_factor}, final_div_factor={final_div_factor}"
        )
        
    else:
        logger.warning(f"未知的调度器类型: {scheduler_type}, 将不使用调度器")
        return None
    
    return scheduler


def _add_warmup_if_needed(optimizer, scheduler, scheduler_type, scheduler_params):
    """
    如果配置了预热，则添加预热阶段
    
    Args:
        optimizer: 优化器
        scheduler: 主调度器
        scheduler_type (str): 调度器类型
        scheduler_params (dict): 调度器参数
        
    Returns:
        可能包含预热的调度器
    """
    warmup_epochs = scheduler_params.get('warmup_epochs', 0)
    
    # OneCycleLR 自带 warmup，不需要额外包装
    if warmup_epochs <= 0 or scheduler_type == 'OneCycleLR':
        return scheduler
    
    # 创建预热调度器
    def warmup_lambda(epoch):
        """线性预热"""
        return (epoch + 1) / warmup_epochs
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # 组合预热和主调度器
    combined_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs]
    )
    
    logger.info(f"添加学习率预热: warmup_epochs={warmup_epochs}")
    
    return combined_scheduler


def step_scheduler(scheduler, epoch, val_loss=None):
    """
    统一的调度器更新接口
    
    Args:
        scheduler: 学习率调度器
        epoch (int): 当前epoch
        val_loss (float, optional): 验证损失，用于ReduceLROnPlateau
        
    Returns:
        bool: 是否成功更新
    """
    if scheduler is None:
        return False
    
    # ReduceLROnPlateau 需要验证损失
    if isinstance(scheduler, ReduceLROnPlateau):
        if val_loss is not None:
            scheduler.step(val_loss)
            return True
        else:
            logger.warning("ReduceLROnPlateau 需要 val_loss 参数，跳过本次更新")
            return False
    
    # OneCycleLR 在每个batch后更新，这里不处理
    elif isinstance(scheduler, OneCycleLR):
        # OneCycleLR 应该在训练循环中每个batch后调用
        return False
    
    # 其他调度器按epoch更新
    else:
        scheduler.step()
        return True


def get_current_lr(optimizer):
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        float: 当前学习率
    """
    return optimizer.param_groups[0]['lr']


def print_lr_schedule_info(config, train_loader=None):
    """
    打印学习率调度信息（用于调试）
    
    Args:
        config (dict): 配置字典
        train_loader: 训练数据加载器
    """
    scheduler_params = config.get('scheduler_params', {})
    scheduler_type = scheduler_params.get('type', 'None')
    num_epochs = config['train_params']['num_epochs']
    initial_lr = config['train_params']['learning_rate']
    
    logger.info("=" * 60)
    logger.info("学习率调度信息:")
    logger.info(f"  初始学习率: {initial_lr}")
    logger.info(f"  总训练轮数: {num_epochs}")
    logger.info(f"  调度器类型: {scheduler_type}")
    
    if scheduler_type == 'CosineAnnealingLR':
        T_max = scheduler_params.get('T_max', num_epochs)
        eta_min = scheduler_params.get('eta_min', 0)
        logger.info(f"  T_max: {T_max}")
        logger.info(f"  最小学习率: {eta_min}")
        
        if T_max < num_epochs:
            logger.warning(f"  ⚠️  警告: T_max < num_epochs, 学习率会在第 {T_max} 轮后反弹！")
            
    elif scheduler_type == 'ReduceLROnPlateau':
        patience = scheduler_params.get('patience', 10)
        factor = scheduler_params.get('gamma', 0.5)
        logger.info(f"  耐心值: {patience}")
        logger.info(f"  衰减因子: {factor}")
        
    warmup_epochs = scheduler_params.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        logger.info(f"  预热轮数: {warmup_epochs}")
        
    logger.info("=" * 60)

