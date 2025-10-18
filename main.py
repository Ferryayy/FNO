"""
FNO 训练主入口
配置驱动的训练流程
"""

import argparse
import logging
import os
import sys
import torch

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import FNO2d
from src.datasets import get_data_loaders
from src.trainer import Trainer
from src.utils import (
    setup_seed, 
    load_config, 
    save_config, 
    create_experiment_dir,
    count_parameters
)


def setup_logger(exp_dir):
    """
    设置日志系统，同时输出到控制台和文件
    
    Args:
        exp_dir (str): 实验目录
    """
    log_file = os.path.join(exp_dir, 'train.log')
    
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_optimizer(model, config):
    """
    根据配置创建优化器
    
    Args:
        model (nn.Module): 模型
        config (dict): 配置字典
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    train_params = config['train_params']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params['learning_rate'],
        weight_decay=train_params['weight_decay']
    )
    return optimizer


def create_scheduler(optimizer, config):
    """
    根据配置创建学习率调度器
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器
        config (dict): 配置字典
        
    Returns:
        torch.optim.lr_scheduler: 调度器
    """
    scheduler_params = config['scheduler_params']
    scheduler_type = scheduler_params['type']
    
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max']
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_params['patience'],
            factor=scheduler_params['gamma']
        )
    else:
        scheduler = None
        logging.warning(f"未知的调度器类型: {scheduler_type}, 将不使用调度器")
    
    return scheduler


def create_loss_fn(config=None):
    """
    复合损失：
      L = w_data * SmoothL1 + w_curl * ||curl||^2 + w_lap * ||Δu||^2 + w_range * hinge_out_of_bounds
    适用于 predict_mode='offset' 或 'absolute'。
    """
    cfg = (config or {}).get('loss_params', {}) if isinstance(config, dict) else {}
    w_data  = cfg.get('w_data', 1.0)
    w_curl  = cfg.get('w_curl', 0.0)
    w_lap   = cfg.get('w_lap', 0.0)
    w_range = cfg.get('w_range', 0.0)
    huber_beta = cfg.get('huber_beta', 0.5)
    def grad_xy(f):
        # f: (B,1,H,W)
        fx = f[..., 1:, :] - f[..., :-1, :]
        fy = f[..., :, 1:] - f[..., :, :-1]
        return fx, fy

    def laplacian(u):
        # u: (B,2,H,W)
        import torch.nn.functional as F
        k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=u.dtype, device=u.device).view(1,1,3,3)
        k = k.repeat(u.size(1), 1, 1, 1)
        return torch.conv2d(u, k, padding=1, groups=u.size(1))

    def hinge_out_of_bounds(abs_coords):
        oob_low  = torch.relu(0.0 - abs_coords)
        oob_high = torch.relu(abs_coords - 1.0)
        return (oob_low + oob_high).abs().mean()

    smooth_l1 = torch.nn.SmoothL1Loss(beta=huber_beta)
    mse_loss = torch.nn.MSELoss()
    predict_mode = (config or {}).get('train_params', {}).get('predict_mode', 'offset')

    def loss_fn(pred, target):
        # pred, target: (B,2,H,W)
        L = 0.0
        L = L + w_data * mse_loss(pred, target)

        if w_curl > 0.0:
            px_x, px_y = grad_xy(pred[:, 0:1])
            py_x, py_y = grad_xy(pred[:, 1:2])
            curl = (py_x[..., :, :-1] - px_y[..., :-1, :])
            L = L + w_curl * (curl.pow(2).mean())

        if w_lap > 0.0:
            L = L + w_lap * (laplacian(pred).pow(2).mean())

        if w_range > 0.0 and predict_mode == 'absolute':
            L = L + w_range * hinge_out_of_bounds(pred)

        return L

    return loss_fn


def main(args):
    """主函数"""
    # 1. 加载配置
    print("=" * 60)
    print("FNO 训练框架")
    print("=" * 60)
    print(f"\n加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 设置随机种子
    setup_seed(config['seed'])
    
    # 3. 创建实验目录
    exp_dir = create_experiment_dir(
        config['log_params']['log_dir_base'],
        config['log_params']['experiment_name']
    )
    print(f"实验目录: {exp_dir}")
    
    # 4. 设置日志系统
    logger = setup_logger(exp_dir)
    logger.info("=" * 60)
    logger.info("FNO 训练开始")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"实验目录: {exp_dir}")
    
    # 5. 保存配置快照
    config_snapshot_path = os.path.join(exp_dir, 'config_snapshot.yaml')
    save_config(config, config_snapshot_path)
    logger.info(f"配置快照已保存: {config_snapshot_path}")
    
    # 6. 设置设备
    device = torch.device(config['train_params']['device'] 
                         if torch.cuda.is_available() 
                         else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 7. 创建数据加载器
    logger.info("\n初始化数据集...")
    train_loader, val_loader = get_data_loaders(config)
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    logger.info(f"批次大小: {config['train_params']['batch_size']}")
    
    # 8. 创建模型
    logger.info("\n初始化模型...")
    model_params = config['model_params']
    model = FNO2d(
        modes1=model_params['modes1'],
        modes2=model_params['modes2'],
        width=model_params['width']
    ).to(device)
    
    num_params = count_parameters(model)
    logger.info(f"模型参数量: {num_params:,}")
    logger.info(f"模型结构:\n{model}")
    
    # 9. 创建优化器
    optimizer = create_optimizer(model, config)
    logger.info(f"\n优化器: {optimizer.__class__.__name__}")
    logger.info(f"学习率: {config['train_params']['learning_rate']}")
    logger.info(f"权重衰减: {config['train_params']['weight_decay']}")
    
    # 10. 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"学习率调度器: {scheduler.__class__.__name__}")
    
    # 11. 创建损失函数
    loss_fn = create_loss_fn(config)
    logger.info("损失函数: 复合损失(可配置)")
    
    # 12. 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        exp_dir=exp_dir
    )
    
    # 13. 开始训练
    logger.info("\n" + "=" * 60)
    logger.info(f"训练轮数: {config['train_params']['num_epochs']}")
    logger.info(f"验证频率: 每 {config['log_params']['val_freq']} 轮")
    logger.info(f"保存频率: 每 {config['log_params']['save_freq']} 轮")
    logger.info("=" * 60 + "\n")
    
    try:
        trainer.train()
        logger.info("\n训练成功完成！")
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"\n训练过程中发生错误: {str(e)}", exc_info=True)
        raise
    
    logger.info(f"\n所有结果已保存至: {exp_dir}")
    logger.info("您可以使用以下命令查看 TensorBoard:")
    logger.info(f"  tensorboard --logdir={os.path.join(exp_dir, 'tensorboard')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO 训练主程序')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='YAML 配置文件路径'
    )
    args = parser.parse_args()
    
    main(args)

