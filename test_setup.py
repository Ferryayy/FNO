"""
快速测试脚本 - 验证框架是否正确安装和配置
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FNO 训练框架 - 环境测试")
print("=" * 60)

# 1. 测试依赖导入
print("\n1. 测试依赖包...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch 导入失败: {e}")

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy 导入失败: {e}")

try:
    import yaml
    print(f"   ✓ PyYAML")
except ImportError as e:
    print(f"   ✗ PyYAML 导入失败: {e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    print(f"   ✓ TensorBoard")
except ImportError as e:
    print(f"   ✗ TensorBoard 导入失败: {e}")

try:
    from tqdm import tqdm
    print(f"   ✓ tqdm")
except ImportError as e:
    print(f"   ✗ tqdm 导入失败: {e}")

# 2. 测试模块导入
print("\n2. 测试项目模块...")
try:
    from src.model import FNO2d, SpectralConv2d
    print("   ✓ 模型模块导入成功")
except ImportError as e:
    print(f"   ✗ 模型模块导入失败: {e}")

try:
    from src.datasets import OTDataset, get_data_loaders
    print("   ✓ 数据集模块导入成功")
except ImportError as e:
    print(f"   ✗ 数据集模块导入失败: {e}")

try:
    from src.trainer import Trainer
    print("   ✓ 训练器模块导入成功")
except ImportError as e:
    print(f"   ✗ 训练器模块导入失败: {e}")

try:
    from src.utils import setup_seed, load_config
    print("   ✓ 工具模块导入成功")
except ImportError as e:
    print(f"   ✗ 工具模块导入失败: {e}")

# 3. 测试配置文件
print("\n3. 测试配置文件...")
try:
    from src.utils import load_config
    config = load_config('configs/default_config.yaml')
    print("   ✓ 配置文件加载成功")
    print(f"   - 训练轮数: {config['train_params']['num_epochs']}")
    print(f"   - 批次大小: {config['train_params']['batch_size']}")
    print(f"   - 学习率: {config['train_params']['learning_rate']}")
except Exception as e:
    print(f"   ✗ 配置文件加载失败: {e}")

# 4. 测试模型实例化
print("\n4. 测试模型实例化...")
try:
    from src.model import FNO2d
    import torch
    model = FNO2d(modes1=12, modes2=12, width=20)
    print("   ✓ 模型实例化成功")
    
    # 测试前向传播
    x = torch.randn(2, 64, 64, 1)
    y = model(x)
    print(f"   ✓ 前向传播成功: 输入 {x.shape} -> 输出 {y.shape}")
except Exception as e:
    print(f"   ✗ 模型测试失败: {e}")

# 5. 测试数据集
print("\n5. 测试数据集...")
try:
    from src.datasets import OTDataset
    dataset = OTDataset(num_samples=5, size=64, verbose=False)
    x, y = dataset[0]
    print(f"   ✓ 数据集创建成功")
    print(f"   - 输入形状: {x.shape}")
    print(f"   - 输出形状: {y.shape}")
except Exception as e:
    print(f"   ✗ 数据集测试失败: {e}")

# 6. 检查 CUDA
print("\n6. 检查 GPU 可用性...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 可用")
        print(f"   - GPU 数量: {torch.cuda.device_count()}")
        print(f"   - 当前 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ! CUDA 不可用，将使用 CPU 训练")
except Exception as e:
    print(f"   ✗ GPU 检查失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n如果所有测试通过，您可以运行:")
print("  python main.py")
print("\n查看更多选项:")
print("  python main.py --help")
print("=" * 60)

