"""
测试真实数据读取功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import read_image, read_ot_mesh_npy, RealOTDataset
from pathlib import Path
import torch
import numpy as np


def test_single_pair():
    """测试读取单个图像和网格文件对"""
    print("=" * 60)
    print("测试1: 读取单个图像和NPY网格文件对")
    print("=" * 60)
    
    # 使用实际数据路径
    image_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray"
    mesh_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray_ot_mesh_npy"
    
    # 查找第一个可用的文件对
    image_files = list(Path(image_dir).glob("*.jpg"))
    if len(image_files) == 0:
        print(f"✗ 图像目录为空: {image_dir}")
        print("请确保数据目录存在并包含图像文件")
        return
    
    # 获取第一个图像文件
    image_path = str(image_files[0])
    image_name = Path(image_path).stem  # 例如：9985_cifar_airplane
    
    # 查找对应的mesh文件
    mesh_path = os.path.join(mesh_dir, f"{image_name}_mesh.npy")
    
    if not os.path.exists(mesh_path):
        print(f"✗ 找不到对应的mesh文件: {mesh_path}")
        return
    
    print(f"\n读取图像: {image_path}")
    print(f"读取网格: {mesh_path}")
    
    try:
        # 读取图像
        image = read_image(image_path, target_size=None)
        print(f"✓ 成功读取图像")
        print(f"  - 形状: {image.shape}")
        print(f"  - 数据类型: {image.dtype}")
        print(f"  - 值范围: [{image.min():.4f}, {image.max():.4f}]")
        
        # 读取OT网格
        ot_map = read_ot_mesh_npy(mesh_path, target_size=None)
        print(f"\n✓ 成功读取OT网格")
        print(f"  - 形状: {ot_map.shape}")
        print(f"  - 数据类型: {ot_map.dtype}")
        print(f"  - X坐标范围: [{ot_map[0].min():.4f}, {ot_map[0].max():.4f}]")
        print(f"  - Y坐标范围: [{ot_map[1].min():.4f}, {ot_map[1].max():.4f}]")
        
        # 检查形状是否匹配
        if image.shape[1:] == ot_map.shape[1:]:
            print(f"\n✓ 图像和OT网格形状匹配: {image.shape[1:]}")
        else:
            print(f"\n✗ 形状不匹配！图像: {image.shape[1:]}, OT网格: {ot_map.shape[1:]}")
        
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return


def test_dataset():
    """测试数据集类"""
    print("\n" + "=" * 60)
    print("测试2: 数据集类")
    print("=" * 60)
    
    # 设置路径
    image_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray"
    mesh_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray_ot_mesh_npy"
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"✗ 图像目录不存在: {image_dir}")
        return
    
    if not os.path.exists(mesh_dir):
        print(f"✗ 网格目录不存在: {mesh_dir}")
        return
    
    try:
        # 创建数据集
        print(f"\n创建数据集...")
        dataset = RealOTDataset(
            image_dir=image_dir,
            mesh_dir=mesh_dir,
            target_size=None,  # 使用原始尺寸
            verbose=True
        )
        
        if len(dataset) == 0:
            print("✗ 没有找到任何匹配的数据对")
            print("\n提示：确保文件命名格式正确")
            print("  - 图像：9985_cifar_airplane.jpg")
            print("  - 网格：9985_cifar_airplane_mesh.npy")
            return
        
        print(f"\n✓ 数据集创建成功，共 {len(dataset)} 个样本")
        
        # 读取前几个样本测试
        num_test = min(3, len(dataset))
        print(f"\n测试读取前 {num_test} 个样本...")
        
        for i in range(num_test):
            image, ot_map = dataset[i]
            print(f"\n样本 {i+1}:")
            print(f"  - 图像形状: {image.shape}")
            print(f"  - OT映射形状: {ot_map.shape}")
            print(f"  - 图像值范围: [{image.min():.4f}, {image.max():.4f}]")
            print(f"  - OT X范围: [{ot_map[0].min():.4f}, {ot_map[0].max():.4f}]")
            print(f"  - OT Y范围: [{ot_map[1].min():.4f}, {ot_map[1].max():.4f}]")
        
        # 测试批处理
        print(f"\n测试批处理...")
        from torch.utils.data import DataLoader
        batch_size = min(4, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batch_images, batch_ot_maps = next(iter(loader))
        print(f"✓ 批处理成功")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 批次图像形状: {batch_images.shape}")
        print(f"  - 批次OT映射形状: {batch_ot_maps.shape}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_split():
    """测试数据集划分"""
    print("\n" + "=" * 60)
    print("测试3: 数据集划分")
    print("=" * 60)
    
    from src.datasets import split_dataset_indices
    
    total_size = 100
    train_indices, val_indices = split_dataset_indices(
        total_size, 
        train_ratio=0.8, 
        val_ratio=0.2,
        random_seed=42
    )
    
    print(f"总样本数: {total_size}")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"验证集样本数: {len(val_indices)}")
    print(f"训练集索引示例: {train_indices[:10]}")
    print(f"验证集索引示例: {val_indices[:10]}")
    
    # 检查是否有重叠
    overlap = set(train_indices) & set(val_indices)
    if len(overlap) == 0:
        print("✓ 训练集和验证集没有重叠")
    else:
        print(f"✗ 训练集和验证集有 {len(overlap)} 个重叠样本")
    
    # 检查是否覆盖所有索引
    all_indices = set(train_indices) | set(val_indices)
    if len(all_indices) == total_size:
        print("✓ 所有索引都被覆盖")
    else:
        print(f"✗ 缺失 {total_size - len(all_indices)} 个索引")


def test_resize():
    """测试图像和网格的尺寸调整功能"""
    print("\n" + "=" * 60)
    print("测试4: 尺寸调整功能")
    print("=" * 60)
    
    image_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray"
    mesh_dir = "/home/project_cx/G_IR_Diffusion/data/fig_gray_ot_mesh_npy"
    
    if not os.path.exists(image_dir) or not os.path.exists(mesh_dir):
        print("✗ 数据目录不存在，跳过此测试")
        return
    
    try:
        # 测试不同的目标尺寸
        target_sizes = [None, (32, 32), (64, 64)]
        
        for target_size in target_sizes:
            print(f"\n目标尺寸: {target_size}")
            dataset = RealOTDataset(
                image_dir=image_dir,
                mesh_dir=mesh_dir,
                target_size=target_size,
                verbose=False
            )
            
            if len(dataset) > 0:
                image, ot_map = dataset[0]
                print(f"  - 图像形状: {image.shape}")
                print(f"  - OT映射形状: {ot_map.shape}")
                
                if target_size is not None:
                    expected_shape = (target_size[0], target_size[1])
                    if image.shape[1:] == expected_shape and ot_map.shape[1:] == expected_shape:
                        print(f"  ✓ 尺寸调整正确")
                    else:
                        print(f"  ✗ 尺寸调整失败")
            else:
                print(f"  ✗ 数据集为空")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("=" * 60)
    print("FNO 真实数据读取测试（简化版 - JPG + NPY）")
    print("=" * 60)
    print("\n数据格式要求：")
    print("  - 图像：JPG 灰度图")
    print("  - 网格：NPY 文件，形状 (2, H, W)")
    print("  - 命名：{前缀}_{名称}.jpg 对应 {前缀}_{名称}_mesh.npy")
    print("  - 例如：9985_cifar_airplane.jpg ↔ 9985_cifar_airplane_mesh.npy\n")
    
    # 运行测试
    test_single_pair()
    test_dataset()
    test_split()
    test_resize()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n如果所有测试通过，您可以：")
    print("1. 运行训练: python main.py --config configs/real_data_config.yaml")
    print("2. 或使用默认配置（伪数据）: python main.py")
    print("=" * 60)