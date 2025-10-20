#!/usr/bin/env python3
"""
显示 npy 文件的坐标信息
用法: python show_data.py <path_to_npy_file>
"""
import sys
import numpy as np

def show_npy_info(npy_path):
    """显示 (2, H, W) 格式 npy 文件的坐标信息"""
    data = np.load(npy_path)
    print(f"文件: {npy_path}")
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"X 坐标范围: [{data[0].min():.6f}, {data[0].max():.6f}]")
    print(f"Y 坐标范围: [{data[1].min():.6f}, {data[1].max():.6f}]")
    print()
    
    if data.ndim != 3 or data.shape[0] != 2:
        print("警告: 数据不是 (2, H, W) 格式")
        return
    
    _, H, W = data.shape
    
    # 打印函数：显示一行的前3个和后3个
    def print_row(h_idx, name):
        print(f"{name} (H={h_idx}):")
        print(f"  前3个: ", end="")
        for w in range(min(3, W)):
            print(f"({data[0,h_idx,w]:.4f}, {data[1,h_idx,w]:.4f}) ", end="")
        print()
        print(f"  后3个: ", end="")
        for w in range(max(0, W-3), W):
            print(f"({data[0,h_idx,w]:.4f}, {data[1,h_idx,w]:.4f}) ", end="")
        print()
    
    # 第一行 (H=0)
    print_row(0, "第一行")
    print()
    
    # 第二行 (H=1)
    if H > 1:
        print_row(1, "第二行")
        print()
    
    # 倒数第二行
    if H > 1:
        print_row(H-2, "倒数第二行")
        print()
    
    # 倒数第一行
    print_row(H-1, "倒数第一行")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python show_data.py <path_to_npy_file>")
        print("示例: python show_data.py ../data/gauss/test/mesh_npy/2_mesh.npy")
        sys.exit(1)
    
    show_npy_info(sys.argv[1])

