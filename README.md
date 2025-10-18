# FNO 训练框架

一个完整的、科学的、工程化的 Fourier Neural Operator (FNO) 深度学习训练框架。

## 📁 项目结构

```
FNO/
├── configs/
│   └── default_config.yaml      # 默认配置文件
├── src/
│   ├── datasets.py              # 数据集定义
│   ├── model.py                 # FNO 模型定义
│   ├── trainer.py               # 训练器核心逻辑
│   └── utils.py                 # 工具函数
├── main.py                      # 主入口程序
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
└── fno_demo.py                  # 原始演示代码
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 使用伪数据（演示）

默认配置使用伪数据生成器，无需准备数据即可运行：
```bash
python main.py
```

#### 使用真实数据

如果您有真实的OT数据，请按以下格式组织：

**图像目录结构：**
```
/path/to/images/
├── 1_image_name.jpg
├── 2_image_name.png
├── 3_another_image.jpg
└── ...
```

**OT网格目录结构：**
```
/path/to/meshes/
├── 1_mesh_name.m
├── 2_mesh_name.m
├── 3_another_mesh.m
└── ...
```

**要求：**
- 图像必须是灰度图（或可转换为灰度）
- 文件名前缀数字用于匹配图像和网格文件（如 `1_xxx.jpg` 对应 `1_xxx.m`）
- 网格文件格式：每行为 `Vertex N x y z {rgb=(...) uv=(x y)}`
- 网格文件按行展开，对应图像从左上角按行扫描的顺序

**测试数据读取：**
```bash
# 修改 test_real_data.py 中的路径，然后运行
python test_real_data.py
```

### 3. 运行训练

使用默认配置（伪数据）：
```bash
python main.py
```

使用真实数据：
```bash
# 1. 修改 configs/real_data_config.yaml 中的路径
# 2. 运行训练
python main.py --config configs/real_data_config.yaml
```

使用自定义配置：
```bash
python main.py --config configs/my_config.yaml
```

### 4. 查看训练日志

训练过程中，所有输出会自动保存到 `outputs/` 目录下的时间戳文件夹中。

使用 TensorBoard 查看训练曲线：
```bash
tensorboard --logdir=outputs/2025-10-04_15-30-00_FNO_OT/tensorboard
```

## 📊 输出目录结构

每次训练会创建一个带时间戳的实验目录：

```
outputs/
└── 2025-10-04_15-30-00_FNO_OT/
    ├── train.log                # 训练日志
    ├── config_snapshot.yaml     # 配置快照（用于复现）
    ├── checkpoints/
    │   ├── last.pth            # 最新的 checkpoint
    │   └── best_model.pth      # 最优模型
    └── tensorboard/            # TensorBoard 日志
```

## ⚙️ 配置说明

配置文件 `configs/default_config.yaml` 包含以下部分：

### 训练参数 (train_params)
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `device`: 设备 ('cuda' 或 'cpu')
- `weight_decay`: 权重衰减
- `gradient_clip`: 梯度裁剪阈值

### 数据参数 (data_params)
- `use_real_data`: 是否使用真实数据（true/false）
- **真实数据参数（use_real_data=true时）：**
  - `image_dir`: 图像目录路径
  - `mesh_dir`: OT网格目录路径
  - `target_size`: 目标图像尺寸 [H, W]（null则保持原尺寸）
  - `use_uv`: 使用uv坐标还是xyz坐标
  - `train_ratio`: 训练集比例
  - `val_ratio`: 验证集比例
- **伪数据参数（use_real_data=false时）：**
  - `num_train_samples`: 训练样本数
  - `num_val_samples`: 验证样本数
  - `image_size`: 图像尺寸
- **DataLoader参数：**
  - `num_workers`: 数据加载线程数
  - `pin_memory`: 是否固定内存

### 模型参数 (model_params)
- `modes1`: 傅里叶模式数 (x方向)
- `modes2`: 傅里叶模式数 (y方向)
- `width`: 隐层宽度

### 日志参数 (log_params)
- `log_dir_base`: 日志基础目录
- `experiment_name`: 实验名称
- `save_freq`: checkpoint 保存频率
- `val_freq`: 验证频率

### Checkpoint 参数 (checkpoint_params)
- `resume_from_checkpoint`: 断点续训路径
- `save_best`: 是否保存最优模型

### 学习率调度器参数 (scheduler_params)
- `type`: 调度器类型 ('StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau')
- 其他参数根据调度器类型而定

## 🔄 断点续训

要从之前的 checkpoint 继续训练，修改配置文件中的 `resume_from_checkpoint`：

```yaml
checkpoint_params:
  resume_from_checkpoint: './outputs/2025-10-04_15-30-00_FNO_OT/checkpoints/last.pth'
```

## 📝 核心功能

### 1. 配置驱动
所有实验参数通过 YAML 配置文件管理，无需修改代码。

### 2. 自动化日志
- 训练日志同时输出到控制台和文件
- TensorBoard 实时记录训练/验证损失和学习率
- 每次运行自动创建唯一的实验目录

### 3. Checkpoint 管理
- 自动保存最新模型 (`last.pth`)
- 自动保存最优模型 (`best_model.pth`)
- 支持断点续训

### 4. 模块化设计
- 数据、模型、训练器完全解耦
- 易于扩展和修改
- 代码清晰，注释完整

## 🎯 模型说明

本框架实现了 2D Fourier Neural Operator (FNO)，用于学习最优传输问题：

- **输入**: 密度函数 ρ_T，shape (1, H, W)
- **输出**: 最优传输映射 T，shape (2, H, W)
- **损失函数**: 相对 L2 损失

核心组件：
- `SpectralConv2d`: 傅里叶谱卷积层
- `FNO2d`: 完整的 FNO 模型

## 🔧 扩展建议

### 使用真实数据
框架已内置对真实OT数据的支持：

1. **从图像和网格文件读取**（推荐）
   - 使用 `RealOTDataset` 类
   - 修改 `configs/real_data_config.yaml` 中的路径
   - 设置 `use_real_data: true`

2. **自定义数据加载器**
   - 继承 `torch.utils.data.Dataset`
   - 在 `src/datasets.py` 中添加新的数据集类
   - 在 `get_data_loaders` 函数中添加相应逻辑

### 添加新的模型
1. 在 `src/model.py` 中定义新模型
2. 在 `main.py` 中切换模型初始化
3. 在配置文件中添加对应的参数

### 自定义损失函数
在 `main.py` 的 `create_loss_fn()` 函数中定义新的损失函数。

## 📚 参考文献

Fourier Neural Operator 相关论文：
- Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021.

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提出问题和改进建议！

