# 使用指南

## 快速开始

### 1. 使用伪数据测试框架

如果您只是想测试框架是否正常工作：

```bash
# 使用默认配置运行（伪数据）
python main.py

# 或者明确指定配置文件
python main.py --config configs/default_config.yaml
```

### 2. 使用真实数据

#### 步骤1：准备数据

确保您的数据按以下格式组织：

```
项目目录/
├── images/
│   ├── 1_xxx.jpg
│   ├── 2_xxx.png
│   └── ...
└── ot_meshes/
    ├── 1_xxx.m
    ├── 2_xxx.m
    └── ...
```

**重要：** 文件名的数字前缀必须匹配（如 `1_xxx.jpg` 对应 `1_xxx.m`）

#### 步骤2：修改配置文件

编辑 `configs/real_data_config.yaml`：

```yaml
data_params:
  use_real_data: true
  image_dir: '/home/project_cx/datasets/cifar10/images'  # 修改为您的图像目录
  mesh_dir: '/home/project_cx/datasets/cifar10/ot_mesh'  # 修改为您的网格目录
  target_size: [32, 32]  # 设置目标尺寸，或 null 保持原尺寸
  use_uv: true  # true: 使用uv坐标, false: 使用xyz坐标
```

#### 步骤3：测试数据读取

在运行训练之前，先测试数据是否能正确读取：

```bash
# 修改 test_real_data.py 中的路径为您的实际路径
python test_real_data.py
```

如果测试通过，您应该看到：
- 成功找到匹配的图像和网格文件对
- 正确读取数据形状和范围
- 数据集划分成功

#### 步骤4：运行训练

```bash
python main.py --config configs/real_data_config.yaml
```

### 3. 查看训练结果

训练过程中会自动创建一个带时间戳的输出目录：

```
outputs/
└── 2025-10-04_15-30-00_FNO_OT_RealData/
    ├── train.log              # 训练日志
    ├── config_snapshot.yaml   # 配置快照
    ├── checkpoints/
    │   ├── last.pth          # 最新checkpoint
    │   └── best_model.pth    # 最优模型
    └── tensorboard/          # TensorBoard日志
```

使用 TensorBoard 查看训练曲线和网格可视化：

```bash
tensorboard --logdir=outputs/2025-10-04_15-30-00_FNO_OT_RealData/tensorboard
```

在 TensorBoard 中，您可以查看：
- **训练/验证损失曲线**：Train/Loss 和 Val/Loss
- **学习率变化**：Learning_Rate
- **网格可视化对比**：Val/Mesh_Comparison_* 
  - 蓝色线条：Ground Truth 网格
  - 红色线条：模型预测网格
  - 对于多通道数据（如RGB），会为每个通道生成独立的可视化图

## 数据格式说明

### 图像格式
- 支持格式：JPG, PNG, BMP
- 类型：灰度图（彩色图会自动转换）
- 命名：`{数字前缀}_{任意名称}.{扩展名}`，如 `1_image.jpg`

### OT网格格式（.m文件）
每行格式：
```
Vertex N x y z {rgb=(r g b) uv=(u v)}
```

例如：
```
Vertex 1 -4.83312 4.82049 0 {rgb=(0.67 0.67 0.67) uv=(-4.83312 4.82049)}
Vertex 2 -4.5299 4.83409 0 {rgb=(0.52 0.52 0.52) uv=(-4.5299 4.83409)}
...
```

**重要信息：**
- 顶点按行展开顺序排列（左上角开始，按行扫描）
- 总顶点数应等于图像高度 × 宽度
- 可以使用 `uv` 坐标或 `xyz` 的前两个坐标
- 命名：`{数字前缀}_{任意名称}.m`，如 `1_mesh.m`

## 常见问题

### Q1: 如何确定图像尺寸？
- 如果设置 `target_size: [32, 32]`，所有图像会被调整到 32×32
- 如果设置 `target_size: null`，保持原始图像尺寸

### Q2: 网格文件的顶点数不匹配怎么办？
代码会自动处理：
- 顶点数少于图像像素数：用零填充
- 顶点数多于图像像素数：截断多余部分
- 会打印警告信息

### Q3: 使用uv坐标还是xyz坐标？
- `use_uv: true` - 从 `uv=(x y)` 读取坐标
- `use_uv: false` - 从 `Vertex N x y z` 的 x, y 读取坐标
- 两者数值应该相同，选择其一即可

### Q4: 如何调整训练集/验证集比例？
修改配置文件中的：
```yaml
data_params:
  train_ratio: 0.8  # 80% 训练
  val_ratio: 0.2    # 20% 验证
```

### Q5: 如何断点续训？
```yaml
checkpoint_params:
  resume_from_checkpoint: './outputs/xxx/checkpoints/last.pth'
```

## 性能调优

### 批次大小
```yaml
train_params:
  batch_size: 16  # 根据GPU内存调整
```

### 数据加载
```yaml
data_params:
  num_workers: 4    # 数据加载线程数
  pin_memory: true  # GPU训练时建议开启
```

### 模型大小
```yaml
model_params:
  modes1: 12  # 傅里叶模式数（越大越精细，但计算量大）
  modes2: 12
  width: 32   # 隐层宽度（越大容量越大，但计算量大）
```

## 高级用法

### 自定义损失函数
修改 `main.py` 中的 `create_loss_fn()` 函数。

### 自定义学习率调度器
修改配置文件中的 `scheduler_params`：
```yaml
scheduler_params:
  type: 'CosineAnnealingLR'  # 或 'StepLR', 'ReduceLROnPlateau'
```

### 添加数据增强
在 `src/datasets.py` 的 `read_image()` 函数中添加数据增强逻辑。

### 可视化预测结果
参考 `fno_demo.py` 中的可视化代码，在训练完成后加载模型并可视化预测结果。

