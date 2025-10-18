# 网格可视化功能指南

## 功能概述

在训练FNO模型学习最优传输（OT）映射时，我们添加了自动的网格可视化功能，可以直观地对比Ground Truth网格和模型预测网格。

## 功能特性

### 1. 自动网格绘制
- **横向连接**：每一行的相邻顶点连接
- **纵向连接**：每一列的相邻顶点连接
- **对角连接**：每个方格的右上角与左下角连接（三角化）

### 2. 对比可视化
- **Ground Truth网格**：蓝色线条
- **预测网格**：红色线条
- 两者叠加显示，便于直观对比差异

### 3. 多通道支持
- **单通道数据**（灰度图）：生成一张对比图
- **多通道数据**（如RGB）：为每个通道独立生成对比图
  - Channel 0 (R)：第一张可视化图
  - Channel 1 (G)：第二张可视化图
  - Channel 2 (B)：第三张可视化图

## 配置说明

在配置文件中（`configs/default_config.yaml` 或 `configs/real_data_config.yaml`），添加以下参数：

```yaml
# 可视化参数
visualization_params:
  enable_mesh_vis: true  # 是否启用网格可视化
  vis_num_samples: 1     # 可视化的样本数量（单通道时有效）
```

### 配置参数说明
- `enable_mesh_vis`: 
  - `true`: 启用网格可视化（默认）
  - `false`: 禁用网格可视化（可提升验证速度）
  
- `vis_num_samples`: 
  - 指定要可视化的样本数量（仅对单通道数据有效）
  - 对于多通道数据，会自动为每个通道生成可视化

## 使用方法

### 1. 训练时自动生成

只需正常运行训练：

```bash
python main.py --config configs/real_data_config.yaml
```

在每次验证阶段（根据`val_freq`配置），会自动生成网格可视化并保存到TensorBoard。

### 2. 在TensorBoard中查看

启动TensorBoard：

```bash
tensorboard --logdir=outputs/YOUR_EXPERIMENT_DIR/tensorboard
```

然后在浏览器中打开TensorBoard（通常是 http://localhost:6006），导航到 **IMAGES** 标签页。

你会看到以下可视化：

**单通道数据：**
- `Val/Mesh_Comparison_Sample_0`
- `Val/Mesh_Comparison_Sample_1`
- ...（根据`vis_num_samples`配置）

**多通道数据（如RGB）：**
- `Val/Mesh_Comparison_Channel_0` (R通道)
- `Val/Mesh_Comparison_Channel_1` (G通道)
- `Val/Mesh_Comparison_Channel_2` (B通道)

## 可视化示例解读

### 理想情况
如果模型训练良好，你应该看到：
- 红色线条（预测）与蓝色线条（GT）高度重合
- 网格结构保持一致
- 形变模式相似

### 需要改进的情况
如果看到以下现象，可能需要调整训练：
- 红色和蓝色线条偏差较大
- 预测网格出现明显扭曲或不合理的形变
- 网格拓扑结构错乱

## 技术细节

### 网格连接算法

对于一个 `(2, H, W)` 形状的网格坐标张量：

```python
# 横向连接
for i in range(H):
    for j in range(W-1):
        连接 [i,j] 到 [i,j+1]

# 纵向连接
for j in range(W):
    for i in range(H-1):
        连接 [i,j] 到 [i+1,j]

# 对角连接（三角化）
for i in range(H-1):
    for j in range(W-1):
        连接 [i,j+1] 到 [i+1,j]
```

### 坐标转换

- 如果使用 `predict_mode: offset`，预测的是相对于标准网格的偏移量
- 可视化时会自动转换为绝对坐标：`absolute = offset + base_grid`
- Ground Truth始终以绝对坐标形式存储和显示

## 性能考虑

### 可视化开销
- 网格可视化会增加少量验证时间（主要是matplotlib绘图）
- 仅在验证阶段执行，不影响训练速度
- 建议在最终评估时启用，日常快速迭代时可关闭

### 禁用可视化

如果需要加快验证速度，可以临时禁用：

```yaml
visualization_params:
  enable_mesh_vis: false
```

## 故障排查

### Q: TensorBoard中看不到网格可视化？
A: 检查以下几点：
1. 确认 `enable_mesh_vis: true`
2. 确认已运行至少一次验证（根据`val_freq`）
3. 刷新TensorBoard页面
4. 检查日志中是否有"已保存网格可视化"的消息

### Q: 可视化图像模糊或不清晰？
A: 可以在 `src/utils.py` 的 `visualize_mesh_comparison()` 函数中调整参数：
```python
fig_size=(10, 10)  # 增大图像尺寸
dpi=150            # 提高分辨率
```

### Q: 网格线条太细看不清？
A: 在 `src/utils.py` 的 `draw_mesh()` 函数中调整：
```python
linewidth=1.0  # 增大线宽（默认0.5）
alpha=0.8      # 增大透明度（默认0.6）
```

## 扩展功能

如果需要更多定制化的可视化，可以参考：
- `src/utils.py`: 包含 `visualize_mesh_comparison()` 和 `visualize_mesh_multichannel()` 函数
- `src/trainer.py`: 包含 `_visualize_meshes()` 方法

你可以根据需要修改这些函数来：
- 改变颜色方案
- 添加更多统计信息
- 生成不同格式的输出
- 添加误差热图等高级可视化

