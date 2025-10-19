import torch
from functools import lru_cache
import torch.nn.functional as F

def compute_beltrami_coefficients_pytorch(
    vertices_src: torch.Tensor, # (N, 2), float
    faces: torch.Tensor,        # (M, 3), long
    vertices_tgt: torch.Tensor, # (N, 2), float
    epsilon: float = 1e-12
) -> torch.Tensor:
    if vertices_src.shape[1] != 2 or vertices_tgt.shape[1] != 2:
        raise ValueError("Vertex tensors must have 2 columns (x, y).")
    if vertices_src.shape[0] != vertices_tgt.shape[0]:
        raise ValueError("Source and target vertex tensors must have the same number of vertices N.")

    z1 = vertices_src[faces[:, 0]]
    z2 = vertices_src[faces[:, 1]]
    z3 = vertices_src[faces[:, 2]]

    w1 = vertices_tgt[faces[:, 0]]
    w2 = vertices_tgt[faces[:, 1]]
    w3 = vertices_tgt[faces[:, 2]]

    z21_vec = z2 - z1
    z32_vec = z3 - z2
    z13_vec = z1 - z3

    z1x, z1y = z1[:, 0], z1[:, 1]
    z2x, z2y = z2[:, 0], z2[:, 1]
    z3x, z3y = z3[:, 0], z3[:, 1]

    # 两倍有向面积
    dT = z1x * z2y - z1y * z2x + \
         z2x * z3y - z2y * z3x + \
         z3x * z1y - z3y * z1x

    dT_safe = dT + torch.copysign(torch.full_like(dT, epsilon), dT)
    dT_safe[dT_safe == 0] = epsilon

    w1x, w1y = w1[:, 0], w1[:, 1]
    w2x, w2y = w2[:, 0], w2[:, 1]
    w3x, w3y = w3[:, 0], w3[:, 1]

    z21x, z21y = z21_vec[:, 0], z21_vec[:, 1]
    z32x, z32y = z32_vec[:, 0], z32_vec[:, 1]
    z13x, z13y = z13_vec[:, 0], z13_vec[:, 1]

    u_x = (-z32y * w1x - z13y * w2x - z21y * w3x) / dT_safe
    u_y = ( z32x * w1x + z13x * w2x + z21x * w3x) / dT_safe
    v_x = (-z32y * w1y - z13y * w2y - z21y * w3y) / dT_safe
    v_y = ( z32x * w1y + z13x * w2y + z21x * w3y) / dT_safe

    fz_bar = torch.complex((u_x - v_y) / 2.0, (v_x + u_y) / 2.0)
    fz     = torch.complex((u_x + v_y) / 2.0, (v_x - u_y) / 2.0)

    fz_safe_denom = fz + torch.complex(torch.full_like(fz.real, epsilon), torch.zeros_like(fz.imag))
    face_mu = fz_bar / fz_safe_denom
    return face_mu


# 把 (2,H,W) 展成 (N,2) 
def grid2_2hw_to_vertices_n2(grid_2hw: torch.Tensor) -> torch.Tensor:
    """
    grid_2hw: (2, H, W)，grid_2hw[0] = x, grid_2hw[1] = y
    返回: (N,2) = (H*W, 2)，按行主序展平（y从上到下，x从左到右）
    """
    if grid_2hw.dim() != 3 or grid_2hw.shape[0] != 2:
        raise ValueError("Expected grid of shape (2, H, W).")
    _, H, W = grid_2hw.shape
    # (2,H,W) -> (H,W,2) -> (H*W,2)
    return grid_2hw.permute(1, 2, 0).reshape(H*W, 2)


# 生成标准网格 faces（右上到左下对角） 
@lru_cache(maxsize=128)
def build_standard_faces(H: int, W: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    标准网格拓扑（右上->左下对角线）。每个小方格两个三角：
      设小方格左上角为 (y,x)，则四个顶点的线性索引：
        v00 = y*W + x
        v10 = y*W + (x+1)
        v01 = (y+1)*W + x
        v11 = (y+1)*W + (x+1)
      采用对角线 v10 -> v01（即右上到左下），三角面为：
        T1 = (v00, v10, v01)
        T2 = (v11, v01, v10)
    说明：
      - 这是行主序展开下的统一拓扑；只要 src/tgt 的顶点顺序一致，就可直接套用。
      - 图像坐标 y 向下增大时，上述顶点顺序在几何上是顺时针，但对本实现不影响，
        关键是“源/目标”用同一拓扑与同一顺序（保持一致性）。
    """
    if H < 2 or W < 2:
        raise ValueError("H and W must be >= 2 to form triangles.")

    rows = H - 1
    cols = W - 1
    # 每个小方格 2 个三角
    M = rows * cols * 2

    faces = torch.empty((M, 3), dtype=torch.long, device=device)
    k = 0
    for y in range(rows):
        base0 = y * W
        base1 = (y + 1) * W
        for x in range(cols):
            v00 = base0 + x
            v10 = base0 + x + 1
            v01 = base1 + x
            v11 = base1 + x + 1

            # 三角 1： (v00, v10, v01)
            faces[k, 0] = v00
            faces[k, 1] = v10
            faces[k, 2] = v01
            k += 1
            # 三角 2： (v11, v01, v10)
            faces[k, 0] = v11
            faces[k, 1] = v01
            faces[k, 2] = v10
            k += 1

    return faces


# 对外的便捷入口（支持传 (2,H,W)，可不传 faces)
def compute_beltrami_from_grids(
    grid_src_2hw: torch.Tensor,   # (2,H,W) float
    grid_tgt_2hw: torch.Tensor,   # (2,H,W) float（与上面一一对应）
    faces: torch.Tensor = None,   # 可选；不传则按标准网格自动生成
    epsilon: float = 1e-12
) -> torch.Tensor:
    """
    计算标准网格上每个三角形的 Beltrami 系数（复数），返回形状 (M,)
    """
    if grid_src_2hw.shape != grid_tgt_2hw.shape:
        raise ValueError("grid_src_2hw and grid_tgt_2hw must have the same shape (2,H,W).")
    if grid_src_2hw.dim() != 3 or grid_src_2hw.size(0) != 2:
        raise ValueError("Expected inputs of shape (2,H,W).")

    _, H, W = grid_src_2hw.shape
    device = grid_src_2hw.device

    # (2,H,W) -> (N,2)
    vertices_src = grid2_2hw_to_vertices_n2(grid_src_2hw)
    vertices_tgt = grid2_2hw_to_vertices_n2(grid_tgt_2hw)

    # 自动生成拓扑
    if faces is None:
        faces = build_standard_faces(H, W, device=device)

    # 调用你已有的实现
    return compute_beltrami_coefficients_pytorch(vertices_src, faces, vertices_tgt, epsilon)

def compute_constraint_penalty(mu, penalty_type='relu', wc=100.0, wb=1.0, delta=1e-6, epsilon=1e-9):
    mu_norm = torch.norm(mu, p=2, dim=-1)

    if penalty_type == 'relu':
        violation = mu_norm - (1.0 - delta)
        activated_violation = F.relu(violation) ** 2
        penalty = wc * torch.sum(activated_violation)
        return penalty
        
    elif penalty_type == 'log':
        mu_norm_sq = mu_norm.pow(2)
        inside_log = 1.0 - mu_norm_sq
        if torch.any(inside_log <= 0):
            return torch.tensor(float('inf'), device=mu.device)
            
        log_barrier = torch.log(inside_log + epsilon)
        penalty = -wb * torch.sum(log_barrier)
        
        return penalty
        
    else:
        raise ValueError("penalty_type must be 'relu' or 'log'")