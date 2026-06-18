"""GracePINN utilities for graph-based curriculum weighting."""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

import deepxde as dde
from deepxde import backend as bkd
from deepxde.backend import backend_name
from deepxde.data.pde import PDE, TimePDE
from deepxde.utils import get_num_args

from src.pde.baseclass import BasePDE, BaseTimePDE  # for helper factory

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Config & weighting
# -----------------------------------------------------------------------------

@dataclass
class GracePINNConfig:
    """Configuration bundle for GracePINN weighting.

    total_iterations : int
        Tổng số epoch/iteration dùng để chuẩn hóa progress p(k) ∈ [0,1].
    alpha : float
        Hệ số cân bằng không gian–thời gian, implement bằng cách
        scale các chiều thời gian bởi sqrt(alpha).
    time_dims : Optional[Sequence[int]]
        Chỉ số các chiều được coi là thời gian.
    radius : float
        Bán kính δ để dựng graph. Bắt buộc phải có để dựng đồ thị.
    schedule : str
        Chiến lược chuyển đổi từ global (residual) sang local (roughness).
    """

    total_iterations: int
    alpha: float = 0.5
    radius: float = 0.15
    time_dims: Optional[Sequence[int]] = None
    schedule: str = "linear"

    def __post_init__(self) -> None:
        self.total_iterations = max(int(self.total_iterations), 1)
        self.alpha = max(float(self.alpha), 0.0)
        
        if self.radius <= 0:
            raise ValueError("radius must be > 0 để xây dựng đồ thị cục bộ.")
        self.radius = float(self.radius)

        if self.time_dims is not None:
            self.time_dims = tuple(int(dim) for dim in self.time_dims)
            
        if self.schedule not in ["linear", "cosine", "step", "exponential"]:
            raise ValueError("schedule must be either 'linear', 'cosine', 'step' or 'exponential'")


class GracePINNWeighting:
    """Graph-based difficulty estimator and curriculum mapper."""

    def __init__(self, config: GracePINNConfig) -> None:
        self.config = config
        self.last_progress: float = 0.0
        self.last_difficulty: Optional[torch.Tensor] = None
        self.last_weights: Optional[torch.Tensor] = None
        
        # --- CACHING VARIABLES ---
        self.cached_coords: Optional[torch.Tensor] = None
        self.cached_edge_index: Optional[torch.Tensor] = None
        self.cached_edge_dists: Optional[torch.Tensor] = None

    def __call__(
        self,
        inputs: torch.Tensor,
        residuals: Sequence[torch.Tensor],
        model: dde.Model,
    ) -> Optional[torch.Tensor]:
        """Compute curriculum weights for the PDE residuals in a batch."""
        if inputs.numel() == 0:
            return None

        device = inputs.device
        coords = inputs.detach()
        num_points = coords.shape[0]
        if num_points <= 1:
            return torch.ones(num_points, device=device)

        # 1) Aggregate residual components -> R(i)
        res_components: List[torch.Tensor] = []
        for res in residuals:
            if res is None or res.numel() == 0:
                continue
            res_components.append(res.detach().reshape(res.shape[0], -1))

        if not res_components:
            return None

        residual_matrix = torch.cat(res_components, dim=1)   # (N, D_total)
        residual_norm = torch.linalg.norm(residual_matrix, dim=1)  # (N,) L2 Norm gộp mọi chiều PDE

        # 2) Normalize coords + space–time scaling via alpha
        coords_norm = self._normalize(coords)  # scale từng chiều về ~[0,1]
        coords_scaled = coords_norm.clone()

        if self.config.time_dims is not None and len(self.config.time_dims) > 0:
            time_dims = torch.tensor(
                self.config.time_dims, device=device, dtype=torch.long
            )
            scale = torch.sqrt(torch.tensor(self.config.alpha, device=device))
            if torch.isfinite(scale) and scale > 0:
                coords_scaled[:, time_dims] = coords_scaled[:, time_dims] * scale
            else:
                coords_scaled[:, time_dims] = 0.0

        # =====================================================================
        # KIỂM TRA ĐIỀU KIỆN ĐỂ ĐO MEMORY VÀ DUMP DỮ LIỆU
        # =====================================================================
        current_epoch = model.train_state.epoch
        
        # Bắt dính các epoch 0 đến 10, epoch 50, epoch 100, epoch 500, và các epoch chia hết cho 1000
        is_early_epoch = (current_epoch <= 10) or (current_epoch == 50) or (current_epoch == 100) or (current_epoch == 500)
        is_thousand_epoch = (current_epoch > 0) and (current_epoch % 1000 == 0)
        is_target_epoch = is_early_epoch or is_thousand_epoch
        
        track_mem = is_target_epoch and (device.type == "cuda")

        # 3) Build spatial-temporal graph (Với cơ chế Caching và Profiling)
        # Ép xây lại đồ thị nếu đang ở mốc cần đo Memory để có Peak Memory chuẩn xác
        force_rebuild = track_mem 
        
        # Kiểm tra Cache
        use_cache = (not force_rebuild) and (self.cached_coords is not None) and \
                    (coords_scaled.shape == self.cached_coords.shape) and \
                    torch.allclose(coords_scaled, self.cached_coords)

        if use_cache:
            edge_index = self.cached_edge_index
            edge_dists = self.cached_edge_dists
        else:
            if track_mem:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_start_graph = torch.cuda.memory_allocated()
                
            edge_index, edge_dists = self._build_edge_index(coords_scaled)
            
            if track_mem:
                torch.cuda.synchronize()
                # Tính bằng số Byte (B) chính xác thay vì MB
                graph_alloc_b = torch.cuda.memory_allocated() - mem_start_graph
                graph_peak_b = torch.cuda.max_memory_allocated() - mem_start_graph
            
            # Cập nhật Cache
            self.cached_coords = coords_scaled.clone()
            self.cached_edge_index = edge_index
            self.cached_edge_dists = edge_dists

        if edge_index.shape[1] == 0:
            return torch.ones(num_points, device=device)

        # 4) Kernel weights + roughness L(i)
        if track_mem:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_start_agg = torch.cuda.memory_allocated()

        sigma = self._compute_sigma(edge_dists)
        edge_kernel = torch.exp(-(edge_dists**2) / (2 * sigma**2 + _EPS))

        di = torch.zeros(num_points, device=device, dtype=edge_kernel.dtype)
        di.scatter_add_(0, edge_index[0], edge_kernel)

        sum_wr = torch.zeros(num_points, device=device, dtype=residual_norm.dtype)
        sum_wr.scatter_add_(0, edge_index[0], edge_kernel * residual_norm[edge_index[1]])

        sum_diff = residual_norm * di - sum_wr
        roughness = sum_diff / (di + _EPS)

        if track_mem:
            torch.cuda.synchronize()
            # Tính bằng số Byte (B)
            agg_alloc_b = torch.cuda.memory_allocated() - mem_start_agg
            agg_peak_b = torch.cuda.max_memory_allocated() - mem_start_agg
            total_current_b = torch.cuda.memory_allocated()

        # 5) Normalize R and L → [0,1] bằng robust sigmoid
        level_score = self._robust_normalize(residual_norm)
        roughness_score = self._robust_normalize(roughness)

        # 6) Difficulty fusion D_i(k) với eta(k) theo schedule
        progress = min(
            float(model.train_state.epoch) / max(self.config.total_iterations - 1, 1),
            1.0,
        )
        self.last_progress = progress

        # --- CÁC CHIẾN LƯỢC SCHEDULE (eta đi từ 1.0 về 0.0) ---
        if self.config.schedule == "cosine":
            eta = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.config.schedule == "step":
            num_steps = 5.0
            eta = 1.0 - (math.floor(progress * num_steps) / num_steps)
        elif self.config.schedule == "exponential":
            c = 5.0
            eta = (math.exp(-c * progress) - math.exp(-c)) / (1.0 - math.exp(-c))
        else:
            eta = 1.0 - progress

        # Gắn eta cho Residual (đi từ 1->0), và (1 - eta) cho Roughness (đi từ 0->1)
        difficulty = eta * level_score + (1.0 - eta) * roughness_score  # (N,)
        self.last_difficulty = difficulty
        weights = difficulty
        self.last_weights = weights

        # =====================================================================
        # XUẤT DỮ LIỆU TỰ ĐỘNG (Dành cho Reviewer 1 & 2)
        # =====================================================================
        if is_target_epoch:
            base_dir = getattr(model, "model_save_path", None)
            
            if base_dir:
                save_dir = os.path.join(base_dir, "gracepinn_logs")
                os.makedirs(save_dir, exist_ok=True)
                
                log_flag = f"_logged_{current_epoch}"
                if not getattr(self, log_flag, False):
                    # 1. Lưu Heatmap (.npz)
                    npz_file = os.path.join(save_dir, f"epoch_{current_epoch}.npz")
                    np.savez(
                        npz_file,
                        coords=coords.cpu().numpy(),
                        r_norm=level_score.cpu().numpy(),
                        l_norm=roughness_score.cpu().numpy(),
                        difficulty=difficulty.cpu().numpy(),
                        raw_residuals=residual_matrix.cpu().numpy()
                    )

                    # 2. Lưu Memory log (.txt) bằng Byte
                    if track_mem:
                        mem_file = os.path.join(save_dir, "memory.txt")
                        with open(mem_file, "a") as f:
                            f.write(f"--- Epoch: {current_epoch} ---\n")
                            f.write(f"Num Points (N): {num_points} | Edges (E): {edge_index.shape[1]}\n")
                            f.write(f"  [Total GPU Mem    ] {total_current_b} B ({total_current_b / (1024**2):.2f} MB)\n")
                            f.write(f"  [Graph Construction] Add: {graph_alloc_b} B | Peak: {graph_peak_b} B\n")
                            f.write(f"  [Neigh Aggregation ] Add: {agg_alloc_b} B | Peak: {agg_peak_b} B\n\n")

                    setattr(self, log_flag, True)
                    
        return weights

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-dimension normalization to ~[0,1]."""
        min_vals = coords.min(dim=0).values
        max_vals = coords.max(dim=0).values
        span = torch.where(
            (max_vals - min_vals) > _EPS,
            max_vals - min_vals,
            torch.ones_like(max_vals),
        )
        return (coords - min_vals) / span

    def _build_edge_index(self, coords_scaled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = coords_scaled.device
        N = coords_scaled.shape[0]
        radius = self.config.radius
        chunk_size = 512  # Larger chunk for better performance

        row_list = []
        col_list = []
        dist_list = []

        for i0 in range(0, N, chunk_size):
            i1 = min(i0 + chunk_size, N)
            chunk_coords = coords_scaled[i0:i1]
            chunk_dist = torch.cdist(chunk_coords, coords_scaled, p=2)
            chunk_row = torch.arange(i0, i1, device=device).unsqueeze(1).expand(-1, N)
            chunk_col = torch.arange(N, device=device).unsqueeze(0).expand(i1 - i0, -1)
            chunk_mask = (chunk_dist < radius) & (chunk_row != chunk_col)
            row_list.append(chunk_row[chunk_mask])
            col_list.append(chunk_col[chunk_mask])
            dist_list.append(chunk_dist[chunk_mask])
            del chunk_dist, chunk_mask  # Free memory

        if row_list:
            row = torch.cat(row_list)
            col = torch.cat(col_list)
            edge_attr = torch.cat(dist_list)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.empty(0, device=device, dtype=coords_scaled.dtype)

        degrees = torch.zeros(N, device=device, dtype=torch.long)
        if edge_index.numel() > 0:
            degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device, dtype=torch.long))

        has_neighbors = degrees > 0

        # Xử lý điểm cô lập dựa vào min_k của tập đã có hàng xóm
        if has_neighbors.any():
            min_k = int(degrees[has_neighbors].min().item())
            if min_k > 0:
                isolated_idx = (~has_neighbors).nonzero(as_tuple=False).squeeze(-1)
                if isolated_idx.numel() > 0:
                    M = isolated_idx.numel()
                    iso_chunk_size = 512
                    add_row_list = []
                    add_col_list = []
                    add_d_list = []
                    for m0 in range(0, M, iso_chunk_size):
                        m1 = min(m0 + iso_chunk_size, M)
                        iso_ids = isolated_idx[m0:m1]
                        dist_iso = torch.cdist(coords_scaled[iso_ids], coords_scaled, p=2)
                        rows_iso = torch.arange(m0, m1, device=device)
                        for ii in range(m1 - m0):
                            global_i = iso_ids[ii]
                            dist_iso[ii, global_i] = float("inf")
                        values, idx_knn = torch.topk(dist_iso, min_k, dim=1, largest=False)
                        add_row = rows_iso.unsqueeze(1).expand(-1, min_k).flatten()
                        add_col = idx_knn.flatten()
                        add_d = values.flatten()
                        add_row_list.append(add_row)
                        add_col_list.append(add_col)
                        add_d_list.append(add_d)
                        del dist_iso
                    if add_row_list:
                        add_row = torch.cat(add_row_list)
                        add_col = torch.cat(add_col_list)
                        add_d = torch.cat(add_d_list)
                        add_index = torch.stack([isolated_idx[add_row], add_col], dim=0)
                        edge_index = torch.cat([edge_index, add_index], dim=1)
                        edge_attr = torch.cat([edge_attr, add_d])

        # Add reverse edges for undirected graph
        row, col = edge_index
        rev_mask = row != col  # No self-loops
        row_rev = col[rev_mask]
        col_rev = row[rev_mask]
        attr_rev = edge_attr[rev_mask]
        edge_index = torch.cat([edge_index, torch.stack([row_rev, col_rev], dim=0)], dim=1)
        edge_attr = torch.cat([edge_attr, attr_rev])

        return edge_index, edge_attr

    def _compute_sigma(self, dists: torch.Tensor) -> torch.Tensor:
        """Kernel length-scale sigma từ các cạnh đang active."""
        if dists.numel() == 0:
            sigma = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
        else:
            positive = dists[dists > _EPS]
            if positive.numel() == 0:
                sigma = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
            else:
                sigma = torch.median(positive)
        if not torch.isfinite(sigma) or sigma <= 0:
            sigma = torch.tensor(1.0, device=sigma.device, dtype=sigma.dtype)
        return sigma

    def _robust_normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Robust (0,1) normalization via median + IQR + sigmoid."""
        values = values.detach()

        q_low = torch.quantile(values, 0.25)
        q_high = torch.quantile(values, 0.75)

        if not torch.isfinite(q_low) or not torch.isfinite(q_high):
            return torch.zeros_like(values)

        iqr = q_high - q_low
        if iqr < _EPS:
            return torch.zeros_like(values)

        median = torch.quantile(values, 0.5)
        z = (values - median) / (iqr + _EPS)
        return torch.sigmoid(z)


# -----------------------------------------------------------------------------
# Data wrappers: inject GracePINN into DeepXDE PDE / TimePDE
# -----------------------------------------------------------------------------

class GracePDEData(PDE):
    """PDE dataset with GracePINN curriculum support (steady PDE)."""

    def __init__(
        self,
        *args,
        weight_strategy: Optional[GracePINNWeighting] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_strategy = weight_strategy
        self.model: Optional[dde.Model] = None
        self.current_weights: Optional[torch.Tensor] = None

    def attach_model(self, model: dde.Model) -> None:
        self.model = model

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif backend_name == "jax":
            outputs_pde = (outputs, aux[0])

        f = []
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(inputs, outputs_pde)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    if aux is None or len(aux) == 1:
                        raise ValueError("Auxiliary variable function not defined.")
                    f = self.pde(inputs, outputs_pde, unknowns=aux[1])
                else:
                    f = self.pde(inputs, outputs_pde, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                f"There are {len(f) + len(self.bcs)} errors, "
                f"but only {len(loss_fn)} losses."
            )

        bcs_start = torch.cumsum(torch.tensor([0] + self.num_bcs), dim=0).tolist()
        error_f = [fi[bcs_start[-1] :] for fi in f]

        weights = None
        if (
            self.weight_strategy is not None
            and self.model is not None
            and self.model.net.training
        ):
            with torch.no_grad():
                inputs_pde = inputs[bcs_start[-1] :]
                weights = self.weight_strategy(inputs_pde, error_f, self.model)
        self.current_weights = weights

        losses = []
        for i, error in enumerate(error_f):
            if weights is None:
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                sqrt_w = torch.sqrt(weights.clamp_min(0.0)).to(
                    error.device, dtype=error.dtype
                )
                expand_shape = [weights.shape[0]] + [1] * (error.ndim - 1)
                sqrt_w_expanded = sqrt_w.view(*expand_shape)
                scaled_error = sqrt_w_expanded * error

                base = loss_fn[i](bkd.zeros_like(error), scaled_error)
                denom = torch.sum(weights) + _EPS
                normalizer = (
                    weights.new_tensor(weights.numel(), dtype=base.dtype)
                    / denom.to(base.dtype)
                )
                losses.append(base * normalizer)

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(
                loss_fn[len(error_f) + i](bkd.zeros_like(error), error)
            )
        return losses


class GraceTimePDEData(TimePDE):
    """Time-dependent PDE dataset with GracePINN curriculum support."""

    def __init__(
        self,
        *args,
        weight_strategy: Optional[GracePINNWeighting] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_strategy = weight_strategy
        self.model: Optional[dde.Model] = None
        self.current_weights: Optional[torch.Tensor] = None

    def attach_model(self, model: dde.Model) -> None:
        self.model = model

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif backend_name == "jax":
            outputs_pde = (outputs, aux[0])

        f = []
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(inputs, outputs_pde)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    if aux is None or len(aux) == 1:
                        raise ValueError("Auxiliary variable function not defined.")
                    f = self.pde(inputs, outputs_pde, unknowns=aux[1])
                else:
                    f = self.pde(inputs, outputs_pde, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                f"There are {len(f) + len(self.bcs)} errors, "
                f"but only {len(loss_fn)} losses."
            )

        bcs_start = torch.cumsum(torch.tensor([0] + self.num_bcs), dim=0).tolist()
        error_f = [fi[bcs_start[-1] :] for fi in f]

        weights = None
        if (
            self.weight_strategy is not None
            and self.model is not None
            and self.model.net.training
        ):
            with torch.no_grad():
                inputs_pde = inputs[bcs_start[-1] :]
                weights = self.weight_strategy(inputs_pde, error_f, self.model)
        self.current_weights = weights

        losses = []
        for i, error in enumerate(error_f):
            if weights is None:
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                sqrt_w = torch.sqrt(weights.clamp_min(0.0)).to(
                    error.device, dtype=error.dtype
                )
                expand_shape = [weights.shape[0]] + [1] * (error.ndim - 1)
                sqrt_w_expanded = sqrt_w.view(*expand_shape)
                scaled_error = sqrt_w_expanded * error

                base = loss_fn[i](bkd.zeros_like(error), scaled_error)
                denom = torch.sum(weights) + _EPS
                normalizer = (
                    weights.new_tensor(weights.numel(), dtype=base.dtype)
                    / denom.to(base.dtype)
                )
                losses.append(base * normalizer)

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(
                loss_fn[len(error_f) + i](bkd.zeros_like(error), error)
            )
        return losses


# -----------------------------------------------------------------------------
# Convenience factory: build config từ PDE object
# -----------------------------------------------------------------------------

def build_gracepinn_for_pde(
    pde: BasePDE,
    total_iterations: int,
) -> GracePINNWeighting:
    """Factory gắn GracePINN cho một PDE cụ thể. Nơi duy nhất truyền cấu hình là Config."""
    
    # Khởi tạo config với các giá trị hard-code bên trong class (bao gồm alpha, radius mặc định)
    cfg = GracePINNConfig(total_iterations=total_iterations)

    # Chỉ override khi không phải TimePDE hoặc set time_dims
    if isinstance(pde, BaseTimePDE):
        cfg.time_dims = [pde.input_dim - 1]
        # alpha được giữ nguyên chính xác bằng giá trị ghi trong Config
    else:
        cfg.time_dims = None
        cfg.alpha = 0.0  # Nếu không phải TimePDE thì mới đổi thành 0

    return GracePINNWeighting(cfg)