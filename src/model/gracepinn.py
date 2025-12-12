"""GracePINN utilities for graph-based curriculum weighting."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
    k : int
        Số hàng xóm K trong KNN (hoặc fallback khi radius quá nhỏ).
    alpha : float
        Hệ số cân bằng không gian–thời gian, implement bằng cách
        scale các chiều thời gian bởi sqrt(alpha).
    percentiles : (float, float)
        Percentile (vd 25, 75) cho robust normalization (median + IQR + sigmoid).
    time_dims : Optional[Sequence[int]]
        Chỉ số các chiều được coi là thời gian.
    radius : Optional[float]
        Bán kính δ để dựng graph. Nếu None hoặc <= 0 → pure KNN.
    """

    total_iterations: int
    k: int = 6
    alpha: float = 0.5
    percentiles: Tuple[float, float] = (25.0, 75.0)
    time_dims: Optional[Sequence[int]] = None
    radius: Optional[float] = 0.15

    def __post_init__(self) -> None:
        self.total_iterations = max(int(self.total_iterations), 1)
        self.k = max(int(self.k), 1)
        self.alpha = max(float(self.alpha), 0.0)

        low, high = self.percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

        if self.time_dims is not None:
            self.time_dims = tuple(int(dim) for dim in self.time_dims)

        if self.radius is not None:
            self.radius = float(self.radius)


class GracePINNWeighting:
    """Graph-based difficulty estimator and curriculum mapper."""

    def __init__(self, config: GracePINNConfig) -> None:
        self.config = config
        self.last_progress: float = 0.0
        self.last_difficulty: Optional[torch.Tensor] = None
        self.last_weights: Optional[torch.Tensor] = None

    def __call__(
        self,
        inputs: torch.Tensor,
        residuals: Sequence[torch.Tensor],
        model: dde.Model,
    ) -> Optional[torch.Tensor]:
        """Compute curriculum weights for the PDE residuals in a batch.

        Parameters
        ----------
        inputs : (N, d_in)
            Tọa độ collocation (chỉ phần PDE, đã bỏ BC/IC phía trước).
        residuals : sequence of tensor
            Các thành phần residual PDE, mỗi cái shape (N, *).
        model : dde.Model
            Model DeepXDE, dùng để đọc epoch hiện tại.

        Returns
        -------
        weights : (N,) hoặc None
            Trọng số v_i(k) trên mỗi điểm PDE.
        """
        if inputs.numel() == 0:
            return None

        device = inputs.device
        coords = inputs.detach()
        num_points = coords.shape[0]
        if num_points <= 1:
            return torch.ones(num_points, device=device)

        # ------------------------------------------------------------------
        # 1) Aggregate residual components -> R(i)
        # ------------------------------------------------------------------
        res_components: List[torch.Tensor] = []
        for res in residuals:
            if res is None or res.numel() == 0:
                continue
            res_components.append(res.detach().reshape(res.shape[0], -1))

        if not res_components:
            return None

        residual_matrix = torch.cat(res_components, dim=1)   # (N, D_total)
        residual_norm = torch.linalg.norm(residual_matrix, dim=1)  # (N,)

        # ------------------------------------------------------------------
        # 2) Normalize coords + space–time scaling via alpha
        # ------------------------------------------------------------------
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
                # alpha degenerate → bỏ luôn khác biệt theo thời gian
                coords_scaled[:, time_dims] = 0.0

        # ------------------------------------------------------------------
        # 3) Build spatial-temporal graph
        # ------------------------------------------------------------------
        dist = torch.cdist(coords_scaled, coords_scaled, p=2)  # (N, N)
        mask = self._build_adjacency_mask(dist)

        if not mask.any():
            # graph rỗng hoàn toàn → quay về uniform
            return torch.ones(num_points, device=device)

        # ------------------------------------------------------------------
        # 4) Kernel weights + roughness L(i)
        # ------------------------------------------------------------------
        edge_dists = dist[mask]
        sigma = self._compute_sigma(edge_dists)

        kernel_full = torch.exp(-(dist**2) / (2 * sigma**2 + _EPS))
        kernel = kernel_full * mask.float()

        diff_full = residual_norm.unsqueeze(1) - residual_norm.unsqueeze(0)
        diff = diff_full * mask.float()

        roughness = (kernel * diff).sum(dim=1) / (kernel.sum(dim=1) + _EPS)

        # ------------------------------------------------------------------
        # 5) Normalize R and L → [0,1] bằng robust sigmoid
        # ------------------------------------------------------------------
        level_score = self._robust_normalize(residual_norm)
        roughness_score = self._robust_normalize(roughness)

        # ------------------------------------------------------------------
        # 6) Difficulty fusion D_i(k) với eta(k) = progress
        # ------------------------------------------------------------------
        progress = min(
            float(model.train_state.epoch) / max(self.config.total_iterations - 1, 1),
            1.0,
        )
        self.last_progress = progress

        eta = progress
        difficulty = (1.0 - eta) * level_score + eta * roughness_score  # (N,)
        self.last_difficulty = difficulty

        # ------------------------------------------------------------------
        # 7) Trọng số v_i(k) = D_i(k) (đã ở (0,1) nhờ sigmoid)
        # ------------------------------------------------------------------
        weights = difficulty
        self.last_weights = weights
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

    def _build_adjacency_mask(self, dist: torch.Tensor) -> torch.Tensor:
        """Build adjacency mask.

        - Nếu radius > 0: dùng rule d(i,j) < radius trước,
          node nào cô lập thì fallback KNN với K = min degree của các node có hàng xóm.
        - Nếu radius <= 0 hoặc radius fail: pure KNN với k neighbors.
        """
        N = dist.shape[0]
        device = dist.device
        eye = torch.eye(N, device=device, dtype=torch.bool)

        radius = self.config.radius
        if radius is not None and radius > 0.0:
            mask = (dist < radius) & (~eye)
            neighbor_counts = mask.sum(dim=1)
            has_neighbors = neighbor_counts > 0

            if has_neighbors.any():
                K = int(neighbor_counts[has_neighbors].min().item())
                if K > 0:
                    # Node cô lập: fallback KNN, nhưng chỉ loop trên ít node này
                    isolated_idx = (~has_neighbors).nonzero(as_tuple=False).view(-1)
                    for i in isolated_idx.tolist():
                        row = dist[i].clone()
                        row[i] = float("inf")
                        _, idx_knn = torch.topk(row, K, largest=False)
                        mask[i, idx_knn] = True
                return mask
            # nếu tất cả đều cô lập → rơi xuống pure KNN

        # ----------------- Pure KNN (vectorized) -----------------
        k = min(self.config.k, max(N - 1, 1))
        if k <= 0:
            return torch.zeros_like(dist, dtype=torch.bool)

        # tránh self-edge bằng cách cộng inf lên đường chéo
        dist_knn = dist + eye.to(dist.dtype) * float("inf")
        _, idx = torch.topk(dist_knn, k, dim=1, largest=False)  # (N, k)

        mask = torch.zeros_like(dist, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

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

        q_low = torch.quantile(values, self.config.percentiles[0] / 100.0)
        q_high = torch.quantile(values, self.config.percentiles[1] / 100.0)

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
        """Override PDE.losses to apply GracePINN weights on PDE residuals only."""
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif backend_name == "jax":
            outputs_pde = (outputs, aux[0])

        # 1) PDE residuals f
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

        # 2) Normalize loss_fn
        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                f"There are {len(f) + len(self.bcs)} errors, "
                f"but only {len(loss_fn)} losses."
            )

        # 3) Split BC/IC vs PDE collocation region
        bcs_start = torch.cumsum(torch.tensor([0] + self.num_bcs), dim=0).tolist()
        error_f = [fi[bcs_start[-1] :] for fi in f]

        # 4) Compute GracePINN weights trên PDE points
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

        # 5) Xây losses: PDE (weighted) + BC/IC (unweighted)
        losses = []

        # PDE terms
        for i, error in enumerate(error_f):
            if weights is None:
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                # L = (sum v_i e_i^2) / (sum v_i)
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

        # BC/IC terms
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
        """Same logic as GracePDEData.losses but cho TimePDE."""
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

# -----------------------------------------------------------------------------
# Convenience factory: build config từ PDE object
# -----------------------------------------------------------------------------

def build_gracepinn_for_pde(
    pde: BasePDE,
    total_iterations: int,
    **overrides,
) -> GracePINNWeighting:
    """Factory gắn GracePINN cho một PDE cụ thể.

    - Nhận duy nhất pde + total_iterations + optional overrides.
    - Default:
        + Nếu là BaseTimePDE: alpha=0.5, time_dims=[dim cuối].
        + Nếu là BasePDE (elliptic / steady): alpha=0.0, time_dims=None.
        + k=6, radius=0.15, percentiles=(25,75).
    - Nếu sau này muốn ablation hyper thì truyền overrides:
        pde.enable_gracepinn(total_iterations=T, k=8, radius=0.2, ...)
    """

    # 1) Phân biệt PDE theo thời gian hay không
    if isinstance(pde, BaseTimePDE):
        default_alpha = 0.5
        default_time_dims: Optional[Sequence[int]] = [pde.input_dim - 1]
    else:
        default_alpha = 0.0
        default_time_dims = None

    # 2) Lấy hyper từ overrides nếu có, ngược lại dùng default
    k = overrides.pop("k", 6)
    radius = overrides.pop("radius", 0.15)
    percentiles = overrides.pop("percentiles", (25.0, 75.0))
    alpha = overrides.pop("alpha", default_alpha)
    time_dims = overrides.pop("time_dims", default_time_dims)

    # 3) Nếu caller truyền key lạ → báo lỗi rõ, tránh bug câm
    if overrides:
        unknown = ", ".join(overrides.keys())
        raise ValueError(f"Unknown GracePINN hyperparameters: {unknown}")

    cfg = GracePINNConfig(
        total_iterations=total_iterations,
        k=k,
        alpha=alpha,
        percentiles=percentiles,
        time_dims=time_dims,
        radius=radius,
    )
    return GracePINNWeighting(cfg)

