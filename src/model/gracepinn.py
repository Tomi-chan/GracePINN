"""GracePINN utilities (model-side) for graph-based curriculum weighting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

import deepxde as dde
from deepxde import backend as bkd
from deepxde.backend import backend_name
from deepxde.data.pde import PDE, TimePDE
from deepxde.utils import get_num_args

_EPS = 1e-12


@dataclass
class GracePINNConfig:
    """Configuration bundle for GracePINN weighting.

    Parameters
    ----------
    total_iterations : int
        Total training iterations (used to normalize progress p(k) in [0,1]).
    k : int, default=12
        Fallback K for pure KNN mode (or for fully KNN if radius is None
        and no node has any neighbor under a radius rule).
    alpha : float, default=0.1
        Space-time balance coefficient in the distance:
        d^2 = ||x_i' - x_j'||^2 + alpha * (t_i' - t_j')^2
        (implemented via scaling time dims by sqrt(alpha)).
    sigma_scale : float, default=1.0
        Multiplicative factor on the median edge length used as kernel length scale.
    percentiles : (float, float), default=(5.0, 95.0)
        Percentiles for robust normalization of residual level and roughness.
    weight_clip : (float, float), default=(0.2, 0.8)
        Lower/upper bounds to clip curriculum weights v_i(k).
    time_dims : Optional[Sequence[int]], default=None
        Indices of coordinates that should be treated as "time" and scaled by sqrt(alpha).
    radius : Optional[float], default=None
        Radius δ for graph construction. If > 0:
          - Edges first formed by d(i,j) < δ (excluding self).
          - For nodes with no neighbors, we use KNN with
            K = min number of neighbors among nodes that did satisfy radius rule.
        If None or <= 0: pure KNN mode with k neighbors.
    """

    total_iterations: int
    k: int = 12
    alpha: float = 0.1
    sigma_scale: float = 1.0
    percentiles: Tuple[float, float] = (5.0, 95.0)
    weight_clip: Tuple[float, float] = (0.2, 0.8)
    time_dims: Optional[Sequence[int]] = None
    radius: Optional[float] = None

    def __post_init__(self) -> None:
        self.total_iterations = max(int(self.total_iterations), 1)
        self.k = max(int(self.k), 1)
        self.alpha = max(float(self.alpha), 0.0)
        self.sigma_scale = max(float(self.sigma_scale), _EPS)

        low, high = self.percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

        vmin, vmax = self.weight_clip
        if not 0.0 <= vmin <= vmax <= 1.0:
            raise ValueError("weight_clip must satisfy 0 <= vmin <= vmax <= 1")

        if self.time_dims is not None:
            self.time_dims = tuple(int(dim) for dim in self.time_dims)

        if self.radius is not None:
            self.radius = float(self.radius)


class GracePINNWeighting:
    """Graph-based difficulty estimator and curriculum mapper.

    This class implements:
      - Spatial-temporal distance with alpha balance.
      - Graph construction with radius rule + KNN fallback (or pure KNN).
      - Neighborhood-based roughness L(i).
      - Difficulty fusion D_i(k).
      - Easy→hard curriculum weights v_i(k).
    """

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
        inputs : torch.Tensor
            Collocation coordinates (after BC/IC indices are stripped), shape (N, d_in).
        residuals : sequence of torch.Tensor
            PDE residual components; each of shape (N, *).
        model : dde.Model
            DeepXDE model, used here to read training progress (epoch).

        Returns
        -------
        weights : Optional[torch.Tensor]
            Vector of per-sample curriculum weights v_i(k), shape (N,).
            None if no residual or inputs are empty.
        """
        if inputs.numel() == 0:
            return None

        device = inputs.device
        coords = inputs.detach()
        num_points = coords.shape[0]
        if num_points <= 1:
            return torch.ones(num_points, device=device)

        # --- 1) Aggregate residual components and compute level R(i) ---
        res_components: List[torch.Tensor] = []
        for res in residuals:
            if res is None or res.numel() == 0:
                continue
            # Flatten each residual component to (N, D_i)
            res_components.append(res.detach().reshape(res.shape[0], -1))

        if not res_components:
            return None

        residual_matrix = torch.cat(res_components, dim=1)  # (N, D_total)
        residual_norm = torch.linalg.norm(residual_matrix, dim=1)  # (N,)

        # --- 2) Normalize coordinates and apply space-time scaling via alpha ---
        coords_norm = self._normalize(coords)  # global [0,1] scaling per dim
        coords_scaled = coords_norm.clone()

        if self.config.time_dims is not None and len(self.config.time_dims) > 0:
            time_dims = torch.tensor(self.config.time_dims, device=device, dtype=torch.long)
            scale = torch.sqrt(torch.tensor(self.config.alpha, device=device))
            if torch.isfinite(scale) and scale > 0:
                coords_scaled[:, time_dims] = coords_scaled[:, time_dims] * scale
            else:
                # If alpha is degenerate, effectively ignore time difference
                coords_scaled[:, time_dims] = 0.0

        # --- 3) Build spatial-temporal graph (radius rule + KNN fallback or pure KNN) ---
        dist = torch.cdist(coords_scaled, coords_scaled, p=2)  # (N, N)
        mask = self._build_adjacency_mask(dist)

        # If still no edges at all (pathological), fallback to uniform weights
        if not mask.any():
            return torch.ones(num_points, device=device)

        # --- 4) Compute kernel weights and roughness L(i) ---
        # sigma from all active edges (distances where mask == True)
        edge_dists = dist[mask]
        sigma = self._compute_sigma(edge_dists)

        # Gaussian kernel on all pairs, then zeroed by mask
        kernel_full = torch.exp(-(dist**2) / (2 * sigma**2 + _EPS))
        kernel = kernel_full * mask.float()

        # Diff matrix: R(i) - R(j)
        # residual_norm: (N,) -> (N,1) and (1,N)
        diff_full = residual_norm.unsqueeze(1) - residual_norm.unsqueeze(0)
        diff = diff_full * mask.float()

        # L(i) = sum_j w_ij (R(i) - R(j)) / sum_j w_ij
        # (no clamp; signed neighborhood contrast as in proposal)
        roughness = (kernel * diff).sum(dim=1) / (kernel.sum(dim=1) + _EPS)

        # --- 5) Normalize level and roughness to [0,1] with robust min-max ---
        level_score = self._robust_normalize(residual_norm)
        roughness_score = self._robust_normalize(roughness)

        # --- 6) Difficulty fusion D_i(k) with epoch-dependent weights ---
        # progress p(k) in [0,1]
        progress = min(
            float(model.train_state.epoch) / max(self.config.total_iterations - 1, 1),
            1.0,
        )
        self.last_progress = progress

        eta = progress  # η(k)
        difficulty = (1.0 - eta) * level_score + eta * roughness_score
        self.last_difficulty = difficulty

        # --- 7) Curriculum map v_i(k) = (1-p)(1-D) + p D ---
        curriculum = (1.0 - progress) * (1.0 - difficulty) + progress * difficulty

        weights = torch.clamp(
            curriculum,
            self.config.weight_clip[0],
            self.config.weight_clip[1],
        )
        self.last_weights = weights
        return weights

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Global per-dimension normalization to [0,1]-ish range."""
        min_vals = coords.min(dim=0).values
        max_vals = coords.max(dim=0).values
        span = torch.where(
            (max_vals - min_vals) > _EPS,
            max_vals - min_vals,
            torch.ones_like(max_vals),
        )
        return (coords - min_vals) / span

    def _build_adjacency_mask(self, dist: torch.Tensor) -> torch.Tensor:
        """Build adjacency mask for the graph.

        If radius > 0:
            - Use radius rule first: edge (i,j) if dist(i,j) < radius and i != j.
            - For nodes with zero neighbors, use KNN with
              K = min number of neighbors among nodes that had at least one neighbor
              under radius rule.
        Else:
            - Pure KNN: each node connects to k nearest neighbors (excluding self).
        """
        N = dist.shape[0]
        device = dist.device
        eye = torch.eye(N, device=device, dtype=torch.bool)

        radius = self.config.radius
        if radius is not None and radius > 0.0:
            # Radius-based mask (excluding self)
            mask = (dist < radius) & (~eye)
            neighbor_counts = mask.sum(dim=1)
            has_neighbors = neighbor_counts > 0

            if has_neighbors.any():
                # For nodes with no neighbors under radius, use KNN with K =
                # min neighbor count among nodes that had neighbors.
                K = int(neighbor_counts[has_neighbors].min().item())
                if K > 0:
                    isolated_idx = (~has_neighbors).nonzero(as_tuple=False).view(-1)
                    for i in isolated_idx.tolist():
                        row = dist[i].clone()
                        row[i] = float("inf")  # exclude self
                        _, idx_knn = torch.topk(row, K, largest=False)
                        mask[i, idx_knn] = True
                return mask

            # If nobody has any neighbor under radius at all, fall back to pure KNN below.

        # Pure KNN mode (or fallback if radius is too small)
        k = min(self.config.k, N - 1)
        if k <= 0:
            return torch.zeros_like(dist, dtype=torch.bool)

        mask = torch.zeros_like(dist, dtype=torch.bool)
        for i in range(N):
            row = dist[i].clone()
            row[i] = float("inf")  # exclude self
            _, idx_knn = torch.topk(row, k, largest=False)
            mask[i, idx_knn] = True
        return mask

    def _compute_sigma(self, dists: torch.Tensor) -> torch.Tensor:
        """Compute kernel length-scale sigma from active edge distances."""
        if dists.numel() == 0:
            sigma = torch.tensor(1.0, device=dists.device if dists.is_cuda or dists.device.type != "cpu" else None)
        else:
            positive = dists[dists > _EPS]
            if positive.numel() == 0:
                sigma = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
            else:
                sigma = torch.median(positive)
        sigma = sigma * self.config.sigma_scale
        if not torch.isfinite(sigma) or sigma <= 0:
            sigma = torch.tensor(1.0, device=sigma.device, dtype=sigma.dtype)
        return sigma

    def _robust_normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Robust [0,1] normalization using percentile clipping."""
        values = values.detach()
        q_low = torch.quantile(values, self.config.percentiles[0] / 100.0)
        q_high = torch.quantile(values, self.config.percentiles[1] / 100.0)

        if not torch.isfinite(q_low):
            q_low = torch.tensor(0.0, device=values.device, dtype=values.dtype)
        if not torch.isfinite(q_high):
            q_high = torch.tensor(1.0, device=values.device, dtype=values.dtype)

        if q_high - q_low < _EPS:
            return torch.zeros_like(values)

        clipped = values.clamp(q_low, q_high)
        normalized = (clipped - q_low) / (q_high - q_low + _EPS)
        return normalized


class GracePDEData(PDE):
    """PDE dataset with GracePINN curriculum support.

    This class wraps DeepXDE's PDE data object and injects GracePINN weights
    into PDE residual terms only, leaving IC/BC losses unchanged.
    """

    def __init__(self, *args, weight_strategy: Optional[GracePINNWeighting] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_strategy = weight_strategy
        self.model: Optional[dde.Model] = None
        self.current_weights: Optional[torch.Tensor] = None

    def attach_model(self, model: dde.Model) -> None:
        """Attach DeepXDE model to access training state (epoch)."""
        self.model = model

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Override PDE.losses to apply GracePINN weights on PDE residuals only."""
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
            outputs_pde = outputs
        elif backend_name == "jax":
            outputs_pde = (outputs, aux[0])

        # 1) Compute PDE residuals f
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

        # 2) Normalize loss_fn list
        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss_fn)
                )
            )

        # 3) Split BC/IC vs PDE collocation region
        bcs_start = torch.cumsum(torch.tensor([0] + self.num_bcs), dim=0)
        bcs_start = bcs_start.tolist()
        # error_f: PDE errors only (strip BC/IC region)
        error_f = [fi[bcs_start[-1] :] for fi in f]

        # 4) Compute GracePINN weights on PDE points (if enabled)
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

        # 5) Build losses: PDE terms (weighted) + BC terms (unweighted)
        losses = []
        # PDE terms
        for i, error in enumerate(error_f):
            if weights is None:
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                # Weighted PDE loss with v_i(k):
                # L = sum v_i e_i^2 / sum v_i
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

        # BC/IC terms (unchanged)
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(
                loss_fn[len(error_f) + i](bkd.zeros_like(error), error)
            )
        return losses


class GraceTimePDEData(TimePDE, GracePDEData):
    """Time-dependent PDE dataset with GracePINN curriculum support."""

    def __init__(self, *args, weight_strategy: Optional[GracePINNWeighting] = None, **kwargs):
        # TimePDE handles underlying PDE data setup
        TimePDE.__init__(self, *args, **kwargs)
        # Grace-specific fields
        self.weight_strategy = weight_strategy
        self.model: Optional[dde.Model] = None
        self.current_weights: Optional[torch.Tensor] = None

    # `losses` and `attach_model` are inherited from GracePDEData via MRO.
