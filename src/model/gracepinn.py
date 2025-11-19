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
    """Configuration bundle for GracePINN weighting."""

    total_iterations: int
    k: int = 12
    alpha: float = 0.1
    sigma_scale: float = 1.0
    percentiles: Tuple[float, float] = (5.0, 95.0)
    weight_clip: Tuple[float, float] = (0.2, 0.8)
    time_dims: Optional[Sequence[int]] = None

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
        """Compute curriculum weights for the PDE residuals."""
        if inputs.numel() == 0:
            return None
        device = inputs.device
        coords = inputs.detach()
        num_points = coords.shape[0]
        if num_points <= 1:
            return torch.ones(num_points, device=device)

        res_components: List[torch.Tensor] = []
        for res in residuals:
            if res is None or res.numel() == 0:
                continue
            res_components.append(res.detach().reshape(res.shape[0], -1))
        if not res_components:
            return None
        residual_matrix = torch.cat(res_components, dim=1)
        residual_norm = torch.linalg.norm(residual_matrix, dim=1)

        coords_norm = self._normalize(coords)
        coords_scaled = coords_norm.clone()
        if self.config.time_dims:
            time_dims = torch.tensor(self.config.time_dims, device=device)
            scale = torch.sqrt(torch.tensor(self.config.alpha, device=device))
            if torch.isfinite(scale) and scale > 0:
                coords_scaled[:, time_dims] = coords_scaled[:, time_dims] * scale
            else:
                coords_scaled[:, time_dims] = 0.0

        dist = torch.cdist(coords_scaled, coords_scaled, p=2)
        k = min(self.config.k, num_points - 1)
        if k <= 0:
            return torch.ones(num_points, device=device)
        dists, indices = torch.topk(dist, k + 1, dim=1, largest=False)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        sigma = self._compute_sigma(dists)
        kernel = torch.exp(-(dists**2) / (2 * sigma**2 + _EPS))
        diff = (residual_norm.unsqueeze(1) - residual_norm[indices]).clamp(min=0.0)
        roughness = (kernel * diff).sum(1) / (kernel.sum(1) + _EPS)

        level_score = self._robust_normalize(residual_norm)
        roughness_score = self._robust_normalize(roughness)

        progress = min(
            float(model.train_state.epoch) / max(self.config.total_iterations - 1, 1),
            1.0,
        )
        self.last_progress = progress
        eta = progress
        difficulty = (1.0 - eta) * level_score + eta * roughness_score
        self.last_difficulty = difficulty

        curriculum = (1.0 - progress) * (1.0 - difficulty) + progress * difficulty
        weights = torch.clamp(
            curriculum,
            self.config.weight_clip[0],
            self.config.weight_clip[1],
        )
        self.last_weights = weights
        return weights

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        min_vals = coords.min(dim=0).values
        max_vals = coords.max(dim=0).values
        span = torch.where(
            (max_vals - min_vals) > _EPS,
            max_vals - min_vals,
            torch.ones_like(max_vals),
        )
        return (coords - min_vals) / span

    def _compute_sigma(self, dists: torch.Tensor) -> torch.Tensor:
        positive = dists[dists > _EPS]
        if positive.numel() == 0:
            sigma = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
        else:
            sigma = torch.median(positive)
        sigma = sigma * self.config.sigma_scale
        if not torch.isfinite(sigma) or sigma <= 0:
            sigma = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
        return sigma

    def _robust_normalize(self, values: torch.Tensor) -> torch.Tensor:
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
    """PDE dataset with GracePINN curriculum support."""

    def __init__(self, *args, weight_strategy: Optional[GracePINNWeighting] = None, **kwargs):
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
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss_fn)
                )
            )

        bcs_start = torch.cumsum(torch.tensor([0] + self.num_bcs), dim=0)
        bcs_start = bcs_start.tolist()
        error_f = [fi[bcs_start[-1] :] for fi in f]

        weights = None
        if (
            self.weight_strategy is not None
            and self.model is not None
            and self.model.net.training
        ):
            with torch.no_grad():
                weights = self.weight_strategy(inputs[bcs_start[-1] :], error_f, self.model)
        self.current_weights = weights

        losses = []
        for i, error in enumerate(error_f):
            if weights is None:
                losses.append(loss_fn[i](bkd.zeros_like(error), error))
            else:
                sqrt_w = torch.sqrt(weights.clamp_min(0.0)).to(error.device, dtype=error.dtype)
                expand_shape = [weights.shape[0]] + [1] * (error.ndim - 1)
                sqrt_w_expanded = sqrt_w.view(*expand_shape)
                scaled_error = sqrt_w_expanded * error
                base = loss_fn[i](bkd.zeros_like(error), scaled_error)
                denom = torch.sum(weights) + _EPS
                normalizer = weights.new_tensor(weights.numel(), dtype=base.dtype) / denom.to(base.dtype)
                losses.append(base * normalizer)
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(loss_fn[len(error_f) + i](bkd.zeros_like(error), error))
        return losses

class GraceTimePDEData(TimePDE, GracePDEData):
    """Time-dependent PDE dataset with GracePINN curriculum support."""

    def __init__(self, *args, weight_strategy: Optional[GracePINNWeighting] = None, **kwargs):
        TimePDE.__init__(self, *args, **kwargs)
        self.weight_strategy = weight_strategy
        self.model: Optional[dde.Model] = None
        self.current_weights: Optional[torch.Tensor] = None

    # ``losses`` and ``attach_model`` inherited from ``GracePDEData`` via MRO.
