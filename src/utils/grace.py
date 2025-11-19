"""Grace curriculum utilities and callback."""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from deepxde.callbacks import Callback

_EPS = 1e-12


def percentile_normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Normalize ``values`` to [0, 1] after percentile clipping."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    low = float(low)
    high = float(high)
    if high <= low:
        return np.zeros_like(arr)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if not np.isfinite(hi - lo) or hi - lo < _EPS:
        return np.zeros_like(arr)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + _EPS)


def cosine_schedule(step: int, total_steps: int, start: float = 0.0, end: float = 1.0) -> float:
    """Cosine annealing schedule between ``start`` and ``end``."""

    total_steps = max(int(total_steps), 1)
    step = min(max(int(step), 0), total_steps)
    progress = 0.5 * (1 - np.cos(np.pi * step / total_steps))
    return float(start + (end - start) * progress)


def build_grace_graph(
    points: np.ndarray,
    radius: Optional[float],
    k_neighbors: int,
) -> Dict[str, Iterable[np.ndarray]]:
    """Construct a neighborhood graph with Gaussian edge weights."""

    pts = np.asarray(points, dtype=np.float64)
    num_points = pts.shape[0]
    if num_points == 0:
        return {"neighbors": [], "weights": [], "sigma": 1.0, "radius": 0.0, "k": 0}

    tree = cKDTree(pts)
    k_neighbors = max(int(k_neighbors), 0)
    effective_k = min(k_neighbors, max(num_points - 1, 0))

    use_radius = None if radius is None else float(radius)
    if use_radius is None or use_radius <= 0:
        if effective_k > 0:
            dist_knn, _ = tree.query(pts, k=min(effective_k + 1, num_points))
            dist_knn = np.atleast_2d(dist_knn)
            if dist_knn.shape[1] > 1:
                use_radius = float(np.median(dist_knn[:, 1:]))
            else:
                use_radius = 1.0
        else:
            use_radius = 1.0
    use_radius = max(use_radius, _EPS)

    neighbors: List[np.ndarray] = []
    distances: List[np.ndarray] = []
    all_dists: List[float] = []

    for idx in range(num_points):
        row_idx: List[int] = []
        row_dist: List[float] = []

        candidate = tree.query_ball_point(pts[idx], r=use_radius)
        if candidate:
            row_idx = [j for j in candidate if j != idx]
            row_dist = [float(np.linalg.norm(pts[idx] - pts[j])) for j in row_idx]

        if not row_idx and effective_k > 0:
            dist_knn, idx_knn = tree.query(pts[idx], k=min(effective_k + 1, num_points))
            dist_knn = np.atleast_1d(dist_knn)
            idx_knn = np.atleast_1d(idx_knn)
            row_idx = idx_knn[1:].astype(int).tolist()
            row_dist = dist_knn[1:].astype(float).tolist()

        neighbors.append(np.array(row_idx, dtype=np.int64))
        distances.append(np.array(row_dist, dtype=np.float64))
        if row_dist:
            all_dists.extend(row_dist)

    if not all_dists:
        sigma = 1.0
    else:
        sigma = float(np.median(np.asarray(all_dists)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

    weights: List[np.ndarray] = []
    for dist in distances:
        if dist.size == 0:
            weights.append(dist)
        else:
            weights.append(np.exp(-np.square(dist) / (2.0 * sigma**2 + _EPS)))

    return {
        "neighbors": neighbors,
        "weights": weights,
        "sigma": sigma,
        "radius": use_radius,
        "k": effective_k,
    }


def compute_graph_difficulty(values: np.ndarray, graph: Dict[str, Iterable[np.ndarray]]) -> np.ndarray:
    """Compute directional roughness based on neighboring residuals."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    diffs = np.zeros_like(arr)
    neighbors = graph.get("neighbors", [])
    weights = graph.get("weights", [])
    for i, (nbrs, w) in enumerate(zip(neighbors, weights)):
        if nbrs.size == 0:
            continue
        local_weights = w
        if local_weights.size == 0 or local_weights.shape[0] != nbrs.shape[0]:
            local_weights = np.ones_like(nbrs, dtype=np.float64)
        delta = np.maximum(arr[i] - arr[nbrs], 0.0)
        weight_sum = float(np.sum(local_weights))
        if weight_sum <= 0:
            diffs[i] = float(np.mean(delta)) if delta.size else 0.0
        else:
            diffs[i] = float(np.dot(local_weights, delta) / (weight_sum + _EPS))
    return diffs


def weights_to_scale(weights: np.ndarray) -> np.ndarray:
    """Convert normalized weights to scaling factors used in the loss."""

    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, _EPS, None)
    arr = arr / np.sum(arr)
    num = arr.shape[0]
    return np.sqrt(num * arr)


class GraceCurriculumCallback(Callback):
    """Graph-based adaptive curriculum for PDE residuals."""

    def __init__(
        self,
        total_iterations: int,
        alpha: float = 0.5,
        delta: float = 0.1,
        radius: Optional[float] = None,
        percentiles: Tuple[float, float] = (5.0, 95.0),
        v_bounds: Tuple[float, float] = (0.2, 0.8),
        k_neighbors: int = 8,
        dump_debug: bool = False,
    ) -> None:
        super().__init__()
        self.total_iterations = max(int(total_iterations), 1)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.delta = float(np.clip(delta, 0.0, 1.0))
        low, high = percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        self.percentiles = (float(low), float(high))
        vmin, vmax = v_bounds
        if vmin < 0 or vmax < vmin:
            raise ValueError("v_bounds must satisfy 0 <= vmin <= vmax")
        self.v_bounds = (float(vmin), float(vmax))
        self.radius = radius
        self.k_neighbors = max(int(k_neighbors), 0)
        self.dump_debug = dump_debug
        self.debug_records: List[Dict[str, float]] = []
        self.debug_path: Optional[str] = None

        self.pde_points: Optional[np.ndarray] = None
        self.graph: Optional[Dict[str, Iterable[np.ndarray]]] = None
        self.current_weights: Optional[np.ndarray] = None

    def on_train_begin(self):
        data = getattr(self.model, "data", None)
        train_state = getattr(self.model, "train_state", None)
        if data is None or train_state is None:
            return
        if not hasattr(data, "set_pde_weight_scale"):
            return
        X_train = getattr(train_state, "X_train", None)
        num_bc = int(np.sum(data.num_bcs)) if data.num_bcs is not None else 0
        if X_train is None or len(X_train) <= num_bc:
            return
        self.pde_points = np.asarray(X_train[num_bc:], dtype=np.float64)
        if self.pde_points.size == 0:
            return
        self.graph = build_grace_graph(
            self.pde_points,
            self.radius,
            self.k_neighbors,
        )
        num_points = self.pde_points.shape[0]
        base_weights = np.full(num_points, 1.0 / num_points, dtype=np.float64)
        self.current_weights = base_weights
        data.set_pde_weight_scale(weights_to_scale(base_weights))
        if self.dump_debug:
            self.debug_records = []
            self.debug_path = os.path.join(
                getattr(self.model, "model_save_path", os.getcwd()),
                "grace_debug.npy",
            )

    def on_epoch_begin(self):
        if self.pde_points is None or self.pde_points.size == 0:
            return
        if not hasattr(self.model.data, "set_pde_weight_scale"):
            return
        pde = getattr(self.model, "pde", None)
        if pde is None or getattr(pde, "pde", None) is None:
            return
        residuals = self.model.predict(self.pde_points, operator=pde.pde)
        residual_norm = self._compute_residual_norm(residuals)
        if residual_norm.size == 0:
            return
        if (
            self.current_weights is not None
            and residual_norm.shape[0] != self.current_weights.shape[0]
        ):
            return
        if self.graph is None:
            self.graph = build_grace_graph(self.pde_points, self.radius, self.k_neighbors)
        graph_score = compute_graph_difficulty(residual_norm, self.graph)
        level = percentile_normalize(residual_norm, *self.percentiles)
        graph_norm = percentile_normalize(graph_score, *self.percentiles)

        total_steps = max(self.total_iterations - 1, 1)
        current_step = min(max(int(self.model.train_state.epoch), 0), total_steps)
        p_k = cosine_schedule(current_step, total_steps)
        eta = self.delta + (1.0 - self.delta) * p_k

        difficulty = (1.0 - eta) * level + eta * graph_norm
        new_weights = np.clip(difficulty, self.v_bounds[0], self.v_bounds[1])
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        else:
            new_weights = np.full_like(new_weights, 1.0 / new_weights.shape[0])

        if self.current_weights is None:
            updated = new_weights
        else:
            updated = (1.0 - self.alpha) * self.current_weights + self.alpha * new_weights
        updated = np.clip(updated, _EPS, None)
        updated = updated / np.sum(updated)
        self.current_weights = updated
        self.model.data.set_pde_weight_scale(weights_to_scale(updated))

        if self.dump_debug and self.debug_path is not None:
            stats = {
                "epoch": float(self.model.train_state.epoch),
                "progress": float(p_k),
                "eta": float(eta),
                "weight_min": float(np.min(updated)),
                "weight_max": float(np.max(updated)),
                "weight_mean": float(updated.mean()),
            }
            self.debug_records.append(stats)

    def on_train_end(self):
        if self.dump_debug and self.debug_records and self.debug_path is not None:
            os.makedirs(os.path.dirname(self.debug_path), exist_ok=True)
            np.save(self.debug_path, np.array(self.debug_records, dtype=object), allow_pickle=True)

    def _compute_residual_norm(self, residuals) -> np.ndarray:
        if residuals is None:
            return np.array([])
        if isinstance(residuals, (list, tuple)):
            mats = []
            for res in residuals:
                if res is None:
                    continue
                arr = np.asarray(res)
                if arr.ndim == 1:
                    arr = arr[:, None]
                else:
                    arr = arr.reshape(arr.shape[0], -1)
                mats.append(arr)
            if not mats:
                return np.array([])
            joined = np.concatenate(mats, axis=1)
        else:
            joined = np.asarray(residuals)
            if joined.ndim == 1:
                joined = joined[:, None]
            else:
                joined = joined.reshape(joined.shape[0], -1)
        return np.linalg.norm(joined, axis=1)
