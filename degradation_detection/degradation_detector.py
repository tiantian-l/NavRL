"""
Online Degradation Detector using likelihood-based Markov transition residuals.

Given a fitted nominal model (Linear or MLP), computes:
  - Instantaneous degradation score:  D_t = r_t^T Q^{-1} r_t   (Mahalanobis distance squared)
  - Windowed degradation score:       D_t^{(W)} = mean(D_{t-W+1}, ..., D_t)

Provides threshold computation from nominal data for chi-squared / empirical percentiles.
"""

import torch
import numpy as np
from typing import Optional
from transition_models import LinearTransitionModel, MLPTransitionModel


class DegradationDetector:
    """
    Online degradation detector.

    Usage:
        detector = DegradationDetector(model, window_size=20)
        detector.compute_thresholds(nominal_data)   # offline
        ...
        for each control step:
            score, level = detector.step(v_prev, u_prev, v_next)
    """

    def __init__(self, model, window_size: int = 20, device: str = "cpu"):
        """
        Args:
            model: a fitted LinearTransitionModel or MLPTransitionModel
            window_size: W for sliding window averaging
            device: torch device
        """
        self.model = model
        self.W = window_size
        self.device = device

        # Q_inv from the fitted model
        if isinstance(model, LinearTransitionModel):
            self.Q_inv = model.Q_inv.to(device)
        else:
            self.Q_inv = model.Q_inv.to(device)

        # Ring buffer for instantaneous scores
        self._scores_buf = torch.zeros(window_size, device=device)
        self._buf_ptr = 0
        self._buf_filled = 0

        # Thresholds (set by compute_thresholds)
        self.thresholds = {
            "q95": float("inf"),
            "q99": float("inf"),
            "q999": float("inf"),
        }

    def _mahalanobis_sq(self, r: torch.Tensor) -> torch.Tensor:
        """Compute r^T Q^{-1} r for a single residual vector (d,) or batch (N, d)."""
        if r.dim() == 1:
            return r @ self.Q_inv @ r
        # Batched: (N, d) @ (d, d) -> (N, d), then sum
        return (r @ self.Q_inv * r).sum(dim=-1)

    def step(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
             v_next: torch.Tensor) -> dict:
        """
        Process one control step.

        Args:
            v_prev: (3,) or (1,3) previous velocity
            u_prev: (3,) or (1,3) velocity command
            v_next: (3,) or (1,3) current velocity

        Returns:
            dict with keys: D_inst, D_window, residual, level
        """
        v_prev = v_prev.view(-1).to(self.device)
        u_prev = u_prev.view(-1).to(self.device)
        v_next = v_next.view(-1).to(self.device)

        # Residual
        if isinstance(self.model, LinearTransitionModel):
            v_hat = self.model.predict(v_prev.unsqueeze(0), u_prev.unsqueeze(0)).squeeze(0)
        else:
            v_hat = self.model.predict(v_prev.unsqueeze(0), u_prev.unsqueeze(0)).squeeze(0)

        r = v_next - v_hat

        # Instantaneous score
        D_inst = self._mahalanobis_sq(r).item()

        # Update ring buffer
        self._scores_buf[self._buf_ptr] = D_inst
        self._buf_ptr = (self._buf_ptr + 1) % self.W
        self._buf_filled = min(self._buf_filled + 1, self.W)

        # Windowed score
        D_window = self._scores_buf[:self._buf_filled].mean().item()

        # Degradation level
        level = 0
        if D_window >= self.thresholds["q999"]:
            level = 3
        elif D_window >= self.thresholds["q99"]:
            level = 2
        elif D_window >= self.thresholds["q95"]:
            level = 1

        return {
            "D_inst": D_inst,
            "D_window": D_window,
            "residual": r.detach().cpu(),
            "level": level,
        }

    def compute_thresholds(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
                           v_next: torch.Tensor):
        """
        Compute empirical thresholds from nominal data using the windowed score distribution.

        Args:
            v_prev, u_prev, v_next: (N, d) tensors of nominal transitions
        """
        v_prev = v_prev.to(self.device)
        u_prev = u_prev.to(self.device)
        v_next = v_next.to(self.device)

        # Compute all instantaneous scores
        if isinstance(self.model, LinearTransitionModel):
            residuals = self.model.residual(v_prev, u_prev, v_next)
        else:
            residuals = self.model.residual(v_prev, u_prev, v_next)

        D_all = self._mahalanobis_sq(residuals)  # (N,)

        # Compute windowed scores via convolution
        N = D_all.shape[0]
        if N >= self.W:
            # Moving average with window W
            D_cumsum = torch.cumsum(D_all, dim=0)
            D_windowed = (D_cumsum[self.W - 1:] - torch.cat([torch.zeros(1, device=self.device), D_cumsum[:-1]])[: N - self.W + 1]) / self.W
            # Simpler: use unfold
            D_windowed = D_all.unfold(0, self.W, 1).mean(dim=-1)
        else:
            D_windowed = D_all

        self.thresholds["q95"] = torch.quantile(D_windowed, 0.95).item()
        self.thresholds["q99"] = torch.quantile(D_windowed, 0.99).item()
        self.thresholds["q999"] = torch.quantile(D_windowed, 0.999).item()

        # Also store chi-squared reference (theoretical, for state_dim degrees of freedom)
        d = v_prev.shape[-1]
        from scipy.stats import chi2
        self.thresholds["chi2_95"] = chi2.ppf(0.95, d)
        self.thresholds["chi2_99"] = chi2.ppf(0.99, d)

        stats = {
            "D_inst_mean": D_all.mean().item(),
            "D_inst_std": D_all.std().item(),
            "D_window_mean": D_windowed.mean().item(),
            "D_window_std": D_windowed.std().item(),
            **self.thresholds,
        }
        return stats

    def reset(self):
        """Reset the ring buffer (e.g. at episode start)."""
        self._scores_buf.zero_()
        self._buf_ptr = 0
        self._buf_filled = 0

    def save(self, path: str):
        # Convert all threshold values to plain Python floats for safe serialization
        safe_thresholds = {k: float(v) for k, v in self.thresholds.items()}
        torch.save({"thresholds": safe_thresholds, "W": int(self.W)}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.thresholds = {k: float(v) for k, v in ckpt["thresholds"].items()}
        self.W = int(ckpt["W"])
        self._scores_buf = torch.zeros(self.W, device=self.device)


class BatchDegradationDetector:
    """
    Vectorized detector for parallel environments (Isaac Sim training).

    Maintains per-environment ring buffers.
    """

    def __init__(self, model, num_envs: int, window_size: int = 20, device: str = "cpu"):
        self.model = model
        self.num_envs = num_envs
        self.W = window_size
        self.device = device

        if isinstance(model, LinearTransitionModel):
            self.Q_inv = model.Q_inv.to(device)
        else:
            self.Q_inv = model.Q_inv.to(device)

        # Ring buffers: (num_envs, W)
        self._scores_buf = torch.zeros(num_envs, window_size, device=device)
        self._buf_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._buf_filled = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.thresholds = {"q95": float("inf"), "q99": float("inf"), "q999": float("inf")}

    def _mahalanobis_sq_batch(self, r: torch.Tensor) -> torch.Tensor:
        """r: (num_envs, d) -> (num_envs,)"""
        return (r @ self.Q_inv * r).sum(dim=-1)

    def step(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
             v_next: torch.Tensor) -> dict:
        """
        Batch step for all environments.

        Args:
            v_prev: (num_envs, 3)
            u_prev: (num_envs, 3)
            v_next: (num_envs, 3)

        Returns:
            dict with D_inst (num_envs,), D_window (num_envs,), level (num_envs,)
        """
        # Residuals
        if isinstance(self.model, LinearTransitionModel):
            v_hat = self.model.predict(v_prev, u_prev)
        else:
            v_hat = self.model.predict(v_prev, u_prev)
        r = v_next - v_hat

        # Instantaneous scores
        D_inst = self._mahalanobis_sq_batch(r)  # (num_envs,)

        # Update ring buffers
        idx = torch.arange(self.num_envs, device=self.device)
        self._scores_buf[idx, self._buf_ptr] = D_inst
        self._buf_ptr = (self._buf_ptr + 1) % self.W
        self._buf_filled = torch.clamp(self._buf_filled + 1, max=self.W)

        # Windowed scores
        filled = self._buf_filled.float().clamp(min=1)
        D_window = self._scores_buf.sum(dim=-1) / filled

        # Levels
        level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        level[D_window >= self.thresholds["q95"]] = 1
        level[D_window >= self.thresholds["q99"]] = 2
        level[D_window >= self.thresholds["q999"]] = 3

        return {
            "D_inst": D_inst,
            "D_window": D_window,
            "level": level,
        }

    def reset_envs(self, env_ids: torch.Tensor):
        """Reset buffers for specific environments."""
        self._scores_buf[env_ids] = 0.0
        self._buf_ptr[env_ids] = 0
        self._buf_filled[env_ids] = 0

    def set_thresholds(self, thresholds: dict):
        self.thresholds = thresholds
