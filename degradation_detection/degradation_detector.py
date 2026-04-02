"""
Online Degradation Detector using Max-Z anomaly score + count-based persistence.

Method:
  Step 1 — Single-point anomaly detection:
    r_t = v_t - f(v_{t-1}, u_{t-1})          residual
    z_i = |r_i| / sigma_i                     per-axis standardized residual
    A_t = max(z_x, z_y, z_z)                  anomaly score (worst-axis deviation)
    a_t = 1 if A_t > tau_point else 0         binary anomaly flag

  Step 2 — Temporal persistence detection:
    C_t = sum(a_{t-W+1}, ..., a_t)            anomaly count in window W
    level = f(C_t)                            degradation level (0-3)

Thresholds computed from nominal validation data (empirical quantiles).
"""

import torch
import numpy as np
from typing import Optional
from transition_models import LinearTransitionModel, MLPTransitionModel


class DegradationDetector:
    """
    Online degradation detector (single environment).

    Usage:
        detector = DegradationDetector(model, window_size=20)
        detector.compute_thresholds(nominal_data)   # offline
        ...
        for each control step:
            result = detector.step(v_prev, u_prev, v_next)
    """

    def __init__(self, model, window_size: int = 20, device: str = "cpu"):
        self.model = model
        self.W = window_size
        self.device = device

        # Per-axis residual std from model's Q diagonal
        Q = model.Q.to(device)
        self.sigma = torch.sqrt(Q.diag()).clamp(min=1e-8)  # (d,)

        # Ring buffer for binary anomaly flags
        self._flag_buf = torch.zeros(window_size, dtype=torch.long, device=device)
        self._score_buf = torch.zeros(window_size, device=device)  # A_t values
        self._buf_ptr = 0
        self._buf_filled = 0

        # Thresholds (set by compute_thresholds)
        self.tau_point = float("inf")       # single-step A_t threshold
        self.C_levels = [4, 8, 14]          # count thresholds for levels 1,2,3

    def step(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
             v_next: torch.Tensor) -> dict:
        """
        Process one control step.

        Returns:
            dict with keys: A_t, a_t, C_t, level, residual, z
        """
        v_prev = v_prev.view(-1).to(self.device)
        u_prev = u_prev.view(-1).to(self.device)
        v_next = v_next.view(-1).to(self.device)

        # Predict and compute residual
        v_hat = self.model.predict(v_prev.unsqueeze(0), u_prev.unsqueeze(0)).squeeze(0)
        r = v_next - v_hat

        # Standardized residual per axis
        z = torch.abs(r) / self.sigma           # (d,)
        A_t = z.max().item()                     # max-z score

        # Binary anomaly flag
        a_t = 1 if A_t > self.tau_point else 0

        # Update ring buffer
        self._flag_buf[self._buf_ptr] = a_t
        self._score_buf[self._buf_ptr] = A_t
        self._buf_ptr = (self._buf_ptr + 1) % self.W
        self._buf_filled = min(self._buf_filled + 1, self.W)

        # Anomaly count in window
        C_t = self._flag_buf[:self._buf_filled].sum().item()

        # Degradation level from count
        level = 0
        if C_t >= self.C_levels[2]:
            level = 3
        elif C_t >= self.C_levels[1]:
            level = 2
        elif C_t >= self.C_levels[0]:
            level = 1

        return {
            "A_t": A_t,
            "a_t": a_t,
            "C_t": int(C_t),
            "level": level,
            "residual": r.detach().cpu(),
            "z": z.detach().cpu(),
        }

    def compute_thresholds(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
                           v_next: torch.Tensor,
                           ep_lengths: Optional[torch.Tensor] = None):
        """
        Compute empirical thresholds from nominal validation data.

        Args:
            v_prev, u_prev, v_next: (N, d) flat tensors
            ep_lengths: (num_episodes,) number of transitions per episode.
                If provided, sliding windows are computed within each episode
                (avoids cross-episode contamination). If None, falls back to
                global unfold (backward compatible).

        Sets:
          - tau_point: q99 of A_t distribution (single-step threshold)
          - C_levels: based on nominal anomaly rate p and Binomial distribution
        """
        v_prev = v_prev.to(self.device)
        u_prev = u_prev.to(self.device)
        v_next = v_next.to(self.device)

        # Compute all residuals
        residuals = self.model.residual(v_prev, u_prev, v_next)  # (N, d)

        # Per-axis standardized residuals
        z_all = torch.abs(residuals) / self.sigma.unsqueeze(0)   # (N, d)
        A_all = z_all.max(dim=-1).values                         # (N,)

        # Point threshold: q99 of A_t
        self.tau_point = float(torch.quantile(A_all, 0.99).item())

        # Compute nominal anomaly rate
        a_all = (A_all > self.tau_point).long()
        p_nominal = a_all.float().mean().item()  # should be ~0.01

        # Compute windowed anomaly counts on nominal data
        N = A_all.shape[0]
        C_parts = []

        if ep_lengths is not None and len(ep_lengths) > 0:
            # Episode-aware windowing: only slide within each episode
            offset = 0
            for ep_len in ep_lengths.tolist():
                ep_len = int(ep_len)
                if ep_len >= self.W:
                    a_ep = a_all[offset:offset + ep_len]
                    C_ep = a_ep.unfold(0, self.W, 1).sum(dim=-1)  # (ep_len - W + 1,)
                    C_parts.append(C_ep)
                offset += ep_len
        elif N >= self.W:
            # Fallback: global unfold (no episode info available)
            C_parts.append(a_all.unfold(0, self.W, 1).sum(dim=-1))

        if C_parts:
            C_all = torch.cat(C_parts, dim=0)
            # Set C_levels from empirical quantiles of C distribution
            c95 = int(torch.quantile(C_all.float(), 0.95).item()) + 1
            c99 = int(torch.quantile(C_all.float(), 0.99).item()) + 1
            c999 = int(torch.quantile(C_all.float(), 0.999).item()) + 1
            # Ensure monotonic and at least 2
            self.C_levels = [max(c95, 2), max(c99, c95 + 1), max(c999, c99 + 1)]
        else:
            self.C_levels = [4, 8, 14]

        stats = {
            "tau_point": self.tau_point,
            "p_nominal": p_nominal,
            "C_levels": self.C_levels,
            "A_mean": A_all.mean().item(),
            "A_std": A_all.std().item(),
            "A_q95": torch.quantile(A_all, 0.95).item(),
            "A_q99": torch.quantile(A_all, 0.99).item(),
            "A_q999": torch.quantile(A_all, 0.999).item(),
            "sigma": self.sigma.cpu().tolist(),
        }
        if C_parts:
            stats["C_mean"] = C_all.float().mean().item()
            stats["C_std"] = C_all.float().std().item()
            stats["C_num_windows"] = C_all.shape[0]
        return stats

    def reset(self):
        """Reset the ring buffer (e.g. at episode start)."""
        self._flag_buf.zero_()
        self._score_buf.zero_()
        self._buf_ptr = 0
        self._buf_filled = 0

    def save(self, path: str):
        torch.save({
            "tau_point": float(self.tau_point),
            "C_levels": [int(c) for c in self.C_levels],
            "sigma": self.sigma.cpu(),
            "W": int(self.W),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.tau_point = float(ckpt["tau_point"])
        self.C_levels = [int(c) for c in ckpt["C_levels"]]
        if "sigma" in ckpt:
            self.sigma = ckpt["sigma"].to(self.device)
        self.W = int(ckpt["W"])
        self._flag_buf = torch.zeros(self.W, dtype=torch.long, device=self.device)
        self._score_buf = torch.zeros(self.W, device=self.device)


class BatchDegradationDetector:
    """
    Vectorized detector for parallel environments (Isaac Sim eval/training).

    Maintains per-environment ring buffers of binary anomaly flags.
    """

    def __init__(self, model, num_envs: int, window_size: int = 20, device: str = "cpu"):
        self.model = model
        self.num_envs = num_envs
        self.W = window_size
        self.device = device

        Q = model.Q.to(device)
        self.sigma = torch.sqrt(Q.diag()).clamp(min=1e-8)  # (d,)

        # Ring buffers: (num_envs, W)
        self._flag_buf = torch.zeros(num_envs, window_size, dtype=torch.long, device=device)
        self._score_buf = torch.zeros(num_envs, window_size, device=device)
        self._buf_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._buf_filled = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.tau_point = float("inf")
        self.C_levels = [4, 8, 14]

    def step(self, v_prev: torch.Tensor, u_prev: torch.Tensor,
             v_next: torch.Tensor) -> dict:
        """
        Batch step for all environments.

        Args:
            v_prev: (num_envs, 3)
            u_prev: (num_envs, 3)
            v_next: (num_envs, 3)

        Returns:
            dict with A_t (num_envs,), a_t (num_envs,), C_t (num_envs,), level (num_envs,)
        """
        v_hat = self.model.predict(v_prev, u_prev)
        r = v_next - v_hat

        # Per-axis standardized residual
        z = torch.abs(r) / self.sigma.unsqueeze(0)          # (num_envs, d)
        A_t = z.max(dim=-1).values                           # (num_envs,)

        # Binary anomaly flags
        a_t = (A_t > self.tau_point).long()                  # (num_envs,)

        # Update ring buffers
        idx = torch.arange(self.num_envs, device=self.device)
        self._flag_buf[idx, self._buf_ptr] = a_t
        self._score_buf[idx, self._buf_ptr] = A_t
        self._buf_ptr = (self._buf_ptr + 1) % self.W
        self._buf_filled = torch.clamp(self._buf_filled + 1, max=self.W)

        # Anomaly count in window
        C_t = self._flag_buf.sum(dim=-1)                     # (num_envs,)

        # Degradation levels
        level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        level[C_t >= self.C_levels[0]] = 1
        level[C_t >= self.C_levels[1]] = 2
        level[C_t >= self.C_levels[2]] = 3

        return {
            "A_t": A_t,
            "a_t": a_t,
            "C_t": C_t,
            "level": level,
        }

    def reset_envs(self, env_ids: torch.Tensor):
        """Reset buffers for specific environments."""
        self._flag_buf[env_ids] = 0
        self._score_buf[env_ids] = 0.0
        self._buf_ptr[env_ids] = 0
        self._buf_filled[env_ids] = 0

    def set_thresholds(self, tau_point: float, C_levels: list):
        self.tau_point = tau_point
        self.C_levels = C_levels
