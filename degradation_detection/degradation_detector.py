"""
Online Degradation Detector using Max-Z anomaly score + count-based persistence.

Method:
  Step 1 — Single-point anomaly detection:
    r_t = v_t - f(v_{t-1}, u_{t-1})          residual
    z_i = |r_{t,i} - mu_i| / sigma_i         per-axis standardized residual
    A_t = max(z_x, z_y, z_z)                  anomaly score (worst-axis deviation)
    a_t = 1 if A_t > tau_point else 0         binary anomaly flag

  Step 2 — Temporal persistence detection:
    C_t = sum(a_{t-W+1}, ..., a_t)            anomaly count in window W
    level = f(C_t)                            degradation level (0-3)

Thresholds:
  tau_point — theoretical: P(A_t > tau | H0) = alpha, under Gaussian assumption
              tau = Phi^{-1}(1 - alpha/6)  (Bonferroni for d=3 axes)
              Validated against empirical false-positive rate on nominal data.
  C_levels  — from Binomial(W, p) survival function, where p = actual anomaly rate.

Optional: input-dependent residual model
  If a residual_model (HeteroscedasticMLP or SparseGPResidualModel) is provided,
  the detector uses locally-adaptive z-scores:
    mu(x_t), sigma(x_t) = residual_model.predict(x_t)
    z_i = |r_{t,i} - mu_i(x_t)| / sigma_i(x_t)
  This resolves heavy-tail issues caused by heteroscedastic residuals.
"""

import torch
import numpy as np
from typing import Optional
from scipy import stats as sp_stats
from transition_models import LinearTransitionModel, MLPTransitionModel


# ──────────────────────────────────────────────────────────────
#  Utility: theoretical thresholds
# ──────────────────────────────────────────────────────────────

def compute_tau_theoretical(alpha: float, d: int = 3) -> float:
    """
    Compute tau such that P(max_i |z_i| > tau) = alpha
    under z_i ~ N(0,1) independent (Bonferroni bound).

    tau = Phi^{-1}(1 - alpha / (2*d))

    This guarantees P(false positive) <= alpha.
    """
    return float(sp_stats.norm.ppf(1.0 - alpha / (2.0 * d)))


def compute_C_levels_binomial(W: int, p: float,
                              beta_levels: tuple = (1e-2, 1e-3, 1e-4)) -> list:
    """
    Compute C thresholds from Binomial(W, p) survival function.

    C_level_k = min{c : P(C >= c | Binom(W, p)) <= beta_k}

    Args:
        W: window size
        p: per-step anomaly probability under H0
        beta_levels: false-alarm probabilities for level 1, 2, 3
    Returns:
        [c1, c2, c3] thresholds
    """
    levels = []
    for beta in beta_levels:
        # Find smallest c such that P(C >= c) <= beta
        # P(C >= c) = 1 - CDF(c-1) = sf(c-1)
        for c in range(1, W + 1):
            if sp_stats.binom.sf(c - 1, W, p) <= beta:
                levels.append(c)
                break
        else:
            levels.append(W)
    # Ensure monotonic increasing and at least 2
    for i in range(len(levels)):
        levels[i] = max(levels[i], 2)
    for i in range(1, len(levels)):
        levels[i] = max(levels[i], levels[i - 1] + 1)
    return levels


class DegradationDetector:
    """
    Online degradation detector (single environment).

    Thresholds are set via theoretical Gaussian + Binomial model,
    then validated on nominal data (empirical false-positive rate).

    If residual_model is provided, uses input-dependent mu(x) and sigma(x)
    for locally-adaptive z-scores. Otherwise falls back to global statistics.

    Usage:
        detector = DegradationDetector(model, window_size=20)
        detector.compute_thresholds(nominal_data, alpha=1e-3)
        ...
        for each control step:
            result = detector.step(v_prev, u_prev, v_next)
    """

    def __init__(self, model, window_size: int = 20, device: str = "cpu",
                 residual_model=None):
        self.model = model
        self.W = window_size
        self.device = device
        self.d = 3  # state dimension
        self.residual_model = residual_model  # optional input-dependent model

        # Per-axis residual stats from model's Q diagonal (global fallback)
        Q = model.Q.to(device)
        self.sigma = torch.sqrt(Q.diag()).clamp(min=1e-8)  # (d,)
        self.mu = torch.zeros(self.d, device=device)        # residual mean (estimated offline)

        # Ring buffer for binary anomaly flags
        self._flag_buf = torch.zeros(window_size, dtype=torch.long, device=device)
        self._score_buf = torch.zeros(window_size, device=device)
        self._buf_ptr = 0
        self._buf_filled = 0

        # Thresholds (set by compute_thresholds)
        self.tau_point = float("inf")
        self.C_levels = [4, 8, 14]
        self.alpha = 1e-3  # nominal single-step false-positive rate

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
        if self.residual_model is not None:
            # Input-dependent: mu(x), sigma(x) from residual model
            x_t = torch.cat([v_prev, u_prev], dim=-1)  # (6,)
            mu_local, sigma_local = self.residual_model.predict(x_t)
            z = torch.abs(r - mu_local) / sigma_local   # (d,)
        else:
            # Global fallback
            z = torch.abs(r - self.mu) / self.sigma      # (d,)
        A_t = z.max().item()                              # max-z score

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
                           ep_lengths: Optional[torch.Tensor] = None,
                           alpha: float = 1e-3):
        """
        Compute thresholds using theoretical Gaussian model + empirical validation.

        Step 1: Estimate mu (residual mean) and sigma from data.
        Step 2: tau_point = Phi^{-1}(1 - alpha/(2d))  (theoretical, Bonferroni).
        Step 3: Validate on data — compute actual false-positive rate p_actual.
        Step 4: C_levels from Binomial(W, p_actual) survival function.

        Args:
            v_prev, u_prev, v_next: (N, d) nominal validation data
            ep_lengths: (num_episodes,) for episode-aware C validation
            alpha: desired single-step false-positive rate (default 1e-3)
        """
        self.alpha = alpha
        v_prev = v_prev.to(self.device)
        u_prev = u_prev.to(self.device)
        v_next = v_next.to(self.device)

        # --- Step 1: Estimate residual statistics ---
        residuals = self.model.residual(v_prev, u_prev, v_next)  # (N, d)
        self.mu = residuals.mean(dim=0)                           # (d,)
        self.sigma = residuals.std(dim=0).clamp(min=1e-8)         # (d,)

        # --- Step 1b: Fit residual model if provided ---
        if self.residual_model is not None:
            x_all = torch.cat([v_prev, u_prev], dim=-1)  # (N, 6)
            rm_stats = self.residual_model.fit(x_all, residuals)
            print(f"\n[ResidualModel] Fitting complete:")
            for k, v in rm_stats.items():
                print(f"  {k}: {v}")

        # --- Step 2: Theoretical tau_point ---
        tau_theory = compute_tau_theoretical(alpha, d=self.d)
        self.tau_point = tau_theory

        # --- Step 3: Empirical validation ---
        if self.residual_model is not None:
            x_all = torch.cat([v_prev, u_prev], dim=-1)
            mu_local, sigma_local = self.residual_model.predict(x_all)
            z_all = torch.abs(residuals - mu_local) / sigma_local
        else:
            z_all = torch.abs(residuals - self.mu.unsqueeze(0)) / self.sigma.unsqueeze(0)
        A_all = z_all.max(dim=-1).values  # (N,)

        a_all = (A_all > self.tau_point).long()
        p_actual = a_all.float().mean().item()

        # --- Step 4: C_levels from Binomial(W, p_actual) ---
        p_for_binom = max(p_actual, 1e-6)  # avoid degenerate case
        self.C_levels = compute_C_levels_binomial(
            self.W, p_for_binom, beta_levels=(1e-2, 1e-3, 1e-4)
        )

        # --- Empirical C statistics for reporting ---
        N = A_all.shape[0]
        C_parts = []
        if ep_lengths is not None and len(ep_lengths) > 0:
            offset = 0
            for ep_len in ep_lengths.tolist():
                ep_len = int(ep_len)
                if ep_len >= self.W:
                    a_ep = a_all[offset:offset + ep_len]
                    C_ep = a_ep.unfold(0, self.W, 1).sum(dim=-1)
                    C_parts.append(C_ep)
                offset += ep_len
        elif N >= self.W:
            C_parts.append(a_all.unfold(0, self.W, 1).sum(dim=-1))

        stats = {
            "alpha": alpha,
            "tau_theory": tau_theory,
            "tau_point": self.tau_point,
            "p_actual": p_actual,
            "p_ratio": p_actual / alpha if alpha > 0 else float("inf"),
            "C_levels": self.C_levels,
            "mu": self.mu.cpu().tolist(),
            "sigma": self.sigma.cpu().tolist(),
            "A_mean": A_all.mean().item(),
            "A_std": A_all.std().item(),
            "A_q95": torch.quantile(A_all, 0.95).item(),
            "A_q99": torch.quantile(A_all, 0.99).item(),
            "A_q999": torch.quantile(A_all, 0.999).item(),
        }
        if C_parts:
            C_all = torch.cat(C_parts, dim=0)
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

    def save(self, path: str, residual_model_path: Optional[str] = None):
        save_dict = {
            "tau_point": float(self.tau_point),
            "alpha": float(self.alpha),
            "C_levels": [int(c) for c in self.C_levels],
            "mu": self.mu.cpu(),
            "sigma": self.sigma.cpu(),
            "W": int(self.W),
            "has_residual_model": self.residual_model is not None,
        }
        torch.save(save_dict, path)
        # Save residual model separately
        if self.residual_model is not None and residual_model_path is not None:
            self.residual_model.save(residual_model_path)

    def load(self, path: str, residual_model_path: Optional[str] = None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.tau_point = float(ckpt["tau_point"])
        self.alpha = float(ckpt.get("alpha", 1e-3))
        self.C_levels = [int(c) for c in ckpt["C_levels"]]
        if "mu" in ckpt:
            self.mu = ckpt["mu"].to(self.device)
        if "sigma" in ckpt:
            self.sigma = ckpt["sigma"].to(self.device)
        self.W = int(ckpt["W"])
        self._flag_buf = torch.zeros(self.W, dtype=torch.long, device=self.device)
        self._score_buf = torch.zeros(self.W, device=self.device)
        # Load residual model if saved and path provided
        if ckpt.get("has_residual_model", False) and residual_model_path is not None:
            if self.residual_model is not None:
                self.residual_model.load(residual_model_path)


class BatchDegradationDetector:
    """
    Vectorized detector for parallel environments (Isaac Sim eval/training).

    Maintains per-environment ring buffers of binary anomaly flags.
    """

    def __init__(self, model, num_envs: int, window_size: int = 20, device: str = "cpu",
                 residual_model=None):
        self.model = model
        self.num_envs = num_envs
        self.W = window_size
        self.device = device
        self.d = 3
        self.residual_model = residual_model  # optional input-dependent model

        Q = model.Q.to(device)
        self.sigma = torch.sqrt(Q.diag()).clamp(min=1e-8)  # (d,)
        self.mu = torch.zeros(self.d, device=device)

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
        if self.residual_model is not None:
            x_t = torch.cat([v_prev, u_prev], dim=-1)  # (num_envs, 6)
            mu_local, sigma_local = self.residual_model.predict(x_t)
            z = torch.abs(r - mu_local) / sigma_local   # (num_envs, d)
        else:
            z = torch.abs(r - self.mu.unsqueeze(0)) / self.sigma.unsqueeze(0)  # (num_envs, d)
        A_t = z.max(dim=-1).values                                              # (num_envs,)

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

    def set_thresholds(self, tau_point: float, C_levels: list,
                       mu: Optional[torch.Tensor] = None,
                       sigma: Optional[torch.Tensor] = None,
                       residual_model=None):
        self.tau_point = tau_point
        self.C_levels = C_levels
        if mu is not None:
            self.mu = mu.to(self.device)
        if sigma is not None:
            self.sigma = sigma.to(self.device)
        if residual_model is not None:
            self.residual_model = residual_model
