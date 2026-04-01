"""
Integration helper: add Max-Z degradation detection to NavigationEnv.

Method: Max-Z anomaly score + count-based persistence.
  A_t = max(|r_i|/sigma_i)  →  a_t = 1{A_t > tau}  →  C_t = count in window

Usage in env.py:
    from degradation_integration import (
        init_degradation_detector, update_degradation_scores,
        reset_degradation, get_degradation_episode_stats
    )

In NavigationEnv.__init__:
    init_degradation_detector(self, model_dir="./nominal_models")

In _compute_state_and_obs (after velocity data available):
    update_degradation_scores(self)

In _reset_idx:
    reset_degradation(self, env_ids)
"""

import os
import sys
import torch

_dd_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "degradation_detection")
if _dd_dir not in sys.path:
    sys.path.insert(0, _dd_dir)

from transition_models import LinearTransitionModel, MLPTransitionModel
from degradation_detector import BatchDegradationDetector


def init_degradation_detector(env, model_dir: str = None, model_type: str = "both",
                              window_size: int = 20):
    """
    Initialize degradation detectors on the env object.

    Args:
        env: NavigationEnv instance
        model_dir: directory containing model and detector .pt files
        model_type: "linear", "mlp", or "both"
        window_size: W for sliding window
    """
    env._deg_window = window_size
    env._deg_enabled = model_dir is not None and os.path.isdir(model_dir)
    env._deg_model_type = model_type

    # Previous velocity/command buffers
    env._prev_vel = torch.zeros(env.num_envs, 3, device=env.device)
    env._prev_cmd = torch.zeros(env.num_envs, 3, device=env.device)
    env._deg_step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Per-env episode history for logging
    env._deg_A_history = {name: [[] for _ in range(env.num_envs)]
                          for name in (["linear", "mlp"] if model_type == "both"
                                       else [model_type])}
    env._deg_C_history = {name: [[] for _ in range(env.num_envs)]
                          for name in env._deg_A_history}

    env._deg_detectors = {}

    if not env._deg_enabled:
        print("[DegDetector] No model_dir provided or not found. Detector disabled.")
        return

    # Load linear model
    if model_type in ("linear", "both"):
        linear_path = os.path.join(model_dir, "linear_model.pt")
        if os.path.exists(linear_path):
            linear_model = LinearTransitionModel(device=env.device)
            linear_model.load(linear_path)
            det = BatchDegradationDetector(linear_model, env.num_envs, window_size, env.device)
            det_path = os.path.join(model_dir, "linear_detector.pt")
            if os.path.exists(det_path):
                ckpt = torch.load(det_path, map_location=env.device, weights_only=True)
                det.set_thresholds(
                    tau_point=float(ckpt["tau_point"]),
                    C_levels=[int(c) for c in ckpt["C_levels"]],
                )
            env._deg_detectors["linear"] = det
            print(f"[DegDetector] Linear: tau={det.tau_point:.4f}, C_levels={det.C_levels}")

    # Load MLP model
    if model_type in ("mlp", "both"):
        mlp_path = os.path.join(model_dir, "mlp_model.pt")
        if os.path.exists(mlp_path):
            mlp_model = MLPTransitionModel(device=env.device)
            mlp_model.load(mlp_path)
            det = BatchDegradationDetector(mlp_model, env.num_envs, window_size, env.device)
            det_path = os.path.join(model_dir, "mlp_detector.pt")
            if os.path.exists(det_path):
                ckpt = torch.load(det_path, map_location=env.device, weights_only=True)
                det.set_thresholds(
                    tau_point=float(ckpt["tau_point"]),
                    C_levels=[int(c) for c in ckpt["C_levels"]],
                )
            env._deg_detectors["mlp"] = det
            print(f"[DegDetector] MLP: tau={det.tau_point:.4f}, C_levels={det.C_levels}")


def update_degradation_scores(env):
    """
    Call each step after velocity data is available.

    Returns dict of results per model type, each with:
      A_t (num_envs,), a_t (num_envs,), C_t (num_envs,), level (num_envs,)
    """
    v_real = env.drone.vel_w[..., :3].squeeze(1).detach()  # (num_envs, 3)
    v_cmd = env.vel_cmd.squeeze(1).detach()                # (num_envs, 3)

    results = {}
    valid_mask = env._deg_step_count >= 1

    if env._deg_enabled and valid_mask.any():
        for name, det in env._deg_detectors.items():
            out = det.step(env._prev_vel, env._prev_cmd, v_real)
            results[name] = out

            # Accumulate per-env history
            for i in range(env.num_envs):
                if valid_mask[i]:
                    env._deg_A_history[name][i].append(out["A_t"][i].item())
                    env._deg_C_history[name][i].append(out["C_t"][i].item())

    # Update buffers for next step
    env._prev_vel[:] = v_real
    env._prev_cmd[:] = v_cmd
    env._deg_step_count += 1

    return results


def reset_degradation(env, env_ids: torch.Tensor):
    """Reset detector buffers for reset environments."""
    env._prev_vel[env_ids] = 0.0
    env._prev_cmd[env_ids] = 0.0
    env._deg_step_count[env_ids] = 0

    for name in env._deg_A_history:
        for eid in env_ids.tolist():
            env._deg_A_history[name][eid] = []
            env._deg_C_history[name][eid] = []

    if env._deg_enabled:
        for det in env._deg_detectors.values():
            det.reset_envs(env_ids)


def get_degradation_episode_stats(env) -> dict:
    """
    Compute episode-level degradation stats for logging.
    Call at terminal steps (before reset).

    Returns dict with keys like 'deg_mlp_A_mean', 'deg_mlp_anomaly_rate', 'deg_mlp_max_C'.
    """
    stats = {}
    for name in env._deg_A_history:
        all_A_means = []
        all_anomaly_rates = []
        all_max_C = []
        for i in range(env.num_envs):
            A_hist = env._deg_A_history[name][i]
            C_hist = env._deg_C_history[name][i]
            if len(A_hist) > 0:
                t_A = torch.tensor(A_hist)
                all_A_means.append(t_A.mean().item())
                if hasattr(env._deg_detectors.get(name, None), 'tau_point'):
                    tau = env._deg_detectors[name].tau_point
                    all_anomaly_rates.append((t_A > tau).float().mean().item())
                if len(C_hist) > 0:
                    all_max_C.append(max(C_hist))
        if all_A_means:
            stats[f"deg_{name}_A_mean"] = np.mean(all_A_means)
        if all_anomaly_rates:
            stats[f"deg_{name}_anomaly_rate"] = np.mean(all_anomaly_rates)
        if all_max_C:
            stats[f"deg_{name}_max_C"] = np.mean(all_max_C)
    return stats


# Need numpy for stats
import numpy as np
