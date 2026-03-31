"""
Integration helper: add Markov transition degradation detection to NavigationEnv.

Usage in env.py:
    from degradation_integration import DegradationEnvMixin

Then in NavigationEnv.__init__, after the Uc buffer setup:
    self._init_degradation_detector()

In _compute_state_and_obs, after the Uc(t) block:
    self._update_degradation_scores()

In _reset_idx:
    self._reset_degradation(env_ids)

This module adds D_inst, D_window stats alongside the existing Uc_mean, U95, R_exc.
"""

import os
import sys
import torch

# Ensure degradation_detection package is importable
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
        env: NavigationEnv instance (self)
        model_dir: directory containing linear_model.pt, mlp_model.pt, *_detector.pt
                   If None, detectors run with identity model (for data collection phase).
        model_type: "linear", "mlp", or "both"
        window_size: W for sliding window
    """
    env._deg_window = window_size
    env._deg_enabled = model_dir is not None and os.path.isdir(model_dir)
    env._deg_model_type = model_type

    # Previous velocity buffer for transition: v_{t-1}
    env._prev_vel = torch.zeros(env.num_envs, 3, device=env.device)
    env._prev_cmd = torch.zeros(env.num_envs, 3, device=env.device)
    env._deg_step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # History accumulators for episode stats
    env._deg_D_history_linear = [[] for _ in range(env.num_envs)]
    env._deg_D_history_mlp = [[] for _ in range(env.num_envs)]

    env._deg_detectors = {}

    if not env._deg_enabled:
        print("[DegDetector] No model_dir provided or not found. Detector disabled (data collection mode).")
        return

    # Load linear model
    if model_type in ("linear", "both"):
        linear_path = os.path.join(model_dir, "linear_model.pt")
        if os.path.exists(linear_path):
            linear_model = LinearTransitionModel(device=env.device)
            linear_model.load(linear_path)
            det = BatchDegradationDetector(linear_model, env.num_envs, window_size, env.device)
            # Load thresholds
            det_path = os.path.join(model_dir, "linear_detector.pt")
            if os.path.exists(det_path):
                ckpt = torch.load(det_path, map_location=env.device, weights_only=True)
                det.set_thresholds(ckpt["thresholds"])
            env._deg_detectors["linear"] = det
            print(f"[DegDetector] Linear model loaded. Thresholds: {det.thresholds}")

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
                det.set_thresholds(ckpt["thresholds"])
            env._deg_detectors["mlp"] = det
            print(f"[DegDetector] MLP model loaded. Thresholds: {det.thresholds}")


def update_degradation_scores(env):
    """
    Call each step after velocity data is available.
    Uses env.vel_cmd (current command) and env.drone.vel_w (current real velocity).

    Must be called AFTER _pre_sim_step has set vel_cmd for this step.
    """
    if not env._deg_enabled:
        # Still update prev buffers for data collection
        v_real = env.drone.vel_w[..., :3].squeeze(1).detach()  # (num_envs, 3)
        v_cmd = env.vel_cmd.squeeze(1).detach()                # (num_envs, 3)
        env._prev_vel[:] = v_real
        env._prev_cmd[:] = v_cmd
        env._deg_step_count += 1
        return {}

    v_real = env.drone.vel_w[..., :3].squeeze(1).detach()  # (num_envs, 3)
    v_cmd = env.vel_cmd.squeeze(1).detach()                # (num_envs, 3)

    results = {}
    # Only compute for envs that have at least 1 previous step
    valid_mask = env._deg_step_count >= 1

    if valid_mask.any():
        for name, det in env._deg_detectors.items():
            # Use previous vel and cmd as (s_{t-1}, u_{t-1}), current vel as s_t
            out = det.step(
                env._prev_vel,
                env._prev_cmd,
                v_real,
            )
            results[name] = out

            # Accumulate per-env history (only for valid envs)
            for i in range(env.num_envs):
                if valid_mask[i]:
                    hist = env._deg_D_history_linear if name == "linear" else env._deg_D_history_mlp
                    hist[i].append(out["D_window"][i].item())

    # Update previous step buffers
    env._prev_vel[:] = v_real
    env._prev_cmd[:] = v_cmd
    env._deg_step_count += 1

    return results


def reset_degradation(env, env_ids: torch.Tensor):
    """Reset detector buffers for reset environments."""
    env._prev_vel[env_ids] = 0.0
    env._prev_cmd[env_ids] = 0.0
    env._deg_step_count[env_ids] = 0

    for eid in env_ids.tolist():
        env._deg_D_history_linear[eid] = []
        env._deg_D_history_mlp[eid] = []

    if env._deg_enabled:
        for det in env._deg_detectors.values():
            det.reset_envs(env_ids)


def get_degradation_episode_stats(env) -> dict:
    """
    Compute episode-level degradation stats for logging.
    Call at terminal steps (before reset).

    Returns dict with keys like 'D_linear_mean', 'D_linear_q95', 'D_mlp_mean', 'D_mlp_q95'.
    """
    stats = {}
    for name, hist_list in [("linear", env._deg_D_history_linear),
                            ("mlp", env._deg_D_history_mlp)]:
        for i in range(env.num_envs):
            if len(hist_list[i]) > 0:
                t = torch.tensor(hist_list[i])
                stats.setdefault(f"D_{name}_mean", torch.zeros(env.num_envs))
                stats.setdefault(f"D_{name}_q95", torch.zeros(env.num_envs))
                stats[f"D_{name}_mean"][i] = t.mean()
                stats[f"D_{name}_q95"][i] = torch.quantile(t, 0.95) if len(hist_list[i]) >= 20 else t.max()
    return stats
