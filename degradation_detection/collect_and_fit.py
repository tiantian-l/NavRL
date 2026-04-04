"""
Collect nominal transition data from Isaac Sim training environment,
then fit both Linear and MLP transition models and compute detection thresholds.

Usage (inside Isaac Sim Python env):
    python collect_and_fit.py --checkpoint <path_to_policy.pt> --output_dir ./nominal_models

Or standalone fitting from a saved .pt data file:
    python collect_and_fit.py --data_file nominal_data.pt --output_dir ./nominal_models
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add parent directory so we can import transition_models
sys.path.insert(0, os.path.dirname(__file__))

from transition_models import LinearTransitionModel, MLPTransitionModel
from quadrotor_dynamics import QuadrotorODETransitionModel
from degradation_detector import DegradationDetector
from residual_model import HeteroscedasticMLP


def _print_threshold_stats(label: str, s: dict):
    """Pretty-print threshold computation results."""
    print(f"  --- {label} Threshold Report ---")
    print(f"  mu    = [{s['mu'][0]:.6f}, {s['mu'][1]:.6f}, {s['mu'][2]:.6f}]")
    print(f"  sigma = [{s['sigma'][0]:.6f}, {s['sigma'][1]:.6f}, {s['sigma'][2]:.6f}]")
    print(f"  alpha (target)     = {s['alpha']:.1e}")
    print(f"  tau   (theoretical)= {s['tau_theory']:.4f}")
    print(f"  p_actual           = {s['p_actual']:.6f}  (ratio to alpha: {s['p_ratio']:.2f}x)")
    if s['p_ratio'] > 3.0:
        print(f"  ⚠ Heavy-tail warning: actual FP rate is {s['p_ratio']:.1f}x higher than Gaussian prediction")
    print(f"  A_t  mean={s['A_mean']:.4f}  std={s['A_std']:.4f}  q99={s['A_q99']:.4f}")
    print(f"  C_levels (warn/degrade/severe) = {s['C_levels']}")
    if 'C_mean' in s:
        print(f"  C_t  mean={s['C_mean']:.4f}  std={s['C_std']:.4f}  ({s['C_num_windows']} windows)")


def fit_and_save(data_path: str, output_dir: str, device: str = "cpu",
                 mlp_epochs: int = 1000, mlp_hidden: int = 32, window_size: int = 20,
                 alpha: float = 1e-3, residual_model_type: str = "none",
                 residual_hidden: int = 64, residual_epochs: int = 500):
    """
    Load nominal data, fit both models, compute thresholds, save everything.

    Args:
        data_path: path to .pt file with keys 'v_prev', 'u_prev', 'v_next'
        output_dir: directory to save models and thresholds
        device: torch device
        mlp_epochs: training epochs for MLP
        mlp_hidden: MLP hidden layer width
        window_size: W for degradation detector sliding window
        alpha: single-step false-positive rate for theoretical threshold
        residual_model_type: 'none', 'hetero_mlp', or 'sparse_gp'
        residual_hidden: hidden dim for heteroscedastic MLP
        residual_epochs: training epochs for residual model
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading nominal data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    v_prev = data["v_prev"]  # (N, 3)
    u_prev = data["u_prev"]  # (N, 3)
    v_next = data["v_next"]  # (N, 3)
    ep_lengths = data.get("ep_lengths", None)  # (num_episodes,) or None
    N = v_prev.shape[0]
    print(f"  Loaded {N} transition samples, state_dim={v_prev.shape[1]}")
    if ep_lengths is not None:
        print(f"  Episode info: {len(ep_lengths)} episodes, lengths: min={ep_lengths.min().item()}, "
              f"max={ep_lengths.max().item()}, mean={ep_lengths.float().mean().item():.1f}")

    # --- Train / validation split ---
    if ep_lengths is not None and len(ep_lengths) > 1:
        # Split by episode to preserve temporal structure in val set
        num_eps = len(ep_lengths)
        ep_perm = torch.randperm(num_eps)
        ep_split = max(1, int(0.9 * num_eps))
        train_ep_ids = ep_perm[:ep_split]
        val_ep_ids = ep_perm[ep_split:]

        # Compute sample indices for each episode
        ep_offsets = torch.cat([torch.tensor([0]), ep_lengths.cumsum(0)])
        train_idx = torch.cat([torch.arange(ep_offsets[i], ep_offsets[i] + ep_lengths[i]) for i in train_ep_ids])
        val_idx = torch.cat([torch.arange(ep_offsets[i], ep_offsets[i] + ep_lengths[i]) for i in val_ep_ids])

        # Val episode lengths (for episode-aware threshold computation)
        val_ep_lengths = ep_lengths[val_ep_ids]
        print(f"  Episode split: {len(train_ep_ids)} train eps ({len(train_idx)} samples), "
              f"{len(val_ep_ids)} val eps ({len(val_idx)} samples)")
    else:
        # Fallback: random sample split (no episode info)
        perm = torch.randperm(N)
        split = int(0.9 * N)
        train_idx, val_idx = perm[:split], perm[split:]
        val_ep_lengths = None
        print(f"  No episode info — random sample split: {len(train_idx)} train, {len(val_idx)} val")

    # Build val data in episode order (contiguous per episode) for threshold computation
    val_v_prev = v_prev[val_idx]
    val_u_prev = u_prev[val_idx]
    val_v_next = v_next[val_idx]

    # ========== 1. Linear Model ==========
    print("\n===== Fitting Linear Transition Model =====")
    linear_model = LinearTransitionModel(state_dim=3, input_dim=3, device=device)
    result = linear_model.fit(v_prev[train_idx], u_prev[train_idx], v_next[train_idx])

    print(f"  A =\n{result['A'].numpy()}")
    print(f"  B =\n{result['B'].numpy()}")
    print(f"  Q diag = {result['Q'].diag().numpy()}")

    # Validation MSE
    with torch.no_grad():
        val_pred = linear_model.predict(val_v_prev.to(device), val_u_prev.to(device))
        val_mse = ((val_pred - val_v_next.to(device)) ** 2).mean().item()
    print(f"  Validation MSE = {val_mse:.6f}")

    linear_path = os.path.join(output_dir, "linear_model.pt")
    linear_model.save(linear_path)
    print(f"  Saved to {linear_path}")

    # Thresholds
    print("  Computing thresholds ...")
    linear_detector = DegradationDetector(linear_model, window_size=window_size, device=device)
    linear_stats = linear_detector.compute_thresholds(
        val_v_prev.to(device), val_u_prev.to(device), val_v_next.to(device),
        ep_lengths=val_ep_lengths, alpha=alpha,
    )
    _print_threshold_stats("Linear", linear_stats)

    linear_det_path = os.path.join(output_dir, "linear_detector.pt")
    linear_detector.save(linear_det_path)

    # ========== 2. MLP Model ==========
    print(f"\n===== Fitting MLP Transition Model (hidden={mlp_hidden}, epochs={mlp_epochs}) =====")
    mlp_model = MLPTransitionModel(state_dim=3, input_dim=3, hidden_dim=mlp_hidden, device=device)
    mlp_result = mlp_model.fit(
        v_prev[train_idx], u_prev[train_idx], v_next[train_idx],
        lr=1e-3, epochs=mlp_epochs, batch_size=4096, verbose=True,
    )
    print(f"  Q diag = {mlp_result['Q'].diag().cpu().numpy()}")

    # Validation MSE
    with torch.no_grad():
        val_pred_mlp = mlp_model.predict(val_v_prev.to(device), val_u_prev.to(device))
        val_mse_mlp = ((val_pred_mlp - val_v_next.to(device)) ** 2).mean().item()
    print(f"  Validation MSE = {val_mse_mlp:.6f}")
    print(f"  Improvement over linear: {(1 - val_mse_mlp / val_mse) * 100:.1f}%")

    mlp_path = os.path.join(output_dir, "mlp_model.pt")
    mlp_model.save(mlp_path)
    print(f"  Saved to {mlp_path}")

    # Thresholds
    print("  Computing thresholds ...")
    mlp_detector = DegradationDetector(mlp_model, window_size=window_size, device=device)
    mlp_stats = mlp_detector.compute_thresholds(
        val_v_prev.to(device), val_u_prev.to(device), val_v_next.to(device),
        ep_lengths=val_ep_lengths, alpha=alpha,
    )
    _print_threshold_stats("MLP (global sigma)", mlp_stats)

    mlp_det_path = os.path.join(output_dir, "mlp_detector.pt")
    mlp_detector.save(mlp_det_path)

    # ========== 3. Physics-Based ODE Model ==========
    print(f"\n===== Fitting Quadrotor ODE Transition Model (dt={1/62.5:.4f}) =====")
    ode_model = QuadrotorODETransitionModel(
        state_dim=3, input_dim=3, dt=1/62.5,
        mass=0.716, gravity=9.81,
        K_v=[2.2, 2.2, 2.2],
        tau_att=0.05, tau_thrust=0.03,
        drag=[0.1, 0.1, 0.1],
        num_substeps=4, device=device,
    )
    ode_result = ode_model.fit(
        v_prev[train_idx], u_prev[train_idx], v_next[train_idx],
        lr=5e-3, epochs=500, batch_size=4096, verbose=True,
    )
    print(f"  Q diag = {ode_result['Q'].diag().cpu().numpy()}")

    # Validation MSE
    with torch.no_grad():
        val_pred_ode = ode_model.predict(val_v_prev.to(device), val_u_prev.to(device))
        val_mse_ode = ((val_pred_ode - val_v_next.to(device)) ** 2).mean().item()
    print(f"  Validation MSE = {val_mse_ode:.6f}")
    print(f"  Improvement over linear: {(1 - val_mse_ode / val_mse) * 100:.1f}%")
    print(f"  Improvement over MLP:    {(1 - val_mse_ode / val_mse_mlp) * 100:.1f}%")

    ode_path = os.path.join(output_dir, "ode_model.pt")
    ode_model.save(ode_path)
    print(f"  Saved to {ode_path}")

    # Thresholds
    print("  Computing thresholds ...")
    ode_detector = DegradationDetector(ode_model, window_size=window_size, device=device)
    ode_stats = ode_detector.compute_thresholds(
        val_v_prev.to(device), val_u_prev.to(device), val_v_next.to(device),
        ep_lengths=val_ep_lengths, alpha=alpha,
    )
    _print_threshold_stats("ODE (physics-based)", ode_stats)

    ode_det_path = os.path.join(output_dir, "ode_detector.pt")
    ode_detector.save(ode_det_path)

    # ========== 4. Residual Model (input-dependent sigma) ==========
    res_model = None
    res_stats = None
    if residual_model_type != "none":
        print(f"\n===== Fitting Residual Model ({residual_model_type}) =====")

        if residual_model_type == "hetero_mlp":
            res_model = HeteroscedasticMLP(
                input_dim=6, output_dim=3, hidden_dim=residual_hidden, device=device
            )
        elif residual_model_type == "sparse_gp":
            from residual_model import SparseGPResidualModel
            res_model = SparseGPResidualModel(
                input_dim=6, output_dim=3, num_inducing=500, device=device
            )
        elif residual_model_type == "exact_gp":
            from residual_model import ExactGPResidualModel
            res_model = ExactGPResidualModel(
                input_dim=6, max_train_size=3000, device=device
            )
        else:
            raise ValueError(f"Unknown residual_model_type: {residual_model_type}")

        # Fit detector with residual model (fits model inside compute_thresholds)
        mlp_detector_rm = DegradationDetector(
            mlp_model, window_size=window_size, device=device,
            residual_model=res_model,
        )
        res_stats = mlp_detector_rm.compute_thresholds(
            val_v_prev.to(device), val_u_prev.to(device), val_v_next.to(device),
            ep_lengths=val_ep_lengths, alpha=alpha,
        )
        _print_threshold_stats(f"MLP + {residual_model_type}", res_stats)

        # Print GP diagnostics if applicable
        if hasattr(res_model, 'print_diagnostics'):
            res_model.print_diagnostics()

        # Save detector and residual model
        rm_det_path = os.path.join(output_dir, "mlp_detector_rm.pt")
        rm_model_path = os.path.join(output_dir, "residual_model.pt")
        mlp_detector_rm.save(rm_det_path, residual_model_path=rm_model_path)
        print(f"  Residual model saved to {rm_model_path}")
        print(f"  Detector (with RM) saved to {rm_det_path}")

    # ========== Summary ==========
    print("\n===== Comparison Summary =====")
    print(f"  {'Metric':<25} {'Linear':>12} {'MLP':>12} {'ODE':>12}")
    print(f"  {'Val MSE':<25} {val_mse:>12.6f} {val_mse_mlp:>12.6f} {val_mse_ode:>12.6f}")
    print(f"  {'Q trace':<25} {result['Q'].trace().item():>12.6f} {mlp_result['Q'].trace().item():>12.6f} {ode_result['Q'].trace().item():>12.6f}")
    print(f"  {'tau (theory)':<25} {linear_stats['tau_theory']:>12.4f} {mlp_stats['tau_theory']:>12.4f} {ode_stats['tau_theory']:>12.4f}")
    print(f"  {'p_actual':<25} {linear_stats['p_actual']:>12.6f} {mlp_stats['p_actual']:>12.6f} {ode_stats['p_actual']:>12.6f}")
    print(f"  {'p_ratio (actual/alpha)':<25} {linear_stats['p_ratio']:>12.2f} {mlp_stats['p_ratio']:>12.2f} {ode_stats['p_ratio']:>12.2f}")
    print(f"  {'A_t q99 (empirical)':<25} {linear_stats['A_q99']:>12.4f} {mlp_stats['A_q99']:>12.4f} {ode_stats['A_q99']:>12.4f}")
    linear_cl = linear_stats['C_levels']
    mlp_cl = mlp_stats['C_levels']
    ode_cl = ode_stats['C_levels']
    print(f"  {'C_levels':<25} {str(linear_cl):>12} {str(mlp_cl):>12} {str(ode_cl):>12}")

    # Print fitted physical parameters
    print(f"\n  --- ODE Model Physical Parameters ---")
    ode_model._print_params()

    if res_stats is not None:
        print(f"\n  --- With {residual_model_type} residual model ---")
        print(f"  {'p_actual (RM)':<25} {res_stats['p_actual']:>12.6f}")
        print(f"  {'p_ratio (RM)':<25} {res_stats['p_ratio']:>12.2f}")
        print(f"  {'A_t q99 (RM)':<25} {res_stats['A_q99']:>12.4f}")
        print(f"  {'C_levels (RM)':<25} {str(res_stats['C_levels']):>12}")
        if 'C_mean' in res_stats:
            print(f"  {'C_mean/std (RM)':<25} {res_stats['C_mean']:>6.4f} / {res_stats['C_std']:>6.4f}")

    print(f"\nAll models saved to {output_dir}/")
    return {
        "linear": {"model": linear_model, "stats": linear_stats, "val_mse": val_mse},
        "mlp": {"model": mlp_model, "stats": mlp_stats, "val_mse": val_mse_mlp},
        "ode": {"model": ode_model, "stats": ode_stats, "val_mse": val_mse_ode},
        "residual_model": res_model,
        "residual_stats": res_stats,
    }


def collect_from_env_buffers(env, num_steps: int, output_path: str):
    """
    Collect nominal transition data from the Isaac Sim env's existing ring buffers.

    Call this AFTER running the policy for num_steps in the env.
    Extracts (v_{t-1}, u_{t-1}, v_t) triples from the env's vel_cmd_buf and vel_real_buf.

    Args:
        env: the NavigationEnv instance (must have _vel_cmd_buf, _vel_real_buf populated)
        num_steps: how many total steps have been run
        output_path: where to save the .pt file
    """
    vel_cmd = env._vel_cmd_buf.detach().cpu()    # (num_envs, W, 3)
    vel_real = env._vel_real_buf.detach().cpu()   # (num_envs, W, 3)
    filled = env._buf_filled.detach().cpu()       # (num_envs,)

    all_v_prev = []
    all_u_prev = []
    all_v_next = []

    for i in range(env.num_envs):
        n = filled[i].item()
        if n < 2:
            continue
        # The ring buffer may wrap; read in order from oldest to newest
        ptr = env._buf_ptr[i].item()
        if n < env._uc_window:
            # Buffer not full yet, data is at indices [0, n)
            indices = list(range(n))
        else:
            # Buffer full, oldest is at ptr, newest at ptr-1
            indices = [(ptr + j) % env._uc_window for j in range(n)]

        v_real_ordered = vel_real[i][indices]   # (n, 3)
        v_cmd_ordered = vel_cmd[i][indices]     # (n, 3)

        # Construct transition tuples: (v_{t-1}, u_{t-1}, v_t)
        all_v_prev.append(v_real_ordered[:-1])
        all_u_prev.append(v_cmd_ordered[:-1])
        all_v_next.append(v_real_ordered[1:])

    v_prev = torch.cat(all_v_prev, dim=0)
    u_prev = torch.cat(all_u_prev, dim=0)
    v_next = torch.cat(all_v_next, dim=0)

    print(f"Collected {v_prev.shape[0]} transition samples from {env.num_envs} envs")
    torch.save({"v_prev": v_prev, "u_prev": u_prev, "v_next": v_next}, output_path)
    print(f"Saved to {output_path}")
    return v_prev, u_prev, v_next


def collect_extended(env, policy, collector, num_frames: int, output_path: str):
    """
    Run the policy in the environment for num_frames steps and collect ALL transitions
    (not limited by ring buffer size).

    Args:
        env: NavigationEnv (the base unwrapped env)
        policy: the PPO policy module
        collector: the SyncDataCollector
        num_frames: total frames to collect
        output_path: where to save
    """
    all_v_prev = []
    all_u_prev = []
    all_v_next = []

    prev_vel = None
    prev_cmd = None
    collected = 0

    for i, data in enumerate(collector):
        # data contains per-step info
        v_cmd = data.get(("info", "vel_cmd"))  # (num_envs, 1, 3) or (batch, 3)
        drone_state = data.get(("info", "drone_state"))  # (num_envs, 1, 13)

        if v_cmd is None or drone_state is None:
            continue

        # Current velocity from drone state (indices 7:10 are linear vel in world frame)
        v_real = drone_state[..., 7:10].reshape(-1, 3).detach().cpu()
        v_cmd_flat = v_cmd.reshape(-1, 3).detach().cpu()

        if prev_vel is not None:
            all_v_prev.append(prev_vel)
            all_u_prev.append(prev_cmd)
            all_v_next.append(v_real)
            collected += v_real.shape[0]

        prev_vel = v_real
        prev_cmd = v_cmd_flat

        if collected >= num_frames:
            break

    v_prev = torch.cat(all_v_prev, dim=0)
    u_prev = torch.cat(all_u_prev, dim=0)
    v_next = torch.cat(all_v_next, dim=0)

    print(f"Collected {v_prev.shape[0]} transition samples")
    torch.save({"v_prev": v_prev, "u_prev": u_prev, "v_next": v_next}, output_path)
    print(f"Saved to {output_path}")
    return v_prev, u_prev, v_next


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit nominal transition models")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to nominal_data.pt with keys v_prev, u_prev, v_next")
    parser.add_argument("--output_dir", type=str, default="./nominal_models",
                        help="Directory to save fitted models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mlp_epochs", type=int, default=1000)
    parser.add_argument("--mlp_hidden", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=20,
                        help="Sliding window W for degradation score")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Single-step false-positive rate (default: 1e-3, tau≈3.4)")
    parser.add_argument("--residual_model", type=str, default="none",
                        choices=["none", "hetero_mlp", "sparse_gp", "exact_gp"],
                        help="Type of input-dependent residual model (default: none)")
    parser.add_argument("--residual_hidden", type=int, default=64,
                        help="Hidden dim for hetero_mlp residual model")
    parser.add_argument("--residual_epochs", type=int, default=500,
                        help="Training epochs for residual model")

    args = parser.parse_args()
    fit_and_save(
        data_path=args.data_file,
        output_dir=args.output_dir,
        device=args.device,
        mlp_epochs=args.mlp_epochs,
        mlp_hidden=args.mlp_hidden,
        window_size=args.window_size,
        alpha=args.alpha,
        residual_model_type=args.residual_model,
        residual_hidden=args.residual_hidden,
        residual_epochs=args.residual_epochs,
    )
