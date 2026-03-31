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
from degradation_detector import DegradationDetector


def fit_and_save(data_path: str, output_dir: str, device: str = "cpu",
                 mlp_epochs: int = 1000, mlp_hidden: int = 32, window_size: int = 20):
    """
    Load nominal data, fit both models, compute thresholds, save everything.

    Args:
        data_path: path to .pt file with keys 'v_prev', 'u_prev', 'v_next'
        output_dir: directory to save models and thresholds
        device: torch device
        mlp_epochs: training epochs for MLP
        mlp_hidden: MLP hidden layer width
        window_size: W for degradation detector sliding window
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading nominal data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    v_prev = data["v_prev"]  # (N, 3)
    u_prev = data["u_prev"]  # (N, 3)
    v_next = data["v_next"]  # (N, 3)
    N = v_prev.shape[0]
    print(f"  Loaded {N} transition samples, state_dim={v_prev.shape[1]}")

    # Shuffle and split: 90% train, 10% validation
    perm = torch.randperm(N)
    split = int(0.9 * N)
    train_idx, val_idx = perm[:split], perm[split:]

    # ========== 1. Linear Model ==========
    print("\n===== Fitting Linear Transition Model =====")
    linear_model = LinearTransitionModel(state_dim=3, input_dim=3, device=device)
    result = linear_model.fit(v_prev[train_idx], u_prev[train_idx], v_next[train_idx])

    print(f"  A =\n{result['A'].numpy()}")
    print(f"  B =\n{result['B'].numpy()}")
    print(f"  Q diag = {result['Q'].diag().numpy()}")

    # Validation MSE
    with torch.no_grad():
        val_pred = linear_model.predict(v_prev[val_idx].to(device), u_prev[val_idx].to(device))
        val_mse = ((val_pred - v_next[val_idx].to(device)) ** 2).mean().item()
    print(f"  Validation MSE = {val_mse:.6f}")

    linear_path = os.path.join(output_dir, "linear_model.pt")
    linear_model.save(linear_path)
    print(f"  Saved to {linear_path}")

    # Thresholds
    print("  Computing thresholds ...")
    linear_detector = DegradationDetector(linear_model, window_size=window_size, device=device)
    linear_stats = linear_detector.compute_thresholds(
        v_prev[val_idx].to(device), u_prev[val_idx].to(device), v_next[val_idx].to(device)
    )
    print(f"  D_inst  mean={linear_stats['D_inst_mean']:.4f}  std={linear_stats['D_inst_std']:.4f}")
    print(f"  D_window mean={linear_stats['D_window_mean']:.4f}  std={linear_stats['D_window_std']:.4f}")
    print(f"  Thresholds: q95={linear_stats['q95']:.4f}  q99={linear_stats['q99']:.4f}  q999={linear_stats['q999']:.4f}")
    print(f"  Chi2 reference: 95%={linear_stats['chi2_95']:.4f}  99%={linear_stats['chi2_99']:.4f}")

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
        val_pred_mlp = mlp_model.predict(v_prev[val_idx].to(device), u_prev[val_idx].to(device))
        val_mse_mlp = ((val_pred_mlp - v_next[val_idx].to(device)) ** 2).mean().item()
    print(f"  Validation MSE = {val_mse_mlp:.6f}")
    print(f"  Improvement over linear: {(1 - val_mse_mlp / val_mse) * 100:.1f}%")

    mlp_path = os.path.join(output_dir, "mlp_model.pt")
    mlp_model.save(mlp_path)
    print(f"  Saved to {mlp_path}")

    # Thresholds
    print("  Computing thresholds ...")
    mlp_detector = DegradationDetector(mlp_model, window_size=window_size, device=device)
    mlp_stats = mlp_detector.compute_thresholds(
        v_prev[val_idx].to(device), u_prev[val_idx].to(device), v_next[val_idx].to(device)
    )
    print(f"  D_inst  mean={mlp_stats['D_inst_mean']:.4f}  std={mlp_stats['D_inst_std']:.4f}")
    print(f"  D_window mean={mlp_stats['D_window_mean']:.4f}  std={mlp_stats['D_window_std']:.4f}")
    print(f"  Thresholds: q95={mlp_stats['q95']:.4f}  q99={mlp_stats['q99']:.4f}  q999={mlp_stats['q999']:.4f}")

    mlp_det_path = os.path.join(output_dir, "mlp_detector.pt")
    mlp_detector.save(mlp_det_path)

    # ========== Summary ==========
    print("\n===== Comparison Summary =====")
    print(f"  {'Metric':<25} {'Linear':>12} {'MLP':>12}")
    print(f"  {'Val MSE':<25} {val_mse:>12.6f} {val_mse_mlp:>12.6f}")
    print(f"  {'Q trace':<25} {result['Q'].trace().item():>12.6f} {mlp_result['Q'].trace().item():>12.6f}")
    print(f"  {'D_window q95':<25} {linear_stats['q95']:>12.4f} {mlp_stats['q95']:>12.4f}")
    print(f"  {'D_window q99':<25} {linear_stats['q99']:>12.4f} {mlp_stats['q99']:>12.4f}")

    print(f"\nAll models saved to {output_dir}/")
    return {
        "linear": {"model": linear_model, "stats": linear_stats, "val_mse": val_mse},
        "mlp": {"model": mlp_model, "stats": mlp_stats, "val_mse": val_mse_mlp},
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

    args = parser.parse_args()
    fit_and_save(
        data_path=args.data_file,
        output_dir=args.output_dir,
        device=args.device,
        mlp_epochs=args.mlp_epochs,
        mlp_hidden=args.mlp_hidden,
        window_size=args.window_size,
    )
