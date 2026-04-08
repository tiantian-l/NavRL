"""
Collect transition data (v_prev, u_prev, v_next) with uniformly distributed
velocity commands, bypassing the trained policy.

Instead of relying on the policy's natural command distribution (biased toward
navigation goals), this script sends random velocity commands sampled uniformly
from [-action_limit, action_limit] in each axis. This produces a much more
uniform coverage of the (vx, vy) input space for fitting transition models.

Usage (from Isaac Sim python):
    python collect_uniform_data.py --output ./uniform_data.pt --num_steps 20000
    python collect_uniform_data.py --output ./uniform_data.pt --num_steps 50000 --hold_steps 30
"""

import argparse
import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def collect_uniform_transitions(cfg, args):
    from omni.isaac.kit import SimulationApp
    sim_app = SimulationApp({"headless": True, "anti_aliasing": 0})

    from env import NavigationEnv
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from torchrl.envs.transforms import TransformedEnv, Compose

    # --- Build environment (same as training) ---
    env = NavigationEnv(cfg)
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform))
    transformed_env.set_seed(cfg.seed)

    # Parameters
    action_limit = cfg.algo.actor.action_limit  # 2.0 m/s
    num_envs = cfg.env.num_envs
    device = cfg.device
    total_steps = args.num_steps
    hold_steps = args.hold_steps  # how many sim steps to hold each command
    settle_steps = args.settle_steps  # initial settle steps (discard)

    print(f"[Collect] num_envs={num_envs}, action_limit={action_limit}")
    print(f"[Collect] total_steps={total_steps}, hold_steps={hold_steps}, settle_steps={settle_steps}")
    print(f"[Collect] Sampling mode: {args.sample_mode}")

    # Reset env
    env.eval()
    td = transformed_env.reset()

    # Let the drone settle from spawn
    print(f"[Collect] Settling for {settle_steps} steps...")
    zero_cmd = torch.zeros(num_envs, 3, device=device)
    for _ in range(settle_steps):
        td.set(("agents", "action"), zero_cmd)
        td = transformed_env.step(td)
        td = td["next"].clone()

    # Storage
    all_v_prev = []
    all_u_prev = []
    all_v_next = []

    step = 0
    cmd_counter = 0
    current_cmd = torch.zeros(num_envs, 3, device=device)

    print(f"[Collect] Starting data collection...")

    while step < total_steps:
        # Generate new random velocity command every `hold_steps`
        if cmd_counter % hold_steps == 0:
            if args.sample_mode == "uniform":
                # Uniform over [-limit, limit] for vx, vy; small range for vz
                current_cmd = torch.empty(num_envs, 3, device=device)
                current_cmd[:, 0].uniform_(-action_limit, action_limit)  # vx
                current_cmd[:, 1].uniform_(-action_limit, action_limit)  # vy
                current_cmd[:, 2].uniform_(-0.5, 0.5)                   # vz (small)
            elif args.sample_mode == "grid":
                # Cycle through a grid of (vx, vy) with random vz
                grid_n = args.grid_n
                vx_vals = torch.linspace(-action_limit, action_limit, grid_n)
                vy_vals = torch.linspace(-action_limit, action_limit, grid_n)
                grid_idx = (cmd_counter // hold_steps) % (grid_n * grid_n)
                ix = grid_idx % grid_n
                iy = grid_idx // grid_n
                current_cmd[:, 0] = vx_vals[ix]
                current_cmd[:, 1] = vy_vals[iy]
                current_cmd[:, 2] = torch.empty(num_envs).uniform_(-0.3, 0.3).to(device)
            elif args.sample_mode == "latin":
                # Latin hypercube style: stratified uniform
                n_bins = args.grid_n
                bin_size = 2 * action_limit / n_bins
                bin_idx = (cmd_counter // hold_steps) % (n_bins * n_bins)
                ix = bin_idx % n_bins
                iy = bin_idx // n_bins
                vx_lo = -action_limit + ix * bin_size
                vy_lo = -action_limit + iy * bin_size
                current_cmd[:, 0] = vx_lo + torch.rand(num_envs, device=device) * bin_size
                current_cmd[:, 1] = vy_lo + torch.rand(num_envs, device=device) * bin_size
                current_cmd[:, 2] = torch.empty(num_envs).uniform_(-0.3, 0.3).to(device)

        cmd_counter += 1

        # Read pre-step velocity
        drone_state = env.drone.get_state(env_frame=False)
        v_prev = drone_state[..., 7:10].squeeze(1).clone()  # (num_envs, 3)

        # Step with velocity command
        td.set(("agents", "action"), current_cmd)
        td = transformed_env.step(td)
        td_next = td["next"]

        # Read post-step velocity
        drone_state_next = env.drone.get_state(env_frame=False)
        v_next = drone_state_next[..., 7:10].squeeze(1).clone()  # (num_envs, 3)

        # Store transitions
        all_v_prev.append(v_prev.cpu())
        all_u_prev.append(current_cmd.cpu())
        all_v_next.append(v_next.cpu())

        step += num_envs  # each step yields num_envs transitions

        # Reset drone position if it flies too far or too low/high
        pos = drone_state_next[..., :3].squeeze(1)  # (num_envs, 3)
        out_of_bounds = (
            (pos[:, 0].abs() > 15.0) |
            (pos[:, 1].abs() > 15.0) |
            (pos[:, 2] < 0.3) |
            (pos[:, 2] > 5.0)
        )
        if out_of_bounds.any():
            reset_ids = torch.where(out_of_bounds)[0]
            env._reset_idx(reset_ids)
            # Re-settle after reset
            for _ in range(20):
                td_next.set(("agents", "action"), zero_cmd)
                td_next = transformed_env.step(td_next)
                td_next = td_next["next"].clone()

        td = td_next.clone()

        if step % (num_envs * 500) == 0:
            n_collected = len(all_v_prev) * num_envs
            print(f"  [{step}/{total_steps}] collected {n_collected} transitions")

    # Concatenate and save
    v_prev_all = torch.cat(all_v_prev, dim=0)  # (N, 3)
    u_prev_all = torch.cat(all_u_prev, dim=0)  # (N, 3)
    v_next_all = torch.cat(all_v_next, dim=0)  # (N, 3)

    N = v_prev_all.shape[0]
    print(f"\n[Collect] Total transitions: {N}")
    print(f"[Collect] u_prev (cmd) stats:")
    print(f"  vx: [{u_prev_all[:, 0].min():.3f}, {u_prev_all[:, 0].max():.3f}], mean={u_prev_all[:, 0].mean():.3f}, std={u_prev_all[:, 0].std():.3f}")
    print(f"  vy: [{u_prev_all[:, 1].min():.3f}, {u_prev_all[:, 1].max():.3f}], mean={u_prev_all[:, 1].mean():.3f}, std={u_prev_all[:, 1].std():.3f}")
    print(f"  vz: [{u_prev_all[:, 2].min():.3f}, {u_prev_all[:, 2].max():.3f}], mean={u_prev_all[:, 2].mean():.3f}, std={u_prev_all[:, 2].std():.3f}")
    print(f"[Collect] v_prev (real vel) stats:")
    print(f"  vx: [{v_prev_all[:, 0].min():.3f}, {v_prev_all[:, 0].max():.3f}], mean={v_prev_all[:, 0].mean():.3f}, std={v_prev_all[:, 0].std():.3f}")
    print(f"  vy: [{v_prev_all[:, 1].min():.3f}, {v_prev_all[:, 1].max():.3f}], mean={v_prev_all[:, 1].mean():.3f}, std={v_prev_all[:, 1].std():.3f}")

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save (compatible with collect_and_fit.py format)
    save_dict = {
        "v_prev": v_prev_all,
        "u_prev": u_prev_all,
        "v_next": v_next_all,
    }

    # Optionally append to existing file
    if args.append and os.path.exists(output_path):
        existing = torch.load(output_path, map_location="cpu", weights_only=True)
        save_dict["v_prev"] = torch.cat([existing["v_prev"], v_prev_all], dim=0)
        save_dict["u_prev"] = torch.cat([existing["u_prev"], u_prev_all], dim=0)
        save_dict["v_next"] = torch.cat([existing["v_next"], v_next_all], dim=0)
        print(f"[Collect] Appended to existing data. New total: {save_dict['v_prev'].shape[0]}")

    torch.save(save_dict, output_path)
    print(f"[Collect] Saved to {output_path}")

    sim_app.close()


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    parser = argparse.ArgumentParser(description="Collect uniform transition data from Isaac Sim")
    parser.add_argument("--output", type=str, default="./uniform_nominal_data.pt",
                        help="Output path for the .pt data file")
    parser.add_argument("--num_steps", type=int, default=20000,
                        help="Total number of transition samples to collect (across all envs)")
    parser.add_argument("--hold_steps", type=int, default=20,
                        help="Number of sim steps to hold each velocity command "
                             "(~0.32s at dt=0.016). Captures transient response.")
    parser.add_argument("--settle_steps", type=int, default=100,
                        help="Initial settling steps after spawn (discarded)")
    parser.add_argument("--sample_mode", type=str, default="uniform",
                        choices=["uniform", "grid", "latin"],
                        help="How to sample velocity commands: "
                             "uniform=random uniform, grid=sweep grid, latin=stratified")
    parser.add_argument("--grid_n", type=int, default=20,
                        help="Grid resolution per axis for grid/latin modes")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of overwriting")

    # Hydra consumes its own args; we parse the rest
    args = parser.parse_args(sys.argv[sys.argv.index("--output"):] if "--output" in sys.argv else [])

    collect_uniform_transitions(cfg, args)


if __name__ == "__main__":
    main()
