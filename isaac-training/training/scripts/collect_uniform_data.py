"""
Collect transition data (v_prev, u_prev, v_next) with uniformly distributed
velocity commands using "drive-then-probe" strategy.

The key insight: the transition model learns v_next = f(v_prev, u_prev), where
the Lee velocity controller translates u_prev into motor commands based on the
DIFFERENCE (u_prev - v_prev). Therefore both v_prev AND u_prev must be
uniformly covered for the model to generalize.

Strategy:
  1. Teleport drone to center, zero velocity
  2. Command v_target for approach_steps (~1.3s) so drone reaches v_target
  3. Issue random u_probe, record (v_actual, u_probe, v_next) -- one transition
  4. Repeat probe num_probes times per approach to amortize cost

This ensures the (v_prev, u_prev) 4D space is uniformly covered, and each
transition is physically meaningful (drone is in a stable state before probing).

NOTE: This functionality is also integrated into evaluate() in utils.py.
      Set `uniform_data.num_steps > 0` in train.yaml to collect during eval.

Usage (from Isaac Sim python):
    python collect_uniform_data.py --output ./uniform_data.pt --num_steps 20000
    python collect_uniform_data.py --output ./uniform_data.pt --num_steps 50000 --approach_steps 100
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
    approach_steps = args.approach_steps
    num_probes = args.num_probes

    print(f"[Collect] Strategy: drive-then-probe")
    print(f"[Collect] num_envs={num_envs}, action_limit={action_limit}")
    print(f"[Collect] total_steps={total_steps}, approach_steps={approach_steps}, "
          f"num_probes={num_probes}")
    print(f"[Collect] approach time = {approach_steps * 0.016:.2f}s "
          f"(Lee K_v=2.2 -> τ≈0.45s -> {approach_steps * 0.016 / 0.45:.1f}τ)")

    # Reset env
    env.eval()
    td = transformed_env.reset()

    # Storage
    all_v_prev = []
    all_u_prev = []
    all_v_next = []

    all_ids = torch.arange(num_envs, device=device)
    # Center position for teleportation between cycles
    center_pos = torch.zeros(num_envs, 1, 3, device=device)
    center_pos[..., 2] = 2.0  # hover height
    default_rot = torch.zeros(num_envs, 1, 4, device=device)
    default_rot[..., 0] = 1.0  # identity quaternion (w=1)

    collected = 0
    cycle = 0

    print(f"[Collect] Starting data collection...")

    while collected < total_steps:
        # === Phase 1: Teleport + Drive to random initial velocity ===
        # Teleport drone to center with zero velocity
        env.drone.set_world_poses(center_pos, default_rot, all_ids)
        env.drone.set_velocities(env.init_vels, all_ids)

        # Sample random target velocity for v_prev coverage
        v_target = torch.empty(num_envs, 3, device=device)
        v_target[:, 0].uniform_(-action_limit, action_limit)  # vx
        v_target[:, 1].uniform_(-action_limit, action_limit)  # vy
        v_target[:, 2].uniform_(-0.5, 0.5)                   # vz

        # Approach: command v_target until drone tracks it
        for _ in range(approach_steps):
            td.set(("agents", "action"), v_target)
            td = transformed_env.step(td)
            td = td["next"].clone()

        # === Phase 2: Probe with random commands ===
        for _ in range(num_probes):
            # Sample random probe command for u_prev coverage
            u_probe = torch.empty(num_envs, 3, device=device)
            u_probe[:, 0].uniform_(-action_limit, action_limit)
            u_probe[:, 1].uniform_(-action_limit, action_limit)
            u_probe[:, 2].uniform_(-0.5, 0.5)

            # Read pre-step velocity (drone should be near v_target)
            drone_state_pre = env.drone.get_state(env_frame=False)
            v_pre = drone_state_pre[..., 7:10].squeeze(1).clone()  # (num_envs, 3)

            # Apply probe command (goes through VelController -> Lee -> motor)
            td.set(("agents", "action"), u_probe)
            td = transformed_env.step(td)
            td = td["next"].clone()

            # Read post-step velocity
            drone_state_post = env.drone.get_state(env_frame=False)
            v_post = drone_state_post[..., 7:10].squeeze(1).clone()  # (num_envs, 3)

            all_v_prev.append(v_pre.cpu())
            all_u_prev.append(u_probe.cpu().clone())
            all_v_next.append(v_post.cpu())
            collected += num_envs

        cycle += 1
        if cycle % 20 == 0:
            print(f"  [cycle {cycle}] collected {collected}/{total_steps} transitions")

    # Concatenate and save
    v_prev_all = torch.cat(all_v_prev, dim=0)  # (N, 3)
    u_prev_all = torch.cat(all_u_prev, dim=0)  # (N, 3)
    v_next_all = torch.cat(all_v_next, dim=0)  # (N, 3)

    N = v_prev_all.shape[0]
    print(f"\n[Collect] Total transitions: {N}")
    print(f"[Collect] v_prev (real velocity before probe):")
    print(f"  vx: [{v_prev_all[:, 0].min():.3f}, {v_prev_all[:, 0].max():.3f}], "
          f"mean={v_prev_all[:, 0].mean():.3f}, std={v_prev_all[:, 0].std():.3f}")
    print(f"  vy: [{v_prev_all[:, 1].min():.3f}, {v_prev_all[:, 1].max():.3f}], "
          f"mean={v_prev_all[:, 1].mean():.3f}, std={v_prev_all[:, 1].std():.3f}")
    print(f"[Collect] u_prev (probe command):")
    print(f"  vx: [{u_prev_all[:, 0].min():.3f}, {u_prev_all[:, 0].max():.3f}], "
          f"mean={u_prev_all[:, 0].mean():.3f}, std={u_prev_all[:, 0].std():.3f}")
    print(f"  vy: [{u_prev_all[:, 1].min():.3f}, {u_prev_all[:, 1].max():.3f}], "
          f"mean={u_prev_all[:, 1].mean():.3f}, std={u_prev_all[:, 1].std():.3f}")

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
    parser = argparse.ArgumentParser(
        description="Collect uniform transition data from Isaac Sim (drive-then-probe)")
    parser.add_argument("--output", type=str, default="./uniform_nominal_data.pt",
                        help="Output path for the .pt data file")
    parser.add_argument("--num_steps", type=int, default=20000,
                        help="Total number of transition samples to collect (across all envs)")
    parser.add_argument("--approach_steps", type=int, default=80,
                        help="Sim steps to drive toward target velocity before probing "
                             "(~1.28s at dt=0.016, ≈3 time constants of Lee controller)")
    parser.add_argument("--num_probes", type=int, default=3,
                        help="Number of probe commands per approach cycle")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of overwriting")

    # Hydra consumes its own args; we parse the rest
    args = parser.parse_args(sys.argv[sys.argv.index("--output"):] if "--output" in sys.argv else [])

    collect_uniform_transitions(cfg, args)


if __name__ == "__main__":
    main()
