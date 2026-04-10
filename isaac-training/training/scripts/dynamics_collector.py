"""
Dynamics Data Collector for GP Training.

Collects drone state transition data: V_t = f(V_{t-1}, U_{t-1})
where V is the drone velocity (vx, vy) in world frame and U is the
velocity command (ux, uy) in world frame.

Design:
- Callback-based: plugs into the training loop with minimal code invasion.
- Each transition is (V_{t-1}, U_{t-1}) -> V_t, stored per-episode as trajectories.
- Supports uniform state-space coverage via optional grid-based filtering.
- Saves data incrementally to disk; easy to extend for more state dimensions.

Usage:
    from dynamics_collector import DynamicsCollector

    collector = DynamicsCollector(save_dir="dynamics_data", num_envs=4)

    # In training loop, after each batch:
    for i, data in enumerate(rl_collector):
        collector.collect_batch(data)
        ...

    collector.save()  # final save
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, List
from tensordict.tensordict import TensorDictBase


class DynamicsCollector:
    """Collects (V_{t-1}, U_{t-1}) -> V_t transitions during RL training.

    State:  vx_w, vy_w  (world frame velocity, 2D)
    Action: ux_w, uy_w  (world frame velocity command, 2D)

    Stores both flat transitions (for GP training) and per-episode
    trajectory sequences (for sequence-level analysis).
    """

    def __init__(
        self,
        save_dir: str = "dynamics_data",
        num_envs: int = 4,
        save_interval: int = 500,
        uniform_grid_bins: int = 0,
        uniform_max_per_bin: int = 200,
        state_range: Optional[tuple] = None,
        action_range: Optional[tuple] = None,
    ):
        """
        Args:
            save_dir: Directory to save collected data.
            num_envs: Number of parallel environments.
            save_interval: Auto-save every N training iterations (0 = no auto-save).
            uniform_grid_bins: If > 0, enable grid-based uniform sampling with this
                many bins per dimension. The state-action space is discretized into
                a grid; bins that exceed `uniform_max_per_bin` are skipped.
                Set to 0 to collect ALL transitions.
            uniform_max_per_bin: Max transitions per grid bin.
            state_range: (min, max) range for velocity states. Default: (-3.0, 3.0).
            action_range: (min, max) range for velocity commands. Default: (-2.0, 2.0).
        """
        self.save_dir = save_dir
        self.num_envs = num_envs
        self.save_interval = save_interval
        self._step_count = 0

        # Flat transition buffer: list of (state, action, next_state) tensors
        self._transitions: List[torch.Tensor] = []  # each: (4+2,) = [vx,vy,ux,uy, vx',vy']

        # Per-episode trajectory tracking
        # episode_buffers[env_id] = list of (V_t, U_t) tuples for current episode
        self._episode_buffers: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        # Completed episode trajectories: list of tensors, each (T, 6) 
        # columns: [vx, vy, ux, uy, vx_next, vy_next]
        self._trajectories: List[torch.Tensor] = []

        # Prev-step state tracking per env (needed because data comes in batches)
        self._prev_vel: Optional[torch.Tensor] = None   # (num_envs, 2)
        self._prev_cmd: Optional[torch.Tensor] = None   # (num_envs, 2)

        # Uniform coverage grid
        self._use_grid = uniform_grid_bins > 0
        self._grid_bins = uniform_grid_bins
        self._max_per_bin = uniform_max_per_bin
        sr = state_range or (-3.0, 3.0)
        ar = action_range or (-2.0, 2.0)
        self._state_min = sr[0]
        self._state_max = sr[1]
        self._action_min = ar[0]
        self._action_max = ar[1]
        if self._use_grid:
            # 4D grid: vx, vy, ux, uy
            shape = (uniform_grid_bins,) * 4
            self._grid_counts = np.zeros(shape, dtype=np.int32)

        os.makedirs(save_dir, exist_ok=True)

    def _to_bin_idx(self, values: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        """Map continuous values to bin indices."""
        clamped = values.clamp(vmin, vmax)
        normalized = (clamped - vmin) / (vmax - vmin + 1e-8)  # [0, 1]
        indices = (normalized * (self._grid_bins - 1)).long()
        return indices.clamp(0, self._grid_bins - 1)

    def _check_grid(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return boolean mask: True if the bin has room for more samples."""
        vx_bin = self._to_bin_idx(state[:, 0], self._state_min, self._state_max)
        vy_bin = self._to_bin_idx(state[:, 1], self._state_min, self._state_max)
        ux_bin = self._to_bin_idx(action[:, 0], self._action_min, self._action_max)
        uy_bin = self._to_bin_idx(action[:, 1], self._action_min, self._action_max)

        mask = torch.zeros(state.shape[0], dtype=torch.bool)
        for i in range(state.shape[0]):
            count = self._grid_counts[vx_bin[i], vy_bin[i], ux_bin[i], uy_bin[i]]
            if count < self._max_per_bin:
                mask[i] = True
        return mask

    def _update_grid(self, state: torch.Tensor, action: torch.Tensor):
        """Increment grid counts for accepted transitions."""
        vx_bin = self._to_bin_idx(state[:, 0], self._state_min, self._state_max)
        vy_bin = self._to_bin_idx(state[:, 1], self._state_min, self._state_max)
        ux_bin = self._to_bin_idx(action[:, 0], self._action_min, self._action_max)
        uy_bin = self._to_bin_idx(action[:, 1], self._action_min, self._action_max)
        for i in range(state.shape[0]):
            self._grid_counts[vx_bin[i], vy_bin[i], ux_bin[i], uy_bin[i]] += 1

    def collect_batch(self, data: TensorDictBase, iteration: int = 0):
        """Process one training batch from SyncDataCollector.

        Expected data shape: (num_envs, num_frames, ...)

        Key tensordict paths used:
            ("agents", "observation", "state")  -> drone internal state, vel_g at [..., 5:8]
            ("agents", "action")                -> velocity command (world frame, 3D)
            ("next", "agents", "observation", "state") -> next state
            ("next", "done")                    -> episode termination
        
        However, vel_g is in goal frame. We need world frame velocity.
        The world frame velocity is available from root_state[..., 7:10],
        but that is NOT stored in the tensordict by default.

        ** IMPORTANT **: This collector reads from the key 
        ("agents", "observation", "dynamics") that is added to the env output.
        See the modifications in env.py for details.
        """
        device = data.device

        # Shape: (num_envs, T)
        vel_w_xy = data["agents", "observation", "dynamics", "vel_w_xy"]        # (E, T, 2)
        cmd_w_xy = data["agents", "observation", "dynamics", "cmd_w_xy"]        # (E, T, 2)
        next_vel_w_xy = data["next", "agents", "observation", "dynamics", "vel_w_xy"]  # (E, T, 2)
        done = data["next", "done"].squeeze(-1)                       # (E, T)

        num_envs, T = vel_w_xy.shape[:2]

        for t in range(T):
            v_t = vel_w_xy[:, t].cpu()       # (E, 2)
            u_t = cmd_w_xy[:, t].cpu()       # (E, 2)
            v_tp1 = next_vel_w_xy[:, t].cpu()  # (E, 2)
            d_t = done[:, t].cpu()            # (E,)

            # Build transition: [vx, vy, ux, uy, vx', vy']
            transition = torch.cat([v_t, u_t, v_tp1], dim=-1)  # (E, 6)

            # Grid-based filtering
            if self._use_grid:
                accept_mask = self._check_grid(v_t, u_t)
            else:
                accept_mask = torch.ones(num_envs, dtype=torch.bool)

            # Store accepted transitions
            for env_idx in range(num_envs):
                if accept_mask[env_idx]:
                    self._transitions.append(transition[env_idx])
                    self._episode_buffers[env_idx].append(transition[env_idx])

                # If episode ends, finalize the trajectory
                if d_t[env_idx]:
                    if len(self._episode_buffers[env_idx]) > 0:
                        traj = torch.stack(self._episode_buffers[env_idx], dim=0)
                        self._trajectories.append(traj)
                        self._episode_buffers[env_idx] = []

            if self._use_grid:
                accepted = transition[accept_mask]
                if accepted.shape[0] > 0:
                    self._update_grid(accepted[:, :2], accepted[:, 2:4])

        self._step_count += 1

        # Save a "latest" snapshot frequently so data is available if training is stopped early
        # Tagged snapshots are saved less often (save_interval) for versioning
        if self._step_count % max(self.save_interval // 10, 1) == 0:
            self.save(tag="latest")
        if self.save_interval > 0 and self._step_count % self.save_interval == 0:
            self.save(tag=f"iter_{self._step_count}")

    @property
    def num_transitions(self) -> int:
        return len(self._transitions)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    def get_dataset(self) -> Dict[str, torch.Tensor]:
        """Return flat dataset as dict of tensors.

        Returns:
            {
                "state":      (N, 2)  - [vx, vy]
                "action":     (N, 2)  - [ux, uy]
                "next_state": (N, 2)  - [vx', vy']
            }
        """
        if len(self._transitions) == 0:
            return {
                "state": torch.zeros(0, 2),
                "action": torch.zeros(0, 2),
                "next_state": torch.zeros(0, 2),
            }
        all_data = torch.stack(self._transitions, dim=0)  # (N, 6)
        return {
            "state": all_data[:, :2],
            "action": all_data[:, 2:4],
            "next_state": all_data[:, 4:6],
        }

    def get_trajectories(self) -> List[torch.Tensor]:
        """Return list of episode trajectories.

        Each element is a tensor of shape (T_i, 6) with columns:
        [vx, vy, ux, uy, vx_next, vy_next]
        """
        return self._trajectories

    def save(self, tag: str = "final"):
        """Save collected data to disk."""
        # Flat transitions
        dataset = self.get_dataset()
        flat_path = os.path.join(self.save_dir, f"dynamics_transitions_{tag}.pt")
        torch.save(dataset, flat_path)

        # Episode trajectories
        traj_path = os.path.join(self.save_dir, f"dynamics_trajectories_{tag}.pt")
        torch.save(self._trajectories, traj_path)

        # Grid coverage info
        if self._use_grid:
            grid_path = os.path.join(self.save_dir, f"grid_coverage_{tag}.npy")
            np.save(grid_path, self._grid_counts)

        print(f"[DynamicsCollector] Saved {self.num_transitions} transitions, "
              f"{self.num_trajectories} trajectories -> {self.save_dir} (tag={tag})")

    def get_grid_coverage_stats(self) -> Dict[str, float]:
        """Return statistics about grid coverage (only if grid mode is enabled)."""
        if not self._use_grid:
            return {}
        total_bins = self._grid_counts.size
        filled_bins = np.count_nonzero(self._grid_counts)
        return {
            "total_bins": total_bins,
            "filled_bins": filled_bins,
            "coverage_ratio": filled_bins / total_bins,
            "min_count": int(self._grid_counts.min()),
            "max_count": int(self._grid_counts.max()),
            "mean_count": float(self._grid_counts.mean()),
        }
