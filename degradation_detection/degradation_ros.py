"""
ROS Degradation Detector Node — online degradation monitoring for deployment.

Works with both ROS1 and ROS2 navigation_runner.
Standalone usage: import and call in the control loop.

Example integration in navigation.py control_callback:
    from degradation_ros import ROSDegradationMonitor
    self.deg_monitor = ROSDegradationMonitor(model_dir, model_type="mlp", device="cpu")

    # In control_callback, after getting vel_world and safe_cmd_vel_world:
    result = self.deg_monitor.step(vel_world_prev, cmd_vel_prev, vel_world_current)
"""

import os
import sys
import torch
import numpy as np

# Ensure degradation_detection package is importable
_dd_dir = os.path.dirname(__file__)
if _dd_dir not in sys.path:
    sys.path.insert(0, _dd_dir)

from transition_models import LinearTransitionModel, MLPTransitionModel
from degradation_detector import DegradationDetector


class ROSDegradationMonitor:
    """
    Lightweight online degradation monitor for ROS deployment.

    Maintains (v_{t-1}, u_{t-1}) state internally, so the caller just needs to
    provide current velocity and current command each step.
    """

    def __init__(self, model_dir: str, model_type: str = "mlp",
                 window_size: int = 20, device: str = "cpu"):
        """
        Args:
            model_dir: directory containing linear_model.pt / mlp_model.pt and detector files
            model_type: "linear" or "mlp"
            window_size: sliding window W
            device: "cpu" or "cuda:0"
        """
        self.device = device
        self.model_type = model_type

        # Load model
        if model_type == "linear":
            model_path = os.path.join(model_dir, "linear_model.pt")
            self.model = LinearTransitionModel(device=device)
            self.model.load(model_path)
            det_path = os.path.join(model_dir, "linear_detector.pt")
        elif model_type == "mlp":
            model_path = os.path.join(model_dir, "mlp_model.pt")
            self.model = MLPTransitionModel(device=device)
            self.model.load(model_path)
            det_path = os.path.join(model_dir, "mlp_detector.pt")
        else:
            raise ValueError(f"model_type must be 'linear' or 'mlp', got '{model_type}'")

        # Create detector
        self.detector = DegradationDetector(self.model, window_size=window_size, device=device)
        if os.path.exists(det_path):
            self.detector.load(det_path)

        # Internal state
        self._prev_vel = None  # (3,) torch tensor
        self._prev_cmd = None  # (3,) torch tensor
        self._initialized = False

        print(f"[DegMonitor] Loaded {model_type} model from {model_dir}")
        print(f"[DegMonitor] Thresholds: {self.detector.thresholds}")
        print(f"[DegMonitor] Window size: {window_size}")

    def step(self, vel_world: np.ndarray, cmd_vel_world: np.ndarray) -> dict:
        """
        Process one control step.

        Args:
            vel_world: current velocity in world frame, shape (3,), numpy or torch
            cmd_vel_world: current velocity command (after safe action), shape (3,)

        Returns:
            dict with D_inst, D_window, level, residual.
            Returns None if not enough history yet (first step).
        """
        # Convert to torch
        if isinstance(vel_world, np.ndarray):
            vel_world = torch.tensor(vel_world, dtype=torch.float32, device=self.device)
        if isinstance(cmd_vel_world, np.ndarray):
            cmd_vel_world = torch.tensor(cmd_vel_world, dtype=torch.float32, device=self.device)

        vel_world = vel_world.view(-1)[:3]
        cmd_vel_world = cmd_vel_world.view(-1)[:3]

        result = None
        if self._initialized:
            # We have v_{t-1} and u_{t-1}, current v_t is vel_world
            result = self.detector.step(self._prev_vel, self._prev_cmd, vel_world)

        # Update internal state for next step
        self._prev_vel = vel_world.clone()
        self._prev_cmd = cmd_vel_world.clone()
        self._initialized = True

        return result

    def reset(self):
        """Reset detector state (e.g. when a new goal is received)."""
        self._prev_vel = None
        self._prev_cmd = None
        self._initialized = False
        self.detector.reset()

    @property
    def current_level(self) -> int:
        """Current degradation level (0-3)."""
        if not self._initialized:
            return 0
        return 0  # Will be updated after first step call
