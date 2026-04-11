"""
Physics-Based Quadrotor Dynamics ODE Transition Model.

Derives closed-loop velocity dynamics by combining:
  ① Newton-Euler rigid-body equations of a quadrotor
  ② Lee PD velocity-tracking controller
  ③ First-order actuator dynamics (attitude & thrust lag)
  ④ Quadratic aerodynamic drag

This provides a structured, interpretable alternative to purely data-driven
(Linear / MLP) transition models.  Physical parameters can be fit from data
while respecting known constraints.

==========================================================================
DERIVATION
==========================================================================

1) Full Quadrotor Rigid-Body Dynamics (Newton-Euler, world frame)
-----------------------------------------------------------------
   m · dv/dt  =  R · [0, 0, T]^T  -  m·g·e_3  -  F_drag
   J · dω/dt  =  τ  -  ω × J·ω

   where:
     v     = [v_x, v_y, v_z]   linear velocity (world frame)
     R     = rotation matrix (body → world)
     T     = total thrust magnitude
     g     = 9.81 m/s²
     F_drag = aerodynamic drag force
     ω     = angular velocity (body frame)
     J     = diagonal inertia matrix
     τ     = torques from rotors

2) Lee PD Velocity Controller
------------------------------
   The RL policy outputs velocity commands u = [u_x, u_y, u_z].
   The Lee controller computes desired acceleration:

     a_des = -K_v ⊙ (v - u)

   From the desired acceleration, the controller derives:
     - Desired roll:    φ_des = -(1/g) · a_des_y
     - Desired pitch:   θ_des =  (1/g) · a_des_x
     - Desired thrust:  T_des = m · (g + a_des_z)

   (Small-angle linearization of the rotation matrix, valid for
    hover and moderate maneuvers.)

3) Actuator Dynamics (First-Order Lag)
---------------------------------------
   Real actuators cannot instantaneously achieve desired attitude/thrust.
   We model this with first-order dynamics:

     dφ/dt = (φ_des - φ) / τ_att
     dθ/dt = (θ_des - θ) / τ_att
     dT/dt = (T_des - T) / τ_thrust

   where τ_att ≈ 0.05s (attitude loop bandwidth) and
         τ_thrust ≈ 0.03s (thrust response time).

4) Resulting Closed-Loop Velocity ODE
--------------------------------------
   Substituting the rotation matrix R(φ, θ) applied to thrust [0, 0, T/m]:

     dv_x/dt = (T/m)·sin(θ)                   - D_x · v_x · |v_x|
     dv_y/dt = -(T/m)·sin(φ)·cos(θ)           - D_y · v_y · |v_y|
     dv_z/dt = (T/m)·cos(φ)·cos(θ) - g        - D_z · v_z · |v_z|

   The quadratic drag terms (D · v · |v|) capture aerodynamic resistance
   and provide genuine nonlinearity beyond the linear model.

5) Full Latent State ODE
--------------------------
   State x = [v_x, v_y, v_z, φ, θ, T]  (6D latent)
   Input u = [u_x, u_y, u_z]            (3D velocity command)

   The system is integrated using RK4 with sub-stepping over dt.
   Observable output: v_next = x[0:3] at t + dt.

6) Why This Is Better Than Linear/MLP
---------------------------------------
   - Linear model:  v_{t+1} = A·v_t + B·u_t  (no nonlinearity)
   - MLP model:     v_{t+1} = f_θ(v_t, u_t)   (black-box, no physics)
   - ODE model:     v_{t+1} = ∫(physics ODE)   (structured, interpretable)

   The ODE model captures:
     a) Transient attitude dynamics (finite actuator bandwidth)
     b) Thrust-direction coupling (sin/cos nonlinearity)
     c) Quadratic aerodynamic drag
     d) Physically meaningful parameters for diagnostics

   For degradation detection, parameter drift is interpretable:
     - mass ↑   →  payload change
     - drag ↑   →  structural damage
     - τ_att ↑  →  actuator degradation
     - K_v ↓    →  controller performance loss

References:
  [1] Lee et al., "Geometric Tracking Control of a Quadrotor UAV on SE(3)",
      IEEE CDC 2010. https://arxiv.org/abs/1003.2005
  [2] Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for
      Quadrotors", ICRA 2011.
==========================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class QuadrotorODETransitionModel(nn.Module):
    r"""
    Physics-based closed-loop quadrotor velocity transition model.

    Models the chain:  velocity command → Lee PD controller →
    desired attitude/thrust → actuator lag → rigid-body dynamics → next velocity.

    Latent state (6D):  x = [v_x, v_y, v_z, φ, θ, T]
    Observed (3D):      v = [v_x, v_y, v_z]
    Input (3D):         u = [u_x, u_y, u_z]  (velocity command)

    Learnable physical parameters:
        K_v       : velocity controller gains [3]          (default from Hummingbird)
        τ_att     : attitude actuator time constant        (default: 0.05 s)
        τ_thrust  : thrust actuator time constant          (default: 0.03 s)
        D         : quadratic drag coefficients [3]        (default: [0.1, 0.1, 0.1])
        m         : mass                                   (default: 0.716 kg)

    Integration:  4th-order Runge-Kutta with configurable sub-steps.

    Interface identical to LinearTransitionModel / MLPTransitionModel:
        fit(v_prev, u_prev, v_next)  →  fit parameters from nominal data
        predict(v_prev, u_prev)      →  one-step velocity prediction
        residual(v_prev, u_prev, v_next)  →  r_t = v_t - v̂_t
    """

    def __init__(
        self,
        state_dim: int = 3,
        input_dim: int = 3,
        dt: float = 0.016,
        mass: float = 0.716,
        gravity: float = 9.81,
        K_v: list = None,
        tau_att: float = 0.05,
        tau_thrust: float = 0.03,
        drag: list = None,
        num_substeps: int = 4,
        learnable_params: list = None,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim:  observed state dimension (3 = velocity)
            input_dim:  command dimension (3 = velocity command)
            dt:         simulation time step in seconds (Isaac Sim default: 0.016)
            mass:       drone mass in kg (Hummingbird: 0.716)
            gravity:    gravitational acceleration in m/s²
            K_v:        velocity PD gains [K_vx, K_vy, K_vz]
                        (Hummingbird default: [2.2, 2.2, 2.2])
            tau_att:    attitude actuator lag time constant in seconds
            tau_thrust: thrust actuator lag time constant in seconds
            drag:       quadratic drag coefficients [D_x, D_y, D_z]
            num_substeps: number of RK4 sub-steps per dt
            learnable_params: which parameters to optimise during fit().
                        List of names from {'K_v','tau_att','tau_thrust','drag','mass'}.
                        None → all learnable.
            device:     torch device string
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt
        self.num_substeps = num_substeps
        self.device_str = device

        # ---- Default values (Hummingbird drone) ----
        if K_v is None:
            K_v = [2.2, 2.2, 2.2]
        if drag is None:
            drag = [0.1, 0.1, 0.1]

        # Store as log-space nn.Parameters → exp() guarantees positivity
        self._log_K_v = nn.Parameter(
            torch.log(torch.tensor(K_v, dtype=torch.float32))
        )
        self._log_tau_att = nn.Parameter(
            torch.log(torch.tensor(tau_att, dtype=torch.float32))
        )
        self._log_tau_thrust = nn.Parameter(
            torch.log(torch.tensor(tau_thrust, dtype=torch.float32))
        )
        self._log_drag = nn.Parameter(
            torch.log(torch.tensor(drag, dtype=torch.float32).clamp(min=1e-6))
        )
        self._log_mass = nn.Parameter(
            torch.log(torch.tensor(mass, dtype=torch.float32))
        )

        # Gravity is NOT learnable
        self.register_buffer(
            "gravity", torch.tensor(gravity, dtype=torch.float32)
        )

        # Noise covariance (estimated from residuals after fit)
        self.Q = torch.eye(state_dim, device=device)
        self.Q_inv = torch.eye(state_dim, device=device)

        # Freeze non-learnable params if requested
        if learnable_params is not None:
            self._set_learnable(learnable_params)

        self.to(device)

    # ------------------------------------------------------------------
    #  Learnable-parameter management
    # ------------------------------------------------------------------

    _PARAM_MAP = {
        "K_v": "_log_K_v",
        "tau_att": "_log_tau_att",
        "tau_thrust": "_log_tau_thrust",
        "drag": "_log_drag",
        "mass": "_log_mass",
    }

    def _set_learnable(self, param_names: list):
        """Freeze all physics params except those in *param_names*."""
        for name, attr in self._PARAM_MAP.items():
            getattr(self, attr).requires_grad_(name in param_names)

    # ------------------------------------------------------------------
    #  Property accessors  (log-space → positive real)
    # ------------------------------------------------------------------

    @property
    def K_v(self) -> torch.Tensor:
        """Velocity controller gains [3]."""
        return self._log_K_v.exp()

    @property
    def tau_att(self) -> torch.Tensor:
        """Attitude actuator time constant (scalar)."""
        return self._log_tau_att.exp()

    @property
    def tau_thrust(self) -> torch.Tensor:
        """Thrust actuator time constant (scalar)."""
        return self._log_tau_thrust.exp()

    @property
    def drag(self) -> torch.Tensor:
        """Quadratic drag coefficients [3]."""
        return self._log_drag.exp()

    @property
    def mass(self) -> torch.Tensor:
        """Drone mass (scalar)."""
        return self._log_mass.exp()

    # ------------------------------------------------------------------
    #  Latent state initialisation
    # ------------------------------------------------------------------

    def _init_latent_state(
        self, v: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialise full 6D latent state [v, φ, θ, T] from observables.

        Uses the quasi-steady-state assumption: at the start of each step
        the attitude and thrust have converged to the values demanded by
        the Lee controller for the current velocity error.

        Args:
            v: (batch, 3) current velocity
            u: (batch, 3) velocity command
        Returns:
            state: (batch, 6)
        """
        K_v = self.K_v
        m = self.mass
        g = self.gravity

        # Lee controller desired acceleration
        a_des = -K_v * (v - u)  # (batch, 3)

        # Small-angle desired attitude
        phi_des = -(1.0 / g) * a_des[:, 1:2]      # (batch, 1)
        theta_des = (1.0 / g) * a_des[:, 0:1]     # (batch, 1)

        # Desired thrust
        T_des = m * (g + a_des[:, 2:3])            # (batch, 1)

        return torch.cat([v, phi_des, theta_des, T_des], dim=-1)

    # ------------------------------------------------------------------
    #  ODE right-hand side
    # ------------------------------------------------------------------

    def _ode_rhs(
        self, state: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        dx/dt = f(x, u)

        State layout: x = [v_x, v_y, v_z, φ, θ, T]

        Returns dstate/dt of shape (batch, 6).
        """
        v = state[:, :3]
        phi = state[:, 3:4]
        theta = state[:, 4:5]
        T = state[:, 5:6]

        K_v = self.K_v
        m = self.mass
        g = self.gravity
        tau_a = self.tau_att
        tau_t = self.tau_thrust
        D = self.drag

        # ---- Lee PD controller: desired acceleration ----
        a_des = -K_v * (v - u)  # (batch, 3)

        # ---- Desired attitude & thrust ----
        phi_des = -(1.0 / g) * a_des[:, 1:2]
        theta_des = (1.0 / g) * a_des[:, 0:1]
        T_des = m * (g + a_des[:, 2:3])

        # ---- Actuator dynamics (first-order lag) ----
        dphi = (phi_des - phi) / tau_a
        dtheta = (theta_des - theta) / tau_a
        dT = (T_des - T) / tau_t

        # ---- Newton-Euler velocity dynamics ----
        #
        #  Rotation matrix R(φ,θ) applied to body-frame thrust [0, 0, T]:
        #    a_x =  (T/m) · sin(θ)
        #    a_y = -(T/m) · sin(φ) · cos(θ)
        #    a_z =  (T/m) · cos(φ) · cos(θ)  -  g
        #
        #  Plus quadratic aerodynamic drag:
        #    F_drag_i = D_i · v_i · |v_i|

        T_over_m = T / m
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        dvx = (
            T_over_m * sin_theta
            - D[0] * v[:, 0:1] * v[:, 0:1].abs()
        )
        dvy = (
            -T_over_m * sin_phi * cos_theta
            - D[1] * v[:, 1:2] * v[:, 1:2].abs()
        )
        dvz = (
            T_over_m * cos_phi * cos_theta
            - g
            - D[2] * v[:, 2:3] * v[:, 2:3].abs()
        )

        return torch.cat([dvx, dvy, dvz, dphi, dtheta, dT], dim=-1)

    # ------------------------------------------------------------------
    #  RK4 integrator
    # ------------------------------------------------------------------

    def _rk4_step(
        self, state: torch.Tensor, u: torch.Tensor, h: float
    ) -> torch.Tensor:
        """Single 4th-order Runge-Kutta step of size h."""
        k1 = self._ode_rhs(state, u)
        k2 = self._ode_rhs(state + 0.5 * h * k1, u)
        k3 = self._ode_rhs(state + 0.5 * h * k2, u)
        k4 = self._ode_rhs(state + h * k3, u)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate(
        self, state: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate ODE from t to t + dt using RK4 with sub-stepping.

        Args:
            state: (batch, 6) latent state at t
            u:     (batch, 3) velocity command (constant over [t, t+dt])
        Returns:
            (batch, 6) latent state at t + dt
        """
        h = self.dt / self.num_substeps
        for _ in range(self.num_substeps):
            state = self._rk4_step(state, u, h)
        return state

    # ------------------------------------------------------------------
    #  Public interface  (compatible with Linear / MLP models)
    # ------------------------------------------------------------------

    def forward(
        self, v_prev: torch.Tensor, u_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        One-step velocity prediction:  v̂_{t+1} = ODE(v_t, u_t; θ_phys).

        Args:
            v_prev: (batch, 3) velocity at time t
            u_prev: (batch, 3) velocity command at time t
        Returns:
            v_next_pred: (batch, 3) predicted velocity at time t + dt
        """
        state = self._init_latent_state(v_prev, u_prev)
        state_next = self._integrate(state, u_prev)
        return state_next[:, :3]

    def predict(
        self, v_prev: torch.Tensor, u_prev: torch.Tensor
    ) -> torch.Tensor:
        """No-grad prediction (same as forward but detached)."""
        with torch.no_grad():
            return self.forward(v_prev, u_prev)

    def residual(
        self,
        v_prev: torch.Tensor,
        u_prev: torch.Tensor,
        v_next: torch.Tensor,
    ) -> torch.Tensor:
        """r_t = v_t - v̂_t"""
        with torch.no_grad():
            return v_next - self.predict(v_prev, u_prev)

    # ------------------------------------------------------------------
    #  Sequential (episode-level) prediction
    # ------------------------------------------------------------------

    def predict_sequence(
        self,
        v_init: torch.Tensor,
        u_sequence: torch.Tensor,
        latent_init: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict a trajectory while propagating latent state across steps.

        Unlike single-step predict(), this does NOT re-initialise the latent
        attitude / thrust state at each step — it carries φ, θ, T forward,
        giving a more physically accurate multi-step forecast.

        Args:
            v_init:      (batch, 3) velocity at t = 0
            u_sequence:  (batch, T, 3) velocity commands for T steps
            latent_init: optional (batch, 6) full initial latent state.
                         If None, quasi-steady-state initialisation is used.
        Returns:
            v_pred: (batch, T, 3) predicted velocity at each step
        """
        T_steps = u_sequence.shape[1]

        if latent_init is None:
            state = self._init_latent_state(v_init, u_sequence[:, 0])
        else:
            state = latent_init

        v_pred = []
        for t in range(T_steps):
            state = self._integrate(state, u_sequence[:, t])
            v_pred.append(state[:, :3])

        return torch.stack(v_pred, dim=1)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def fit(
        self,
        v_prev: torch.Tensor,
        u_prev: torch.Tensor,
        v_next: torch.Tensor,
        lr: float = 5e-3,
        epochs: int = 500,
        batch_size: int = 4096,
        verbose: bool = True,
    ):
        """
        Fit physical parameters by minimising prediction MSE.

        Uses Adam + cosine-annealing LR schedule with gradient clipping
        for stable optimisation of the log-space parameters.
        """
        device = self.device_str
        v_prev = v_prev.float().to(device)
        u_prev = u_prev.float().to(device)
        v_next = v_next.float().to(device)

        N = v_prev.shape[0]
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        dataset = torch.utils.data.TensorDataset(v_prev, u_prev, v_next)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.train()
        best_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for vp_b, up_b, vn_b in loader:
                pred = self.forward(vp_b, up_b)
                loss = nn.functional.mse_loss(pred, vn_b)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                optimizer.step()
                epoch_loss += loss.item() * vp_b.shape[0]

            scheduler.step()
            avg_loss = epoch_loss / N
            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [ODE] epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")
                self._print_params()

        self.eval()

        # Estimate Q from residuals on full dataset
        with torch.no_grad():
            full_pred = self.forward(v_prev, u_prev)
            residuals = v_next - full_pred
            self.Q = (residuals.T @ residuals) / N
            self.Q_inv = torch.inverse(self.Q)

        if verbose:
            print("  Final physical parameters:")
            self._print_params()

        return {"Q": self.Q, "best_loss": best_loss}

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def _print_params(self):
        """Pretty-print current physical parameters."""
        K = self.K_v.detach().cpu().numpy()
        D = self.drag.detach().cpu().numpy()
        print(
            f"    K_v=[{K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}]  "
            f"τ_att={self.tau_att.item():.4f}s  "
            f"τ_thrust={self.tau_thrust.item():.4f}s  "
            f"D=[{D[0]:.4f}, {D[1]:.4f}, {D[2]:.4f}]  "
            f"m={self.mass.item():.4f}kg"
        )

    def get_params_dict(self) -> dict:
        """Return all physical parameters as a plain dict (for logging)."""
        return {
            "K_v": self.K_v.detach().cpu().numpy().tolist(),
            "tau_att": self.tau_att.item(),
            "tau_thrust": self.tau_thrust.item(),
            "drag": self.drag.detach().cpu().numpy().tolist(),
            "mass": self.mass.item(),
            "gravity": self.gravity.item(),
            "dt": self.dt,
            "num_substeps": self.num_substeps,
        }

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "Q": self.Q,
                "Q_inv": self.Q_inv,
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "dt": self.dt,
                "num_substeps": self.num_substeps,
                "params": self.get_params_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device_str, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
        self.Q = ckpt["Q"].to(self.device_str)
        self.Q_inv = torch.inverse(self.Q)
        self.dt = ckpt["dt"]
        self.num_substeps = ckpt["num_substeps"]
        self.eval()
