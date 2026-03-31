"""
Nominal Markov Transition Models for Degradation Detection.

Two models:
  1. LinearTransitionModel:  v_t = A @ v_{t-1} + B @ u_{t-1} + w,  w ~ N(0, Q)
  2. MLPTransitionModel:     v_t = f_theta(v_{t-1}, u_{t-1}) + w,   w ~ N(0, Q)

Both provide:
  - fit(v_prev, u_prev, v_next):  offline fitting from nominal data
  - predict(v_prev, u_prev):      one-step prediction  v_hat_t
  - residual(v_prev, u_prev, v_next):  r_t = v_t - v_hat_t
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class LinearTransitionModel:
    """v_t = A @ v_{t-1} + B @ u_{t-1},  noise covariance Q estimated from residuals."""

    def __init__(self, state_dim: int = 3, input_dim: int = 3, device: str = "cpu"):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.device = device
        self.A = torch.eye(state_dim, device=device)
        self.B = torch.zeros(state_dim, input_dim, device=device)
        self.Q = torch.eye(state_dim, device=device)
        self.Q_inv = torch.eye(state_dim, device=device)

    def fit(self, v_prev: torch.Tensor, u_prev: torch.Tensor, v_next: torch.Tensor):
        """
        Least-squares fit:  v_next = M @ [v_prev; u_prev]^T
        Args:
            v_prev: (N, d)  previous velocity
            u_prev: (N, m)  velocity command at t-1
            v_next: (N, d)  next velocity
        """
        # Phi = [v_prev, u_prev]  shape (N, d+m)
        Phi = torch.cat([v_prev, u_prev], dim=-1).double()
        Y = v_next.double()

        # M = (Phi^T Phi)^{-1} Phi^T Y  -> shape (d+m, d)
        M, _, _, _ = torch.linalg.lstsq(Phi, Y)
        # M shape: (d+m, d)
        self.A = M[: self.state_dim, :].float().to(self.device)
        self.B = M[self.state_dim :, :].float().to(self.device)

        # Estimate Q from residuals
        with torch.no_grad():
            residuals = v_next.to(self.device) - self.predict(v_prev.to(self.device), u_prev.to(self.device))
            N = residuals.shape[0]
            self.Q = (residuals.T @ residuals) / N
            self.Q_inv = torch.inverse(self.Q)

        return {"A": self.A, "B": self.B, "Q": self.Q}

    def predict(self, v_prev: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        """v_hat_t = A @ v_{t-1} + B @ u_{t-1}"""
        return v_prev @ self.A.T + u_prev @ self.B.T

    def residual(self, v_prev: torch.Tensor, u_prev: torch.Tensor, v_next: torch.Tensor) -> torch.Tensor:
        return v_next - self.predict(v_prev, u_prev)

    def save(self, path: str):
        torch.save({
            "A": self.A, "B": self.B, "Q": self.Q, "Q_inv": self.Q_inv,
            "state_dim": self.state_dim, "input_dim": self.input_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.A = ckpt["A"].to(self.device)
        self.B = ckpt["B"].to(self.device)
        self.Q = ckpt["Q"].to(self.device)
        self.Q_inv = torch.inverse(self.Q)
        self.state_dim = ckpt["state_dim"]
        self.input_dim = ckpt["input_dim"]


class MLPTransitionModel(nn.Module):
    """v_t = f_theta(v_{t-1}, u_{t-1}),  noise covariance Q estimated from residuals."""

    def __init__(self, state_dim: int = 3, input_dim: int = 3,
                 hidden_dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.device_str = device

        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        ).to(device)

        self.Q = torch.eye(state_dim, device=device)
        self.Q_inv = torch.eye(state_dim, device=device)

    def forward(self, v_prev: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([v_prev, u_prev], dim=-1)
        return self.net(x)

    def predict(self, v_prev: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(v_prev, u_prev)

    def residual(self, v_prev: torch.Tensor, u_prev: torch.Tensor, v_next: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return v_next - self.predict(v_prev, u_prev)

    def fit(self, v_prev: torch.Tensor, u_prev: torch.Tensor, v_next: torch.Tensor,
            lr: float = 1e-3, epochs: int = 1000, batch_size: int = 4096, verbose: bool = True):
        """Train MLP with MSE loss, then estimate Q from residuals."""
        device = self.device_str
        v_prev = v_prev.float().to(device)
        u_prev = u_prev.float().to(device)
        v_next = v_next.float().to(device)

        N = v_prev.shape[0]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(v_prev, u_prev, v_next)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for vp_b, up_b, vn_b in loader:
                pred = self.forward(vp_b, up_b)
                loss = nn.functional.mse_loss(pred, vn_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * vp_b.shape[0]
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  [MLP] epoch {epoch+1}/{epochs}  loss={epoch_loss/N:.6f}")
        self.eval()

        # Estimate Q from residuals on full dataset
        with torch.no_grad():
            residuals = v_next - self.forward(v_prev, u_prev)
            self.Q = (residuals.T @ residuals) / N
            self.Q_inv = torch.inverse(self.Q)

        return {"Q": self.Q}

    def save(self, path: str):
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "Q": self.Q, "Q_inv": self.Q_inv,
            "state_dim": self.state_dim, "input_dim": self.input_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device_str, weights_only=False)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.Q = ckpt["Q"].to(self.device_str)
        self.Q_inv = torch.inverse(self.Q)
        self.state_dim = ckpt["state_dim"]
        self.input_dim = ckpt["input_dim"]
        self.eval()
