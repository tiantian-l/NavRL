"""
Input-dependent residual distribution models.

Instead of assuming r_t ~ N(mu, Q) with global fixed parameters,
these models learn mu(x) and sigma(x) as functions of the input x = [v_{t-1}, u_{t-1}].

Motivation:
  - In different operating regimes (hover, cruise, maneuver), the transition model
    has different prediction accuracy.
  - A global Q averages over all regimes → appears heavy-tailed overall
  - Input-dependent sigma(x) captures heteroscedasticity → z-scores become
    locally well-calibrated (close to standard normal under H0)

Two implementations:
  1. HeteroscedasticMLP — neural network predicting per-axis mean and log-variance
     (parametric, always available with PyTorch, fast inference)
  2. SparseGPResidualModel — sparse variational GP per axis
     (non-parametric, principled uncertainty, requires gpytorch)

Both provide:
  - fit(x, r): train on nominal residuals
  - predict(x) -> (mu, sigma): per-axis predictive mean and std
  - save(path) / load(path)

Reference:
  - Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian
    Deep Learning for Computer Vision?" NeurIPS.
  - Hensman, J. et al. (2015). "Scalable Variational Gaussian Process
    Classification." AISTATS.
"""

import os
import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────────────────────
#  1. Heteroscedastic MLP (primary, no extra dependencies)
# ──────────────────────────────────────────────────────────────

class HeteroscedasticMLP(nn.Module):
    """
    MLP that predicts per-axis residual mean and log-variance.

    r_i | x ~ N(mu_i(x), sigma_i^2(x))

    where x = [v_{t-1}, u_{t-1}] (6D) and i ∈ {x, y, z}.

    Training loss: Gaussian negative log-likelihood (NLL)
      L = (1/N) sum_t sum_i [ 0.5*logvar_i(x_t) + 0.5*(r_{t,i} - mu_i(x_t))^2 / exp(logvar_i(x_t)) ]

    This is equivalent to maximum likelihood estimation of a heteroscedastic
    Gaussian model, and provides a parametric approximation to GP regression.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 3,
                 hidden_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device_str = device

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)

        # Mean head: predicts E[r_i | x]
        self.mu_head = nn.Linear(hidden_dim, output_dim).to(device)

        # Log-variance head: predicts log(sigma_i^2(x))
        # Initialized to log(1) = 0, so initial sigma = 1
        self.logvar_head = nn.Linear(hidden_dim, output_dim).to(device)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, input_dim) input features [v_{t-1}, u_{t-1}]
        Returns:
            mu: (N, output_dim) per-axis mean
            logvar: (N, output_dim) per-axis log-variance
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def predict(self, x: torch.Tensor):
        """
        Predictive mean and std (for anomaly detection).

        Args:
            x: (N, input_dim) or (input_dim,) — input features
        Returns:
            mu: (N, output_dim) or (output_dim,)
            sigma: (N, output_dim) or (output_dim,) — per-axis std, clamped > 0
        """
        squeeze = (x.dim() == 1)
        if squeeze:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x)
            sigma = torch.exp(0.5 * logvar).clamp(min=1e-6)
        if squeeze:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
        return mu, sigma

    def fit(self, x: torch.Tensor, r: torch.Tensor,
            lr: float = 1e-3, epochs: int = 500, batch_size: int = 4096,
            verbose: bool = True):
        """
        Train with Gaussian NLL loss.

        Args:
            x: (N, input_dim) inputs [v_{t-1}, u_{t-1}]
            r: (N, output_dim) residuals
            lr: learning rate
            epochs: training epochs
            batch_size: mini-batch size
            verbose: print progress every 100 epochs
        Returns:
            dict with training stats
        """
        device = self.device_str
        x = x.float().to(device)
        r = r.float().to(device)
        N = x.shape[0]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(x, r)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_b, r_b in loader:
                mu, logvar = self.forward(x_b)
                # Gaussian NLL: 0.5 * [logvar + (r - mu)^2 / exp(logvar)]
                nll = 0.5 * (logvar + (r_b - mu) ** 2 / torch.exp(logvar).clamp(min=1e-8))
                loss = nll.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_b.shape[0]

            avg_loss = epoch_loss / N
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  [HeteroscedasticMLP] epoch {epoch+1}/{epochs}  NLL={avg_loss:.6f}")

        self.eval()

        # Compute summary stats
        with torch.no_grad():
            mu_all, logvar_all = self.forward(x)
            sigma_all = torch.exp(0.5 * logvar_all)
            z_all = torch.abs(r - mu_all) / sigma_all.clamp(min=1e-6)

        return {
            "final_nll": losses[-1],
            "sigma_mean": sigma_all.mean(dim=0).cpu().tolist(),
            "sigma_std": sigma_all.std(dim=0).cpu().tolist(),
            "sigma_min": sigma_all.min(dim=0).values.cpu().tolist(),
            "sigma_max": sigma_all.max(dim=0).values.cpu().tolist(),
            "z_mean": z_all.mean().item(),
            "z_std": z_all.std().item(),
        }

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.backbone[0].in_features,
        }, path)
        # hidden_dim is inferred from first layer

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device_str, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
        self.eval()


# ──────────────────────────────────────────────────────────────
#  2. Sparse Variational GP (optional, requires gpytorch)
# ──────────────────────────────────────────────────────────────

def _check_gpytorch():
    try:
        import gpytorch
        return True
    except ImportError:
        return False


class SparseGPResidualModel:
    """
    Sparse variational GP for residual modeling.

    Fits 3 independent SVGPs (one per velocity axis) on the residuals.
    Each GP: input = [v_{t-1}, u_{t-1}] (6D), output = r_i (scalar).

    Uses GPyTorch's VariationalStrategy with inducing points.
    RBF kernel with ARD (automatic relevance determination).

    Requires: pip install gpytorch

    Reference:
      Hensman, J., Matthews, A.G., Ghahramani, Z. (2015).
      "Scalable Variational Gaussian Process Classification." AISTATS.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 3,
                 num_inducing: int = 500, device: str = "cpu"):
        if not _check_gpytorch():
            raise ImportError("gpytorch is required for SparseGPResidualModel. "
                              "Install with: pip install gpytorch")
        import gpytorch

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.device = device

        # Will be created during fit() when we know the data
        self.models = None
        self.likelihoods = None

    def _create_gp(self, inducing_points):
        """Create a single-axis SVGP model."""
        import gpytorch

        class _SVGP(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
                )

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _SVGP(inducing_points)

    def fit(self, x: torch.Tensor, r: torch.Tensor,
            lr: float = 0.01, epochs: int = 200, batch_size: int = 4096,
            verbose: bool = True):
        """
        Train 3 independent SVGPs on residuals.

        Args:
            x: (N, 6) inputs [v_{t-1}, u_{t-1}]
            r: (N, 3) residuals per axis
        """
        import gpytorch

        x = x.float().to(self.device)
        r = r.float().to(self.device)
        N = x.shape[0]

        # Select inducing points via k-means or random subset
        perm = torch.randperm(N)[:self.num_inducing]
        inducing_x = x[perm].clone()

        self.models = []
        self.likelihoods = []

        for axis in range(self.output_dim):
            print(f"  [GP] Fitting axis {axis} ({['x','y','z'][axis]}) ...")

            model = self._create_gp(inducing_x.clone()).to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=lr)

            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

            dataset = torch.utils.data.TensorDataset(x, r[:, axis])
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                epoch_loss = 0.0
                for x_b, r_b in loader:
                    optimizer.zero_grad()
                    output = model(x_b)
                    loss = -mll(output, r_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * x_b.shape[0]
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"    epoch {epoch+1}/{epochs}  ELBO={-epoch_loss/N:.6f}")

            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

        return {}

    def predict(self, x: torch.Tensor):
        """
        Predictive mean and std for all 3 axes.

        Args:
            x: (N, 6) or (6,) input features
        Returns:
            mu: (N, 3) or (3,) predictive mean
            sigma: (N, 3) or (3,) predictive std
        """
        import gpytorch

        squeeze = (x.dim() == 1)
        if squeeze:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        mus, sigmas = [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(self.output_dim):
                pred = self.likelihoods[i](self.models[i](x))
                mus.append(pred.mean)
                sigmas.append(pred.stddev)

        mu = torch.stack(mus, dim=-1)
        sigma = torch.stack(sigmas, dim=-1).clamp(min=1e-6)

        if squeeze:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
        return mu, sigma

    def save(self, path: str):
        save_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_inducing": self.num_inducing,
        }
        for i in range(self.output_dim):
            save_dict[f"model_{i}"] = self.models[i].state_dict()
            save_dict[f"likelihood_{i}"] = self.likelihoods[i].state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        import gpytorch

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.output_dim = ckpt["output_dim"]
        self.num_inducing = ckpt["num_inducing"]

        self.models = []
        self.likelihoods = []

        for i in range(self.output_dim):
            # Reconstruct inducing points shape from saved state
            inducing_key = "variational_strategy.inducing_points"
            inducing_pts = ckpt[f"model_{i}"][inducing_key]

            model = self._create_gp(inducing_pts).to(self.device)
            model.load_state_dict(ckpt[f"model_{i}"])
            model.eval()

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            likelihood.load_state_dict(ckpt[f"likelihood_{i}"])
            likelihood.eval()

            self.models.append(model)
            self.likelihoods.append(likelihood)
