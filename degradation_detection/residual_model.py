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

    def get_diagnostics(self) -> dict:
        """
        Extract diagnostic info from trained Sparse GP models.

        Returns dict with per-axis kernel hyperparameters, noise, and
        input relevance — allows checking if training was adequate.
        """
        if self.models is None:
            return {"error": "Models not trained yet"}

        diagnostics = {}
        axis_labels = ["vx", "vy", "vz"]
        input_names = ["v_x", "v_y", "v_z", "u_x", "u_y", "u_z"]

        for i in range(self.output_dim):
            model = self.models[i]
            likelihood = self.likelihoods[i]

            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze()
            os_val = model.covar_module.outputscale.item()
            noise = likelihood.noise.item()

            relevance = 1.0 / ls.numpy()
            relevance_norm = relevance / relevance.sum()

            diagnostics[axis_labels[i]] = {
                "lengthscales": ls.numpy().tolist(),
                "outputscale": os_val,
                "noise": noise,
                "signal_to_noise": os_val / max(noise, 1e-10),
                "input_relevance": {
                    name: f"{rel:.1%}" for name, rel in zip(input_names, relevance_norm)
                },
                "num_inducing": model.variational_strategy.inducing_points.shape[0],
            }

        return diagnostics

    def print_diagnostics(self):
        """Pretty-print Sparse GP diagnostic information."""
        diag = self.get_diagnostics()
        if "error" in diag:
            print(f"  {diag['error']}")
            return

        print("\n  ========== Sparse GP Diagnostics ==========")
        for axis_name, d in diag.items():
            print(f"\n  --- {axis_name} ---")
            print(f"  Kernel: outputscale={d['outputscale']:.6f}  "
                  f"noise={d['noise']:.6f}  "
                  f"SNR={d['signal_to_noise']:.1f}")
            print(f"  Lengthscales: {['%.3f' % l for l in d['lengthscales']]}")
            print(f"  Input relevance: {d['input_relevance']}")
            print(f"  Inducing points: {d['num_inducing']}")


# ──────────────────────────────────────────────────────────────
#  3. Exact (Full) GP for v_x, v_y  (optional, requires gpytorch)
# ──────────────────────────────────────────────────────────────

class ExactGPResidualModel:
    """
    Exact (Full) GP for residual modeling of v_x and v_y.

    Unlike Sparse GP, Full GP uses the exact posterior — no variational
    approximation, no inducing points.  The trade-off is O(N^3) cost, so
    we subsample the training data to a manageable size (default: 3000).

    Two independent single-output GPs:
      GP_vx: input = [v_{t-1}, u_{t-1}] (6D) -> residual r_x (scalar)
      GP_vy: input = [v_{t-1}, u_{t-1}] (6D) -> residual r_y (scalar)

    Kernel: ScaleKernel(RBF(ARD)) — auto learns per-dimension lengthscales.

    Built-in diagnostics:
      - NLPD (Negative Log Predictive Density) on held-out data
      - Learned kernel hyperparameters (lengthscales, outputscale, noise)
      - Standardized residual calibration check
      - Input relevance ranking from ARD lengthscales

    For v_z: falls back to global sigma (least variable axis).

    Reference:
      Rasmussen & Williams, "Gaussian Processes for Machine Learning",
      MIT Press 2006.
    """

    def __init__(self, input_dim: int = 6, max_train_size: int = 3000,
                 device: str = "cpu"):
        if not _check_gpytorch():
            raise ImportError("gpytorch required. Install: pip install gpytorch")

        self.input_dim = input_dim
        self.output_dim = 3  # interface compatibility (predict returns 3D)
        self.max_train_size = max_train_size
        self.device = device
        self.axis_names = ["vx", "vy"]
        self.num_gp_axes = 2  # only vx, vy get GPs

        self.models = None
        self.likelihoods = None
        self.train_x = None
        self.train_y = None  # list of (N,) per axis

        # Global fallback sigma for vz (axis 2)
        self.vz_mu = 0.0
        self.vz_sigma = 1.0

    def _create_exact_gp(self, train_x, train_y):
        """Create a single-axis ExactGP model class."""
        import gpytorch

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
                )

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return _ExactGP

    def fit(self, x: torch.Tensor, r: torch.Tensor,
            lr: float = 0.1, epochs: int = 100,
            verbose: bool = True):
        """
        Train 2 independent ExactGPs (vx, vy) on residuals.

        Args:
            x: (N, 6) inputs [v_{t-1}, u_{t-1}]
            r: (N, 3) residuals per axis
        Returns:
            dict with diagnostics per axis
        """
        import gpytorch

        x = x.float().to(self.device)
        r = r.float().to(self.device)
        N = x.shape[0]

        # --- Subsample if needed ---
        if N > self.max_train_size:
            perm = torch.randperm(N)[:self.max_train_size]
            mask = _mask_from_indices(N, perm, self.device)
            x_train = x[perm]
            r_train = r[perm]
            x_val = x[~mask]
            r_val = r[~mask]
            print(f"  [ExactGP] Subsampled {self.max_train_size}/{N} for training, "
                  f"{N - self.max_train_size} for validation")
        else:
            x_train = x
            r_train = r
            x_val = None
            r_val = None

        self.train_x = x_train
        self.train_y = []
        self.models = []
        self.likelihoods = []

        # vz global fallback
        self.vz_mu = r[:, 2].mean().item()
        self.vz_sigma = r[:, 2].std().clamp(min=1e-8).item()

        diagnostics = {}

        for axis_idx, axis_name in enumerate(self.axis_names):
            print(f"\n  [ExactGP] Fitting axis {axis_idx} ({axis_name}) "
                  f"with {x_train.shape[0]} points ...")

            y_train = r_train[:, axis_idx]
            self.train_y.append(y_train)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            ).to(self.device)
            model_cls = self._create_exact_gp(x_train, y_train)
            model = model_cls(x_train, y_train, likelihood).to(self.device)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(x_train)
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    epoch {epoch+1}/{epochs}  -MLL={loss.item():.4f}")

            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

            # ---- Diagnostics ----
            diag = self._compute_diagnostics(
                model, likelihood, x_train, y_train, x_val,
                r_val[:, axis_idx] if r_val is not None else None,
                axis_name, losses
            )
            diagnostics[axis_name] = diag

        if verbose:
            self._print_diagnostics(diagnostics)

        return diagnostics

    def _compute_diagnostics(self, model, likelihood, x_train, y_train,
                             x_val, y_val, axis_name, losses):
        """Compute per-axis GP quality diagnostics."""
        import gpytorch

        diag = {"final_mll": -losses[-1], "converged": True}

        # Check convergence
        if len(losses) > 20:
            early_loss = np.mean(losses[:10])
            late_loss = np.mean(losses[-10:])
            diag["loss_reduction"] = early_loss - late_loss
            diag["converged"] = late_loss < early_loss

        # Kernel hyperparameters
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze()
        diag["lengthscales"] = ls.numpy().tolist()
        diag["outputscale"] = model.covar_module.outputscale.item()
        diag["noise"] = likelihood.noise.item()
        diag["signal_to_noise"] = diag["outputscale"] / max(diag["noise"], 1e-10)

        # Input relevance ranking (shorter lengthscale = more relevant)
        input_names = ["v_x", "v_y", "v_z", "u_x", "u_y", "u_z"]
        relevance = 1.0 / ls.numpy()
        relevance_norm = relevance / relevance.sum()
        diag["input_relevance"] = {
            name: f"{rel:.1%}" for name, rel in zip(input_names, relevance_norm)
        }

        # Training NLPD
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_pred = likelihood(model(x_train))
            train_nlpd = -train_pred.log_prob(y_train) / len(y_train)
            diag["train_nlpd"] = train_nlpd.item()

            # Standardized residuals on training set
            train_z = (y_train - train_pred.mean) / train_pred.stddev.clamp(min=1e-8)
            diag["train_z_mean"] = train_z.mean().item()
            diag["train_z_std"] = train_z.std().item()

        # Validation NLPD (if available)
        if x_val is not None and y_val is not None and len(y_val) > 0:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_pred = likelihood(model(x_val))
                val_nlpd = -val_pred.log_prob(y_val) / len(y_val)
                diag["val_nlpd"] = val_nlpd.item()

                val_z = (y_val - val_pred.mean) / val_pred.stddev.clamp(min=1e-8)
                diag["val_z_mean"] = val_z.mean().item()
                diag["val_z_std"] = val_z.std().item()

                # Coverage: fraction of val points within 2-sigma
                in_2sigma = (val_z.abs() < 2.0).float().mean().item()
                diag["val_coverage_2sigma"] = in_2sigma  # should be ~0.954

        return diag

    def _print_diagnostics(self, diagnostics):
        """Pretty-print GP diagnostics."""
        print("\n  ========== ExactGP Diagnostics ==========")
        for axis_name, d in diagnostics.items():
            print(f"\n  --- {axis_name} ---")
            print(f"  Final MLL: {d['final_mll']:.4f}  "
                  f"Converged: {d['converged']}")
            print(f"  Kernel: outputscale={d['outputscale']:.6f}  "
                  f"noise={d['noise']:.6f}  "
                  f"SNR={d['signal_to_noise']:.1f}")
            print(f"  Lengthscales: {['%.3f' % l for l in d['lengthscales']]}")
            print(f"  Input relevance: {d['input_relevance']}")
            print(f"  Train: NLPD={d['train_nlpd']:.4f}  "
                  f"z_mean={d['train_z_mean']:.3f}  "
                  f"z_std={d['train_z_std']:.3f}")
            if "val_nlpd" in d:
                print(f"  Val:   NLPD={d['val_nlpd']:.4f}  "
                      f"z_mean={d['val_z_mean']:.3f}  "
                      f"z_std={d['val_z_std']:.3f}  "
                      f"2sigma-coverage={d['val_coverage_2sigma']:.1%}")
                # Quality warnings
                if d["val_z_std"] > 1.5:
                    print(f"  ⚠ Underconfident: val z_std={d['val_z_std']:.2f} >> 1.0 "
                          f"(GP predicts too-wide intervals)")
                elif d["val_z_std"] < 0.7:
                    print(f"  ⚠ Overconfident: val z_std={d['val_z_std']:.2f} << 1.0 "
                          f"(GP predicts too-tight intervals)")
                if d["val_coverage_2sigma"] < 0.90:
                    print(f"  ⚠ Poor coverage: only {d['val_coverage_2sigma']:.1%} within 2sigma "
                          f"(expected ~95.4%)")

    def predict(self, x: torch.Tensor):
        """
        Predictive mean and std for all 3 axes.

        vx, vy: from ExactGP
        vz: from global fallback (constant mu, sigma)

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
        N = x.shape[0]
        mu = torch.zeros(N, 3, device=self.device)
        sigma = torch.ones(N, 3, device=self.device)

        # GP predictions for vx, vy
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(self.num_gp_axes):
                pred = self.likelihoods[i](self.models[i](x))
                mu[:, i] = pred.mean
                sigma[:, i] = pred.stddev

        # vz fallback
        mu[:, 2] = self.vz_mu
        sigma[:, 2] = self.vz_sigma

        sigma = sigma.clamp(min=1e-6)

        if squeeze:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
        return mu, sigma

    def save(self, path: str):
        save_dict = {
            "input_dim": self.input_dim,
            "max_train_size": self.max_train_size,
            "vz_mu": self.vz_mu,
            "vz_sigma": self.vz_sigma,
        }
        for i in range(self.num_gp_axes):
            save_dict[f"model_{i}"] = self.models[i].state_dict()
            save_dict[f"likelihood_{i}"] = self.likelihoods[i].state_dict()
            save_dict[f"train_x_{i}"] = self.train_x.cpu()
            save_dict[f"train_y_{i}"] = self.train_y[i].cpu()
        torch.save(save_dict, path)

    def load(self, path: str):
        import gpytorch

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = ckpt["input_dim"]
        self.max_train_size = ckpt["max_train_size"]
        self.vz_mu = ckpt["vz_mu"]
        self.vz_sigma = ckpt["vz_sigma"]

        self.models = []
        self.likelihoods = []
        self.train_y = []

        for i in range(self.num_gp_axes):
            train_x = ckpt[f"train_x_{i}"].to(self.device)
            train_y = ckpt[f"train_y_{i}"].to(self.device)
            self.train_y.append(train_y)

            if i == 0:
                self.train_x = train_x

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            ).to(self.device)
            model_cls = self._create_exact_gp(train_x, train_y)
            model = model_cls(train_x, train_y, likelihood).to(self.device)
            model.load_state_dict(ckpt[f"model_{i}"])
            likelihood.load_state_dict(ckpt[f"likelihood_{i}"])
            model.eval()
            likelihood.eval()

            self.models.append(model)
            self.likelihoods.append(likelihood)


def _mask_from_indices(N: int, indices: torch.Tensor, device: str) -> torch.Tensor:
    """Create a boolean mask of shape (N,) with True at given indices."""
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask
