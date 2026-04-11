import argparse
import os
import copy
import torch
import gpytorch

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =========================
# 1. Utilities
# =========================

def print_split_info(train_idx, val_idx):
    print(f"Train samples: {len(train_idx)}")
    print(f"Val   samples: {len(val_idx)}")


def make_episode_split(ep_lengths, train_ratio=0.9):
    num_eps = len(ep_lengths)
    if num_eps < 2:
        raise ValueError("Episode-aware split requires at least 2 episodes.")

    ep_perm = torch.randperm(num_eps)
    ep_split = max(1, min(int(train_ratio * num_eps), num_eps - 1))

    train_ep_ids = ep_perm[:ep_split]
    val_ep_ids = ep_perm[ep_split:]

    ep_offsets = torch.cat([torch.tensor([0]), ep_lengths.cumsum(0)])

    train_idx = torch.cat([
        torch.arange(ep_offsets[i], ep_offsets[i] + ep_lengths[i])
        for i in train_ep_ids
    ])

    val_idx = torch.cat([
        torch.arange(ep_offsets[i], ep_offsets[i] + ep_lengths[i])
        for i in val_ep_ids
    ])

    return train_idx, val_idx


def make_random_split(N, train_ratio=0.9):
    if N < 2:
        raise ValueError(f"Need at least 2 samples for train/val split, got N={N}")

    perm = torch.randperm(N)
    split = max(1, min(int(train_ratio * N), N - 1))
    return perm[:split], perm[split:]


def clamp_std(x, eps=1e-6):
    return torch.clamp(x, min=eps)


# =========================
# 2. Mean function
# residual dynamics: mean = 0
# =========================

class ZeroResidualMean(gpytorch.means.Mean):
    def forward(self, x):
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)


# =========================
# 3. Exact GP model (single output)
# =========================

class ExactResidualGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroResidualMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# =========================
# 4. Sparse GP model (single output)
# =========================

class SparseResidualGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ZeroResidualMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# =========================
# 5. Model wrappers
# =========================

class FullGPResidualModel2D:
    """
    训练 2 个独立的 exact GP，拟合 dvx / dvy
    """
    def __init__(self, input_dim=4, output_dim=2, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.models = []
        self.likelihoods = []

    def fit(self, x_train, y_train, training_iter=150, lr=0.05, verbose=True):
        self.models = []
        self.likelihoods = []

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        for d in range(self.output_dim):
            dim_name = "dvx" if d == 0 else "dvy"

            if verbose:
                print(f"\n[Full GP Residual] Training {dim_name} ...")

            y_d = y_train[:, d]

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = ExactResidualGPModel(
                train_x=x_train,
                train_y=y_d,
                likelihood=likelihood,
            ).to(self.device)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(x_train)
                loss = -mll(output, y_d)
                loss.backward()
                optimizer.step()

                if verbose and ((i + 1) % 25 == 0 or i == 0):
                    noise = likelihood.noise.item()
                    lengthscale = model.covar_module.base_kernel.lengthscale.mean().item()
                    outputscale = model.covar_module.outputscale.item()
                    print(
                        f"  iter {i+1:3d}/{training_iter} | "
                        f"loss={loss.item():.6f} | "
                        f"noise={noise:.6f} | "
                        f"ls={lengthscale:.4f} | "
                        f"os={outputscale:.4f}"
                    )

            self.models.append(model)
            self.likelihoods.append(likelihood)

    @torch.no_grad()
    def predict(self, x_test):
        x_test = x_test.to(self.device)

        means = []
        stds = []

        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()

            with gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(x_test))
                means.append(pred_dist.mean.unsqueeze(-1))
                stds.append(pred_dist.stddev.unsqueeze(-1))

        mean = torch.cat(means, dim=-1)
        std = torch.cat(stds, dim=-1)
        return mean, std

    def save(self, path):
        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "target": "delta_v",
            "state_dicts": [],
        }
        for model, likelihood in zip(self.models, self.likelihoods):
            payload["state_dicts"].append({
                "model": model.state_dict(),
                "likelihood": likelihood.state_dict(),
            })
        torch.save(payload, path)


class SparseGPResidualModel2D:
    """
    训练 2 个独立的 sparse variational GP，拟合 dvx / dvy
    """
    def __init__(self, input_dim=4, output_dim=2, num_inducing=512, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.device = device

        self.models = []
        self.likelihoods = []

    def _init_inducing_points(self, x_train, num_inducing, method="kmeans"):
        n = x_train.shape[0]
        if n <= num_inducing:
            return x_train.clone()

        if method == "kmeans" and SKLEARN_AVAILABLE:
            x_np = x_train.detach().cpu().numpy()
            kmeans = MiniBatchKMeans(
                n_clusters=num_inducing,
                batch_size=min(8192, n),
                n_init=10,
                random_state=0,
            )
            kmeans.fit(x_np)
            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=x_train.dtype,
                device=x_train.device,
            )
            return centers

        if method == "kmeans" and not SKLEARN_AVAILABLE:
            print("Warning: sklearn not found, fallback to random inducing initialization.")

        idx = torch.randperm(n, device=x_train.device)[:num_inducing]
        return x_train[idx].clone()

    def fit(
        self,
        x_train,
        y_train,
        training_iter=300,
        lr=1e-2,
        batch_size=4096,
        noise_lower_bound=1e-5,
        inducing_init="kmeans",
        verbose=True,
    ):
        self.models = []
        self.likelihoods = []

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        num_data = x_train.size(0)

        for d in range(self.output_dim):
            dim_name = "dvx" if d == 0 else "dvy"

            if verbose:
                print(f"\n[Sparse GP Residual] Training {dim_name} ...")

            inducing_points = self._init_inducing_points(
                x_train,
                self.num_inducing,
                method=inducing_init,
            )

            model = SparseResidualGPModel(
                inducing_points=inducing_points,
            ).to(self.device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound)
            ).to(self.device)

            model.covar_module.outputscale = 0.01
            model.covar_module.base_kernel.lengthscale = 1.0

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()),
                lr=lr,
            )

            mll = gpytorch.mlls.VariationalELBO(
                likelihood,
                model,
                num_data=num_data,
            )

            best_loss = float("inf")
            best_model_state = None
            best_likelihood_state = None

            for epoch in range(training_iter):
                epoch_loss = 0.0

                for xb, yb in loader:
                    optimizer.zero_grad()
                    output = model(xb)
                    loss = -mll(output, yb[:, d])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)

                epoch_loss /= num_data

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_likelihood_state = copy.deepcopy(likelihood.state_dict())

                if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
                    noise = likelihood.noise.item()
                    ls = model.covar_module.base_kernel.lengthscale.mean().item()
                    os_ = model.covar_module.outputscale.item()
                    print(
                        f"  epoch {epoch+1:3d}/{training_iter} | "
                        f"loss={epoch_loss:.6f} | "
                        f"noise={noise:.6f} | "
                        f"ls={ls:.4f} | "
                        f"os={os_:.4f}"
                    )

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            if best_likelihood_state is not None:
                likelihood.load_state_dict(best_likelihood_state)

            self.models.append(model)
            self.likelihoods.append(likelihood)

    @torch.no_grad()
    def predict(self, x_test):
        x_test = x_test.to(self.device)

        means = []
        stds = []

        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()

            with gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(x_test))
                means.append(pred_dist.mean.unsqueeze(-1))
                stds.append(pred_dist.stddev.unsqueeze(-1))

        mean = torch.cat(means, dim=-1)
        std = torch.cat(stds, dim=-1)
        return mean, std

    def save(self, path):
        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_inducing": self.num_inducing,
            "target": "delta_v",
            "state_dicts": [],
        }
        for model, likelihood in zip(self.models, self.likelihoods):
            payload["state_dicts"].append({
                "model": model.state_dict(),
                "likelihood": likelihood.state_dict(),
            })
        torch.save(payload, path)


# =========================
# 6. Evaluation
# =========================

@torch.no_grad()
def evaluate_probabilistic_regression(y_true, mean, std, prefix="GP"):
    std = clamp_std(std)

    rmse_all = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    rmse_dim = torch.sqrt(torch.mean((mean - y_true) ** 2, dim=0)).cpu().numpy()

    var = std ** 2
    nll = 0.5 * (((y_true - mean) ** 2) / var + torch.log(2 * torch.pi * var))
    nll_all = nll.mean().item()
    nll_dim = nll.mean(dim=0).cpu().numpy()

    z = (y_true - mean) / std
    z_mean = z.mean(dim=0).cpu().numpy()
    z_std = z.std(dim=0).cpu().numpy()

    cover_68 = ((y_true >= mean - 1.0 * std) & (y_true <= mean + 1.0 * std)).float().mean(dim=0).cpu().numpy()
    cover_95 = ((y_true >= mean - 1.96 * std) & (y_true <= mean + 1.96 * std)).float().mean(dim=0).cpu().numpy()
    cover_997 = ((y_true >= mean - 3.0 * std) & (y_true <= mean + 3.0 * std)).float().mean(dim=0).cpu().numpy()

    print(f"\n===== {prefix} Evaluation =====")
    print(f"Overall RMSE: {rmse_all:.6f}")
    print(f"Overall NLL : {nll_all:.6f}")
    print(f"Per-dim RMSE: {rmse_dim}")
    print(f"Per-dim NLL : {nll_dim}")
    print(f"z mean      : {z_mean}")
    print(f"z std       : {z_std}")
    print(f"68% cover   : {cover_68}")
    print(f"95% cover   : {cover_95}")
    print(f"99.7% cover : {cover_997}")

    return {
        "rmse_all": rmse_all,
        "rmse_dim": rmse_dim,
        "nll_all": nll_all,
        "nll_dim": nll_dim,
        "z_mean": z_mean,
        "z_std": z_std,
        "cover_68": cover_68,
        "cover_95": cover_95,
        "cover_997": cover_997,
    }


# =========================
# 7. Data loading
# =========================

def load_xy_data(data_path):
    print(f"Loading data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu")

    required_keys = ["state", "action", "next_state"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in dataset. Expected keys: {required_keys}")

    state = data["state"].float()
    action = data["action"].float()
    next_state = data["next_state"].float()

    v_prev = state[:, :2]
    u_prev = action[:, :2]
    v_next = next_state[:, :2]
    delta_v = v_next - v_prev

    ep_lengths = data.get("ep_lengths", None)

    print(f"Loaded {v_prev.shape[0]} samples")
    print("Using XY dimensions only:")
    print("  input  = [vx_prev, vy_prev, ux_prev, uy_prev]")
    print("  target = [dvx, dvy]")

    return v_prev, u_prev, v_next, delta_v, ep_lengths


def make_split_indices(N, ep_lengths):
    if ep_lengths is not None and len(ep_lengths) > 1:
        print("Using episode-aware split (90% train / 10% val)")
        train_idx, val_idx = make_episode_split(ep_lengths, train_ratio=0.9)
    else:
        print("No episode info, using random split (90% train / 10% val)")
        train_idx, val_idx = make_random_split(N, train_ratio=0.9)

    print_split_info(train_idx, val_idx)
    return train_idx, val_idx


# =========================
# 8. Training pipelines
# =========================

def run_full_gp(
    v_prev, u_prev, v_next, delta_v,
    train_idx, val_idx,
    output_dir,
    device,
    gp_sample_size,
    gp_epochs,
    gp_lr,
):
    x_train_full = torch.cat([v_prev[train_idx], u_prev[train_idx]], dim=1)
    y_train_full = delta_v[train_idx]

    x_val = torch.cat([v_prev[val_idx], u_prev[val_idx]], dim=1)
    y_val_delta = delta_v[val_idx]
    y_val_next = v_next[val_idx]
    v_prev_val = v_prev[val_idx]

    n_train = x_train_full.shape[0]
    gp_sample_size = min(gp_sample_size, n_train)

    sample_idx = torch.randperm(n_train)[:gp_sample_size]
    x_gp_train = x_train_full[sample_idx]
    y_gp_train = y_train_full[sample_idx]

    print(f"Sampled {gp_sample_size} transition points from 90% train set for exact residual GP")

    print(f"\n===== Train Full GP Residual Dynamics Model (XY only) =====")
    gp_model = FullGPResidualModel2D(input_dim=4, output_dim=2, device=device)
    gp_model.fit(
        x_gp_train,
        y_gp_train,
        training_iter=gp_epochs,
        lr=gp_lr,
        verbose=True,
    )

    gp_path = os.path.join(output_dir, "full_gp_residual_dynamics_xy.pt")
    gp_model.save(gp_path)
    print(f"Saved full GP residual model to {gp_path}")

    with torch.no_grad():
        pred_delta_mean, pred_delta_std = gp_model.predict(x_val)
        pred_delta_mean = pred_delta_mean.cpu()
        pred_delta_std = pred_delta_std.cpu()

    print(f"\n===== Evaluate Full GP on held-out 10%: delta_v =====")
    delta_metrics = evaluate_probabilistic_regression(
        y_true=y_val_delta,
        mean=pred_delta_mean,
        std=pred_delta_std,
        prefix="Full GP Residual Model (delta_v)",
    )

    pred_v_next_mean = v_prev_val + pred_delta_mean
    pred_v_next_std = pred_delta_std

    print(f"\n===== Evaluate Full GP on held-out 10%: reconstructed v_next =====")
    next_metrics = evaluate_probabilistic_regression(
        y_true=y_val_next,
        mean=pred_v_next_mean,
        std=pred_v_next_std,
        prefix="Full GP Residual Model (reconstructed v_next)",
    )

    return {
        "gp_model": gp_model,
        "delta_metrics": delta_metrics,
        "next_metrics": next_metrics,
    }


def run_sparse_gp(
    v_prev, u_prev, v_next, delta_v,
    train_idx, val_idx,
    output_dir,
    device,
    num_inducing,
    gp_epochs,
    gp_lr,
    batch_size,
    noise_lower_bound,
    inducing_init,
):
    x_train = torch.cat([v_prev[train_idx], u_prev[train_idx]], dim=1)
    y_train = delta_v[train_idx]

    x_val = torch.cat([v_prev[val_idx], u_prev[val_idx]], dim=1)
    y_val_delta = delta_v[val_idx]
    y_val_next = v_next[val_idx]
    v_prev_val = v_prev[val_idx]

    print(f"\n===== Train Sparse GP Residual Dynamics Model (XY only) =====")
    gp_model = SparseGPResidualModel2D(
        input_dim=4,
        output_dim=2,
        num_inducing=num_inducing,
        device=device,
    )

    gp_model.fit(
        x_train,
        y_train,
        training_iter=gp_epochs,
        lr=gp_lr,
        batch_size=batch_size,
        noise_lower_bound=noise_lower_bound,
        inducing_init=inducing_init,
        verbose=True,
    )

    gp_path = os.path.join(output_dir, "sparse_gp_residual_dynamics_xy.pt")
    gp_model.save(gp_path)
    print(f"Saved sparse GP residual model to {gp_path}")

    with torch.no_grad():
        pred_delta_mean, pred_delta_std = gp_model.predict(x_val)
        pred_delta_mean = pred_delta_mean.cpu()
        pred_delta_std = pred_delta_std.cpu()

    print(f"\n===== Evaluate Sparse GP on held-out 10%: delta_v =====")
    delta_metrics = evaluate_probabilistic_regression(
        y_true=y_val_delta,
        mean=pred_delta_mean,
        std=pred_delta_std,
        prefix="Sparse GP Residual Model (delta_v)",
    )

    pred_v_next_mean = v_prev_val + pred_delta_mean
    pred_v_next_std = pred_delta_std

    print(f"\n===== Evaluate Sparse GP on held-out 10%: reconstructed v_next =====")
    next_metrics = evaluate_probabilistic_regression(
        y_true=y_val_next,
        mean=pred_v_next_mean,
        std=pred_v_next_std,
        prefix="Sparse GP Residual Model (reconstructed v_next)",
    )

    return {
        "gp_model": gp_model,
        "delta_metrics": delta_metrics,
        "next_metrics": next_metrics,
    }


# =========================
# 9. Main
# =========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, fallback to CPU.")
        args.device = "cpu"

    v_prev, u_prev, v_next, delta_v, ep_lengths = load_xy_data(args.data_file)
    N = v_prev.shape[0]
    train_idx, val_idx = make_split_indices(N, ep_lengths)

    if args.model_type == "full_gp":
        results = run_full_gp(
            v_prev=v_prev,
            u_prev=u_prev,
            v_next=v_next,
            delta_v=delta_v,
            train_idx=train_idx,
            val_idx=val_idx,
            output_dir=args.output_dir,
            device=args.device,
            gp_sample_size=args.gp_sample_size,
            gp_epochs=args.full_gp_epochs,
            gp_lr=args.full_gp_lr,
        )
    elif args.model_type == "sparse_gp":
        results = run_sparse_gp(
            v_prev=v_prev,
            u_prev=u_prev,
            v_next=v_next,
            delta_v=delta_v,
            train_idx=train_idx,
            val_idx=val_idx,
            output_dir=args.output_dir,
            device=args.device,
            num_inducing=args.num_inducing,
            gp_epochs=args.sparse_gp_epochs,
            gp_lr=args.sparse_gp_lr,
            batch_size=args.batch_size,
            noise_lower_bound=args.noise_lower_bound,
            inducing_init=args.inducing_init,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"\nAll outputs saved to: {args.output_dir}")
    return results


# =========================
# 10. CLI
# =========================

if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Train full GP / sparse GP residual dynamics model on XY motion"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["full_gp", "sparse_gp"]
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to DynamicsCollector dataset .pt with keys: state, action, next_state"
    )
    parser.add_argument("--output_dir", type=str, default="./gp_residual_results_xy")
    parser.add_argument("--device", type=str, default=default_device)

    parser.add_argument("--gp_sample_size", type=int, default=10000)
    parser.add_argument("--full_gp_epochs", type=int, default=150)
    parser.add_argument("--full_gp_lr", type=float, default=0.05)

    parser.add_argument("--sparse_gp_epochs", type=int, default=300)
    parser.add_argument("--sparse_gp_lr", type=float, default=1e-2)
    parser.add_argument("--num_inducing", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--noise_lower_bound", type=float, default=1e-5)
    parser.add_argument(
        "--inducing_init",
        type=str,
        default="kmeans",
        choices=["kmeans", "random"],
    )

    args = parser.parse_args()
    main(args)