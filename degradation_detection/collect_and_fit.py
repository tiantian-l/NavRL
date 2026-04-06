import argparse
import os
import sys
import torch
import gpytorch

# Add parent directory so we can import transition_models
sys.path.insert(0, os.path.dirname(__file__))

from transition_models import MLPTransitionModel


# =========================
# 1. Utilities
# =========================

def print_split_info(train_idx, val_idx):
    print(f"Train samples: {len(train_idx)}")
    print(f"Val   samples: {len(val_idx)}")


def make_episode_split(ep_lengths, train_ratio=0.9):
    num_eps = len(ep_lengths)
    ep_perm = torch.randperm(num_eps)
    ep_split = max(1, int(train_ratio * num_eps))

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
    perm = torch.randperm(N)
    split = int(train_ratio * N)
    return perm[:split], perm[split:]


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def _safe_std(x, eps=1e-6):
    return torch.clamp(x, min=eps)


# =========================
# 2. Exact GP model (single output)
# =========================

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FullGPResidualModel2D:
    """
    训练 2 个独立的 exact GP，分别拟合 residual_x 和 residual_y
    输入: x = [vx_prev, vy_prev, ux_prev, uy_prev] -> shape (N, 4)
    输出: residual_xy -> shape (N, 2)
    """
    def __init__(self, input_dim=4, output_dim=2, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.models = []
        self.likelihoods = []

    def fit(self, x_train, y_train, training_iter=150, lr=0.1, verbose=True):
        self.models = []
        self.likelihoods = []

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        for d in range(self.output_dim):
            if verbose:
                dim_name = "x" if d == 0 else "y"
                print(f"\n[GP] Training residual_{dim_name} ...")

            y_d = y_train[:, d]

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = ExactGPModel(x_train, y_d, likelihood).to(self.device)

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
                        f"loss={loss.item():.4f} | "
                        f"noise={noise:.6f} | "
                        f"ls={lengthscale:.4f} | "
                        f"os={outputscale:.4f}"
                    )

            self.models.append(model)
            self.likelihoods.append(likelihood)

    @torch.no_grad()
    def predict(self, x_test):
        """
        返回:
            mean: (N, 2)
            std:  (N, 2)
        """
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
            "state_dicts": [],
        }
        for model, likelihood in zip(self.models, self.likelihoods):
            payload["state_dicts"].append({
                "model": model.state_dict(),
                "likelihood": likelihood.state_dict(),
            })
        torch.save(payload, path)


# =========================
# 3. Evaluation
# =========================

@torch.no_grad()
def evaluate_mlp_xy(y_true, pred, prefix="MLP XY"):
    rmse_all = torch.sqrt(torch.mean((pred - y_true) ** 2)).item()
    rmse_dim = torch.sqrt(torch.mean((pred - y_true) ** 2, dim=0)).cpu().numpy()

    print(f"\n===== {prefix} Evaluation =====")
    print(f"Overall RMSE: {rmse_all:.6f}")
    print(f"Per-dim RMSE [vx, vy]: {rmse_dim}")

    return {
        "rmse_all": rmse_all,
        "rmse_dim": rmse_dim,
    }


@torch.no_grad()
def evaluate_probabilistic_regression_xy(y_true, mean, std, prefix="GP Residual XY"):
    std = _safe_std(std)

    # RMSE
    rmse_all = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    rmse_dim = torch.sqrt(torch.mean((mean - y_true) ** 2, dim=0)).cpu().numpy()

    # Gaussian NLL
    var = std ** 2
    nll = 0.5 * (((y_true - mean) ** 2) / var + torch.log(2 * torch.pi * var))
    nll_all = nll.mean().item()
    nll_dim = nll.mean(dim=0).cpu().numpy()

    # Standardized residual
    z = (y_true - mean) / std
    z_mean = z.mean(dim=0).cpu().numpy()
    z_std = z.std(dim=0).cpu().numpy()

    frac_abs_gt_1 = (z.abs() > 1.0).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_2 = (z.abs() > 2.0).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_3 = (z.abs() > 3.0).float().mean(dim=0).cpu().numpy()

    # Coverage
    cover_68 = ((y_true >= mean - 1.0 * std) & (y_true <= mean + 1.0 * std)).float().mean(dim=0).cpu().numpy()
    cover_95 = ((y_true >= mean - 1.96 * std) & (y_true <= mean + 1.96 * std)).float().mean(dim=0).cpu().numpy()
    cover_997 = ((y_true >= mean - 3.0 * std) & (y_true <= mean + 3.0 * std)).float().mean(dim=0).cpu().numpy()

    print(f"\n===== {prefix} Evaluation =====")
    print(f"Overall RMSE: {rmse_all:.6f}")
    print(f"Overall NLL : {nll_all:.6f}")
    print(f"Per-dim RMSE [x_res, y_res]: {rmse_dim}")
    print(f"Per-dim NLL  [x_res, y_res]: {nll_dim}")

    print("\n[Standardized residual z = (y - mu) / sigma]")
    print(f"z mean [x, y]: {z_mean}")
    print(f"z std  [x, y]: {z_std}")
    print(f"P(|z|>1) [x, y]: {frac_abs_gt_1}   (理论约 0.317)")
    print(f"P(|z|>2) [x, y]: {frac_abs_gt_2}   (理论约 0.0455)")
    print(f"P(|z|>3) [x, y]: {frac_abs_gt_3}   (理论约 0.0027)")

    print("\n[Calibration / Coverage]")
    print(f"68% interval coverage  [x, y]: {cover_68}")
    print(f"95% interval coverage  [x, y]: {cover_95}")
    print(f"99.7% interval coverage [x, y]: {cover_997}")

    return {
        "rmse_all": rmse_all,
        "rmse_dim": rmse_dim,
        "nll_all": nll_all,
        "nll_dim": nll_dim,
        "z_mean": z_mean,
        "z_std": z_std,
        "frac_abs_gt_1": frac_abs_gt_1,
        "frac_abs_gt_2": frac_abs_gt_2,
        "frac_abs_gt_3": frac_abs_gt_3,
        "cover_68": cover_68,
        "cover_95": cover_95,
        "cover_997": cover_997,
    }


# =========================
# 4. Main pipeline (XY only)
# =========================

def fit_mlp_then_full_gp_xy_only(
    data_path: str,
    output_dir: str,
    device: str = "cpu",
    mlp_epochs: int = 1000,
    mlp_hidden: int = 32,
    gp_sample_size: int = 10000,
    gp_epochs: int = 150,
    gp_lr: float = 0.1,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)

    # 原始数据还是 (N, 3)，这里只截取前两维 x/y
    v_prev = data["v_prev"].float()[:, :2]   # (N, 2)
    u_prev = data["u_prev"].float()[:, :2]   # (N, 2) 只保留 ux, uy
    v_next = data["v_next"].float()[:, :2]   # (N, 2)
    ep_lengths = data.get("ep_lengths", None)

    N = v_prev.shape[0]
    print(f"Loaded {N} samples")
    print("Only using XY dimensions:")
    print("  v_prev -> [vx, vy]")
    print("  u_prev -> [ux, uy]")
    print("  v_next -> [vx_next, vy_next]")
    print("  z-axis is ignored")

    # 1) 90/10 split
    if ep_lengths is not None and len(ep_lengths) > 1:
        print("Using episode-aware split (90% train / 10% val)")
        train_idx, val_idx = make_episode_split(ep_lengths, train_ratio=0.9)
    else:
        print("No episode info, using random split (90% train / 10% val)")
        train_idx, val_idx = make_random_split(N, train_ratio=0.9)

    print_split_info(train_idx, val_idx)

    train_v_prev = v_prev[train_idx]
    train_u_prev = u_prev[train_idx]
    train_v_next = v_next[train_idx]

    val_v_prev = v_prev[val_idx]
    val_u_prev = u_prev[val_idx]
    val_v_next = v_next[val_idx]

    # 2) Train MLP on 90%
    print(f"\n===== Train MLP on 90% training data (XY only) =====")
    mlp_model = MLPTransitionModel(
        state_dim=2,
        input_dim=2,
        hidden_dim=mlp_hidden,
        device=device
    )

    mlp_model.fit(
        train_v_prev,
        train_u_prev,
        train_v_next,
        lr=1e-3,
        epochs=mlp_epochs,
        batch_size=4096,
        verbose=True,
    )

    mlp_path = os.path.join(output_dir, "mlp_model_xy.pt")
    mlp_model.save(mlp_path)
    print(f"Saved MLP to {mlp_path}")

    # 3) Evaluate MLP on 10%
    with torch.no_grad():
        val_pred_mlp = mlp_model.predict(val_v_prev.to(device), val_u_prev.to(device)).cpu()

    mlp_metrics = evaluate_mlp_xy(
        y_true=val_v_next,
        pred=val_pred_mlp,
        prefix="MLP Transition Model (XY)"
    )

    # 4) Compute residuals on 90% training set
    print(f"\n===== Compute residuals on 90% training data (XY only) =====")
    with torch.no_grad():
        train_pred_mlp = mlp_model.predict(train_v_prev.to(device), train_u_prev.to(device)).cpu()
        train_residual = train_v_next - train_pred_mlp   # (N_train, 2)

    # GP 输入是 [vx_prev, vy_prev, ux_prev, uy_prev]
    x_train_full = torch.cat([train_v_prev, train_u_prev], dim=1)  # (N_train, 4)

    n_train = x_train_full.shape[0]
    gp_sample_size = min(gp_sample_size, n_train)

    sample_idx = torch.randperm(n_train)[:gp_sample_size]
    x_gp_train = x_train_full[sample_idx]
    y_gp_train = train_residual[sample_idx]

    print(f"Sampled {gp_sample_size} residual points from 90% train set for exact GP")

    # 5) Train exact GP on residuals
    print(f"\n===== Train Full GP on sampled XY residuals =====")
    gp_model = FullGPResidualModel2D(input_dim=4, output_dim=2, device=device)
    gp_model.fit(
        x_gp_train,
        y_gp_train,
        training_iter=gp_epochs,
        lr=gp_lr,
        verbose=True,
    )

    gp_path = os.path.join(output_dir, "full_gp_residual_xy.pt")
    gp_model.save(gp_path)
    print(f"Saved full GP to {gp_path}")

    # 6) Evaluate GP on held-out 10%
    print(f"\n===== Evaluate Full GP on held-out 10% (XY residuals) =====")
    with torch.no_grad():
        val_pred_mlp = mlp_model.predict(val_v_prev.to(device), val_u_prev.to(device)).cpu()
        val_residual_true = val_v_next - val_pred_mlp  # (N_val, 2)

        x_val = torch.cat([val_v_prev, val_u_prev], dim=1)  # (N_val, 4)
        gp_mean, gp_std = gp_model.predict(x_val)

        gp_mean = gp_mean.cpu()
        gp_std = gp_std.cpu()

    gp_metrics = evaluate_probabilistic_regression_xy(
        y_true=val_residual_true,
        mean=gp_mean,
        std=gp_std,
        prefix="Full GP Residual Model (XY)"
    )

    print(f"\nAll outputs saved to: {output_dir}")
    return {
        "mlp_model": mlp_model,
        "mlp_metrics": mlp_metrics,
        "gp_model": gp_model,
        "gp_metrics": gp_metrics,
    }


# =========================
# 5. CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP and full GP residual model on XY motion only")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to nominal_data.pt with keys v_prev, u_prev, v_next")
    parser.add_argument("--output_dir", type=str, default="./nominal_models_xy")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--mlp_epochs", type=int, default=1000)
    parser.add_argument("--mlp_hidden", type=int, default=32)

    parser.add_argument("--gp_sample_size", type=int, default=10000,
                        help="Number of residual samples drawn from 90% train set for exact GP")
    parser.add_argument("--gp_epochs", type=int, default=150)
    parser.add_argument("--gp_lr", type=float, default=0.1)

    args = parser.parse_args()

    fit_mlp_then_full_gp_xy_only(
        data_path=args.data_file,
        output_dir=args.output_dir,
        device=args.device,
        mlp_epochs=args.mlp_epochs,
        mlp_hidden=args.mlp_hidden,
        gp_sample_size=args.gp_sample_size,
        gp_epochs=args.gp_epochs,
        gp_lr=args.gp_lr,
    )