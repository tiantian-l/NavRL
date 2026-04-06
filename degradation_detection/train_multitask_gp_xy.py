import argparse
import os
import sys
import torch
import numpy as np
import gpytorch

# 让脚本能 import 你项目里的模型
sys.path.insert(0, os.path.dirname(__file__))
from transition_models import MLPTransitionModel


# =========================
# Utils
# =========================

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


def _safe_std(x, eps=1e-6):
    return torch.clamp(x, min=eps)


@torch.no_grad()
def evaluate_probabilistic_regression_xy(y_true, mean, std, prefix="Multi-output GP Residual XY"):
    std = _safe_std(std)

    # RMSE
    rmse_all = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    rmse_dim = torch.sqrt(torch.mean((mean - y_true) ** 2, dim=0)).cpu().numpy()

    # Gaussian NLL (diagonal marginal NLL)
    var = std ** 2
    nll = 0.5 * (((y_true - mean) ** 2) / var + torch.log(2 * torch.pi * var))
    nll_all = nll.mean().item()
    nll_dim = nll.mean(dim=0).cpu().numpy()

    # standardized residual
    z = (y_true - mean) / std
    z_mean = z.mean(dim=0).cpu().numpy()
    z_std = z.std(dim=0).cpu().numpy()

    frac_abs_gt_1 = (z.abs() > 1.0).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_2 = (z.abs() > 2.0).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_3 = (z.abs() > 3.0).float().mean(dim=0).cpu().numpy()

    # calibration / coverage
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
    print(f"68% interval coverage   [x, y]: {cover_68}")
    print(f"95% interval coverage   [x, y]: {cover_95}")
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
# Multi-output GP
# =========================

class MultitaskExactGPModel(gpytorch.models.ExactGP):
    """
    输入: x = [vx_prev, vy_prev, ux_prev, uy_prev]  shape (N, 4)
    输出: y = [res_x, res_y]                        shape (N, 2)
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(),
            num_tasks=num_tasks
        )

        data_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_kernel,
            num_tasks=num_tasks,
            rank=2,  # 可以调成 1 或 2；2 更灵活
        )

    def forward(self, x):
        mean_x = self.mean_module(x)        # (N, num_tasks)
        covar_x = self.covar_module(x)      # Multitask covariance
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultiOutputGPResidualModel2D:
    def __init__(self, input_dim=4, output_dim=2, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.model = None
        self.likelihood = None

    def fit(self, x_train, y_train, training_iter=150, lr=0.1, verbose=True):
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.output_dim
        ).to(self.device)

        self.model = MultitaskExactGPModel(
            train_x=x_train,
            train_y=y_train,
            likelihood=self.likelihood,
            num_tasks=self.output_dim,
        ).to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            if verbose and ((i + 1) % 25 == 0 or i == 0):
                noise = self.likelihood.task_noises.detach().cpu().numpy()
                ls = self.model.covar_module.data_covar_module.base_kernel.lengthscale.mean().item()
                os = self.model.covar_module.data_covar_module.outputscale.item()
                task_covar = self.model.covar_module.task_covar_module.covar_matrix.detach().cpu().numpy()
                print(
                    f"iter {i+1:3d}/{training_iter} | "
                    f"loss={loss.item():.4f} | "
                    f"task_noises={noise} | "
                    f"ls={ls:.4f} | os={os:.4f}"
                )
                print("task covariance:")
                print(task_covar)

    @torch.no_grad()
    def predict(self, x_test):
        x_test = x_test.to(self.device)
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(x_test))

        mean = pred_dist.mean                      # (N, 2)
        std = pred_dist.stddev                    # (N, 2)
        return mean, std

    @torch.no_grad()
    def predict_full_cov(self, x_test):
        """
        返回联合协方差，后面如果你要做 full covariance anomaly score 会有用
        """
        x_test = x_test.to(self.device)
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(x_test))

        return pred_dist

    def save(self, path):
        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path, x_dummy=None, y_dummy=None):
        """
        如果以后想加载，需要给 dummy 输入构造 model 结构
        """
        payload = torch.load(path, map_location=self.device)
        self.input_dim = payload["input_dim"]
        self.output_dim = payload["output_dim"]

        if x_dummy is None or y_dummy is None:
            raise ValueError("Loading MultiOutputGPResidualModel2D requires x_dummy and y_dummy to build model.")

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.output_dim
        ).to(self.device)

        self.model = MultitaskExactGPModel(
            train_x=x_dummy.to(self.device),
            train_y=y_dummy.to(self.device),
            likelihood=self.likelihood,
            num_tasks=self.output_dim,
        ).to(self.device)

        self.model.load_state_dict(payload["model_state_dict"])
        self.likelihood.load_state_dict(payload["likelihood_state_dict"])


# =========================
# Main
# =========================

def train_multioutput_gp_xy_only(
    data_path: str,
    mlp_model_path: str,
    output_dir: str,
    device: str = "cpu",
    gp_sample_size: int = 10000,
    gp_epochs: int = 150,
    gp_lr: float = 0.1,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)

    # 只保留 XY
    v_prev = data["v_prev"].float()[:, :2]   # (N, 2)
    u_prev = data["u_prev"].float()[:, :2]   # (N, 2)
    v_next = data["v_next"].float()[:, :2]   # (N, 2)
    ep_lengths = data.get("ep_lengths", None)

    N = v_prev.shape[0]
    print(f"Loaded {N} samples")
    print("Only using XY dimensions")
    print("  v_prev -> [vx, vy]")
    print("  u_prev -> [ux, uy]")
    print("  v_next -> [vx_next, vy_next]")
    print("  z-axis ignored")

    # train/val split
    if ep_lengths is not None and len(ep_lengths) > 1:
        print("Using episode-aware split (90% train / 10% val)")
        train_idx, val_idx = make_episode_split(ep_lengths, train_ratio=0.9)
    else:
        print("No episode info, using random split (90% train / 10% val)")
        train_idx, val_idx = make_random_split(N, train_ratio=0.9)

    train_v_prev = v_prev[train_idx]
    train_u_prev = u_prev[train_idx]
    train_v_next = v_next[train_idx]

    val_v_prev = v_prev[val_idx]
    val_u_prev = u_prev[val_idx]
    val_v_next = v_next[val_idx]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val   samples: {len(val_idx)}")

    # 加载已有 MLP
    print(f"\nLoading existing MLP from {mlp_model_path}")
    mlp_model = MLPTransitionModel(
        state_dim=2,
        input_dim=2,
        hidden_dim=32,   # 如果你训练时改过 hidden_dim，这里要对应改
        device=device
    )
    mlp_model.load(mlp_model_path)

    # 用已有 MLP 计算 residual
    print("\nComputing residuals from existing MLP ...")
    with torch.no_grad():
        train_pred_mlp = mlp_model.predict(train_v_prev.to(device), train_u_prev.to(device)).cpu()
        val_pred_mlp = mlp_model.predict(val_v_prev.to(device), val_u_prev.to(device)).cpu()

    train_residual = train_v_next - train_pred_mlp   # (N_train, 2)
    val_residual_true = val_v_next - val_pred_mlp    # (N_val, 2)

    # GP 输入: [vx_prev, vy_prev, ux_prev, uy_prev]
    x_train_full = torch.cat([train_v_prev, train_u_prev], dim=1)  # (N_train, 4)
    x_val = torch.cat([val_v_prev, val_u_prev], dim=1)             # (N_val, 4)

    # sample 10000 / 20000
    n_train = x_train_full.shape[0]
    gp_sample_size = min(gp_sample_size, n_train)

    sample_idx = torch.randperm(n_train)[:gp_sample_size]
    x_gp_train = x_train_full[sample_idx]
    y_gp_train = train_residual[sample_idx]

    print(f"Sampled {gp_sample_size} points for multi-output exact GP training")

    if gp_sample_size > 12000:
        print("Warning: exact multi-output GP with >12000 samples may be slow / memory-heavy.")

    # 训练 multi-output GP
    print("\n===== Train Multi-output GP on XY residuals =====")
    gp_model = MultiOutputGPResidualModel2D(input_dim=4, output_dim=2, device=device)
    gp_model.fit(
        x_gp_train,
        y_gp_train,
        training_iter=gp_epochs,
        lr=gp_lr,
        verbose=True,
    )

    gp_path = os.path.join(output_dir, "multioutput_gp_residual_xy.pt")
    gp_model.save(gp_path)
    print(f"Saved GP to {gp_path}")

    # 评估
    print("\n===== Evaluate Multi-output GP on held-out 10% =====")
    with torch.no_grad():
        gp_mean, gp_std = gp_model.predict(x_val)
        gp_mean = gp_mean.cpu()
        gp_std = gp_std.cpu()

    metrics = evaluate_probabilistic_regression_xy(
        y_true=val_residual_true,
        mean=gp_mean,
        std=gp_std,
        prefix="Multi-output GP Residual Model (XY)"
    )

    print(f"\nAll outputs saved to: {output_dir}")
    return {
        "gp_model": gp_model,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-output GP residual model on XY only, reusing existing MLP")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to nominal_data.pt")
    parser.add_argument("--mlp_model_file", type=str, required=True,
                        help="Path to existing mlp_model_xy.pt")
    parser.add_argument("--output_dir", type=str, default="./nominal_models_xy_multioutput")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--gp_sample_size", type=int, default=10000,
                        help="Residual samples for exact multi-output GP, e.g. 10000 or 20000")
    parser.add_argument("--gp_epochs", type=int, default=150)
    parser.add_argument("--gp_lr", type=float, default=0.1)

    args = parser.parse_args()

    train_multioutput_gp_xy_only(
        data_path=args.data_file,
        mlp_model_path=args.mlp_model_file,
        output_dir=args.output_dir,
        device=args.device,
        gp_sample_size=args.gp_sample_size,
        gp_epochs=args.gp_epochs,
        gp_lr=args.gp_lr,
    )