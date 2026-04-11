import argparse
import os
import sys
import math
import torch
import numpy as np
import gpytorch

sys.path.insert(0, os.path.dirname(__file__))
from transition_models import MLPTransitionModel


# =========================
# Split utils
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

    return train_idx, val_idx, val_ep_ids


def make_random_split(N, train_ratio=0.9):
    perm = torch.randperm(N)
    split = int(train_ratio * N)
    return perm[:split], perm[split:], None


# =========================
# Theoretical chi-square(2) thresholds
# CDF of chi2(df=2): F(x)=1-exp(-x/2)
# Quantile: q(p) = -2 log(1-p)
# =========================

def chi2_df2_quantile(p: float) -> float:
    return -2.0 * math.log(max(1e-12, 1.0 - p))


# =========================
# Multi-output GP definition
# Must match your training script
# =========================

class MultitaskExactGPModel(gpytorch.models.ExactGP):
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
            rank=2,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultiOutputGPResidualModel2D:
    def __init__(self, input_dim=4, output_dim=2, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.model = None
        self.likelihood = None

    def load(self, path, x_dummy, y_dummy):
        payload = torch.load(path, map_location=self.device)

        self.input_dim = payload["input_dim"]
        self.output_dim = payload["output_dim"]

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

        self.model.eval()
        self.likelihood.eval()

    @torch.no_grad()
    def predict_full_cov_per_sample(self, x_test):
        """
        返回：
            mean: (N, 2)
            covs: (N, 2, 2)
        注意：
            这里我们从 multitask GP 的预测分布中提取每个样本对应的 2x2 task covariance。
        """
        x_test = x_test.to(self.device)

        with gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(x_test))

        mean = pred_dist.mean  # (N, 2)

        # pred_dist.lazy_covariance_matrix 对应的是 (N*2, N*2)
        # 我们只取每个样本自己的 2x2 block
        full_covar = pred_dist.lazy_covariance_matrix.to_dense()  # (2N, 2N)

        N = x_test.shape[0]
        covs = []
        for i in range(N):
            sl = slice(i * 2, (i + 1) * 2)
            cov_i = full_covar[sl, sl]
            covs.append(cov_i.unsqueeze(0))

        covs = torch.cat(covs, dim=0)  # (N, 2, 2)
        return mean, covs


# =========================
# Joint evaluation
# =========================

@torch.no_grad()
def compute_joint_mahalanobis(residual_true, mean, covs, jitter=1e-6):
    """
    residual_true: (N, 2)
    mean:          (N, 2)
    covs:          (N, 2, 2)

    返回：
        d2: (N,)    joint Mahalanobis distance squared
    """
    err = (residual_true - mean).unsqueeze(-1)  # (N, 2, 1)

    eye = torch.eye(2, device=covs.device).unsqueeze(0)  # (1,2,2)
    covs_reg = covs + jitter * eye

    inv_covs = torch.linalg.inv(covs_reg)  # (N,2,2)

    # d^2 = e^T Sigma^{-1} e
    d2 = torch.matmul(torch.matmul(err.transpose(1, 2), inv_covs), err).squeeze(-1).squeeze(-1)
    return d2


@torch.no_grad()
def evaluate_joint_distribution(
    d2,
    window_size=None,
    ep_lengths=None,
    prefix="Joint evaluation of multi-output GP"
):
    """
    d2 should approximately follow chi-square(df=2) if joint Gaussian calibration is good.
    """
    d2_np = d2.detach().cpu().numpy()

    # Theoretical stats for chi2(df=2)
    theory_mean = 2.0
    theory_var = 4.0

    q50 = chi2_df2_quantile(0.50)
    q90 = chi2_df2_quantile(0.90)
    q95 = chi2_df2_quantile(0.95)
    q99 = chi2_df2_quantile(0.99)
    q999 = chi2_df2_quantile(0.999)

    p_gt_q50 = float((d2_np > q50).mean())
    p_gt_q90 = float((d2_np > q90).mean())
    p_gt_q95 = float((d2_np > q95).mean())
    p_gt_q99 = float((d2_np > q99).mean())
    p_gt_q999 = float((d2_np > q999).mean())

    print(f"\n===== {prefix} =====")
    print("Target reference: d^2 ~ chi-square(df=2)")
    print(f"Empirical mean(d^2): {d2_np.mean():.6f}   | theory: {theory_mean:.6f}")
    print(f"Empirical var(d^2) : {d2_np.var():.6f}   | theory: {theory_var:.6f}")
    print(f"Empirical median   : {np.median(d2_np):.6f} | theory q50: {q50:.6f}")

    print("\n[Exceedance over chi-square(2) thresholds]")
    print(f"P(d^2 > q50={q50:.4f})  = {p_gt_q50:.6f} | theory: 0.500000")
    print(f"P(d^2 > q90={q90:.4f})  = {p_gt_q90:.6f} | theory: 0.100000")
    print(f"P(d^2 > q95={q95:.4f})  = {p_gt_q95:.6f} | theory: 0.050000")
    print(f"P(d^2 > q99={q99:.4f})  = {p_gt_q99:.6f} | theory: 0.010000")
    print(f"P(d^2 > q999={q999:.4f})= {p_gt_q999:.6f} | theory: 0.001000")

    print("\n[Empirical quantiles of d^2]")
    print(f"q50  = {np.quantile(d2_np, 0.50):.6f}")
    print(f"q90  = {np.quantile(d2_np, 0.90):.6f}")
    print(f"q95  = {np.quantile(d2_np, 0.95):.6f}")
    print(f"q99  = {np.quantile(d2_np, 0.99):.6f}")
    print(f"q999 = {np.quantile(d2_np, 0.999):.6f}")

    out = {
        "mean_d2": float(d2_np.mean()),
        "var_d2": float(d2_np.var()),
        "median_d2": float(np.median(d2_np)),
        "p_gt_q50": p_gt_q50,
        "p_gt_q90": p_gt_q90,
        "p_gt_q95": p_gt_q95,
        "p_gt_q99": p_gt_q99,
        "p_gt_q999": p_gt_q999,
        "emp_q50": float(np.quantile(d2_np, 0.50)),
        "emp_q90": float(np.quantile(d2_np, 0.90)),
        "emp_q95": float(np.quantile(d2_np, 0.95)),
        "emp_q99": float(np.quantile(d2_np, 0.99)),
        "emp_q999": float(np.quantile(d2_np, 0.999)),
    }

    # Optional: window-level evaluation
    if window_size is not None and window_size > 1:
        c_vals = []
        if ep_lengths is None:
            for i in range(0, len(d2_np) - window_size + 1):
                c_vals.append(d2_np[i:i + window_size].mean())
        else:
            start = 0
            for L in ep_lengths:
                L = int(L)
                seq = d2_np[start:start + L]
                if len(seq) >= window_size:
                    for i in range(0, len(seq) - window_size + 1):
                        c_vals.append(seq[i:i + window_size].mean())
                start += L

        c_vals = np.array(c_vals)
        print(f"\n[Window mean of d^2] window_size={window_size}")
        print(f"C_t mean = {c_vals.mean():.6f}")
        print(f"C_t std  = {c_vals.std():.6f}")
        print(f"C_t q95  = {np.quantile(c_vals, 0.95):.6f}")
        print(f"C_t q99  = {np.quantile(c_vals, 0.99):.6f}")

        out["window_mean"] = float(c_vals.mean())
        out["window_std"] = float(c_vals.std())
        out["window_q95"] = float(np.quantile(c_vals, 0.95))
        out["window_q99"] = float(np.quantile(c_vals, 0.99))

    return out


# =========================
# Main
# =========================

def main(
    data_file,
    mlp_model_file,
    gp_model_file,
    device="cpu",
    window_size=None,
):
    print(f"Loading data from {data_file} ...")
    data = torch.load(data_file, map_location="cpu", weights_only=True)

    v_prev = data["v_prev"].float()[:, :2]
    u_prev = data["u_prev"].float()[:, :2]
    v_next = data["v_next"].float()[:, :2]
    ep_lengths = data.get("ep_lengths", None)

    N = v_prev.shape[0]
    print(f"Loaded {N} samples")
    print("Only using XY dimensions")

    if ep_lengths is not None and len(ep_lengths) > 1:
        print("Using episode-aware split (90% train / 10% val)")
        train_idx, val_idx, val_ep_ids = make_episode_split(ep_lengths, train_ratio=0.9)
        val_ep_lengths = ep_lengths[val_ep_ids]
    else:
        print("No episode info, using random split (90% train / 10% val)")
        train_idx, val_idx, _ = make_random_split(N, train_ratio=0.9)
        val_ep_lengths = None

    train_v_prev = v_prev[train_idx]
    train_u_prev = u_prev[train_idx]
    train_v_next = v_next[train_idx]

    val_v_prev = v_prev[val_idx]
    val_u_prev = u_prev[val_idx]
    val_v_next = v_next[val_idx]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val   samples: {len(val_idx)}")

    # Load MLP
    print(f"\nLoading existing MLP from {mlp_model_file}")
    mlp_model = MLPTransitionModel(
        state_dim=2,
        input_dim=2,
        hidden_dim=32,   # 如果你训练MLP时改过，这里要同步改
        device=device
    )
    mlp_model.load(mlp_model_file)

    with torch.no_grad():
        train_pred_mlp = mlp_model.predict(train_v_prev.to(device), train_u_prev.to(device)).cpu()
        val_pred_mlp = mlp_model.predict(val_v_prev.to(device), val_u_prev.to(device)).cpu()

    train_residual = train_v_next - train_pred_mlp
    val_residual_true = val_v_next - val_pred_mlp

    # Build GP inputs
    x_train_full = torch.cat([train_v_prev, train_u_prev], dim=1)  # (N_train, 4)
    x_val = torch.cat([val_v_prev, val_u_prev], dim=1)             # (N_val, 4)

    # Load GP
    print(f"\nLoading multi-output GP from {gp_model_file}")
    gp_model = MultiOutputGPResidualModel2D(input_dim=4, output_dim=2, device=device)

    # 用一小段 dummy 训练数据来构建结构
    dummy_n = min(16, x_train_full.shape[0])
    x_dummy = x_train_full[:dummy_n]
    y_dummy = train_residual[:dummy_n]

    gp_model.load(
        path=gp_model_file,
        x_dummy=x_dummy,
        y_dummy=y_dummy
    )

    print("\nPredicting full covariance on held-out 10% ...")
    with torch.no_grad():
        mean, covs = gp_model.predict_full_cov_per_sample(x_val)
        mean = mean.cpu()
        covs = covs.cpu()

    # Joint Mahalanobis distance
    d2 = compute_joint_mahalanobis(
        residual_true=val_residual_true,
        mean=mean,
        covs=covs,
        jitter=1e-6,
    )

    stats = evaluate_joint_distribution(
        d2=d2,
        window_size=window_size,
        ep_lengths=val_ep_lengths,
        prefix="Joint evaluation of multi-output GP on XY residuals"
    )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint evaluation for multi-output GP on XY residuals")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--mlp_model_file", type=str, required=True)
    parser.add_argument("--gp_model_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window_size", type=int, default=0)

    args = parser.parse_args()

    ws = args.window_size if args.window_size > 1 else None

    main(
        data_file=args.data_file,
        mlp_model_file=args.mlp_model_file,
        gp_model_file=args.gp_model_file,
        device=args.device,
        window_size=ws,
    )