#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import gpytorch
import numpy as np


# =========================
# 1. Utilities
# =========================

def clamp_std(x, eps=1e-6):
    return torch.clamp(x, min=eps)


# =========================
# 2. Mean function
# =========================

class SelectInputMean(gpytorch.means.Mean):
    """
    从输入 x 的某一列直接取值作为 mean
    例如:
      d=0 -> mean = x[:, 0] = vx_prev
      d=1 -> mean = x[:, 1] = vy_prev
    """
    def __init__(self, input_idx: int):
        super().__init__()
        self.input_idx = input_idx

    def forward(self, x):
        return x[..., self.input_idx]


# =========================
# 3. Sparse GP model
# =========================

class SparseTransitionGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_input_idx: int):
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

        self.mean_module = SelectInputMean(mean_input_idx)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPTransitionModel2D:
    """
    与你训练脚本一致的 2 维 sparse GP 包装器
    output = [vx_next, vy_next]
    """
    def __init__(self, input_dim=4, output_dim=2, num_inducing=512, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.device = device

        self.models = []
        self.likelihoods = []

    @torch.no_grad()
    def predict(self, x_test, batch_size=4096):
        x_test = x_test.to(self.device)

        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()

        means_all = []
        stds_all = []

        for start in range(0, x_test.shape[0], batch_size):
            xb = x_test[start:start + batch_size]

            means = []
            stds = []

            for model, likelihood in zip(self.models, self.likelihoods):
                with gpytorch.settings.fast_pred_var():
                    pred_dist = likelihood(model(xb))
                    means.append(pred_dist.mean.unsqueeze(-1))
                    stds.append(pred_dist.stddev.unsqueeze(-1))

            mean_b = torch.cat(means, dim=-1)
            std_b = torch.cat(stds, dim=-1)

            means_all.append(mean_b)
            stds_all.append(std_b)

        mean = torch.cat(means_all, dim=0)
        std = torch.cat(stds_all, dim=0)
        return mean, std

    def load(self, path):
        payload = torch.load(path, map_location=self.device)

        self.input_dim = payload["input_dim"]
        self.output_dim = payload["output_dim"]
        self.num_inducing = payload["num_inducing"]

        self.models = []
        self.likelihoods = []

        state_dicts = payload["state_dicts"]
        if len(state_dicts) != self.output_dim:
            raise ValueError(
                f"Checkpoint output_dim={self.output_dim}, but found {len(state_dicts)} state dict groups."
            )

        for d, sd in enumerate(state_dicts):
            mean_idx = d

            model_state = sd["model"]
            likelihood_state = sd["likelihood"]

            inducing_key = "variational_strategy.inducing_points"
            if inducing_key not in model_state:
                raise KeyError(f"Cannot find '{inducing_key}' in checkpoint.")

            inducing_points = model_state[inducing_key].to(self.device)

            model = SparseTransitionGPModel(
                inducing_points=inducing_points,
                mean_input_idx=mean_idx,
            ).to(self.device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

            model.load_state_dict(model_state)
            likelihood.load_state_dict(likelihood_state)

            model.eval()
            likelihood.eval()

            self.models.append(model)
            self.likelihoods.append(likelihood)


# =========================
# 4. Data loading
# =========================

def load_xy_data(data_path):
    print(f"Loading data from {data_path} ...")
    data = torch.load(data_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {data_path}, but got {type(data)}")

    print(f"Available keys: {list(data.keys())}")

    # Format A: v_prev / u_prev / v_next
    if all(k in data for k in ["v_prev", "u_prev", "v_next"]):
        v_prev = data["v_prev"].float()[:, :2]
        u_prev = data["u_prev"].float()[:, :2]
        v_next = data["v_next"].float()[:, :2]

        x = torch.cat([v_prev, u_prev], dim=1)
        y = v_next

        print(f"Loaded {x.shape[0]} samples")
        print("Detected format: [v_prev, u_prev, v_next]")
        print("Using XY dimensions only:")
        print("  input  = [vx_prev, vy_prev, ux_prev, uy_prev]")
        print("  target = [vx_next, vy_next]")
        return x, y

    # Format B: state / action / next_state
    if all(k in data for k in ["state", "action", "next_state"]):
        v_prev = data["state"].float()[:, :2]
        u_prev = data["action"].float()[:, :2]
        v_next = data["next_state"].float()[:, :2]

        x = torch.cat([v_prev, u_prev], dim=1)
        y = v_next

        print(f"Loaded {x.shape[0]} samples")
        print("Detected format: [state, action, next_state]")
        print("Using XY dimensions only:")
        print("  input  = [vx_prev, vy_prev, ux_prev, uy_prev]")
        print("  target = [vx_next, vy_next]")
        return x, y

    # Format C: v_prev / u_prev / dv
    if all(k in data for k in ["v_prev", "u_prev", "dv"]):
        v_prev = data["v_prev"].float()[:, :2]
        u_prev = data["u_prev"].float()[:, :2]
        dv = data["dv"].float()[:, :2]
        v_next = v_prev + dv

        x = torch.cat([v_prev, u_prev], dim=1)
        y = v_next

        print(f"Loaded {x.shape[0]} samples")
        print("Detected format: [v_prev, u_prev, dv]")
        print("Using XY dimensions only:")
        print("  input  = [vx_prev, vy_prev, ux_prev, uy_prev]")
        print("  target = [vx_next, vy_next] reconstructed from v_prev + dv")
        return x, y

    raise KeyError(
        "Could not find a supported key pattern in the data file.\n"
        f"Available keys are: {list(data.keys())}"
    )


# =========================
# 5. Anomaly statistics
# =========================

@torch.no_grad()
def estimate_anomaly_rates(y_true, mean, std):
    std = clamp_std(std)

    z = (y_true - mean) / std
    abs_z = z.abs()

    thr_1 = 1.0
    thr_99 = 2.5758293035489004
    thr_3 = 3.0

    rmse_all = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    rmse_dim = torch.sqrt(torch.mean((mean - y_true) ** 2, dim=0)).cpu().numpy()

    z_mean = z.mean(dim=0).cpu().numpy()
    z_std = z.std(dim=0).cpu().numpy()

    # 单维异常率
    frac_abs_gt_1 = (abs_z > thr_1).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_99 = (abs_z > thr_99).float().mean(dim=0).cpu().numpy()
    frac_abs_gt_3 = (abs_z > thr_3).float().mean(dim=0).cpu().numpy()

    # 单维覆盖率
    cover_68 = (abs_z <= thr_1).float().mean(dim=0).cpu().numpy()
    cover_99 = (abs_z <= thr_99).float().mean(dim=0).cpu().numpy()
    cover_997 = (abs_z <= thr_3).float().mean(dim=0).cpu().numpy()

    # 联合异常率：任一维超阈值就算异常
    joint_anom_1 = (abs_z > thr_1).any(dim=1).float().mean().item()
    joint_anom_99 = (abs_z > thr_99).any(dim=1).float().mean().item()
    joint_anom_3 = (abs_z > thr_3).any(dim=1).float().mean().item()

    # 联合正常率：两维都在区间内
    joint_cover_68 = (abs_z <= thr_1).all(dim=1).float().mean().item()
    joint_cover_99 = (abs_z <= thr_99).all(dim=1).float().mean().item()
    joint_cover_997 = (abs_z <= thr_3).all(dim=1).float().mean().item()

    return {
        "rmse_all": rmse_all,
        "rmse_dim": rmse_dim.tolist(),
        "z_mean": z_mean.tolist(),
        "z_std": z_std.tolist(),

        "per_dim": {
            "anomaly_rate_outside_1sigma": frac_abs_gt_1.tolist(),
            "anomaly_rate_outside_99pct": frac_abs_gt_99.tolist(),
            "anomaly_rate_outside_3sigma": frac_abs_gt_3.tolist(),
            "coverage_68": cover_68.tolist(),
            "coverage_99": cover_99.tolist(),
            "coverage_997": cover_997.tolist(),
        },

        "joint": {
            "anomaly_rate_outside_1sigma": joint_anom_1,
            "anomaly_rate_outside_99pct": joint_anom_99,
            "anomaly_rate_outside_3sigma": joint_anom_3,
            "coverage_68": joint_cover_68,
            "coverage_99": joint_cover_99,
            "coverage_997": joint_cover_997,
        },

        "theory": {
            "outside_1sigma": 0.3173,
            "outside_99pct": 0.01,
            "outside_3sigma": 0.0027,
            "coverage_68": 0.6827,
            "coverage_99": 0.99,
            "coverage_997": 0.9973,
        }
    }


def pretty_print_stats(stats):
    print("\n===== Sparse GP Anomaly Rate Estimation =====")
    print(f"Overall RMSE: {stats['rmse_all']:.6f}")
    print(f"Per-dim RMSE [vx_next, vy_next]: {np.array(stats['rmse_dim'])}")

    print("\n[Standardized residual z = (y - mu) / sigma]")
    print(f"z mean [vx, vy]: {np.array(stats['z_mean'])}")
    print(f"z std  [vx, vy]: {np.array(stats['z_std'])}")

    print("\n[Per-dimension anomaly rates]")
    print(
        f"P(|z|>1)        [vx, vy]: {np.array(stats['per_dim']['anomaly_rate_outside_1sigma'])}   "
        f"(theory ~ {stats['theory']['outside_1sigma']})"
    )
    print(
        f"P(|z|>2.576)    [vx, vy]: {np.array(stats['per_dim']['anomaly_rate_outside_99pct'])}   "
        f"(theory ~ {stats['theory']['outside_99pct']})"
    )
    print(
        f"P(|z|>3)        [vx, vy]: {np.array(stats['per_dim']['anomaly_rate_outside_3sigma'])}   "
        f"(theory ~ {stats['theory']['outside_3sigma']})"
    )

    print("\n[Per-dimension coverage]")
    print(
        f"68% cover       [vx, vy]: {np.array(stats['per_dim']['coverage_68'])}   "
        f"(theory ~ {stats['theory']['coverage_68']})"
    )
    print(
        f"99% cover       [vx, vy]: {np.array(stats['per_dim']['coverage_99'])}   "
        f"(theory ~ {stats['theory']['coverage_99']})"
    )
    print(
        f"99.7% cover     [vx, vy]: {np.array(stats['per_dim']['coverage_997'])}   "
        f"(theory ~ {stats['theory']['coverage_997']})"
    )

    print("\n[Joint sample-level rates: any dimension exceeds threshold => anomaly]")
    print(
        f"Joint anomaly outside 1σ     : {stats['joint']['anomaly_rate_outside_1sigma']:.6f}"
    )
    print(
        f"Joint anomaly outside 99% CI : {stats['joint']['anomaly_rate_outside_99pct']:.6f}"
    )
    print(
        f"Joint anomaly outside 3σ     : {stats['joint']['anomaly_rate_outside_3sigma']:.6f}"
    )

    print("\n[Joint sample-level coverage: both dimensions inside interval]")
    print(
        f"Joint 68% coverage   : {stats['joint']['coverage_68']:.6f}"
    )
    print(
        f"Joint 99% coverage   : {stats['joint']['coverage_99']:.6f}"
    )
    print(
        f"Joint 99.7% coverage : {stats['joint']['coverage_997']:.6f}"
    )


# =========================
# 6. Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate anomaly rates on new samples using a trained sparse GP transition model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained sparse GP checkpoint, e.g. gp_results_sparse/sparse_gp_transition_xy.pt"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to another sampled dataset (.pt) with keys v_prev, u_prev, v_next"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=""
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    x, y = load_xy_data(args.data_file)

    model = SparseGPTransitionModel2D(device=device)
    print(f"Loading sparse GP model from: {args.model_path}")
    model.load(args.model_path)

    with torch.no_grad():
        mean, std = model.predict(x, batch_size=args.batch_size)
        mean = mean.cpu()
        std = std.cpu()

    stats = estimate_anomaly_rates(y_true=y, mean=mean, std=std)
    pretty_print_stats(stats)

    if args.save_json:
        save_dir = os.path.dirname(args.save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nSaved stats to: {args.save_json}")


if __name__ == "__main__":
    main()