import torch
import numpy as np
import argparse
import os
import sys

# import your model
sys.path.insert(0, os.path.dirname(__file__))
from transition_models import MLPTransitionModel


@torch.no_grad()
def analyze_residual_correlation_xy(
    v_prev, u_prev, v_next, mlp_model, device="cpu", num_bins=4
):
    pred = mlp_model.predict(v_prev.to(device), u_prev.to(device)).cpu()
    residual = v_next - pred

    rx = residual[:, 0].numpy()
    ry = residual[:, 1].numpy()

    print("\n===== Global Correlation =====")
    print("Pearson(res_x, res_y) =", np.corrcoef(rx, ry)[0, 1])
    print("Pearson(|res_x|, |res_y|) =", np.corrcoef(np.abs(rx), np.abs(ry))[0, 1])

    # Spearman
    rx_rank = np.argsort(np.argsort(rx))
    ry_rank = np.argsort(np.argsort(ry))
    print("Spearman(res_x, res_y) =", np.corrcoef(rx_rank, ry_rank)[0, 1])

    # 局部分析
    v_norm = torch.norm(v_prev, dim=1).numpy()
    u_norm = torch.norm(u_prev, dim=1).numpy()

    def bin_edges(x):
        qs = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(x, qs)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return edges

    print("\n===== Local Correlation by |v_prev| =====")
    v_edges = bin_edges(v_norm)
    for i in range(num_bins):
        mask = (v_norm > v_edges[i]) & (v_norm <= v_edges[i + 1])
        if mask.sum() < 50:
            continue
        corr = np.corrcoef(rx[mask], ry[mask])[0, 1]
        print(f"Bin {i}: corr={corr:.4f}, n={mask.sum()}")

    print("\n===== Local Correlation by |u_prev| =====")
    u_edges = bin_edges(u_norm)
    for i in range(num_bins):
        mask = (u_norm > u_edges[i]) & (u_norm <= u_edges[i + 1])
        if mask.sum() < 50:
            continue
        corr = np.corrcoef(rx[mask], ry[mask])[0, 1]
        print(f"Bin {i}: corr={corr:.4f}, n={mask.sum()}")


def main(data_file, model_file, device="cpu"):
    print(f"Loading data from {data_file}")
    data = torch.load(data_file, map_location="cpu", weights_only=True)

    # 只取 XY
    v_prev = data["v_prev"].float()[:, :2]
    u_prev = data["u_prev"].float()[:, :2]
    v_next = data["v_next"].float()[:, :2]

    print(f"Loaded {v_prev.shape[0]} samples")

    print(f"\nLoading MLP model from {model_file}")
    mlp_model = MLPTransitionModel(state_dim=2, input_dim=2, hidden_dim=32, device=device)
    mlp_model.load(model_file)

    analyze_residual_correlation_xy(
        v_prev=v_prev,
        u_prev=u_prev,
        v_next=v_next,
        mlp_model=mlp_model,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="nominal_data.pt")
    parser.add_argument("--model_file", type=str, default="./nominal_models_xy/mlp_model_xy.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args.data_file, args.model_file, args.device)