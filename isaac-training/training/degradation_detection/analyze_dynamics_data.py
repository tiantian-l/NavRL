import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# =========================
# 1. Load data
# =========================
def load_data(path):
    print(f"Loading data from {path} ...")
    data = torch.load(path, map_location="cpu")

    required_keys = ["state", "action", "next_state"]
    for k in required_keys:
        if k not in data:
            raise KeyError(
                f"Missing key '{k}' in dataset. "
                f"Expected keys: {required_keys}"
            )

    state = data["state"].float()
    action = data["action"].float()
    next_state = data["next_state"].float()

    if state.ndim != 2 or state.shape[1] < 2:
        raise ValueError(f"'state' should have shape (N, >=2), got {state.shape}")
    if action.ndim != 2 or action.shape[1] < 2:
        raise ValueError(f"'action' should have shape (N, >=2), got {action.shape}")
    if next_state.ndim != 2 or next_state.shape[1] < 2:
        raise ValueError(f"'next_state' should have shape (N, >=2), got {next_state.shape}")

    v_prev = state[:, :2]       # vx, vy
    u_prev = action[:, :2]      # ux, uy
    v_next = next_state[:, :2]  # vx_next, vy_next
    delta_v = v_next - v_prev   # dvx, dvy

    return v_prev, u_prev, v_next, delta_v


# =========================
# 2. Stats
# =========================
def get_stats_dict(x):
    x_np = x.numpy()
    return {
        "mean": x_np.mean(axis=0),
        "std": x_np.std(axis=0),
        "min": x_np.min(axis=0),
        "max": x_np.max(axis=0),
    }


def print_stats(name, x):
    stats = get_stats_dict(x)
    print(f"\n[{name}]")
    print(f"  mean: {stats['mean']}")
    print(f"  std : {stats['std']}")
    print(f"  min : {stats['min']}")
    print(f"  max : {stats['max']}")


def format_stats_for_file(name, x):
    stats = get_stats_dict(x)
    lines = [
        f"[{name}]",
        f"  mean: {stats['mean']}",
        f"  std : {stats['std']}",
        f"  min : {stats['min']}",
        f"  max : {stats['max']}",
        "",
    ]
    return "\n".join(lines)


def compute_near_zero_ratio(x, threshold=0.1):
    # x: (N, 2)
    norm = torch.norm(x, dim=1)
    return (norm < threshold).float().mean().item()


# =========================
# 3. Plot helpers
# =========================
def plot_hist(x, title, save_path, bins=100):
    x = x.numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_scatter(x, y, title, save_path, max_points=50000):
    x = x.numpy()
    y = y.numpy()

    if len(x) > max_points:
        idx = np.random.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=1, alpha=0.3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_hist(x, y, title, save_path, bins=100):
    x = x.numpy()
    y = y.numpy()

    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =========================
# 4. Save summary text
# =========================
def save_summary(
    output_path,
    v_prev,
    u_prev,
    v_next,
    delta_v,
    near_zero_v_ratio,
    near_zero_u_ratio,
    dataset_size,
):
    lines = []
    lines.append("Dynamics Dataset Analysis Summary")
    lines.append("=" * 40)
    lines.append(f"Number of samples: {dataset_size}")
    lines.append("")

    lines.append(format_stats_for_file("Velocity state (vx, vy)", v_prev))
    lines.append(format_stats_for_file("Control action (ux, uy)", u_prev))
    lines.append(format_stats_for_file("Next velocity (vx_next, vy_next)", v_next))
    lines.append(format_stats_for_file("Delta velocity (dvx, dvy)", delta_v))

    lines.append("[Near-zero ratios]")
    lines.append(f"  ||v|| < 0.1 : {near_zero_v_ratio:.6f}")
    lines.append(f"  ||u|| < 0.1 : {near_zero_u_ratio:.6f}")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


# =========================
# 5. Main
# =========================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    v_prev, u_prev, v_next, delta_v = load_data(args.data_file)

    vx = v_prev[:, 0]
    vy = v_prev[:, 1]

    ux = u_prev[:, 0]
    uy = u_prev[:, 1]

    vx_next = v_next[:, 0]
    vy_next = v_next[:, 1]

    dvx = delta_v[:, 0]
    dvy = delta_v[:, 1]

    dataset_size = v_prev.shape[0]

    print(f"\nDataset size: {dataset_size}")

    # =========================
    # Stats
    # =========================
    print_stats("Velocity state (vx, vy)", v_prev)
    print_stats("Control action (ux, uy)", u_prev)
    print_stats("Next velocity (vx_next, vy_next)", v_next)
    print_stats("Delta velocity (dvx, dvy)", delta_v)

    near_zero_v_ratio = compute_near_zero_ratio(v_prev, threshold=args.zero_threshold)
    near_zero_u_ratio = compute_near_zero_ratio(u_prev, threshold=args.zero_threshold)

    print("\n[Near-zero ratios]")
    print(f"  ||v|| < {args.zero_threshold}: {near_zero_v_ratio:.6f}")
    print(f"  ||u|| < {args.zero_threshold}: {near_zero_u_ratio:.6f}")

    # =========================
    # 1D Histograms
    # =========================
    plot_hist(vx, "vx distribution", os.path.join(args.output_dir, "vx_hist.png"), bins=args.bins)
    plot_hist(vy, "vy distribution", os.path.join(args.output_dir, "vy_hist.png"), bins=args.bins)
    plot_hist(ux, "ux distribution", os.path.join(args.output_dir, "ux_hist.png"), bins=args.bins)
    plot_hist(uy, "uy distribution", os.path.join(args.output_dir, "uy_hist.png"), bins=args.bins)

    plot_hist(vx_next, "vx_next distribution", os.path.join(args.output_dir, "vx_next_hist.png"), bins=args.bins)
    plot_hist(vy_next, "vy_next distribution", os.path.join(args.output_dir, "vy_next_hist.png"), bins=args.bins)

    plot_hist(dvx, "dvx distribution", os.path.join(args.output_dir, "dvx_hist.png"), bins=args.bins)
    plot_hist(dvy, "dvy distribution", os.path.join(args.output_dir, "dvy_hist.png"), bins=args.bins)

    # =========================
    # 2D Scatter Plots
    # =========================
    plot_2d_scatter(vx, vy, "vx vs vy", os.path.join(args.output_dir, "vx_vy_scatter.png"), max_points=args.max_scatter_points)
    plot_2d_scatter(ux, uy, "ux vs uy", os.path.join(args.output_dir, "ux_uy_scatter.png"), max_points=args.max_scatter_points)
    plot_2d_scatter(vx, ux, "vx vs ux", os.path.join(args.output_dir, "vx_ux_scatter.png"), max_points=args.max_scatter_points)
    plot_2d_scatter(vy, uy, "vy vs uy", os.path.join(args.output_dir, "vy_uy_scatter.png"), max_points=args.max_scatter_points)

    plot_2d_scatter(vx_next, vy_next, "vx_next vs vy_next", os.path.join(args.output_dir, "vxnext_vynext_scatter.png"), max_points=args.max_scatter_points)
    plot_2d_scatter(dvx, dvy, "dvx vs dvy", os.path.join(args.output_dir, "dvx_dvy_scatter.png"), max_points=args.max_scatter_points)

    plot_2d_scatter(ux, dvx, "ux vs dvx", os.path.join(args.output_dir, "ux_dvx_scatter.png"), max_points=args.max_scatter_points)
    plot_2d_scatter(uy, dvy, "uy vs dvy", os.path.join(args.output_dir, "uy_dvy_scatter.png"), max_points=args.max_scatter_points)

    # =========================
    # 2D Histograms / Heatmaps
    # =========================
    plot_2d_hist(vx, vy, "vx vs vy heatmap", os.path.join(args.output_dir, "vx_vy_heatmap.png"), bins=args.bins)
    plot_2d_hist(ux, uy, "ux vs uy heatmap", os.path.join(args.output_dir, "ux_uy_heatmap.png"), bins=args.bins)
    plot_2d_hist(vx, ux, "vx vs ux heatmap", os.path.join(args.output_dir, "vx_ux_heatmap.png"), bins=args.bins)
    plot_2d_hist(vy, uy, "vy vs uy heatmap", os.path.join(args.output_dir, "vy_uy_heatmap.png"), bins=args.bins)

    plot_2d_hist(vx_next, vy_next, "vx_next vs vy_next heatmap", os.path.join(args.output_dir, "vxnext_vynext_heatmap.png"), bins=args.bins)
    plot_2d_hist(dvx, dvy, "dvx vs dvy heatmap", os.path.join(args.output_dir, "dvx_dvy_heatmap.png"), bins=args.bins)

    plot_2d_hist(ux, dvx, "ux vs dvx heatmap", os.path.join(args.output_dir, "ux_dvx_heatmap.png"), bins=args.bins)
    plot_2d_hist(uy, dvy, "uy vs dvy heatmap", os.path.join(args.output_dir, "uy_dvy_heatmap.png"), bins=args.bins)

    # =========================
    # Save summary
    # =========================
    summary_path = os.path.join(args.output_dir, "summary.txt")
    save_summary(
        summary_path,
        v_prev,
        u_prev,
        v_next,
        delta_v,
        near_zero_v_ratio,
        near_zero_u_ratio,
        dataset_size,
    )

    print(f"\nAll analysis results saved to: {args.output_dir}")


# =========================
# 6. CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to dynamics_transitions_*.pt")
    parser.add_argument("--output_dir", type=str, default="./dynamics_analysis_plots",
                        help="Directory to save analysis plots")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of bins for histograms / heatmaps")
    parser.add_argument("--max_scatter_points", type=int, default=50000,
                        help="Max points used in scatter plots to avoid huge files")
    parser.add_argument("--zero_threshold", type=float, default=0.1,
                        help="Threshold for near-zero norm ratio")

    args = parser.parse_args()
    main(args)