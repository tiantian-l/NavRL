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

    v_prev = data["v_prev"].float()[:, :2]   # vx, vy
    u_prev = data["u_prev"].float()[:, :2]   # ux, uy

    return v_prev, u_prev


# =========================
# 2. Print stats
# =========================
def print_stats(name, x):
    x_np = x.numpy()
    print(f"\n[{name}]")
    print(f"  mean: {x_np.mean(axis=0)}")
    print(f"  std : {x_np.std(axis=0)}")
    print(f"  min : {x_np.min(axis=0)}")
    print(f"  max : {x_np.max(axis=0)}")


# =========================
# 3. Plot functions (save only)
# =========================
def plot_hist(x, title, save_path):
    x = x.numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=100)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()   # 🔥 关键：防止阻塞 + 防止内存爆

    print(f"Saved: {save_path}")


def plot_2d(x, y, title, save_path):
    x = x.numpy()
    y = y.numpy()
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=1, alpha=0.3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# =========================
# 4. Main
# =========================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    v_prev, u_prev = load_data(args.data_file)

    vx = v_prev[:, 0]
    vy = v_prev[:, 1]
    ux = u_prev[:, 0]
    uy = u_prev[:, 1]

    # =========================
    # Stats
    # =========================
    print_stats("Velocity (vx, vy)", v_prev)
    print_stats("Control  (ux, uy)", u_prev)

    # =========================
    # Histogram
    # =========================
    plot_hist(vx, "vx distribution", os.path.join(args.output_dir, "vx_hist.png"))
    plot_hist(vy, "vy distribution", os.path.join(args.output_dir, "vy_hist.png"))
    plot_hist(ux, "ux distribution", os.path.join(args.output_dir, "ux_hist.png"))
    plot_hist(uy, "uy distribution", os.path.join(args.output_dir, "uy_hist.png"))

    # =========================
    # 2D distributions
    # =========================
    plot_2d(vx, vy, "vx vs vy", os.path.join(args.output_dir, "vx_vy.png"))
    plot_2d(ux, uy, "ux vs uy", os.path.join(args.output_dir, "ux_uy.png"))

    plot_2d(vx, ux, "vx vs ux", os.path.join(args.output_dir, "vx_ux.png"))
    plot_2d(vy, uy, "vy vs uy", os.path.join(args.output_dir, "vy_uy.png"))

    print(f"\nAll plots saved to: {args.output_dir}")


# =========================
# 5. CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data_analysis_plots")

    args = parser.parse_args()
    main(args)