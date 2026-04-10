"""
Sim05 plotting: Tensor network, min-cut, and RT analogue figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from mera_evanescent import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_entropy_comparison(results):
    """Fig 5a: SVD entropy vs QuTiP entropy — exact agreement."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)

    # Panel (a): S_svd vs S_qutip at max entanglement over gap sweep
    ax = axes[0]
    kd = results["kappa_d"]
    ax.plot(kd, results["svd_entropy"] / np.log(2), lw=2,
            label="SVD (MPS)", color="#0072B2")
    ax.plot(kd, results["qutip_entropy"] / np.log(2), lw=2, ls="--",
            label="QuTiP $\\mathrm{Tr}(\\rho_A \\ln \\rho_A)$", color="#D55E00")
    ax.plot(kd, results["mincut_entropy"] / np.log(2), lw=1.5, ls=":",
            label="Min-cut", color="#009E73")
    ax.set_xlabel("$\\kappa d$")
    ax.set_ylabel("$S / \\ln 2$")
    ax.set_title("(a) Three routes, one entropy")
    ax.legend(frameon=False, fontsize=8)

    # Panel (b): time-resolved comparison
    ax2 = axes[1]
    tr = results["time_resolved"]
    gt = tr["g_times_t"] / np.pi
    ax2.plot(gt, tr["svd_entropy"] / np.log(2), lw=2,
             label="SVD", color="#0072B2")
    ax2.plot(gt, tr["qutip_entropy"] / np.log(2), lw=2, ls="--",
             label="QuTiP", color="#D55E00")
    ax2.set_xlabel("$g t / \\pi$")
    ax2.set_ylabel("$S / \\ln 2$")
    ax2.set_title("(b) Time-resolved ($d = 200$ nm)")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim05_entropy_comparison.{ext}"))
    plt.close(fig)
    print("Saved sim05_entropy_comparison")


def plot_tensor_network_diagram(results):
    """Fig 5b: Visual tensor network with bond thickness ~ chi."""
    fig, ax = plt.subplots(figsize=(config.FIG_DOUBLE_COL[0], 3.0))

    # Draw a simple 2-site tensor network for a few kd values
    kd_examples = [0.5, 1.0, 2.0, 3.5]
    y_positions = np.arange(len(kd_examples))[::-1] * 1.5

    for idx, kd in enumerate(kd_examples):
        y = y_positions[idx]
        chi_eff = np.exp(-kd)

        # Site A
        circle_a = plt.Circle((-1.5, y), 0.3, color="#0072B2", zorder=3)
        ax.add_patch(circle_a)
        ax.text(-1.5, y, "A", ha="center", va="center", color="white",
                fontweight="bold", fontsize=10, zorder=4)

        # Site B
        circle_b = plt.Circle((1.5, y), 0.3, color="#D55E00", zorder=3)
        ax.add_patch(circle_b)
        ax.text(1.5, y, "B", ha="center", va="center", color="white",
                fontweight="bold", fontsize=10, zorder=4)

        # Bond line (thickness proportional to chi)
        lw = max(0.5, 8.0 * chi_eff)
        ax.plot([-1.2, 1.2], [y, y], lw=lw, color="#009E73", alpha=0.7, zorder=2)

        # Min-cut indicator (dashed line)
        ax.plot([0, 0], [y - 0.35, y + 0.35], lw=1.5, ls="--",
                color="#CC79A7", zorder=5)

        # Label
        S_val = np.log(2)  # at max entanglement for |1,0>
        ax.text(2.5, y, f"$\\kappa d = {kd:.1f}$\n$\\chi \\propto {chi_eff:.2f}$",
                ha="left", va="center", fontsize=9)

    ax.set_xlim(-3, 5)
    ax.set_ylim(-1, y_positions[0] + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tensor network: bond dimension $\\chi \\propto e^{-\\kappa d}$, "
                 "min-cut (dashed)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim05_tensor_network.{ext}"))
    plt.close(fig)
    print("Saved sim05_tensor_network")


def plot_mincut_scaling(results):
    """Fig 5c: Min-cut entropy scaling with kappa*d."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)

    chain = results["chain"]
    ax.plot(chain["kd_values"], chain["bond_entropies"], "o-",
            lw=2, color="#CC79A7", markersize=5)
    ax.set_xlabel("$\\kappa d$ at bond")
    ax.set_ylabel("Min-cut entropy (nats)")
    ax.set_title("RT analogue: bond entropy vs $\\kappa d$")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim05_mincut_scaling.{ext}"))
    plt.close(fig)
    print("Saved sim05_mincut_scaling")


def main():
    results = run()
    plot_entropy_comparison(results)
    plot_tensor_network_diagram(results)
    plot_mincut_scaling(results)
    print("\n--- Sim05 complete ---")


if __name__ == "__main__":
    main()
