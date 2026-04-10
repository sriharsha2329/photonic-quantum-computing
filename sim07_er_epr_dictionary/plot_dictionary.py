"""
Sim07 plotting: The unified ER=EPR dictionary figure.

4-panel figure with shared x-axis (kappa*d), showing that one parameter
controls coupling, entanglement, geometry, and tensor network structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from correspondence_map import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_unified_dictionary(results):
    """
    THE figure: 4-panel unified ER=EPR dictionary.
    All panels share x-axis = kappa*d.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharex=True)
    kd = results["kappa_d"]

    # --- Panel (a): Coupling g(d) — the "bridge strength" ---
    ax = axes[0, 0]
    ax.semilogy(kd, results["g"], lw=2.5, color="#0072B2")
    ax.set_ylabel("$g$ (Hz)")
    ax.set_title("(a) Coupling (bridge strength)", fontsize=11)
    ax.text(0.95, 0.92, "$g = g_0\\, e^{-\\kappa d}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Panel (b): Entanglement S_max and t_entangle ---
    ax = axes[0, 1]
    ax.plot(kd, results["S_max"] / np.log(2), lw=2.5, color="#D55E00",
            label="$S_{\\max} / \\ln 2$")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_ylabel("$S_{\\max} / \\ln 2$")
    ax.set_title("(b) Entanglement entropy", fontsize=11)

    ax2 = ax.twinx()
    ax2.semilogy(kd, results["t_entangle"], lw=2, ls="--", color="#CC79A7",
                 label="$t_{\\mathrm{ent}}$")
    ax2.set_ylabel("$t_{\\mathrm{ent}}$ (s)", color="#CC79A7")
    ax2.tick_params(axis="y", labelcolor="#CC79A7")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              frameon=False, fontsize=8, loc="center right")

    # --- Panel (c): Fubini-Study distance ---
    ax = axes[1, 0]
    ax.plot(kd, results["D_fs_max"], lw=2.5, color="#009E73")
    ax.axhline(np.pi / 4, ls="--", color="gray", lw=0.8)
    ax.text(3.5, np.pi / 4 + 0.01, "$\\pi/4$", fontsize=9, color="gray")
    ax.set_ylabel("$D_{FS}$ (rad)")
    ax.set_title("(c) Geometric distance (state space)", fontsize=11)
    ax.set_xlabel("$\\kappa d$")
    ax.set_ylim(0, 1.2)

    # --- Panel (d): Min-cut entropy (tensor network) ---
    ax = axes[1, 1]
    ax.plot(kd, results["S_mincut"] / np.log(2), lw=2.5, color="#CC79A7",
            label="Min-cut (TN)")
    ax.plot(kd, results["S_max"] / np.log(2), lw=2, ls="--", color="#D55E00",
            alpha=0.7, label="Von Neumann")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_ylabel("$S_{\\mathrm{min\\text{-}cut}} / \\ln 2$")
    ax.set_title("(d) RT analogue (min-cut entropy)", fontsize=11)
    ax.set_xlabel("$\\kappa d$")
    ax.set_ylim(0, 1.2)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Photonic ER=EPR Dictionary: $\\kappa d$ controls everything",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim07_er_epr_dictionary.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved sim07_er_epr_dictionary")


def plot_correspondence_scatter(results):
    """Supplementary: pairwise scatter plots of all quantities."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    kd = results["kappa_d"]
    g_norm = results["g"] / config.G0
    S_norm = results["S_max"] / np.log(2)
    D_norm = results["D_fs_max"] / (np.pi / 4)
    Smc_norm = results["S_mincut"] / np.log(2)

    # g/g0 vs S_max
    ax = axes[0]
    ax.scatter(g_norm, S_norm, s=8, c=kd, cmap="viridis", alpha=0.7)
    ax.set_xlabel("$g/g_0$")
    ax.set_ylabel("$S_{\\max}/\\ln 2$")
    ax.set_title("Coupling vs entropy")

    # D_FS vs S_max
    ax = axes[1]
    ax.scatter(D_norm, S_norm, s=8, c=kd, cmap="viridis", alpha=0.7)
    ax.set_xlabel("$D_{FS} / (\\pi/4)$")
    ax.set_ylabel("$S_{\\max}/\\ln 2$")
    ax.set_title("Geometry vs entropy")

    # S_mincut vs S_max
    ax = axes[2]
    sc = ax.scatter(Smc_norm, S_norm, s=8, c=kd, cmap="viridis", alpha=0.7)
    ax.plot([0, 1.1], [0, 1.1], ls=":", color="gray", lw=0.8)
    ax.set_xlabel("$S_{\\mathrm{min\\text{-}cut}}/\\ln 2$")
    ax.set_ylabel("$S_{\\max}/\\ln 2$")
    ax.set_title("TN min-cut vs vN entropy")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, label="$\\kappa d$")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim07_correspondence_scatter.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved sim07_correspondence_scatter")


def main():
    results = run()
    plot_unified_dictionary(results)
    plot_correspondence_scatter(results)
    print("\n--- Sim07 complete ---")


if __name__ == "__main__":
    main()
