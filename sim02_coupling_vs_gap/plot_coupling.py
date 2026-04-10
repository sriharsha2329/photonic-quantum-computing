"""
Sim02 plotting: coupling g(d) vs gap width on semilog scale.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from coupling_sweep import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_coupling_semilog(results):
    """Semilog plot of g(d) for both models + fit comparison."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)
    d_nm = results["d"] * 1e9

    # --- Panel (a): g(d) on semilog scale ---
    ax = axes[0]
    ax.semilogy(d_nm, results["g_exp"], lw=2, label="Exponential $g_0 e^{-\\kappa d}$")
    ax.semilogy(d_nm, results["g_cmt"], lw=2, ls="--",
                label="CMT overlap integral")

    ax.axhline(1e9, ls=":", color="gray", lw=0.8)
    ax.text(350, 1.5e9, "$g = 1$ GHz", fontsize=9, color="gray")

    ax.set_xlabel("Gap width $d$ (nm)")
    ax.set_ylabel("Coupling rate $g$ (Hz)")
    ax.set_title("(a) Coupling vs gap width")
    ax.legend(frameon=False, fontsize=9)

    # --- Panel (b): g vs kappa*d (universal scaling) ---
    ax2 = axes[1]
    kd = results["kappa_d"]
    ax2.semilogy(kd, results["g_exp"] / config.G0, lw=2,
                 label="$g/g_0 = e^{-\\kappa d}$")
    ax2.semilogy(kd, results["g_cmt"] / config.G0, lw=2, ls="--",
                 label="CMT / $g_0$")

    ax2.set_xlabel(r"Dimensionless gap $\kappa d$")
    ax2.set_ylabel("$g / g_0$")
    ax2.set_title(r"(b) Universal scaling with $\kappa d$")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim02_coupling_vs_gap.{ext}"))
    plt.close(fig)
    print("Saved sim02_coupling_vs_gap")


def plot_fit_residual(results):
    """Residual plot showing fit quality."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)
    d_nm = results["d"] * 1e9

    # Ratio of CMT to exponential
    ratio = results["g_cmt"] / results["g_exp"]
    ax.plot(d_nm, ratio, lw=2, color="#009E73")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("Gap width $d$ (nm)")
    ax.set_ylabel("$g_{\\mathrm{CMT}} / g_{\\mathrm{exp}}$")
    ax.set_title("CMT vs exponential: constant ratio")
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim02_fit_residual.{ext}"))
    plt.close(fig)
    print("Saved sim02_fit_residual")


def main():
    results = run()
    plot_coupling_semilog(results)
    plot_fit_residual(results)
    print("\n--- Sim02 complete ---")


if __name__ == "__main__":
    main()
