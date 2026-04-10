"""
Sim06 plotting: Room temperature feasibility figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from thermal_occupation import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_thermal_occupation(results):
    """Fig 6a: n_bar vs wavelength for multiple temperatures."""
    fig, ax = plt.subplots(figsize=config.FIG_DOUBLE_COL)
    lam_um = results["lam_arr"] * 1e6  # convert to micrometers

    colors = ["#D55E00", "#0072B2", "#009E73"]
    for T, col in zip(results["T_values"], colors):
        nbar = results["nbar"][T]
        # Mask out zero/tiny values for log plot
        mask = nbar > 1e-100
        ax.semilogy(lam_um[mask], nbar[mask], lw=2, color=col,
                     label=f"$T = {T:.0f}$ K")

    ax.axvline(config.LAMBDA_UV * 1e6, ls="--", color="gray", lw=1,
               label=f"$\\lambda = {config.LAMBDA_UV*1e9:.0f}$ nm")
    ax.axvline(1.55, ls=":", color="gray", lw=1, label="$\\lambda = 1550$ nm")

    ax.set_xlabel("Wavelength ($\\mu$m)")
    ax.set_ylabel("$\\bar{n}(\\omega, T)$")
    ax.set_title("Thermal photon occupation: UV advantage")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0.1, 100)
    ax.set_xscale("log")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim06_thermal_occupation.{ext}"))
    plt.close(fig)
    print("Saved sim06_thermal_occupation")


def plot_decoherence_budget(results):
    """Fig 6b: Bar chart of decoherence sources."""
    budget = results["budget"]
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)

    sources = [k for k in budget if k != "total"]
    rates = [budget[k] for k in sources]
    labels = [s.replace("_", " ").title() for s in sources]

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    bars = ax.barh(labels, rates, color=colors[:len(sources)])
    ax.set_xscale("log")
    ax.set_xlabel("Decoherence rate (Hz)")
    ax.set_title("Decoherence budget ($T=300$ K)")

    # Add total line
    ax.axvline(budget["total"], ls="--", color="black", lw=1.5,
               label=f"Total = {budget['total']:.1e} Hz")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim06_decoherence_budget.{ext}"))
    plt.close(fig)
    print("Saved sim06_decoherence_budget")


def plot_figure_of_merit(results):
    """Fig 6c: g/Gamma vs gap width — the operating window."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)
    d_nm = results["d_arr"] * 1e9

    # Panel (a): g and Gamma on same semilog plot
    ax = axes[0]
    ax.semilogy(d_nm, results["g_arr"], lw=2, color="#0072B2",
                label="$g(d)$")
    ax.axhline(results["gamma_total"], ls="--", color="#D55E00", lw=2,
               label=f"$\\Gamma_{{\\mathrm{{tot}}}} = {results['gamma_total']:.1e}$ Hz")
    ax.set_xlabel("Gap width $d$ (nm)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title("(a) Coupling vs decoherence")
    ax.legend(frameon=False, fontsize=9)

    # Panel (b): g/Gamma ratio
    ax2 = axes[1]
    ax2.semilogy(d_nm, results["fom"], lw=2, color="#009E73")
    ax2.axhline(1.0, ls=":", color="gray", lw=1)
    ax2.fill_between(d_nm, 1e-5, results["fom"],
                     where=(results["fom"] > 1.0),
                     alpha=0.15, color="#009E73")

    if results["d_max_viable"] > 0:
        ax2.axvline(results["d_max_viable"] * 1e9, ls="--", color="#CC79A7",
                    lw=1, label=f"$d_{{\\max}} = {results['d_max_viable']*1e9:.0f}$ nm")
        ax2.legend(frameon=False, fontsize=9)

    ax2.set_xlabel("Gap width $d$ (nm)")
    ax2.set_ylabel("$g / \\Gamma$")
    ax2.set_title("(b) Figure of merit")
    ax2.set_ylim(bottom=0.1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim06_figure_of_merit.{ext}"))
    plt.close(fig)
    print("Saved sim06_figure_of_merit")


def main():
    results = run()
    plot_thermal_occupation(results)
    plot_decoherence_budget(results)
    plot_figure_of_merit(results)
    print("\n--- Sim06 complete ---")


if __name__ == "__main__":
    main()
