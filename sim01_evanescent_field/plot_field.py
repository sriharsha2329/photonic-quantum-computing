"""
Sim01 plotting: evanescent field profile, kappa sweeps.
Produces publication-quality figures for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from evanescent_decay import run

plt.rcParams.update(config.FIGURE_STYLE)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR), exist_ok=True)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)


def plot_field_profile(results):
    """Fig 1a: |E(z)|^2 through the 3-region structure."""
    fig, ax = plt.subplots(figsize=config.FIG_DOUBLE_COL)
    z_nm = results["z"] * 1e9

    for d, E2 in results["profiles"].items():
        ax.plot(z_nm, E2, label=f"$d = {d*1e9:.0f}$ nm")

    # Shade gap region for the widest gap
    d_max = max(results["d_gaps"])
    ax.axvspan(0, d_max * 1e9, alpha=0.08, color="gray")
    ax.axvline(0, ls="--", color="gray", lw=0.8)

    ax.set_xlabel("Position $z$ (nm)")
    ax.set_ylabel(r"$|E(z)|^2$ (normalised)")
    ax.set_title("Evanescent field profile: waveguide A | gap | waveguide B")
    ax.legend(frameon=False)
    ax.set_xlim(z_nm[0], z_nm[-1])
    ax.set_ylim(bottom=0)

    # Annotate regions
    ax.text(-100, 0.85, "WG A", ha="center", fontsize=10, color="#555")
    ax.text(d_max * 1e9 / 2, 0.5, "gap", ha="center", fontsize=10, color="#555")
    ax.text(d_max * 1e9 + 100, 0.15, "WG B", ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim01_field_profile.{ext}"))
    plt.close(fig)
    print("Saved sim01_field_profile")


def plot_kappa_vs_angle(results):
    """Fig 1b: kappa vs incidence angle."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)
    ax.plot(results["theta_deg_arr"], results["kappa_vs_angle"] * 1e-6,
            color="#0072B2", lw=2)
    ax.axvline(config.THETA_C_DEG, ls=":", color="gray", lw=1,
               label=rf"$\theta_c = {config.THETA_C_DEG:.1f}°$")
    ax.axvline(config.THETA_DEG, ls="--", color="#D55E00", lw=1,
               label=rf"operating $\theta = {config.THETA_DEG:.0f}°$")
    ax.set_xlabel(r"Incidence angle $\theta$ (deg)")
    ax.set_ylabel(r"$\kappa$ ($\mu$m$^{-1}$)")
    ax.set_title(r"Evanescent decay constant vs $\theta$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim01_kappa_vs_angle.{ext}"))
    plt.close(fig)
    print("Saved sim01_kappa_vs_angle")


def plot_kappa_vs_wavelength(results):
    """Fig 1c: kappa vs wavelength — demonstrates UV advantage."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)
    lam_nm = results["lambda_arr"] * 1e9
    kap_um = results["kappa_vs_lambda"] * 1e-6

    ax.plot(lam_nm, kap_um, color="#009E73", lw=2)
    ax.axvline(config.LAMBDA_UV * 1e9, ls="--", color="#D55E00", lw=1,
               label=rf"$\lambda = {config.LAMBDA_UV*1e9:.0f}$ nm (operating)")

    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel(r"$\kappa$ ($\mu$m$^{-1}$)")
    ax.set_title(r"UV advantage: shorter $\lambda \Rightarrow$ larger $\kappa$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim01_kappa_vs_wavelength.{ext}"))
    plt.close(fig)
    print("Saved sim01_kappa_vs_wavelength")


def main():
    results = run()
    plot_field_profile(results)
    plot_kappa_vs_angle(results)
    plot_kappa_vs_wavelength(results)
    print("\n--- Sim01 complete ---")


if __name__ == "__main__":
    main()
