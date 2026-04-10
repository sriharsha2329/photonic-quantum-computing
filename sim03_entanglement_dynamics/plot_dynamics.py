"""
Sim03 plotting: entanglement dynamics figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from beam_splitter_evolution import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_entropy_vs_time(results):
    """Fig 3a: Von Neumann entropy S(t) for |1,0> at multiple gap widths."""
    fig, ax = plt.subplots(figsize=config.FIG_DOUBLE_COL)

    for key in sorted(results.keys()):
        if key.endswith("_1ph"):
            r = results[key]
            d_nm = r["d"] * 1e9
            ax.plot(r["g_times_t"] / np.pi, r["von_neumann"] / np.log(2),
                    lw=1.8, label=f"$d = {d_nm:.0f}$ nm")

    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.text(0.05, 1.03, "$S_{\\max} = \\ln 2$", fontsize=9, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("$S(\\rho_A) / \\ln 2$")
    ax.set_title("Von Neumann entropy: single photon $|1,0\\rangle$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0)
    ax.set_ylim(-0.02, 1.15)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim03_entropy_vs_time.{ext}"))
    plt.close(fig)
    print("Saved sim03_entropy_vs_time")


def plot_concurrence_vs_time(results):
    """Fig 3b: Concurrence for |1,0> at multiple gap widths."""
    fig, ax = plt.subplots(figsize=config.FIG_DOUBLE_COL)

    for key in sorted(results.keys()):
        if key.endswith("_1ph"):
            r = results[key]
            d_nm = r["d"] * 1e9
            ax.plot(r["g_times_t"] / np.pi, r["concurrence"],
                    lw=1.8, label=f"$d = {d_nm:.0f}$ nm")

    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("Concurrence $\\mathcal{C}$")
    ax.set_title("Wootters concurrence: $|1,0\\rangle$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim03_concurrence_vs_time.{ext}"))
    plt.close(fig)
    print("Saved sim03_concurrence_vs_time")


def plot_two_photon(results):
    """Fig 3c: |2,0> entropy — richer structure than single photon."""
    r = results["2ph_200nm"]
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)

    ax = axes[0]
    ax.plot(r["g_times_t"] / np.pi, r["von_neumann"] / np.log(2),
            lw=2, color="#D55E00")
    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("$S(\\rho_A) / \\ln 2$")
    ax.set_title("(a) Entropy: $|2,0\\rangle$, $d=200$ nm")
    ax.set_xlim(0)

    ax2 = axes[1]
    ax2.plot(r["g_times_t"] / np.pi, r["photon_number_A"],
             lw=2, label="$\\langle n_A \\rangle$")
    ax2.plot(r["g_times_t"] / np.pi, r["photon_number_B"],
             lw=2, label="$\\langle n_B \\rangle$")
    ax2.plot(r["g_times_t"] / np.pi, r["total_photon"],
             lw=1.5, ls=":", color="gray", label="$\\langle N_{\\mathrm{tot}} \\rangle$")
    ax2.set_xlabel("$g t / \\pi$")
    ax2.set_ylabel("Photon number")
    ax2.set_title("(b) Photon transfer")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlim(0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim03_two_photon.{ext}"))
    plt.close(fig)
    print("Saved sim03_two_photon")


def plot_lindblad(results):
    """Fig 3d: Decoherence effects — entropy decay under photon loss."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)

    ax = axes[0]
    ax2 = axes[1]
    for key in sorted(results.keys()):
        if key.startswith("lindblad_"):
            r = results[key]
            ratio = r["gamma"] / r["g"]
            lbl = f"$\\gamma/g = {ratio:.2f}$"
            ax.plot(r["g_times_t"] / np.pi, r["von_neumann"] / np.log(2),
                    lw=1.8, label=lbl)
            ax2.plot(r["g_times_t"] / np.pi, r["total_photon"],
                     lw=1.8, label=lbl)

    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("$S(\\rho_A) / \\ln 2$")
    ax.set_title("(a) Entropy with photon loss")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0)

    ax2.set_xlabel("$g t / \\pi$")
    ax2.set_ylabel("$\\langle N_{\\mathrm{tot}} \\rangle$")
    ax2.set_title("(b) Photon number decay")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlim(0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim03_lindblad.{ext}"))
    plt.close(fig)
    print("Saved sim03_lindblad")


def plot_photon_conservation(results):
    """Verification plot: total photon number for unitary evolution."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)
    r = results["d200nm_1ph"]
    deviation = np.abs(r["total_photon"] - 1.0)
    ax.semilogy(r["g_times_t"] / np.pi, deviation + 1e-16, lw=1.5)
    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("$|\\langle N \\rangle - 1|$")
    ax.set_title("Photon number conservation check")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim03_conservation_check.{ext}"))
    plt.close(fig)
    print("Saved sim03_conservation_check")


def main():
    results = run()
    plot_entropy_vs_time(results)
    plot_concurrence_vs_time(results)
    plot_two_photon(results)
    plot_lindblad(results)
    plot_photon_conservation(results)
    print("\n--- Sim03 complete ---")


if __name__ == "__main__":
    main()
