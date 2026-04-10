"""
Sim04 plotting: Fubini-Study geometry figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from state_space_geodesic import run

plt.rcParams.update(config.FIGURE_STYLE)
figdir = os.path.join(os.path.dirname(__file__), "..", config.FIGURE_DIR)
os.makedirs(figdir, exist_ok=True)


def plot_fs_distance_and_entropy(results):
    """Fig 4a: D_FS(t) and S(t) overlaid — show monotonic relationship."""
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)

    # Panel (a): D_FS and S vs gt/pi
    ax = axes[0]
    r = results["d200nm"]
    gt = r["g_times_t"] / np.pi

    ax.plot(gt, r["D_fs"], lw=2, color="#0072B2", label="$D_{FS}(t)$")
    ax2 = ax.twinx()
    ax2.plot(gt, r["von_neumann"] / np.log(2), lw=2, ls="--",
             color="#D55E00", label="$S / \\ln 2$")

    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("$D_{FS}$ (rad)", color="#0072B2")
    ax2.set_ylabel("$S(\\rho_A) / \\ln 2$", color="#D55E00")
    ax.set_title("(a) Geometry tracks entanglement")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)

    # Panel (b): D_FS vs S parametric plot
    ax3 = axes[1]
    for key in sorted(results.keys()):
        if key.startswith("d") and key.endswith("nm"):
            r2 = results[key]
            d_nm = r2["d"] * 1e9
            # Only first half-Rabi period (monotonic part)
            idx_half = len(r2["tlist"]) // 4
            ax3.plot(r2["von_neumann"][:idx_half] / np.log(2),
                     r2["D_fs"][:idx_half],
                     lw=1.8, label=f"$d = {d_nm:.0f}$ nm")

    ax3.set_xlabel("$S(\\rho_A) / \\ln 2$")
    ax3.set_ylabel("$D_{FS}$ (rad)")
    ax3.set_title("(b) $D_{FS}$ vs entropy (parametric)")
    ax3.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim04_fs_geometry.{ext}"))
    plt.close(fig)
    print("Saved sim04_fs_geometry")


def plot_fs_at_max_vs_gap(results):
    """Fig 4b: D_FS at maximum entanglement vs gap — should be constant pi/2."""
    sw = results["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_DOUBLE_COL)

    # Panel (a): D_FS at max entanglement vs d
    ax = axes[0]
    ax.plot(sw["d"] * 1e9, sw["D_fs_at_max"], lw=2, color="#009E73")
    ax.axhline(np.pi / 4, ls="--", color="#0072B2", lw=0.8)
    ax.text(300, np.pi / 4 + 0.03, "$\\pi/4$", fontsize=10, color="#0072B2")
    ax.set_xlabel("Gap width $d$ (nm)")
    ax.set_ylabel("$D_{FS}$ to max entanglement (rad)")
    ax.set_title("(a) Constant geodesic distance $\\pi/4$")
    ax.set_ylim(0, 2.0)

    # Panel (b): time to max entanglement vs d (semilog — exponential in kd)
    ax2 = axes[1]
    ax2.semilogy(sw["d"] * 1e9, sw["t_to_max"], lw=2, color="#CC79A7")
    ax2.set_xlabel("Gap width $d$ (nm)")
    ax2.set_ylabel("$t_{\\mathrm{ent}}$ (s)")
    ax2.set_title("(b) Entanglement time $\\propto e^{+\\kappa d}$")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim04_fs_vs_gap.{ext}"))
    plt.close(fig)
    print("Saved sim04_fs_vs_gap")


def plot_quantum_speed(results):
    """Fig 4c: Quantum speed (ds_FS/dt) vs time."""
    fig, ax = plt.subplots(figsize=config.FIG_SINGLE_COL)

    for key in sorted(results.keys()):
        if key.startswith("d") and key.endswith("nm"):
            r = results[key]
            d_nm = r["d"] * 1e9
            gt = r["g_times_t"] / np.pi
            # Normalise speed by g for universal comparison
            ax.plot(gt, r["speed"] / r["g"], lw=1.5, label=f"$d={d_nm:.0f}$ nm")

    ax.set_xlabel("$g t / \\pi$")
    ax.set_ylabel("Quantum speed / $g$")
    ax.set_title("Quantum speed on state manifold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"sim04_quantum_speed.{ext}"))
    plt.close(fig)
    print("Saved sim04_quantum_speed")


def main():
    results = run()
    plot_fs_distance_and_entropy(results)
    plot_fs_at_max_vs_gap(results)
    plot_quantum_speed(results)
    print("\n--- Sim04 complete ---")


if __name__ == "__main__":
    main()
