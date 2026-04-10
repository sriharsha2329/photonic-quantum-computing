"""
Sim02: Coupling strength g(d) vs gap width d.

Computes:
  - g(d) = g0 * exp(-kappa * d)  [simple exponential model]
  - Coupled-mode theory overlap integral  [more rigorous]
  - Numerical fit to extract kappa and compare with analytical
"""

import numpy as np
from scipy.optimize import curve_fit
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def coupling_exponential(d_arr, kappa=config.KAPPA, g0=config.G0):
    """Simple exponential coupling: g(d) = g0 * exp(-kappa * d)."""
    return g0 * np.exp(-kappa * d_arr)


def coupling_cmt_overlap(d_arr, kappa=config.KAPPA, g0=config.G0,
                          n1=config.N1, n2=config.N2, omega=config.OMEGA):
    """
    Coupled-mode theory (CMT) coupling via overlap integral.

    For two identical slab waveguides separated by gap d, the CMT coupling is:
        g_cmt(d) = g0 * (2 * kappa * delta) / (kappa^2 + delta^2) * exp(-kappa * d)

    where delta = (omega/c) * n1 * cos(theta) is the propagation constant
    in the z-direction inside the waveguide. The prefactor accounts for the
    overlap of the evanescent tails with the adjacent waveguide core.
    """
    delta = (omega / config.C) * n1 * np.cos(config.THETA)
    prefactor = 2.0 * kappa * delta / (kappa**2 + delta**2)
    return g0 * prefactor * np.exp(-kappa * d_arr)


def fit_exponential(d_arr, g_arr):
    """
    Fit g(d) = A * exp(-alpha * d) to extract decay constant alpha.
    Returns (A_fit, alpha_fit, covariance).
    """
    def model(d, A, alpha):
        return A * np.exp(-alpha * d)

    # Initial guesses
    p0 = [g_arr[0], config.KAPPA]
    popt, pcov = curve_fit(model, d_arr, g_arr, p0=p0, maxfev=10000)
    return popt, pcov


def run():
    """Execute all sim02 calculations; return dict of results."""
    d = config.GAP_RANGE
    results = {}

    # Exponential model
    g_exp = coupling_exponential(d)
    results["d"] = d
    results["g_exp"] = g_exp

    # CMT overlap model
    g_cmt = coupling_cmt_overlap(d)
    results["g_cmt"] = g_cmt

    # Fit exponential to CMT result to extract effective kappa
    popt_exp, _ = fit_exponential(d, g_exp)
    popt_cmt, _ = fit_exponential(d, g_cmt)

    results["fit_exp"] = {"g0_fit": popt_exp[0], "kappa_fit": popt_exp[1]}
    results["fit_cmt"] = {"g0_fit": popt_cmt[0], "kappa_fit": popt_cmt[1]}

    print(f"Analytical kappa          = {config.KAPPA:.6e} m^-1")
    print(f"Fit kappa (exponential)   = {popt_exp[1]:.6e} m^-1")
    print(f"Fit kappa (CMT overlap)   = {popt_cmt[1]:.6e} m^-1")
    print(f"CMT/exp kappa ratio       = {popt_cmt[1]/config.KAPPA:.6f}")
    print(f"\nCMT prefactor reduction   = {g_cmt[0]/g_exp[0]:.4f}")
    print(f"  -> same kappa, different g0 prefactor")

    # Dimensionless kappa*d product
    kd = config.KAPPA * d
    results["kappa_d"] = kd

    # Key result: at what gap does g drop to 1 GHz? (relevant for sim06)
    g_threshold = 1e9  # 1 GHz
    d_threshold = -np.log(g_threshold / config.G0) / config.KAPPA
    results["d_threshold_1GHz"] = d_threshold
    print(f"\ng drops to 1 GHz at d = {d_threshold*1e9:.1f} nm (kappa*d = {config.KAPPA*d_threshold:.2f})")

    return results


if __name__ == "__main__":
    run()
