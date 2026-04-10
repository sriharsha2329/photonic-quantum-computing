"""
Sim01: Analytical solution of the evanescent field in a 3-region
waveguide-gap-waveguide system (Helmholtz equation).

Computes:
  - Field profile |E(z)|^2 through waveguide A | gap | waveguide B
  - Evanescent decay constant kappa from first principles
  - kappa vs incidence angle theta
  - kappa vs wavelength lambda
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ---------------------------------------------------------------------------
# 1. Evanescent decay constant from first principles
# ---------------------------------------------------------------------------

def kappa_from_params(omega, n1, n2, theta):
    """kappa = (omega/c) * sqrt(n1^2 sin^2(theta) - n2^2)"""
    return (omega / config.C) * np.sqrt(n1**2 * np.sin(theta)**2 - n2**2)


def kappa_vs_angle(theta_arr, lam=config.LAMBDA_UV, n1=config.N1, n2=config.N2):
    """Sweep kappa over incidence angles (must be above critical angle)."""
    omega = 2.0 * np.pi * config.C / lam
    k0 = omega / config.C
    arg = n1**2 * np.sin(theta_arr)**2 - n2**2
    arg = np.clip(arg, 0, None)
    return k0 * np.sqrt(arg)


def kappa_vs_wavelength(lam_arr, theta=config.THETA, n1=config.N1, n2=config.N2):
    """Sweep kappa over wavelength. Shorter lambda -> larger kappa."""
    k0_arr = 2.0 * np.pi / lam_arr
    arg = n1**2 * np.sin(theta)**2 - n2**2
    return k0_arr * np.sqrt(arg)


# ---------------------------------------------------------------------------
# 2. Three-region field profile  (TE polarisation)
#    Region 1 (z < 0):        waveguide A, propagating
#    Region 2 (0 < z < d):    vacuum gap, evanescent
#    Region 3 (z > d):        waveguide B, propagating
# ---------------------------------------------------------------------------

def field_profile(z_arr, d_gap, omega=config.OMEGA, n1=config.N1,
                  n2=config.N2, theta=config.THETA):
    """
    Compute |E(z)|^2 for the 3-region slab geometry.

    Parameters
    ----------
    z_arr : array, spatial coordinate across the structure (m)
    d_gap : float, gap width (m)

    Returns
    -------
    E2 : array, |E(z)|^2 normalised to 1 at z=0
    """
    k0 = omega / config.C
    kz1 = k0 * n1 * np.cos(theta)              # z-component in waveguide
    kappa = k0 * np.sqrt(n1**2 * np.sin(theta)**2 - n2**2)

    E2 = np.zeros_like(z_arr, dtype=float)

    # Region 1: z < 0  — standing wave (incident + reflected)
    mask1 = z_arr < 0
    E2[mask1] = np.cos(kz1 * z_arr[mask1])**2  # simplified standing wave

    # Region 2: 0 <= z <= d  — exponential decay
    mask2 = (z_arr >= 0) & (z_arr <= d_gap)
    E2[mask2] = np.exp(-2.0 * kappa * z_arr[mask2])

    # Region 3: z > d  — transmitted evanescent tail couples into propagating mode
    mask3 = z_arr > d_gap
    transmission = np.exp(-2.0 * kappa * d_gap)
    E2[mask3] = transmission * np.cos(kz1 * (z_arr[mask3] - d_gap))**2

    return E2


# ---------------------------------------------------------------------------
# 3. Run all computations and return results dict
# ---------------------------------------------------------------------------

def run():
    """Execute all sim01 calculations; return dict of results."""
    results = {}

    # Verify kappa at operating point
    kappa_calc = kappa_from_params(config.OMEGA, config.N1, config.N2, config.THETA)
    results["kappa"] = kappa_calc
    results["kappa_config"] = config.KAPPA
    print(f"kappa (calculated)   = {kappa_calc:.4e} m^-1")
    print(f"kappa (from config)  = {config.KAPPA:.4e} m^-1")
    print(f"1/kappa (decay len)  = {1.0/kappa_calc*1e9:.1f} nm")
    print(f"Critical angle       = {config.THETA_C_DEG:.2f} deg")
    print(f"Operating angle      = {config.THETA_DEG:.1f} deg")

    # Field profile for several gap widths
    d_gaps = [100e-9, 200e-9, 300e-9]
    z = np.linspace(-200e-9, 500e-9, 2000)
    profiles = {}
    for d in d_gaps:
        profiles[d] = field_profile(z, d)
    results["z"] = z
    results["profiles"] = profiles
    results["d_gaps"] = d_gaps

    # kappa vs angle
    results["theta_arr"] = config.THETA_RANGE
    results["theta_deg_arr"] = config.THETA_RANGE_DEG
    results["kappa_vs_angle"] = kappa_vs_angle(config.THETA_RANGE)

    # kappa vs wavelength
    results["lambda_arr"] = config.LAMBDA_RANGE
    results["kappa_vs_lambda"] = kappa_vs_wavelength(config.LAMBDA_RANGE)

    return results


if __name__ == "__main__":
    run()
