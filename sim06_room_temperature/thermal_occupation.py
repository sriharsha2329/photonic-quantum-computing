"""
Sim06: Room-temperature feasibility analysis.

Computes:
  - Thermal photon occupation n_bar(omega, T) across spectrum
  - Complete decoherence budget (all noise sources)
  - Figure of merit g/Gamma vs gap width
  - Optimal operating window
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def thermal_occupation(omega, T):
    """Bose-Einstein: n_bar = 1/(exp(hbar*omega/kT) - 1)"""
    x = config.HBAR * omega / (config.KB * T)
    # Avoid overflow for large x
    x = np.clip(x, 0, 700)
    return 1.0 / (np.exp(x) - 1.0)


def thermal_occupation_spectrum(lam_arr, T_arr):
    """
    Compute n_bar for a grid of wavelengths and temperatures.

    Returns dict[T] -> n_bar_array
    """
    result = {}
    for T in T_arr:
        omega_arr = 2.0 * np.pi * config.C / lam_arr
        result[T] = thermal_occupation(omega_arr, T)
    return result


# ---------------------------------------------------------------------------
# Decoherence budget
# ---------------------------------------------------------------------------

def material_absorption_rate(lam=config.LAMBDA_UV):
    """
    Absorption rate in fused silica.
    At 400nm, Im(n) ~ 1e-8 for high-purity fused silica.
    gamma_abs = (omega / c) * 2 * Im(n) * c_group
    ~ omega * Im(n) / n_real
    """
    im_n = 1e-8  # typical for UV-grade fused silica at 400nm
    omega = 2.0 * np.pi * config.C / lam
    # Absorption coefficient alpha = 2 * omega * Im(n) / c
    alpha = 2.0 * omega * im_n / config.C
    # For a waveguide of length L ~ 10 um, rate = alpha * v_group
    v_group = config.C / config.N1
    gamma_abs = alpha * v_group
    return gamma_abs


def rayleigh_scattering_rate(lam=config.LAMBDA_UV):
    """
    Rayleigh scattering rate ~ 1/lambda^4.
    For fused silica, Rayleigh scattering loss ~ 0.7 dB/km at 1550nm.
    Scale to 400nm: factor of (1550/400)^4 ~ 226.
    """
    loss_1550 = 0.7  # dB/km at 1550nm
    loss_uv = loss_1550 * (1550e-9 / lam)**4  # dB/km

    # Convert dB/km to 1/s: alpha = loss * ln(10)/(10) / km, rate = alpha * v_group
    alpha = loss_uv * np.log(10) / (10.0 * 1e3)  # 1/m
    v_group = config.C / config.N1
    return alpha * v_group


def surface_roughness_rate():
    """
    Surface roughness scattering for nanophotonic waveguides.
    Typical: 1-10 dB/cm for silicon nitride at visible wavelengths.
    For high-quality silica: ~ 0.1 dB/cm.
    """
    loss_dbcm = 0.1  # dB/cm, optimistic for polished silica
    alpha = loss_dbcm * np.log(10) / (10.0 * 1e-2)  # 1/m
    v_group = config.C / config.N1
    return alpha * v_group


def detector_dark_count_rate():
    """
    Single-photon detector dark counts.
    Superconducting nanowire (SNSPD): ~ 1 Hz at room temp (N/A), ~0.01 Hz at 2K
    Si-APD (avalanche photodiode): ~ 100 Hz at room temp
    """
    return 100.0  # Hz, Si-APD at room temperature


def thermal_scattering_rate(T=config.T_ROOM, omega=config.OMEGA):
    """
    Thermal photon scattering rate: n_bar * gamma_spontaneous.
    For UV at room temp, n_bar ~ 10^-52, so this is negligible.
    """
    n_bar = thermal_occupation(omega, T)
    # Spontaneous emission rate for a photon in a waveguide ~ omega^3 * d^2 / (3 pi eps0 hbar c^3)
    # For our system this is dominated by material effects, not free-space emission.
    # Use a representative cavity linewidth ~ 1 GHz
    gamma_spont = 1e9  # Hz, representative
    return n_bar * gamma_spont


def decoherence_budget(T=config.T_ROOM, lam=config.LAMBDA_UV):
    """
    Complete decoherence budget. Returns dict of rates (Hz).
    """
    budget = {
        "thermal_scattering": thermal_scattering_rate(T),
        "material_absorption": material_absorption_rate(lam),
        "rayleigh_scattering": rayleigh_scattering_rate(lam),
        "surface_roughness": surface_roughness_rate(),
        "detector_dark_counts": detector_dark_count_rate(),
    }
    budget["total"] = sum(budget.values())
    return budget


def figure_of_merit(d_arr, T=config.T_ROOM):
    """
    Compute g(d) / Gamma_total for the operating window analysis.

    Returns
    -------
    g_arr : coupling rates
    gamma_total : total decoherence rate
    fom : g/Gamma ratio
    """
    g_arr = config.G0 * np.exp(-config.KAPPA * d_arr)
    budget = decoherence_budget(T)
    gamma_total = budget["total"]
    fom = g_arr / gamma_total
    return g_arr, gamma_total, fom


def run():
    """Execute all sim06 calculations."""
    results = {}

    # --- 1. Thermal occupation spectrum ---
    lam_arr = np.logspace(np.log10(100e-9), np.log10(1e-3), 500)  # 100nm to 1mm
    T_values = [config.T_ROOM, config.T_LN2, config.T_LHE]
    nbar = thermal_occupation_spectrum(lam_arr, T_values)
    results["lam_arr"] = lam_arr
    results["nbar"] = nbar
    results["T_values"] = T_values

    print("--- Thermal occupation at key wavelengths ---")
    for T in T_values:
        omega_uv = 2 * np.pi * config.C / config.LAMBDA_UV
        nb_uv = thermal_occupation(omega_uv, T)
        omega_ir = 2 * np.pi * config.C / 1550e-9
        nb_ir = thermal_occupation(omega_ir, T)
        print(f"  T = {T:6.1f} K: n_bar(400nm) = {nb_uv:.2e}, n_bar(1550nm) = {nb_ir:.2e}")

    # --- 2. Decoherence budget ---
    budget = decoherence_budget()
    results["budget"] = budget
    print("\n--- Decoherence budget (T = 300K, lambda = 400nm) ---")
    for key, val in budget.items():
        print(f"  {key:25s}: {val:.3e} Hz")

    # --- 3. Figure of merit vs gap ---
    d_arr = config.GAP_RANGE
    g_arr, gamma_total, fom = figure_of_merit(d_arr)
    results["d_arr"] = d_arr
    results["g_arr"] = g_arr
    results["gamma_total"] = gamma_total
    results["fom"] = fom
    results["kappa_d"] = config.KAPPA * d_arr

    # Find operating window: g/Gamma > 1
    viable = d_arr[fom > 1.0]
    if len(viable) > 0:
        d_max_viable = viable[-1]
        print(f"\n--- Operating window ---")
        print(f"g/Gamma > 1 for d < {d_max_viable*1e9:.1f} nm "
              f"(kappa*d < {config.KAPPA*d_max_viable:.2f})")
        print(f"g/Gamma at d=100nm: {fom[np.argmin(np.abs(d_arr - 100e-9))]:.1f}")
        print(f"g/Gamma at d=200nm: {fom[np.argmin(np.abs(d_arr - 200e-9))]:.1f}")
        results["d_max_viable"] = d_max_viable
    else:
        print("\nWARNING: No viable operating point found!")
        results["d_max_viable"] = 0

    # --- 4. Temperature comparison ---
    fom_temps = {}
    for T in T_values:
        _, gamma_T, fom_T = figure_of_merit(d_arr, T)
        fom_temps[T] = fom_T
    results["fom_temps"] = fom_temps

    return results


if __name__ == "__main__":
    run()
