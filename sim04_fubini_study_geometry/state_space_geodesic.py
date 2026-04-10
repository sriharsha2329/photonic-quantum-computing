"""
Sim04: Fubini-Study metric and geometric phase on the evolving state manifold.

Computes:
  - Fubini-Study infinitesimal distance ds_FS at each timestep
  - Integrated geodesic distance D_FS(t)
  - Berry/geometric phase for cyclic evolution
  - Correlation between D_FS and entanglement entropy
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim03_entanglement_dynamics"))
from beam_splitter_evolution import build_hamiltonian, evolve_pure, compute_entanglement_measures


def fubini_study_distance(psi1, psi2):
    """
    Fubini-Study distance between two pure states:
        d_FS = arccos(|<psi1|psi2>|)
    """
    overlap = np.abs(psi1.overlap(psi2))
    overlap = min(overlap, 1.0)  # numerical safety
    return np.arccos(overlap)


def compute_fs_trajectory(states, tlist):
    """
    Compute the Fubini-Study distance trajectory.

    Returns
    -------
    ds_fs : array, infinitesimal FS distance at each step
    D_fs  : array, cumulative geodesic distance
    speed : array, ds/dt (quantum speed)
    """
    n = len(states)
    ds_fs = np.zeros(n)
    D_fs = np.zeros(n)

    for i in range(1, n):
        ds_fs[i] = fubini_study_distance(states[i-1], states[i])
        D_fs[i] = D_fs[i-1] + ds_fs[i]

    dt = np.diff(tlist, prepend=tlist[0])
    dt[0] = dt[1]  # avoid division by zero
    speed = ds_fs / dt

    return ds_fs, D_fs, speed


def compute_geometric_phase(states, tlist):
    """
    Pancharatnam geometric phase: total phase minus dynamic phase.

    For H = g(a†b + ab†), the dynamic phase is phi_dyn = -E*t/hbar.
    The geometric phase is:
        phi_geo = phi_total - phi_dyn
        phi_total = arg(<psi(0)|psi(t)>)
        phi_dyn = -Im(integral <psi|H|psi> dt) / hbar  [set hbar=1 in natural units]
    """
    n = len(states)
    psi0 = states[0]

    total_phase = np.zeros(n)
    dynamic_phase = np.zeros(n)

    for i in range(n):
        overlap = psi0.overlap(states[i])
        total_phase[i] = np.angle(overlap)

    # Dynamic phase: integral of <H> dt
    # For beam splitter with |1,0>: <H> = g * Re(<1,0|a†b+ab†|psi(t)>)
    # Since we work in units where hbar=1 effectively (g already in Hz),
    # phi_dyn = -integral_0^t <psi|H|psi> dt'
    # But for this Hamiltonian, we can compute it directly from energy expectation.

    geometric_phase = np.zeros(n)
    # For cyclic evolution under beam-splitter, geometric phase = 0 for 1D paths
    # on CP^1, but nonzero for higher-dimensional state spaces.
    # We compute it numerically via the Pancharatnam connection.

    for i in range(1, n):
        # Pancharatnam relative phase between consecutive states
        phi_pan = np.angle(states[i-1].overlap(states[i]))
        dynamic_phase[i] = dynamic_phase[i-1] + phi_pan

    geometric_phase = total_phase - dynamic_phase

    return total_phase, dynamic_phase, geometric_phase


def run():
    """Execute sim04 calculations."""
    N = config.N_FOCK
    results = {}

    # --- FS trajectory for |1,0> at multiple gap widths ---
    gap_widths = [100e-9, 200e-9, 300e-9, 400e-9]
    psi0 = qt.tensor(qt.fock(N, 1), qt.fock(N, 0))

    for d in gap_widths:
        g = config.G0 * np.exp(-config.KAPPA * d)
        T_rabi = np.pi / (2.0 * g)
        tlist = np.linspace(0, 2 * T_rabi, config.N_TIME)

        res = evolve_pure(g, psi0, tlist, N)
        ent = compute_entanglement_measures(res.states, N, is_dm=False)
        ds_fs, D_fs, speed = compute_fs_trajectory(res.states, tlist)
        total_ph, dyn_ph, geo_ph = compute_geometric_phase(res.states, tlist)

        key = f"d{d*1e9:.0f}nm"
        results[key] = {
            "d": d, "g": g, "tlist": tlist,
            "g_times_t": g * tlist,
            "ds_fs": ds_fs, "D_fs": D_fs, "speed": speed,
            "total_phase": total_ph, "dynamic_phase": dyn_ph,
            "geometric_phase": geo_ph,
            "von_neumann": ent["von_neumann"],
        }

        # Key prediction check: D_FS from |1,0> to max entangled state = pi/2
        idx_max_ent = np.argmax(ent["von_neumann"])
        D_at_max_ent = D_fs[idx_max_ent]
        print(f"d = {d*1e9:.0f} nm: D_FS to max entanglement = {D_at_max_ent:.4f} "
              f"(pi/2 = {np.pi/2:.4f}), ratio = {D_at_max_ent/(np.pi/2):.4f}")

    # --- FS distance to max-entangled state vs gap width (sweep) ---
    d_sweep = config.GAP_RANGE
    D_fs_at_max = np.zeros(len(d_sweep))
    t_to_max = np.zeros(len(d_sweep))
    g_sweep = np.zeros(len(d_sweep))

    for i, d in enumerate(d_sweep):
        g = config.G0 * np.exp(-config.KAPPA * d)
        g_sweep[i] = g
        T_quarter = np.pi / (4.0 * g)  # time to max entanglement
        t_to_max[i] = T_quarter
        tlist = np.linspace(0, T_quarter, 200)

        res = evolve_pure(g, psi0, tlist, N)
        _, D_fs_sweep, _ = compute_fs_trajectory(res.states, tlist)
        D_fs_at_max[i] = D_fs_sweep[-1]

    results["sweep"] = {
        "d": d_sweep, "g": g_sweep,
        "D_fs_at_max": D_fs_at_max,
        "t_to_max": t_to_max,
        "kappa_d": config.KAPPA * d_sweep,
    }

    print(f"\nD_FS to max entanglement: mean = {D_fs_at_max.mean():.4f}, "
          f"std = {D_fs_at_max.std():.6f} (should be ~pi/2 = {np.pi/2:.4f})")
    print(f"t_to_max_ent scales as exp(+kappa*d): "
          f"ratio t(500nm)/t(50nm) = {t_to_max[-1]/t_to_max[0]:.2f}, "
          f"exp(kappa*Dd) = {np.exp(config.KAPPA*(d_sweep[-1]-d_sweep[0])):.2f}")

    return results


if __name__ == "__main__":
    run()
