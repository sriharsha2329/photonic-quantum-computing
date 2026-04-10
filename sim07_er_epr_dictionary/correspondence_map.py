"""
Sim07: Unified ER=EPR dictionary — the master correspondence plot.

Collects results from all previous simulations and produces the unified
4-panel figure demonstrating that kappa*d controls everything.
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim03_entanglement_dynamics"))
from beam_splitter_evolution import build_hamiltonian, evolve_pure, compute_entanglement_measures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim04_fubini_study_geometry"))
from state_space_geodesic import compute_fs_trajectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim05_tensor_network"))
from mera_evanescent import state_to_mps, entropy_from_svd


def run():
    """Compute all quantities as function of gap width d (or kappa*d)."""
    N = config.N_FOCK
    d_arr = config.GAP_RANGE
    kd_arr = config.KAPPA * d_arr
    n_d = len(d_arr)

    results = {
        "d": d_arr,
        "kappa_d": kd_arr,
    }

    # Arrays to fill
    g_arr = np.zeros(n_d)
    S_max = np.zeros(n_d)
    t_entangle = np.zeros(n_d)
    D_fs_max = np.zeros(n_d)
    S_mincut = np.zeros(n_d)
    concurrence_max = np.zeros(n_d)

    psi0 = qt.tensor(qt.fock(N, 1), qt.fock(N, 0))

    for i, d in enumerate(d_arr):
        g = config.G0 * np.exp(-config.KAPPA * d)
        g_arr[i] = g

        T_quarter = np.pi / (4.0 * g)
        t_entangle[i] = T_quarter
        T_rabi = np.pi / (2.0 * g)
        tlist = np.linspace(0, T_rabi, 300)

        # Evolve
        res = evolve_pure(g, psi0, tlist, N)

        # Entanglement measures
        ent = compute_entanglement_measures(res.states, N, is_dm=False)
        S_max[i] = np.max(ent["von_neumann"])
        concurrence_max[i] = np.max(ent["concurrence"])

        # Fubini-Study distance to max-entangled state
        tlist_quarter = np.linspace(0, T_quarter, 200)
        res_q = evolve_pure(g, psi0, tlist_quarter, N)
        _, D_fs_q, _ = compute_fs_trajectory(res_q.states, tlist_quarter)
        D_fs_max[i] = D_fs_q[-1]

        # Tensor network min-cut
        psi_max = res_q.states[-1]
        U, S, Vh, chi = state_to_mps(psi_max, [N, N])
        S_mincut[i] = entropy_from_svd(S)

    results["g"] = g_arr
    results["S_max"] = S_max
    results["t_entangle"] = t_entangle
    results["D_fs_max"] = D_fs_max
    results["S_mincut"] = S_mincut
    results["concurrence_max"] = concurrence_max

    # Print the correspondence table
    print("=" * 72)
    print("ER=EPR CORRESPONDENCE DICTIONARY")
    print("=" * 72)
    print(f"{'kd':>6} | {'g (Hz)':>12} | {'S_max/ln2':>9} | "
          f"{'t_ent (s)':>12} | {'D_FS':>8} | {'S_mincut/ln2':>12}")
    print("-" * 72)
    for idx in [0, 25, 50, 75, 99]:
        if idx < n_d:
            print(f"{kd_arr[idx]:6.2f} | {g_arr[idx]:12.3e} | "
                  f"{S_max[idx]/np.log(2):9.4f} | {t_entangle[idx]:12.3e} | "
                  f"{D_fs_max[idx]:8.4f} | {S_mincut[idx]/np.log(2):12.4f}")

    print("-" * 72)
    print(f"D_FS range: [{D_fs_max.min():.6f}, {D_fs_max.max():.6f}] "
          f"(pi/4 = {np.pi/4:.6f})")
    print(f"S_max range: [{S_max.min()/np.log(2):.6f}, {S_max.max()/np.log(2):.6f}] "
          f"(should be 1.0)")
    print(f"|S_mincut - S_max| max = {np.max(np.abs(S_mincut - S_max)):.2e}")

    return results


if __name__ == "__main__":
    run()
