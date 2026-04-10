"""
Sim03: Beam-splitter Hamiltonian evolution and entanglement dynamics.

Core simulation: H = hbar*g*(a†b + ab†) acting on |1,0> and |2,0>
with and without decoherence (Lindblad master equation).
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def build_hamiltonian(g, N=config.N_FOCK):
    """
    Beam-splitter Hamiltonian: H = hbar * g * (a†b + ab†)

    Parameters
    ----------
    g : float, coupling rate (Hz)
    N : int, Fock space truncation per mode

    Returns
    -------
    H : Qobj, Hamiltonian in tensor product space
    a, b : Qobj, annihilation operators for modes A and B
    """
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    H = g * (a.dag() * b + a * b.dag())
    return H, a, b


def evolve_pure(g, psi0, tlist, N=config.N_FOCK):
    """
    Unitary Schrodinger evolution under H_BS.

    Returns
    -------
    result : qutip Result object with states
    """
    H, a, b = build_hamiltonian(g, N)
    result = qt.sesolve(H, psi0, tlist, e_ops=[])
    return result


def evolve_lindblad(g, rho0, tlist, gamma_loss, N=config.N_FOCK):
    """
    Open-system evolution: Lindblad master equation with photon loss.

    Collapse operators: sqrt(gamma) * a  and  sqrt(gamma) * b
    """
    H, a, b = build_hamiltonian(g, N)
    c_ops = [np.sqrt(gamma_loss) * a, np.sqrt(gamma_loss) * b]
    result = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[])
    return result


def compute_entanglement_measures(states_or_rhos, N=config.N_FOCK, is_dm=False):
    """
    Compute entanglement measures at each time step.

    Returns dict with:
      - von_neumann: S(rho_A) = -Tr(rho_A ln rho_A)
      - linear_entropy: S_L = 1 - Tr(rho_A^2)
      - concurrence: Wootters concurrence (for 2-qubit subspace)
      - purity: Tr(rho_A^2)
      - photon_number_A: <a†a>
      - photon_number_B: <b†b>
      - total_photon: <a†a + b†b>  (conservation check)
    """
    n_steps = len(states_or_rhos)
    vn = np.zeros(n_steps)
    lin_ent = np.zeros(n_steps)
    conc = np.zeros(n_steps)
    purity = np.zeros(n_steps)
    nA = np.zeros(n_steps)
    nB = np.zeros(n_steps)

    a_op = qt.tensor(qt.destroy(N), qt.qeye(N))
    b_op = qt.tensor(qt.qeye(N), qt.destroy(N))
    nA_op = a_op.dag() * a_op
    nB_op = b_op.dag() * b_op

    for i, state in enumerate(states_or_rhos):
        if is_dm:
            rho = state
        else:
            rho = qt.ket2dm(state)

        # Partial trace over B to get rho_A
        rho_A = rho.ptrace(0)

        # Von Neumann entropy
        vn[i] = qt.entropy_vn(rho_A, np.e)  # base-e (nats)

        # Linear entropy
        purity[i] = (rho_A * rho_A).tr().real
        lin_ent[i] = 1.0 - purity[i]

        # Concurrence (project into |0>,|1> qubit subspace of each mode)
        # Extract 2x2 blocks from the full density matrix
        rho_qubit = _extract_qubit_subspace(rho, N)
        if rho_qubit is not None:
            conc[i] = qt.concurrence(rho_qubit)
        else:
            conc[i] = 0.0

        # Photon numbers
        nA[i] = qt.expect(nA_op, rho)
        nB[i] = qt.expect(nB_op, rho)

    return {
        "von_neumann": vn,
        "linear_entropy": lin_ent,
        "concurrence": conc,
        "purity": purity,
        "photon_number_A": nA,
        "photon_number_B": nB,
        "total_photon": nA + nB,
    }


def _extract_qubit_subspace(rho, N):
    """
    Extract the 2-qubit (4x4) density matrix from the |00>,|01>,|10>,|11>
    subspace of the full NxN tensor product space.
    """
    rho_full = rho.full()
    # Indices in the tensor product: |n_A, n_B> -> index = n_A * N + n_B
    indices = [0 * N + 0,  # |00>
               0 * N + 1,  # |01>
               1 * N + 0,  # |10>
               1 * N + 1]  # |11>

    rho_sub = np.zeros((4, 4), dtype=complex)
    for ii, idx_i in enumerate(indices):
        for jj, idx_j in enumerate(indices):
            if idx_i < rho_full.shape[0] and idx_j < rho_full.shape[1]:
                rho_sub[ii, jj] = rho_full[idx_i, idx_j]

    # Renormalise
    tr = np.trace(rho_sub).real
    if tr > 1e-10:
        rho_sub /= tr
        return qt.Qobj(rho_sub, dims=[[2, 2], [2, 2]])
    return None


def run():
    """Execute all sim03 calculations."""
    N = config.N_FOCK
    results = {}

    # --- Single photon |1,0> evolution for multiple gap widths ---
    gap_widths = [100e-9, 200e-9, 300e-9, 400e-9]
    psi0_10 = qt.tensor(qt.fock(N, 1), qt.fock(N, 0))

    for d in gap_widths:
        g = config.G0 * np.exp(-config.KAPPA * d)
        T_rabi = np.pi / (2.0 * g)   # half Rabi period
        tlist = np.linspace(0, 4 * T_rabi, config.N_TIME)

        print(f"d = {d*1e9:.0f} nm: g = {g:.3e} Hz, T_Rabi/2 = {T_rabi:.3e} s")

        res = evolve_pure(g, psi0_10, tlist, N)
        ent = compute_entanglement_measures(res.states, N, is_dm=False)

        results[f"d{d*1e9:.0f}nm_1ph"] = {
            "d": d, "g": g, "tlist": tlist,
            "g_times_t": g * tlist,
            **ent,
        }

    # --- Two photon |2,0> evolution at d = 200 nm ---
    d_2ph = 200e-9
    g_2ph = config.G0 * np.exp(-config.KAPPA * d_2ph)
    T_rabi_2ph = np.pi / (2.0 * g_2ph)
    tlist_2ph = np.linspace(0, 4 * T_rabi_2ph, config.N_TIME)
    psi0_20 = qt.tensor(qt.fock(N, 2), qt.fock(N, 0))

    print(f"\n|2,0> at d = {d_2ph*1e9:.0f} nm: g = {g_2ph:.3e} Hz")
    res_2ph = evolve_pure(g_2ph, psi0_20, tlist_2ph, N)
    ent_2ph = compute_entanglement_measures(res_2ph.states, N, is_dm=False)
    results["2ph_200nm"] = {
        "d": d_2ph, "g": g_2ph, "tlist": tlist_2ph,
        "g_times_t": g_2ph * tlist_2ph,
        **ent_2ph,
    }

    # --- Lindblad evolution |1,0> at d = 200 nm with photon loss ---
    d_lind = 200e-9
    g_lind = config.G0 * np.exp(-config.KAPPA * d_lind)
    T_rabi_lind = np.pi / (2.0 * g_lind)
    tlist_lind = np.linspace(0, 4 * T_rabi_lind, config.N_TIME)
    rho0 = qt.ket2dm(psi0_10)

    gamma_values = [0.0, 0.01 * g_lind, 0.05 * g_lind, 0.1 * g_lind]
    for gamma in gamma_values:
        label = f"lindblad_gamma{gamma/g_lind:.2f}"
        print(f"Lindblad: gamma/g = {gamma/g_lind:.2f}")
        res_lind = evolve_lindblad(g_lind, rho0, tlist_lind, gamma, N)
        ent_lind = compute_entanglement_measures(
            res_lind.states, N, is_dm=True)
        results[label] = {
            "d": d_lind, "g": g_lind, "gamma": gamma,
            "tlist": tlist_lind,
            "g_times_t": g_lind * tlist_lind,
            **ent_lind,
        }

    # --- Verification checks ---
    print("\n--- Conservation checks ---")
    for key in results:
        if "total_photon" in results[key]:
            n_tot = results[key]["total_photon"]
            print(f"  {key}: <N_total> range = [{n_tot.min():.6f}, {n_tot.max():.6f}]")

    return results


if __name__ == "__main__":
    run()
