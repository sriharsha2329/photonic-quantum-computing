"""
Sim05: Tensor network representation of the evanescent-coupled waveguide system.

Builds an MPS (matrix product state) for the two-waveguide system where
the bond dimension chi ~ exp(-kappa*d), then:
  - Computes entanglement entropy via SVD of the bond
  - Implements min-cut algorithm (Ryu-Takayanagi analogue)
  - Compares with QuTiP von Neumann entropy from sim03
"""

import numpy as np
from scipy.linalg import svd
import qutip as qt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim03_entanglement_dynamics"))
from beam_splitter_evolution import build_hamiltonian, evolve_pure, compute_entanglement_measures


def state_to_mps(psi, dims):
    """
    Decompose a bipartite pure state |psi> into MPS form via SVD.

    |psi> = sum_ij c_ij |i>_A |j>_B
    Reshape c into matrix C[i,j], then SVD: C = U S V†
    MPS form: A[i,alpha] * Lambda[alpha] * B[alpha,j]

    Parameters
    ----------
    psi : Qobj, state vector in tensor product space
    dims : list of ints, [dim_A, dim_B]

    Returns
    -------
    U, S, Vh : SVD factors
    chi : effective bond dimension (number of nonzero singular values)
    """
    psi_arr = psi.full().flatten()
    C = psi_arr.reshape(dims[0], dims[1])
    U, S, Vh = svd(C, full_matrices=False)

    # Effective bond dimension: count singular values above threshold
    threshold = 1e-12
    chi = np.sum(S > threshold)

    return U, S, Vh, chi


def entropy_from_svd(S):
    """
    Entanglement entropy from singular values (Schmidt coefficients).
    S_ent = -sum_i lambda_i^2 * ln(lambda_i^2)
    """
    p = S**2
    p = p[p > 1e-30]  # avoid log(0)
    p = p / p.sum()   # normalise
    return -np.sum(p * np.log(p))


def min_cut_entropy(bond_weights):
    """
    Ryu-Takayanagi analogue: minimal cut through the tensor network.

    For our simple 2-site MPS, the minimal cut is just the single bond
    between A and B. The cost of cutting bond alpha is:
        cost = sum over cut bonds of log(chi_bond)

    For a single bond with weight vector (singular values S):
        S_mincut = entropy_from_svd(S)

    This equals the von Neumann entropy by construction — which IS the point:
    the RT formula S = Area/(4G) becomes S = min-cut cost in the tensor network.
    """
    return entropy_from_svd(bond_weights)


def bond_dimension_model(kappa_d, chi_max=None):
    """
    Model bond dimension as function of kappa*d.

    chi_eff(kappa*d) ~ g/g0 = exp(-kappa*d)

    In practice, for a Fock space of dimension N, the max bond dimension
    is min(N_A, N_B). We map:
        chi_continuous = exp(-kappa*d) * chi_max
    """
    if chi_max is None:
        chi_max = config.N_FOCK
    return np.clip(np.exp(-kappa_d) * chi_max, 1, chi_max)


def build_extended_tn(n_sites, kappa_d_values):
    """
    Build an extended 1D tensor network (chain) with n_sites,
    where each bond has dimension proportional to exp(-kappa*d).

    This models a chain of waveguide segments with evanescent coupling.
    Returns bond entropies for each cut position.
    """
    bond_dims = []
    for kd in kappa_d_values:
        chi = max(2, int(np.round(10 * np.exp(-kd))))
        bond_dims.append(chi)

    # Generate random MPS with specified bond dimensions
    # For each bond, create random isometry and compute entropy
    bond_entropies = []
    for chi in bond_dims:
        # Random Schmidt coefficients with the right distribution
        # For thermal-like distribution: lambda_i ~ exp(-i * kd / chi)
        lambdas = np.exp(-np.arange(chi) * 0.5)
        lambdas = lambdas / np.linalg.norm(lambdas)
        bond_entropies.append(entropy_from_svd(lambdas))

    return bond_dims, bond_entropies


def run():
    """Execute all sim05 calculations."""
    N = config.N_FOCK
    results = {}

    # --- 1. MPS decomposition of beam-splitter states at max entanglement ---
    psi0 = qt.tensor(qt.fock(N, 1), qt.fock(N, 0))
    d_sweep = config.GAP_RANGE

    svd_entropy = np.zeros(len(d_sweep))
    qutip_entropy = np.zeros(len(d_sweep))
    bond_dims = np.zeros(len(d_sweep), dtype=int)
    singular_values_list = []

    for i, d in enumerate(d_sweep):
        g = config.G0 * np.exp(-config.KAPPA * d)
        t_max_ent = np.pi / (4.0 * g)
        tlist = np.linspace(0, t_max_ent, 100)

        res = evolve_pure(g, psi0, tlist, N)
        psi_max_ent = res.states[-1]

        # MPS decomposition
        U, S, Vh, chi = state_to_mps(psi_max_ent, [N, N])
        svd_entropy[i] = entropy_from_svd(S)
        bond_dims[i] = chi
        singular_values_list.append(S)

        # QuTiP entropy for comparison
        rho_A = qt.ket2dm(psi_max_ent).ptrace(0)
        qutip_entropy[i] = qt.entropy_vn(rho_A, np.e)

    results["d_sweep"] = d_sweep
    results["kappa_d"] = config.KAPPA * d_sweep
    results["svd_entropy"] = svd_entropy
    results["qutip_entropy"] = qutip_entropy
    results["bond_dims"] = bond_dims
    results["singular_values"] = singular_values_list

    # Min-cut entropy (equals SVD entropy for 2-site MPS)
    mincut_entropy = np.array([min_cut_entropy(sv) for sv in singular_values_list])
    results["mincut_entropy"] = mincut_entropy

    print("--- SVD vs QuTiP entropy comparison ---")
    print(f"Max |S_svd - S_qutip| = {np.max(np.abs(svd_entropy - qutip_entropy)):.2e}")
    print(f"Max |S_mincut - S_qutip| = {np.max(np.abs(mincut_entropy - qutip_entropy)):.2e}")
    print(f"S at max entanglement (|1,0>): {svd_entropy[0]:.6f} (should be ln2 = {np.log(2):.6f})")
    print(f"Bond dimension range: [{bond_dims.min()}, {bond_dims.max()}]")

    # --- 2. Time-resolved MPS entropy at d=200nm ---
    d_fixed = 200e-9
    g_fixed = config.G0 * np.exp(-config.KAPPA * d_fixed)
    T_rabi = np.pi / (2.0 * g_fixed)
    tlist = np.linspace(0, 2 * T_rabi, config.N_TIME)

    res = evolve_pure(g_fixed, psi0, tlist, N)
    svd_ent_t = np.zeros(len(tlist))
    chi_t = np.zeros(len(tlist), dtype=int)

    for j, psi_t in enumerate(res.states):
        U, S, Vh, chi = state_to_mps(psi_t, [N, N])
        svd_ent_t[j] = entropy_from_svd(S)
        chi_t[j] = chi

    ent_qutip = compute_entanglement_measures(res.states, N, is_dm=False)

    results["time_resolved"] = {
        "tlist": tlist,
        "g_times_t": g_fixed * tlist,
        "svd_entropy": svd_ent_t,
        "qutip_entropy": ent_qutip["von_neumann"],
        "bond_dim": chi_t,
    }

    print(f"\nTime-resolved: max |S_svd - S_qutip| = "
          f"{np.max(np.abs(svd_ent_t - ent_qutip['von_neumann'])):.2e}")

    # --- 3. Extended chain: min-cut scaling ---
    n_bonds = 10
    kd_values = np.linspace(0.5, 4.0, n_bonds)
    chain_bond_dims, chain_entropies = build_extended_tn(n_bonds + 1, kd_values)
    results["chain"] = {
        "kd_values": kd_values,
        "bond_dims": chain_bond_dims,
        "bond_entropies": chain_entropies,
    }

    return results


if __name__ == "__main__":
    run()
