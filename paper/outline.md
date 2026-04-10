# Photonic ER=EPR via Evanescent Wave Coupling: A Room-Temperature Analogue

## Abstract

We demonstrate a photonic analogue of the ER=EPR correspondence using
evanescent wave coupling between UV photon waveguides. A single dimensionless
parameter kappa*d (evanescent decay constant times gap width) simultaneously
controls the inter-waveguide coupling strength, entanglement generation rate,
Fubini-Study geodesic traversal speed, and tensor network bond dimension.
Through numerical simulations we show exact agreement between von Neumann
entropy, MPS singular value decomposition, and tensor network min-cut —
providing a discrete Ryu-Takayanagi analogue without imposing any bulk metric
by hand. The UV operating regime (lambda = 400 nm) ensures thermal photon
occupation n_bar ~ 10^{-52} at room temperature, making the system effectively
zero-temperature quantum mechanically.

## Key Numerical Results

### Evanescent decay (Sim01)
- kappa = 8.13 x 10^6 m^{-1} at lambda=400nm, theta=50 deg
- Decay length 1/kappa = 123 nm
- Critical angle theta_c = 42.86 deg

### Coupling (Sim02)
- g(d) = g0 * exp(-kappa*d), verified against coupled-mode theory
- CMT gives identical kappa with 0.84x prefactor (overlap integral correction)
- g drops to 1 GHz at d = 849 nm (kappa*d = 6.91)

### Entanglement dynamics (Sim03)
- |1,0> evolves to maximally entangled state at t = pi/(4g)
- S_max = ln(2) exactly (verified to machine precision)
- Entanglement oscillates (Rabi-like) with period pi/g
- Photon number conservation verified: |<N> - 1| < 10^{-15}
- |2,0> shows richer entanglement structure (S_max > ln 2)
- Lindblad decoherence: entropy degrades gracefully with gamma/g up to 0.10

### Fubini-Study geometry (Sim04)
- D_FS from |1,0> to max entangled state = pi/4 (quarter great circle on CP^1)
- D_FS is CONSTANT across all gap widths (std = 0 to machine precision)
- Traversal speed = g = g0*exp(-kappa*d): geometry is universal, speed is not
- KEY INSIGHT: The "bridge length" in state space is fixed; kappa*d controls
  only how fast you traverse it

### Tensor network (Sim05)
- SVD entropy = QuTiP entropy = min-cut entropy to machine precision (~10^{-16})
- Three independent routes to the same entanglement entropy
- This IS the Ryu-Takayanagi analogue: S_ent = min-cut cost in the tensor network

### Room temperature feasibility (Sim06)
- n_bar(400nm, 300K) = 8.3 x 10^{-53} — thermal noise is identically zero
- Total decoherence Gamma = 541 MHz (dominated by surface roughness)
- Operating window: g/Gamma > 1 for d < 500 nm
- Figure of merit g/Gamma = 820 at d = 100 nm

### Unified dictionary (Sim07)
- All four quantities (g, S, D_FS, S_mincut) controlled by kappa*d
- S_max/ln2 = 1.0000 at all gap widths
- D_FS = pi/4 = 0.785398 at all gap widths
- |S_mincut - S_max| < 1.4 x 10^{-5}

## Paper Structure

1. **Introduction**: ER=EPR conjecture, motivation for tabletop analogues
2. **System**: Evanescent coupling geometry, beam-splitter Hamiltonian
3. **Entanglement dynamics**: QuTiP simulations, entropy oscillations
4. **Geometric structure**: Fubini-Study metric, constant geodesic distance
5. **Tensor network description**: MPS, min-cut, RT analogue
6. **Feasibility**: UV advantage, decoherence budget, operating window
7. **ER=EPR dictionary**: Unified correspondence table
8. **Discussion**: What this is and is not (formally analogous, not identical)

## What We Do NOT Claim

- This is NOT a gravitational system — no AdS metric is imposed
- The correspondence is FORMAL ANALOGY, not mathematical identity
- Coherent states remain separable under passive linear optics
- We do not define G_eff or claim "tunneling = wormhole"
- The tensor network geometry is EMERGENT from entanglement, not imposed

## Target Journals

- Physical Review Letters (if concise enough)
- Physical Review A (full version)
- Optica (if emphasising experimental feasibility)
