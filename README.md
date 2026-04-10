# Photonic ER=EPR Evanescent Wave Simulation Suite

Numerical simulations demonstrating a photonic analogue of ER=EPR using evanescent
wave coupling between UV photon waveguides at room temperature.

## Core Claim

The evanescent decay parameter kappa controls both inter-waveguide coupling and
entanglement generation rate. This coupling structure admits a tensor-network
description whose geometry maps onto holographic entanglement (ER=EPR).

## Structure

| Directory | Description |
|-----------|-------------|
| `sim01_evanescent_field/` | Evanescent field profile and decay constant |
| `sim02_coupling_vs_gap/` | Coupling strength g(d) = g0 exp(-kappa d) |
| `sim03_entanglement_dynamics/` | QuTiP beam-splitter entanglement dynamics |
| `sim04_fubini_study_geometry/` | Fubini-Study metric and geometric phase |
| `sim05_tensor_network/` | Tensor network, min-cut, Ryu-Takayanagi analogue |
| `sim06_room_temperature/` | Thermal occupation and decoherence budget |
| `sim07_er_epr_dictionary/` | Unified ER=EPR correspondence plots |

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

All figures are saved to `figures/`.
