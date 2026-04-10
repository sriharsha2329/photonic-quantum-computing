"""
Central configuration for Photonic ER=EPR Evanescent Wave Simulation Suite.

All physical constants and simulation parameters in one place.
"""

import numpy as np

# --- Fundamental constants ---
C = 2.998e8              # speed of light (m/s)
HBAR = 1.055e-34         # reduced Planck constant (J·s)
KB = 1.381e-23           # Boltzmann constant (J/K)

# --- Material and geometry ---
LAMBDA_UV = 400e-9       # operating wavelength: 400 nm UV (m)
N1 = 1.47                # refractive index, fused silica at 400 nm
N2 = 1.0                 # refractive index, vacuum gap
THETA_DEG = 50.0         # incidence angle (degrees), above critical angle
THETA = np.radians(THETA_DEG)

# --- Derived optical quantities ---
OMEGA = 2.0 * np.pi * C / LAMBDA_UV          # angular frequency (rad/s)
K0 = 2.0 * np.pi / LAMBDA_UV                 # free-space wavenumber (1/m)
THETA_C_DEG = np.degrees(np.arcsin(N2 / N1)) # critical angle (~42.8 deg)
THETA_C = np.arcsin(N2 / N1)

# Evanescent decay constant (1/m)
KAPPA = K0 * np.sqrt(N1**2 * np.sin(THETA)**2 - N2**2)

# --- Coupling parameters ---
G0 = 1e12               # bare coupling rate (Hz), nanophotonic waveguides
GAP_RANGE = np.linspace(50e-9, 500e-9, 100)  # gap sweep: 50 nm to 500 nm

# --- Temperature ---
T_ROOM = 300.0           # room temperature (K)
T_LN2 = 77.0            # liquid nitrogen (K)
T_LHE = 4.0             # liquid helium (K)

# --- Fock space truncation ---
N_FOCK = 5              # Fock space dimension per mode (sufficient for |2,0> sims)

# --- Simulation time grid ---
N_TIME = 500            # number of time steps

# --- Decoherence ---
GAMMA_LOSS = 1e9        # photon loss rate (Hz), representative for UV in silica

# --- Wavelength sweep ---
LAMBDA_RANGE = np.linspace(300e-9, 800e-9, 200)   # 300 nm – 800 nm

# --- Angle sweep ---
THETA_RANGE_DEG = np.linspace(THETA_C_DEG + 0.5, 89.5, 100)
THETA_RANGE = np.radians(THETA_RANGE_DEG)

# --- Figure style ---
import matplotlib as mpl

FIGURE_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "text.usetex": False,        # set True if LaTeX is available
    "mathtext.fontset": "cm",
    "axes.prop_cycle": mpl.cycler(
        color=["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    ),
}

FIG_SINGLE_COL = (3.375, 2.8)   # single-column figure (inches)
FIG_DOUBLE_COL = (7.0, 4.5)     # double-column figure (inches)
FIGURE_DIR = "figures"
