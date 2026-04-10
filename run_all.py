#!/usr/bin/env python3
"""
Master script: run all simulations end-to-end and generate all figures.

Usage:
    python run_all.py          # run everything
    python run_all.py sim03    # run only sim03
"""

import sys
import os
import time
import importlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SIMULATIONS = [
    ("sim01_evanescent_field",       "plot_field"),
    ("sim02_coupling_vs_gap",        "plot_coupling"),
    ("sim03_entanglement_dynamics",  "plot_dynamics"),
    ("sim04_fubini_study_geometry",  "plot_geometry"),
    ("sim05_tensor_network",         "plot_network"),
    ("sim06_room_temperature",       "plot_feasibility"),
    ("sim07_er_epr_dictionary",      "plot_dictionary"),
]


def run_sim(sim_dir, module_name):
    """Import and run a simulation's main() function."""
    sys.path.insert(0, sim_dir)
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"  WARNING: {module_name} has no main() function")
    finally:
        sys.path.pop(0)
        # Remove cached module so re-imports work
        if module_name in sys.modules:
            del sys.modules[module_name]


def main():
    os.makedirs("figures", exist_ok=True)

    # Filter to specific sim if argument given
    filter_key = sys.argv[1] if len(sys.argv) > 1 else None

    total_start = time.time()

    for sim_dir, module_name in SIMULATIONS:
        if filter_key and filter_key not in sim_dir:
            continue

        print(f"\n{'='*60}")
        print(f"  Running: {sim_dir}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            run_sim(sim_dir, module_name)
            dt = time.time() - t0
            print(f"  Completed in {dt:.1f}s")
        except Exception as e:
            dt = time.time() - t0
            print(f"  FAILED after {dt:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    total_dt = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All simulations complete in {total_dt:.1f}s")
    print(f"  Figures saved to: figures/")
    print(f"{'='*60}")

    # List generated figures
    figs = sorted(f for f in os.listdir("figures") if f.endswith(".png"))
    print(f"\n  Generated {len(figs)} PNG figures:")
    for f in figs:
        print(f"    {f}")


if __name__ == "__main__":
    main()
