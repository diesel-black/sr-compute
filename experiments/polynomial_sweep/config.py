"""
Thread 7 polynomial sweep configuration.

Baseline parameters from Appendix A, Design Decision 2.
The sweep varies polynomial order n while holding all other parameters fixed.
Predictions are qualitative patterns across n, not fits to specific values.
"""

# Sweep range
N_VALUES = [2, 3, 4, 5, 6, 7, 8]

# Baseline SR parameters (Design Decision 2)
BASELINE_PARAMS = {
    "mu_sq": 1.0,  # Attractor binding strength
    "alpha_phi": 1.0,  # Autopoietic potential coupling
    "gamma": 1.0,  # Nonlinearity coefficient in F_n
    "sigma": 0.5,  # Gaussian smearing resolution scale
    "lambda_B": 0.5,  # Brake coupling strength
    "eta_g": 1.0,  # MFE constraint-contraction coupling
    "xi_g": 0.1,  # MFE reflexive-density expansion coupling
}

# Grid
GRID = {
    "N": 256,  # Spatial grid points
    "L": 10.0,  # Domain length
}

# Integration
INTEGRATION = {
    "t_span": (0.0, 30.0),  # Integration window (may need tuning)
    "method": "RK45",  # Explicit for now; switch to Radau if stiff
    "max_step": 0.1,  # Step size cap for coupled system
    "seed": 42,  # Reproducible initial conditions
}

# Higher polynomial orders stiffen the coupled IVP; implicit Radau reduces step-collapse stalls.
# Merged in run_single as: INTEGRATION -> overrides[n] -> caller integration (e.g. QUICK wins last).
INTEGRATION_OVERRIDES_BY_N = {
    # n=2 brake is singular at psi_bar=0; explicit RK45 often stalls; implicit step is more robust here.
    2: {"method": "Radau", "max_step": 1.0, "rtol": 1e-5, "atol": 1e-8},
    4: {"method": "Radau", "max_step": 2.0, "rtol": 1e-6, "atol": 1e-9},
    5: {"method": "Radau", "max_step": 2.0, "rtol": 1e-6, "atol": 1e-9},
    6: {"method": "Radau", "max_step": 2.0, "rtol": 1e-6, "atol": 1e-9},
    7: {"method": "Radau", "max_step": 2.0, "rtol": 1e-6, "atol": 1e-9},
    8: {"method": "Radau", "max_step": 2.0, "rtol": 1e-6, "atol": 1e-9},
}

# ReconstructionLUT for h(C) during coupled IVP (see models.dim_1plus1.run_simulation)
# Wide C span avoids flat extrapolation when |C| grows before metric events; 10k samples is cheap at init.
RECONSTRUCTION_LUT = {
    "C_min": -15.0,
    "C_max": 15.0,
    "n_samples": 10_000,
}

# Optional wall-clock cap per n for ``python -m experiments.polynomial_sweep.run`` (uses spawn subprocess).
# Default None: run in the main process so monitors show one Python and output is not split across PIDs.
# Enable with CLI ``--wallclock 900`` if a single n can hang the whole sweep.
SWEEP_DEFAULT_WALLCLOCK_SEC: float | None = None

# Sampling range for metastable-state count in psi_bar space (matches tests/test_metrics.py)
METASTABLE_PSI_RANGE = (-2.0, 2.0, 4001)

# Measurement 4 (non-local correction growth) settings
NONLOCAL = {
    "coarsening_factors": [1.0, 1.2, 1.5, 2.0],  # Mild; calibrated in Phase 1 tests
}

# Quick-test settings (for development and CI, not for real results)
QUICK = {
    "N": 32,
    "L": 10.0,
    "t_span": (0.0, 5.0),
    "max_step": 0.5,
}

# Output
RESULTS_DIR = "experiments/polynomial_sweep/results"
