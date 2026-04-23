"""Public diagnostic functions for the SR disorder-spectrum and clinical work.

Primary surface:

    spectral_concentration_ratio  — r(T) = sigma_1^2 / ||T||_HS^2, measurement of
                                    position in the 12-signature space
    interpretive_condition_number — kappa(Pi) = (max|psi|/min|psi|)^(n-3)
    nonlocal_correction_growth    — eta mismatch across coarsening scales
    count_metastable_states       — landscape maxima count (catastrophe class)

All functions accept NumPy arrays; see shared/metrics.py for full parameter docs.

Example::

    from sr_compute.diagnostics import spectral_concentration_ratio
    result = spectral_concentration_ratio(psi_bar_field, n=4, gamma=1.0, sigma=0.5, dx=dx)
    r_T = result["ratio"]  # sigma_1^2 / ||T||_HS^2
"""

from shared.metrics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)

__all__ = [
    "count_metastable_states",
    "interpretive_condition_number",
    "nonlocal_correction_growth",
    "spectral_concentration_ratio",
]
