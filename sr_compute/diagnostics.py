"""Public diagnostic functions for the SR disorder-spectrum and clinical work.

Primary surface:

    spectral_concentration_ratio  — r(T) = sigma_1^2 / ||T||_HS^2
    interpretive_condition_number — kappa(Pi) = (max|psi|/min|psi|)^(n-3)
    nonlocal_correction_growth    — eta mismatch across coarsening scales
    count_metastable_states       — landscape maxima count (catastrophe class)
    arnold_class                  — Arnold A_k classification of V_n critical points

All functions accept NumPy arrays; see shared/metrics.py for full parameter docs.

Example::

    from sr_compute.diagnostics import spectral_concentration_ratio
    result = spectral_concentration_ratio(psi_bar_field, n=4, gamma=1.0, sigma=0.5, dx=dx)
    r_T = result["ratio"]  # sigma_1^2 / ||T||_HS^2
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import numpy.polynomial.polynomial as npp
from scipy.optimize import brentq

from shared.metrics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)

__all__ = [
    "arnold_class",
    "count_metastable_states",
    "interpretive_condition_number",
    "nonlocal_correction_growth",
    "spectral_concentration_ratio",
]


def arnold_class(
    n: int,
    params: dict[str, Any],
    psi_range: tuple[float, float],
    resolution: int = 2000,
    *,
    n_deriv: int = 8,
    polish_tol: float = 1e-12,
    vanishing_abs_tol: float = 1e-8,
) -> dict[str, Any]:
    """Arnold A_k classification of critical points of V_n(psi_bar) = V_eff(F_n(psi_bar)).

    V_n(psi_bar) = (mu_sq/2)*C^2 - alpha_phi*C^4,  C = F_n(psi_bar) = psi_bar + gamma*psi_bar**n.

    V_n is a polynomial in psi_bar; Taylor derivatives at each critical point are computed by
    iterated symbolic differentiation (npp.polyder + npp.polyval) — exact, no finite-difference
    noise or factorial-amplification artifacts from polynomial fitting.

    Algorithm:
      1. Build V_n and its derivative polynomial V_n' symbolically.
      2. Locate critical points as sign-changes in V_n' on a coarse grid.
      3. Polish each to tolerance polish_tol using brentq on V_n'.
      4. At each polished psi*, evaluate V_n^(k)(psi*) for k = 0 .. n_deriv exactly.
      5. Find the lowest order m >= 2 where |V_n^(m)(psi*)| > vanishing_abs_tol.
         Report A_{m-1}: A_1 = Morse (non-degenerate); A_2 = cusp; A_3 = swallowtail.

    Parameters
    ----------
    n : polynomial order of the coarse-graining map F_n.
    params : dict with keys mu_sq, alpha_phi, gamma (and optionally others; extras are ignored).
    psi_range : (psi_min, psi_max) for the coarse landscape grid.
    resolution : number of grid points for coarse V_n.
    n_deriv : highest derivative order computed (default 8; V_eff is degree-4 so contributions
              beyond order 4 from V_eff itself vanish, but the polynomial V_n extends to
              degree 4n, so higher derivatives can be non-zero for large n).
    polish_tol : brentq absolute and relative tolerance for critical point polishing.
    vanishing_abs_tol : absolute threshold below which a derivative is declared vanishing.
                        A_k is detected when |V_n^(k+1)| > vanishing_abs_tol is the first
                        such order. Default 1e-8 is well above floating-point noise (~1e-14)
                        and well below any genuine non-zero derivative for the SR potential.

    Returns
    -------
    dict with:
      critical_points : list of per-critical-point dicts, each containing:
          psi (float), V (float), critical_point_type (str: "maximum"/"minimum"/"inflection"),
          arnold_class (str: "A_1", "A_2", ..., or "undetermined"),
          first_nonvanishing_order (int or None),
          leading_derivative_magnitudes (list[float], orders 2..min(6, n_deriv)).
      landscape_label : summary string counting each (class, type) combination.
      n_maxima : number of local maxima found.
      n_minima : number of local minima found.
      psi_grid : array of shape (resolution,).
      V_grid : array of shape (resolution,).
    """
    psi_min, psi_max = float(psi_range[0]), float(psi_range[1])
    gamma = float(params["gamma"])
    mu_sq = float(params["mu_sq"])
    alpha_phi = float(params["alpha_phi"])

    # Build F_n(psi) = psi + gamma*psi^n as polynomial coefficients.
    # Coefficient array: F[k] = coefficient of psi^k.
    # For n == 1 the two terms share index 1; for n >= 2 they are distinct.
    F = np.zeros(max(2, n + 1), dtype=float)
    F[1] += 1.0       # psi^1 term
    F[n] += gamma     # gamma*psi^n term (adds to F[1] iff n==1)

    # V_n = (mu_sq/2)*F^2 - alpha_phi*F^4
    F2 = npp.polymul(F, F)
    F4 = npp.polymul(F2, F2)
    len_V = max(len(F2), len(F4))
    V = np.zeros(len_V, dtype=float)
    V[: len(F2)] += (mu_sq / 2.0) * F2
    V[: len(F4)] -= alpha_phi * F4

    # Precompute derivative polynomials V^(0), V^(1), ..., V^(n_deriv).
    V_derivs: list[np.ndarray] = [V.copy()]
    for _ in range(n_deriv):
        V_derivs.append(npp.polyder(V_derivs[-1]))

    # Coarse grid and sign-change detection.
    psi_grid = np.linspace(psi_min, psi_max, int(resolution))
    V_grid = npp.polyval(psi_grid, V)
    Vprime_grid = npp.polyval(psi_grid, V_derivs[1])

    sign_arr = np.sign(Vprime_grid)
    change_idx = np.where(np.diff(sign_arr) != 0)[0]

    if change_idx.size == 0:
        return {
            "critical_points": [],
            "landscape_label": "no critical points found",
            "n_maxima": 0,
            "n_minima": 0,
            "psi_grid": psi_grid,
            "V_grid": V_grid,
        }

    # Polish with brentq on the exact derivative polynomial.
    V1 = V_derivs[1]

    def _vprime(p: float) -> float:
        return float(npp.polyval(p, V1))

    polished: list[float] = []
    for idx in change_idx:
        a, b = float(psi_grid[idx]), float(psi_grid[idx + 1])
        if _vprime(a) * _vprime(b) > 0:
            continue
        try:
            root = brentq(_vprime, a, b, xtol=float(polish_tol), rtol=float(polish_tol))
            polished.append(float(root))
        except ValueError:
            pass

    if not polished:
        return {
            "critical_points": [],
            "landscape_label": "polish failed",
            "n_maxima": 0,
            "n_minima": 0,
            "psi_grid": psi_grid,
            "V_grid": V_grid,
        }

    results: list[dict[str, Any]] = []
    n_maxima = 0
    n_minima = 0

    for psi_star in polished:
        # Exact Taylor derivatives at psi_star via symbolic polynomials.
        derivs_at = [
            float(npp.polyval(psi_star, V_derivs[k]))
            for k in range(min(n_deriv + 1, len(V_derivs)))
        ]
        abs_d = [abs(x) for x in derivs_at]

        # Find lowest order m >= 2 with |d_m| > vanishing_abs_tol.
        first_order: Optional[int] = None
        for k in range(2, len(derivs_at)):
            if abs_d[k] > float(vanishing_abs_tol):
                first_order = k
                break

        ak: Optional[int] = (first_order - 1) if first_order is not None else None

        crit_type = "unknown"
        if first_order is not None:
            if first_order % 2 == 0:
                crit_type = "maximum" if derivs_at[first_order] < 0.0 else "minimum"
            else:
                crit_type = "inflection"

        if crit_type == "maximum":
            n_maxima += 1
        elif crit_type == "minimum":
            n_minima += 1

        lead = [abs_d[k] for k in range(2, min(7, len(abs_d)))]

        results.append({
            "psi": float(psi_star),
            "V": float(derivs_at[0]),
            "critical_point_type": crit_type,
            "arnold_class": f"A_{ak}" if ak is not None else "undetermined",
            "first_nonvanishing_order": first_order,
            "leading_derivative_magnitudes": lead,
        })

    class_type_counts: dict[str, int] = {}
    for r in results:
        key = f"{r['arnold_class']} {r['critical_point_type']}"
        class_type_counts[key] = class_type_counts.get(key, 0) + 1
    landscape_label = ", ".join(
        f"{v}x {k}" for k, v in sorted(class_type_counts.items())
    )

    return {
        "critical_points": results,
        "landscape_label": landscape_label,
        "n_maxima": n_maxima,
        "n_minima": n_minima,
        "psi_grid": psi_grid,
        "V_grid": V_grid,
    }
