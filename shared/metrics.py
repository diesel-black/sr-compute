"""Thread 7 polynomial-sweep measurements (R25, R26, R27, RG marginality)."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import svdvals
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax, find_peaks

from shared import brake, coupling
from shared.potentials import V_eff, full_effective_potential
from shared.reconstruction import coarse_grain

ArrayLike = Union[np.ndarray, Sequence[float]]

PsiBarRange = Tuple[float, float, int]


def count_metastable_states(
    psi_bar_range: PsiBarRange,
    n: int,
    gamma: float,
    mu_sq: float,
    alpha_phi: float,
    lambda_b: float,
    zeta: Optional[float],
    *,
    sigma: Optional[float] = None,
    n_quad: int = 512,
    peak_prominence: float = 0.04,
    peak_distance: int = 120,
    use_argrelmax: bool = False,
    argrel_order: int = 80,
) -> dict[str, Any]:
    """Measurement 1: count local maxima of the psi-bar effective landscape (tests R26, catastrophe class).

    Stable equilibria correspond to **maxima** of the landscape as defined here because the CFE reaction
    term carries +V_eff'(C) (uphill flow on V_eff in C-space; see Appendix A.1.6).

    - ``psi_bar_range``: (psi_min, psi_max, num_points) sampling grid.
    - ``zeta``: brake kernel prefactor. If ``None`` and ``n == 3``, ``sigma`` must be given and
      ``zeta = brake.zeta_cubic(gamma, sigma)`` is used. If ``None`` and ``n != 3``, the brake
      integral is omitted and the landscape is ``V_eff(C(psi_bar))`` only (``lambda_b`` ignored).
    - ``n == 2``: brake contribution is ill-posed; landscape is ``V_eff(C(psi_bar))`` only and
      ``n2_brake_excluded`` is set in the result.

    Peak detection defaults to ``scipy.signal.find_peaks`` (``peak_prominence``, ``peak_distance``).
    Set ``use_argrelmax=True`` to use ``argrelmax`` with ``argrel_order`` instead.

    Provisional R26 pattern (subject to parameters): n=3 often shows two maxima; higher n may add
    folds when the brake-free ``V_eff(C(psi))`` shape is resolved cleanly.
    """
    psi_min, psi_max, num_points = psi_bar_range
    psi = np.linspace(float(psi_min), float(psi_max), int(num_points), dtype=float)

    flags: dict[str, Any] = {}

    if n == 2:
        warnings.warn(
            "n=2: metastable count uses V_eff(C(psi_bar)) only (singular brake kernel).",
            UserWarning,
            stacklevel=2,
        )
        c = coarse_grain(psi, n, gamma)
        potential = np.asarray(V_eff(c, mu_sq, alpha_phi), dtype=float)
        flags["n2_brake_excluded"] = True
    elif zeta is None:
        if n == 3:
            if sigma is None:
                raise ValueError("sigma is required when n=3 and zeta is None (to build zeta_cubic).")
            zeta_eff = brake.zeta_cubic(gamma, float(sigma))
            potential = np.asarray(
                full_effective_potential(
                    psi, n, gamma, mu_sq, alpha_phi, lambda_b, zeta_eff, n_quad=n_quad
                ),
                dtype=float,
            )
            flags["zeta_from_cubic"] = True
        else:
            c = coarse_grain(psi, n, gamma)
            potential = np.asarray(V_eff(c, mu_sq, alpha_phi), dtype=float)
            flags["brake_integral_omitted"] = True
    else:
        potential = np.asarray(
            full_effective_potential(psi, n, gamma, mu_sq, alpha_phi, lambda_b, float(zeta), n_quad=n_quad),
            dtype=float,
        )

    valid = np.isfinite(potential)
    if not np.any(valid):
        return {
            "count": 0,
            "positions": np.array([], dtype=float),
            "potential": potential,
            "peak_indices": np.array([], dtype=int),
            **flags,
        }

    if use_argrelmax:
        idx = argrelmax(potential, order=int(argrel_order))[0]
    else:
        idx, _props = find_peaks(
            potential,
            prominence=float(peak_prominence),
            distance=int(peak_distance),
        )

    positions = psi[idx]
    return {
        "count": int(positions.size),
        "positions": positions.astype(float),
        "potential": potential,
        "peak_indices": idx.astype(int),
        **flags,
    }


def interpretive_condition_number(
    psi_bar_field: ArrayLike,
    n: int,
    min_amplitude: float = 1e-10,
) -> dict[str, Any]:
    """Measurement 2: interpretive condition kappa(Pi) = (max|psi|/min|psi|)^(n-3) (tests R25).

    Only grid points with |psi_bar| > min_amplitude enter the minimum. At n=3 the exponent vanishes,
    so kappa = 1 for any non-degenerate field.
    """
    p = np.asarray(psi_bar_field, dtype=float).ravel()
    abs_p = np.abs(p)
    mask = abs_p > float(min_amplitude)
    if not np.any(mask):
        nan = float("nan")
        return {
            "kappa": nan,
            "max_amplitude": nan,
            "min_amplitude": nan,
            "dynamic_range": nan,
        }

    max_amp = float(np.max(abs_p))
    min_amp = float(np.min(abs_p[mask]))
    dynamic_range = max_amp / min_amp if min_amp > 0 else float("inf")
    kappa = float(dynamic_range ** (n - 3))

    return {
        "kappa": kappa,
        "max_amplitude": max_amp,
        "min_amplitude": min_amp,
        "dynamic_range": float(dynamic_range),
    }


def spectral_concentration_ratio(
    psi_bar_field: ArrayLike,
    n: int,
    gamma: float,
    sigma: float,
    dx: float,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Measurement 3: sigma_1^2 / sum sigma_i^2 for coupling tensor K (tests R27, Fisher-Rao identification).

    Uses ``coupling.coupling_tensor_matrix`` and ``scipy.linalg.svdvals``. The ratio is the fraction of
    total Frobenius (Hilbert-Schmidt) energy in the leading singular mode.
    """
    psi = np.asarray(psi_bar_field, dtype=float).ravel()
    n_grid = int(psi.size)
    k_mat = coupling.coupling_tensor_matrix(psi, n, gamma, sigma, dx)
    s = svdvals(k_mat)
    hs_sq = float(np.sum(s**2))
    if hs_sq <= 0.0 or s.size == 0:
        ratio = float("nan")
        s1 = float("nan")
    else:
        s1 = float(s[0])
        ratio = float(s1**2 / hs_sq)

    k_show = min(int(top_k), int(s.size))
    singular_top = s[:k_show].astype(float)

    return {
        "ratio": ratio,
        "sigma_1": s1,
        "hs_norm_sq": hs_sq,
        "N": n_grid,
        "singular_values": singular_top,
    }


def nonlocal_correction_growth(
    psi_bar_field: ArrayLike,
    n: int,
    gamma: float,
    sigma: float,
    dx: float,
    coarsening_factors: Optional[Sequence[float]] = None,
    epsilon: float = 1e-7,
) -> dict[str, Any]:
    """Measurement 4: relative L2 mismatch of numerical vs analytical brake variation across scales (RG marginality).

    For each factor ``f``, ``sigma_current = sigma * f``. When ``f > 1``, the field is smoothed by a
    periodic Gaussian with spatial std ``sqrt(sigma_current^2 - sigma^2)`` (variance addition for
    composed Gaussians), implemented via ``gaussian_filter1d`` in units of ``sigma_pixels = sigma_add/dx``.

    eta(f) = ||B_num - B_ana||_2 / ||B_ana||_2 (NaN if ||B_ana|| is tiny).

    At n=2 the brake kernel is singular; returns NaN-filled lists and ``skipped=True``.
    """
    if n == 2:
        factors = list(coarsening_factors) if coarsening_factors is not None else [1, 2, 4, 8]
        return {
            "coarsening_factors": factors,
            "eta_values": [float("nan")] * len(factors),
            "growth_rates": [float("nan")] * max(0, len(factors) - 1),
            "skipped": True,
        }

    psi0 = np.asarray(psi_bar_field, dtype=float).ravel()
    factors = [float(f) for f in (coarsening_factors if coarsening_factors is not None else [1, 2, 4, 8])]

    eta_values: list[float] = []
    for f in factors:
        sigma_current = float(sigma) * f
        if sigma_current <= 0:
            eta_values.append(float("nan"))
            continue

        psi_s = psi0.copy()
        if f > 1.0:
            var_add = sigma_current**2 - float(sigma) ** 2
            if var_add < 0:
                eta_values.append(float("nan"))
                continue
            sigma_add = float(np.sqrt(var_add))
            sigma_px = sigma_add / float(dx)
            psi_s = gaussian_filter1d(psi_s, sigma=sigma_px, mode="wrap")

        b_a = brake.brake_variation_analytical(psi_s, n, gamma, sigma_current)
        b_n = brake.brake_variation_numerical(psi_s, n, gamma, sigma_current, float(dx), epsilon=epsilon)

        na = float(np.linalg.norm(b_a))
        if na < 1e-14:
            eta_values.append(float("nan"))
        else:
            eta_values.append(float(np.linalg.norm(b_n - b_a) / na))

    growth: list[float] = []
    for i in range(len(eta_values) - 1):
        a, b = eta_values[i], eta_values[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-14:
            growth.append(float("nan"))
        else:
            growth.append(float(b / a))

    return {
        "coarsening_factors": factors,
        "eta_values": eta_values,
        "growth_rates": growth,
    }
