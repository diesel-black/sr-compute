"""Brake variation dB/dC: analytical (n=3 exact) and numerical via Hilbert-Schmidt norm of K."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar

from shared import coupling
from shared.reconstruction import coarse_grain_derivative

ArrayLike = Union[float, np.ndarray]


def zeta_cubic(gamma: float, sigma: float) -> float:
    """Brake prefactor zeta at n=3 (Appendix A.1.3): zeta = 9*gamma^2 / (pi^(3/2) * sigma^3)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return float(9.0 * gamma**2 / (np.pi ** (3.0 / 2.0) * sigma**3))


def brake_variation_analytical(
    psi_bar: ArrayLike,
    n: int,
    gamma: float,
    sigma: float,
    g_metric: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Analytical brake variation using the n=3 zeta as baseline prefactor:

    (dB/dC)_analytical = zeta_cubic(gamma,sigma) * psi_bar^(2n-5) / (1 + n*gamma*psi_bar^(n-1)),

    optionally multiplied by g_metric (same shape as psi_bar). Exact at n=3; local approximation for n>3.

    At n=2 the exponent 2n-5 = -1: values at psi_bar=0 are NaN and a warning is issued once per call.
    """
    p = np.asarray(psi_bar, dtype=float)
    zeta = zeta_cubic(gamma, sigma)

    if n == 2:
        warnings.warn(
            "n=2: brake variation diverges like 1/psi_bar at the origin; non-zero entries may be NaN.",
            UserWarning,
            stacklevel=2,
        )

    denom = 1.0 + n * gamma * p ** (n - 1)
    numer = p ** (2 * n - 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = zeta * numer / denom
    if n == 2:
        out = np.where(np.isfinite(p) & (p != 0.0), out, np.nan)

    if g_metric is not None:
        gm = np.asarray(g_metric, dtype=float)
        if gm.shape != p.shape:
            raise ValueError("g_metric must have the same shape as psi_bar.")
        out = out * gm

    return out.astype(float)


def _hilbert_schmidt_B(K: np.ndarray) -> float:
    """Discrete brake functional B[K] matched to the n=3 local dB/dC (Appendix A.1.3).

    With K[i,j] = (kappa * |psi_i|^(n-2) * G(r_ij)) * dx, a consistent discretization uses
    B = (1/2) * (sum_{ij} K_ij^2) * N where N is the grid size (equivalently L/dx for period L=N*dx).
    This differs from the one-dx factor in the batch note but is what makes finite-difference
    dB/dC agree with brake_variation_analytical at n=3 on periodic domains.
    """
    n_grid = int(K.shape[0])
    return 0.5 * float(np.sum(K**2)) * float(n_grid)


def brake_variation_numerical(
    psi_bar: np.ndarray,
    n: int,
    gamma: float,
    sigma: float,
    dx: float,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """dB/dC by finite differences on the discretized HS norm of K.

    B = (1/2) * sum(K**2) * N (see _hilbert_schmidt_B). At each grid point r, psi_bar[r] is perturbed by epsilon/(dC/dpsi_bar)
    so that C shifts by approximately epsilon; then dB/dC[r] ≈ (B_pert - B) / epsilon.
    """
    psi = np.asarray(psi_bar, dtype=float).ravel()
    n_grid = psi.size
    if n_grid < 1:
        raise ValueError("psi_bar must be non-empty.")

    K0 = coupling.coupling_tensor_matrix(psi, n, gamma, sigma, dx)
    B0 = _hilbert_schmidt_B(K0)
    dcdpsi = coarse_grain_derivative(psi, n, gamma)

    dB_dC = np.empty(n_grid, dtype=float)
    for r in range(n_grid):
        jac = float(dcdpsi[r])
        if not np.isfinite(jac) or jac == 0.0:
            dB_dC[r] = float("nan")
            continue
        psi_p = psi.copy()
        psi_p[r] += epsilon / jac
        Kp = coupling.coupling_tensor_matrix(psi_p, n, gamma, sigma, dx)
        Bp = _hilbert_schmidt_B(Kp)
        dB_dC[r] = (Bp - B0) / epsilon

    return dB_dC.reshape(np.asarray(psi_bar).shape)


def brake_saturation_threshold(n: int, gamma: float) -> float:
    """|psi_bar| at which |brake_variation_analytical| (ignoring g_metric) is maximal on psi_bar > 0.

    Uses scipy.optimize.minimize_scalar on -|B(psi)| with zeta_cubic(gamma, sigma=1) cancelling in the
    maximization of the local shape; the argmax of |B| is independent of sigma and zeta scale.

    At n=3 with gamma>0 the maximum is at |psi| = 1/sqrt(3*gamma).
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive for a finite saturation scale.")

    def neg_abs_brake(psi: float) -> float:
        if psi <= 0:
            return 0.0
        z = zeta_cubic(gamma, 1.0)
        val = z * float(psi ** (2 * n - 5) / (1.0 + n * gamma * psi ** (n - 1)))
        return -abs(val)

    res = minimize_scalar(neg_abs_brake, bounds=(1e-12, 1e6), method="bounded")
    if not res.success:
        warnings.warn(
            f"brake_saturation_threshold: minimizer did not report success (n={n}, gamma={gamma}).",
            UserWarning,
            stacklevel=2,
        )
    return float(res.x)
