"""Scalar and element-wise potential functions for the SR 1+1 toy model (Appendix A notation)."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from scipy.integrate import cumulative_trapezoid

ArrayLike = Union[float, np.ndarray]


def V(C: ArrayLike, mu_sq: float) -> ArrayLike:
    """Attractor potential V(C) = (mu_sq/2) * C**2."""
    return (mu_sq / 2.0) * np.asarray(C, dtype=float) ** 2


def Phi(C: ArrayLike, alpha_phi: float) -> ArrayLike:
    """Autopoietic potential Phi(C) = alpha_phi * C**4."""
    return alpha_phi * np.asarray(C, dtype=float) ** 4


def V_eff(C: ArrayLike, mu_sq: float, alpha_phi: float) -> ArrayLike:
    """Effective potential V_eff(C) = V(C) - Phi(C) = (mu_sq/2)*C**2 - alpha_phi*C**4."""
    return V(C, mu_sq) - Phi(C, alpha_phi)


def V_eff_prime(C: ArrayLike, mu_sq: float, alpha_phi: float) -> ArrayLike:
    """dV_eff/dC = mu_sq*C - 4*alpha_phi*C**3 (Allen-Cahn style reaction term)."""
    C_arr = np.asarray(C, dtype=float)
    return mu_sq * C_arr - 4.0 * alpha_phi * C_arr**3


def attractor_stability(C: ArrayLike, mu_sq: float, alpha_phi: float) -> ArrayLike:
    """A(C) = (12*alpha_phi*C**2 - mu_sq) / (2*mu_sq).

    Basins near ±C_star have A > 0; the inter-basin region near C=0 has A < 0.
    The A=0 locus is |C| = C_star / sqrt(3) with C_star = sqrt(mu_sq) / (2*sqrt(alpha_phi)).
    """
    return (12.0 * alpha_phi * np.asarray(C, dtype=float) ** 2 - mu_sq) / (2.0 * mu_sq)


def equilibrium_C_star(mu_sq: float, alpha_phi: float) -> float:
    """Positive coherence equilibrium C_star = sqrt(mu_sq) / (2*sqrt(alpha_phi)); full set is ±C_star.

    Here mu_sq is mu**2 in Appendix A, so mu = sqrt(mu_sq) and V_eff' vanishes at
    C = ±mu/(2*sqrt(alpha_phi)).
    """
    if mu_sq < 0:
        raise ValueError("mu_sq must be non-negative for a real equilibrium scale.")
    if alpha_phi <= 0:
        raise ValueError("alpha_phi must be positive for finite nonzero equilibria.")
    return float(np.sqrt(mu_sq) / (2.0 * np.sqrt(alpha_phi)))


def _brake_integrand(psi_prime: np.ndarray, n: int, gamma: float, zeta: float) -> np.ndarray:
    """B_n(psi') = zeta * psi'**(2n-5) / (1 + n*gamma*psi'**(n-1)), element-wise."""
    return zeta * psi_prime ** (2 * n - 5) / (1.0 + n * gamma * psi_prime ** (n - 1))


def _brake_integral_scalar(
    psi_bar: float,
    n: int,
    gamma: float,
    zeta: float,
    *,
    n_quad: int = 4096,
) -> float:
    """Integral from 0 to psi_bar of B_n(psi') d(psi') via cumulative trapezoid on a uniform grid."""
    if np.isclose(psi_bar, 0.0):
        return 0.0
    t = np.linspace(0.0, float(psi_bar), max(2, int(n_quad)))
    y = _brake_integrand(t, n, gamma, zeta)
    cum = cumulative_trapezoid(y, t, initial=0.0)
    return float(cum[-1])


def full_effective_potential(
    psi_bar: ArrayLike,
    n: int,
    gamma: float,
    mu_sq: float,
    alpha_phi: float,
    lambda_b: float,
    zeta: float,
    *,
    n_quad: int = 4096,
) -> np.ndarray:
    """Full effective landscape in psi_bar-space at polynomial order n.

    C = psi_bar + gamma*psi_bar**n (forward coarse-graining), base energy V_eff(C), plus
    lambda_b times the integral from 0 to psi_bar of B_n(psi') d(psi') with
    B_n = zeta * psi'**(2n-5) / (1 + n*gamma*psi'**(n-1)).

    For n=2 the brake kernel diverges like 1/psi' at the origin; the brake contribution is
    undefined and this function returns NaNs (after emitting a warning).

    Returns an array with the same shape as psi_bar.
    """
    arr = np.asarray(psi_bar, dtype=float)
    flat = arr.ravel()
    out_flat = np.empty_like(flat)

    if n == 2:
        warnings.warn(
            "n=2: brake integrand B_n diverges as 1/psi' near psi'=0; "
            "returning V_eff(C) only (brake integral omitted).",
            UserWarning,
            stacklevel=2,
        )
        C = flat + gamma * flat**n
        return np.asarray(V_eff(C, mu_sq, alpha_phi), dtype=float).reshape(arr.shape)

    for i, p in enumerate(flat):
        bi = _brake_integral_scalar(float(p), n, gamma, zeta, n_quad=n_quad)
        C_i = float(p + gamma * p**n)
        out_flat[i] = float(V_eff(C_i, mu_sq, alpha_phi)) + float(lambda_b) * bi

    return out_flat.reshape(arr.shape)
