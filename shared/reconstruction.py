"""Forward coarse-graining F_n and numerical inverse h(C) for general polynomial order n."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from scipy.optimize import brentq

ArrayLike = Union[float, np.ndarray]


def coarse_grain(psi_bar: ArrayLike, n: int, gamma: float) -> np.ndarray:
    """Forward map C = F_n(psi_bar) = psi_bar + gamma * psi_bar**n (element-wise)."""
    p = np.asarray(psi_bar, dtype=float)
    return p + gamma * p**n


def coarse_grain_derivative(psi_bar: ArrayLike, n: int, gamma: float) -> np.ndarray:
    """Jacobian dC/d(psi_bar) = 1 + n*gamma*psi_bar**(n-1) (element-wise)."""
    p = np.asarray(psi_bar, dtype=float)
    return 1.0 + n * gamma * p ** (n - 1)


def _even_n_critical_point(n: int, gamma: float) -> float:
    """Real critical point of C(psi) for even n: 1 + n*gamma*psi^(n-1) = 0.

    Here n-1 is odd and gamma > 0, so psi^(n-1) = -1/(n*gamma) has the unique real root
    psi = -(1/(n*gamma))**(1/(n-1)) (avoid complex float roots from negative**fractional).
    """
    if gamma <= 0:
        raise ValueError("Even-n critical point requires gamma > 0.")
    return float(-((1.0 / (n * gamma)) ** (1.0 / (n - 1))))


def _reconstruct_scalar(C: float, n: int, gamma: float, tol: float) -> float:
    """Solve psi + gamma*psi**n = C for scalar C."""

    def f(psi: float) -> float:
        return psi + gamma * psi**n - C

    if np.isnan(C) or np.isinf(C):
        return float("nan")

    if np.isclose(C, 0.0, atol=tol, rtol=0.0) and np.isclose(f(0.0), 0.0, atol=tol, rtol=0.0):
        return 0.0

    if n % 2 == 1:
        lo, hi = -abs(C) * 2.0 - 1.0, abs(C) * 2.0 + 1.0
        flo, fhi = f(lo), f(hi)
        expand = 0
        max_expand = 80
        while flo * fhi > 0 and expand < max_expand:
            lo = lo * 2.0 - 1.0
            hi = hi * 2.0 + 1.0
            flo, fhi = f(lo), f(hi)
            expand += 1
        if flo * fhi > 0:
            raise ValueError(f"Could not bracket a root for C={C}, n={n}, gamma={gamma}.")
        return float(brentq(f, lo, hi, xtol=tol, rtol=tol))

    if gamma <= 0:
        raise ValueError("Even-n reconstruction expects gamma > 0 for the standard SR branch structure.")

    psi_c = _even_n_critical_point(n, gamma)
    C_min = float(psi_c + gamma * psi_c**n)

    if C < C_min - 10.0 * tol * max(1.0, abs(C_min)):
        warnings.warn(
            f"C={C} lies below the minimum value C_min={C_min} of the coarse-grain map on R; "
            "no real preimage exists.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")

    if np.isclose(C, C_min, rtol=0.0, atol=10.0 * tol * max(1.0, abs(C_min))):
        return psi_c

    eps = max(1e-10, abs(psi_c) * 1e-8)
    lo = psi_c + eps
    flo = f(lo)
    hi = max(abs(C), 1.0, abs(lo)) * 4.0 + 1.0
    fhi = f(hi)
    grow = 0
    while flo * fhi > 0 and grow < 80:
        hi = hi * 2.0 + 1.0
        fhi = f(hi)
        grow += 1

    if flo * fhi > 0:
        warnings.warn(
            f"Could not bracket the principal-branch preimage for C={C}, n={n}, gamma={gamma}.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")

    return float(brentq(f, lo, hi, xtol=tol, rtol=tol))


def reconstruct(C: ArrayLike, n: int, gamma: float, tol: float = 1e-12) -> np.ndarray:
    """Inverse map psi_bar = h(C) such that coarse_grain(psi_bar, n, gamma) = C.

    Odd n with gamma > 0: globally strictly increasing coarse_grain; wide bracket with expansion.

    Even n: not globally monotone; this returns the preimage on the principal branch
    psi > psi_c where dC/d(psi) > 0 (psi_c is the leftmost critical point of C(psi)).
    If C is below the minimum value of C on that branch (within tolerance), warns and returns NaN.

    Arrays are handled element-wise (scalar brentq per entry).
    """
    carr = np.asarray(C, dtype=float)

    def _one(x: float) -> float:
        return _reconstruct_scalar(float(x), n, gamma, tol)

    vfunc = np.vectorize(_one, otypes=[float])
    return vfunc(carr)


def is_monotonic_region(psi_bar: ArrayLike, n: int, gamma: float) -> np.ndarray:
    """True where dC/d(psi_bar) > 0 for the coarse-grain map at (n, gamma)."""
    return coarse_grain_derivative(psi_bar, n, gamma) > 0.0
