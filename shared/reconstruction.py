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


def _c_min_even_branch(n: int, gamma: float) -> float:
    """Minimum C on the principal-branch domain for even n (same as `reconstruct` / `_reconstruct_scalar`)."""
    psi_c = _even_n_critical_point(n, gamma)
    return float(psi_c + gamma * psi_c**n)


class ReconstructionLUT:
    """Precomputed 1D table for h(C) = reconstruct(C, n, gamma).

    One-time construction uses the existing scalar `brentq` inverse; `__call__` uses `numpy.interp`
    on the full field so coupled RHS evaluation avoids O(N) root solves per map evaluation.

    Odd n: table spans [C_min, C_max] (caller-chosen). Out-of-range C uses flat extrapolation
    (boundary psi) with a one-time warning per bound when any sample lies outside.

    Even n: the principal branch exists only for C >= C_floor where C_floor matches `reconstruct`.
    The table starts at max(C_min, C_floor). C < C_floor yields NaN, matching `reconstruct`.
    """

    def __init__(
        self,
        n: int,
        gamma: float,
        C_min: float,
        C_max: float,
        n_samples: int = 10_000,
        tol: float = 1e-12,
    ) -> None:
        if C_max <= C_min:
            raise ValueError("ReconstructionLUT requires C_max > C_min.")
        if n_samples < 2:
            raise ValueError("ReconstructionLUT requires n_samples >= 2.")
        self.n = int(n)
        self.gamma = float(gamma)
        self._tol = float(tol)
        self._warned_below = False
        self._warned_above = False

        if self.n % 2 == 1:
            self._c_floor = float("-inf")
            c_lo = float(C_min)
        else:
            self._c_floor = _c_min_even_branch(self.n, self.gamma)
            c_lo = max(float(C_min), self._c_floor)

        c_hi = float(C_max)
        if c_hi <= c_lo:
            raise ValueError(
                f"ReconstructionLUT: effective C range empty (c_lo={c_lo}, C_max={c_hi}); "
                "increase C_max or raise C_min above the even-n branch floor."
            )

        self.C_table = np.linspace(c_lo, c_hi, int(n_samples), dtype=float)
        self.psi_table = np.asarray(
            reconstruct(self.C_table, self.n, self.gamma, tol=self._tol),
            dtype=float,
        )

    def __call__(self, C: ArrayLike) -> np.ndarray:
        """Vectorized h(C); preserves array shape (0-d array in, 0-d array out)."""
        C_arr = np.asarray(C, dtype=float)
        scalar_input = C_arr.ndim == 0
        flat = C_arr.ravel()
        out_flat = np.empty_like(flat, dtype=float)

        if self.n % 2 == 0:
            below = flat < self._c_floor
            out_flat[below] = np.nan
            mid = flat[~below]
            if mid.size:
                interp = np.interp(
                    mid,
                    self.C_table,
                    self.psi_table,
                    left=float(self.psi_table[0]),
                    right=float(self.psi_table[-1]),
                )
                out_flat[~below] = interp
            lo_b = float(self.C_table[0])
            hi_b = float(self.C_table[-1])
            valid = ~below
            if np.any(valid & (flat < lo_b)) and not self._warned_below:
                warnings.warn(
                    f"ReconstructionLUT(n={self.n}): C below table minimum {lo_b} (above C_floor); "
                    "using flat extrapolation at left edge.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_below = True
            if np.any(valid & (flat > hi_b)) and not self._warned_above:
                warnings.warn(
                    f"ReconstructionLUT(n={self.n}): C above table maximum {hi_b}; "
                    "using flat extrapolation at right edge.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_above = True
        else:
            out_flat[:] = np.interp(
                flat,
                self.C_table,
                self.psi_table,
                left=float(self.psi_table[0]),
                right=float(self.psi_table[-1]),
            )
            lo_b = float(self.C_table[0])
            hi_b = float(self.C_table[-1])
            if np.any(flat < lo_b) and not self._warned_below:
                warnings.warn(
                    f"ReconstructionLUT(n={self.n}): C below table minimum {lo_b}; "
                    "using flat extrapolation at left edge.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_below = True
            if np.any(flat > hi_b) and not self._warned_above:
                warnings.warn(
                    f"ReconstructionLUT(n={self.n}): C above table maximum {hi_b}; "
                    "using flat extrapolation at right edge.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_above = True

        out = out_flat.reshape(C_arr.shape).astype(float)
        if scalar_input:
            return out.reshape(())
        return out
