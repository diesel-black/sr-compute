"""Gaussian smearing kernel, coupling prefactor kappa_n, and discretized coupling tensor K (Appendix A)."""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def gaussian_kernel(r: ArrayLike, sigma: float) -> np.ndarray:
    """1D normalized Gaussian kernel G_sigma(r) = (2*pi*sigma^2)^(-1/2) * exp(-r^2/(2*sigma^2)).

    Element-wise in r; sigma > 0.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    r_arr = np.asarray(r, dtype=float)
    norm = (2.0 * np.pi * sigma**2) ** (-0.5)
    return norm * np.exp(-(r_arr**2) / (2.0 * sigma**2))


def kappa_n(n: int, gamma: float, sigma: float) -> float:
    """Prefactor kappa_n = n*(n-1)*gamma / sqrt(2*pi*sigma^2) (coincidence limit of the second variation of F_n).

    At n=3 this is 6*gamma/sqrt(2*pi*sigma^2). At n=2, n*(n-1)=2 gives field-independent coupling strength.
    """
    if n < 2:
        raise ValueError("Polynomial order n must be at least 2 for kappa_n.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return float(n * (n - 1) * gamma / np.sqrt(2.0 * np.pi * sigma**2))


def _periodic_min_image_separation(
    i: np.ndarray,
    j: np.ndarray,
    n: int,
    dx: float,
) -> np.ndarray:
    """Spatial separation |x_i - x_j| with periodic minimum image on indices 0..N-1."""
    diff = np.abs(i - j)
    steps = np.minimum(diff, n - diff)
    return steps.astype(float) * dx


def coupling_tensor_matrix(
    psi_bar: np.ndarray,
    n: int,
    gamma: float,
    sigma: float,
    dx: float,
) -> np.ndarray:
    """Discrete coupling operator K with entries

    K[i,j] = kappa_n * |psi_bar[i]|^(n-2) * G_sigma(r_ij) * dx,

    where r_ij uses periodic minimum image distance on a uniform grid with spacing dx.

    Matrix-vector products approximate the integral operator acting on grid functions (Measurement 3 / SVD).
    """
    psi = np.asarray(psi_bar, dtype=float).ravel()
    n_grid = psi.size
    if n_grid < 1:
        raise ValueError("psi_bar must be non-empty.")
    if dx <= 0:
        raise ValueError("dx must be positive.")

    kap = kappa_n(n, gamma, sigma)
    amp = np.abs(psi) ** (n - 2)

    idx_i = np.arange(n_grid, dtype=int)[:, None]
    idx_j = np.arange(n_grid, dtype=int)[None, :]
    r = _periodic_min_image_separation(idx_i, idx_j, n_grid, dx)
    g = gaussian_kernel(r, sigma)

    return (kap * amp[:, None] * g * dx).astype(float)


def self_coupling(psi_bar: ArrayLike, n: int, gamma: float, sigma: float) -> np.ndarray:
    """Coincidence amplitude kappa_n * |psi_bar|^(n-2) * G_sigma(0) (no dx).

    The matrix diagonal from coupling_tensor_matrix equals self_coupling * dx because each row
    carries the quadrature weight dx in K[i,j].

    G_sigma(0) = (2*pi*sigma^2)^(-1/2). Element-wise on psi_bar.
    """
    p = np.asarray(psi_bar, dtype=float)
    return kappa_n(n, gamma, sigma) * np.abs(p) ** (n - 2) * gaussian_kernel(0.0, sigma)
