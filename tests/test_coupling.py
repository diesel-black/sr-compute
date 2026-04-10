"""Tests for shared.coupling (Gaussian kernel, kappa_n, coupling tensor K)."""

import numpy as np
import pytest

from shared.coupling import (
    coupling_tensor_matrix,
    gaussian_kernel,
    kappa_n,
    self_coupling,
)


@pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0, 2.0])
def test_gaussian_kernel_at_zero_matches_normalization(sigma: float):
    expected = (2.0 * np.pi * sigma**2) ** (-0.5)
    assert float(gaussian_kernel(0.0, sigma)) == pytest.approx(expected)


def test_gaussian_kernel_integrates_to_one():
    sigma = 0.4
    dx = 0.001
    x = np.arange(-30.0, 30.0 + dx, dx)
    g = gaussian_kernel(x, sigma)
    integral = float(np.sum(g) * dx)
    assert integral == pytest.approx(1.0, rel=1e-3, abs=1e-3)


def test_kappa_n_matches_cubic_case():
    n, gamma, sigma = 3, 1.0, 0.5
    expected = 6.0 * gamma / np.sqrt(2.0 * np.pi * sigma**2)
    assert kappa_n(n, gamma, sigma) == pytest.approx(expected)


def test_coupling_tensor_symmetric_uniform_psi_n3():
    n_grid = 32
    dx = 0.1
    psi = np.ones(n_grid)
    K = coupling_tensor_matrix(psi, n=3, gamma=1.0, sigma=0.35, dx=dx)
    assert K.shape == (n_grid, n_grid)
    assert np.allclose(K, K.T)
    assert np.all(K > 0)


def test_coupling_tensor_asymmetric_nonuniform_psi_n3():
    n_grid = 32
    dx = 0.1
    x = np.arange(n_grid, dtype=float) * dx
    psi = 0.5 + 0.3 * np.sin(2.0 * np.pi * x / (n_grid * dx))
    K = coupling_tensor_matrix(psi, n=3, gamma=1.0, sigma=0.35, dx=dx)
    assert not np.allclose(K, K.T)
    assert np.all(K > 0)


def test_self_coupling_matches_diagonal_of_K_n3():
    n_grid = 16
    dx = 0.05
    psi = 0.4 + 0.1 * np.arange(n_grid, dtype=float)
    K = coupling_tensor_matrix(psi, n=3, gamma=0.8, sigma=0.25, dx=dx)
    diag = np.diag(K)
    sc = self_coupling(psi, n=3, gamma=0.8, sigma=0.25)
    np.testing.assert_allclose(diag, sc * dx, rtol=1e-12)
