"""Tests for shared.brake (analytical vs numerical brake variation)."""

import numpy as np
import pytest

from shared.brake import (
    brake_saturation_threshold,
    brake_variation_analytical,
    brake_variation_numerical,
    zeta_cubic,
)


def test_zeta_cubic_numeric_value():
    gamma, sigma = 1.0, 0.5
    expected = 9.0 / (np.pi ** 1.5 * sigma**3)
    assert zeta_cubic(gamma, sigma) == pytest.approx(expected)


def test_brake_variation_analytical_n3_matches_closed_form():
    gamma, sigma = 0.9, 0.35
    zeta = zeta_cubic(gamma, sigma)
    psi = np.linspace(-0.9, 0.9, 11)
    direct = zeta * psi / (1.0 + 3.0 * gamma * psi**2)
    out = brake_variation_analytical(psi, n=3, gamma=gamma, sigma=sigma)
    np.testing.assert_allclose(out, direct, rtol=0.0, atol=1e-12)


def test_brake_saturation_threshold_n3():
    gamma = 1.0
    expected = 1.0 / np.sqrt(3.0 * gamma)
    got = brake_saturation_threshold(3, gamma)
    assert got == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_brake_numerical_matches_analytical_n3_small_grid():
    n_grid = 32
    l_period = 1.0
    dx = l_period / n_grid
    x = np.arange(n_grid, dtype=float) * dx
    psi = 0.3 * np.sin(2.0 * np.pi * x / l_period)
    n_poly = 3
    gamma = 0.85
    sigma = 0.12

    ana = brake_variation_analytical(psi, n_poly, gamma, sigma)
    num = brake_variation_numerical(psi, n_poly, gamma, sigma, dx, epsilon=5e-8)

    mask = np.isfinite(ana) & np.isfinite(num) & (np.abs(ana) > 1e-9)
    rel = np.abs((num[mask] - ana[mask]) / ana[mask])
    assert rel.size > 0
    assert float(np.max(rel)) < 1e-3


def test_brake_numerical_differs_from_analytical_n4():
    n_grid = 48
    l_period = 1.0
    dx = l_period / n_grid
    x = np.arange(n_grid, dtype=float) * dx
    psi = 0.25 * np.sin(2.0 * np.pi * x / l_period) + 0.05
    n_poly = 4
    gamma = 0.9
    sigma = 0.11

    ana = brake_variation_analytical(psi, n_poly, gamma, sigma)
    num = brake_variation_numerical(psi, n_poly, gamma, sigma, dx, epsilon=5e-8)

    mask = np.isfinite(ana) & np.isfinite(num) & (np.abs(ana) > 1e-6)
    rel = np.abs((num[mask] - ana[mask]) / np.maximum(np.abs(ana[mask]), 1e-12))
    assert rel.size > 0
    assert float(np.mean(rel)) > 0.05
