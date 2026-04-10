"""Tests for shared.potentials (effective potential, A(C), psi-bar landscape)."""

import numpy as np
import pytest
from scipy.signal import argrelmax, argrelmin

from shared.potentials import (
    V_eff,
    V_eff_prime,
    attractor_stability,
    equilibrium_C_star,
    full_effective_potential,
)


def test_V_eff_single_minimum_and_twin_barriers_at_plus_minus_C_star():
    """V_eff(C) = (mu_sq/2)C^2 - alpha_phi C^4 has a minimum at C=0 and local maxima at ±C_star."""
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    grid = np.linspace(-1.0, 1.0, 20001)
    values = V_eff(grid, mu_sq, alpha_phi)
    idx_min = argrelmin(values, order=50)[0]
    idx_max = argrelmax(values, order=50)[0]
    assert idx_min.size == 1
    assert abs(float(grid[idx_min[0]])) < 1e-6
    assert idx_max.size == 2
    peaks = np.sort(np.abs(grid[idx_max]))
    np.testing.assert_allclose(peaks, [c_star, c_star], rtol=1e-3, atol=1e-3)


def test_V_eff_prime_zeros():
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    zeros = np.array([-c_star, 0.0, c_star])
    vp = V_eff_prime(zeros, mu_sq, alpha_phi)
    np.testing.assert_allclose(vp, 0.0, atol=1e-9)


def test_attractor_stability_signs():
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    assert float(attractor_stability(0.0, mu_sq, alpha_phi)) < 0
    assert float(attractor_stability(c_star, mu_sq, alpha_phi)) > 0
    assert float(attractor_stability(-c_star, mu_sq, alpha_phi)) > 0


def test_equilibrium_C_star_unit_case():
    assert equilibrium_C_star(1.0, 1.0) == pytest.approx(0.5)


def test_full_effective_potential_n3_cusp_fold_signature():
    """n=3 ψ̄-landscape shows a central minimum with two flanking maxima (cusp / fold geometry)."""
    x = np.linspace(-1.35, 1.35, 3001)
    y = full_effective_potential(
        x,
        n=3,
        gamma=0.85,
        mu_sq=1.0,
        alpha_phi=1.0,
        lambda_b=0.45,
        zeta=1.0,
        n_quad=384,
    )
    idx_min = argrelmin(y, order=25)[0]
    idx_max = argrelmax(y, order=25)[0]
    assert idx_min.size == 1
    assert abs(float(x[idx_min[0]])) < 0.05
    assert idx_max.size == 2
    assert float(np.max(x[idx_max])) > 0.2
    assert float(np.min(x[idx_max])) < -0.2
