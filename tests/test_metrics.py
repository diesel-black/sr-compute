"""Tests for shared.metrics (Thread 7 measurements)."""

import numpy as np
import pytest

from shared import metrics


def test_count_metastable_states_n3_two_maxima_baseline():
    rng = (-2.0, 2.0, 4001)
    out = metrics.count_metastable_states(
        rng,
        n=3,
        gamma=1.0,
        mu_sq=1.0,
        alpha_phi=1.0,
        lambda_b=0.5,
        zeta=None,
        sigma=0.5,
        n_quad=512,
        peak_prominence=0.04,
        peak_distance=120,
    )
    assert out["count"] == 2
    assert out["positions"].size == 2
    assert "zeta_from_cubic" in out
    assert np.max(np.abs(out["positions"])) > 0.3


def test_count_metastable_states_n4_three_maxima_veff_only():
    """Swallowtail-style three-peak fold in V_eff(C(psi)) at n=4 (brake integral omitted)."""
    rng = (-2.0, 2.0, 4001)
    out = metrics.count_metastable_states(
        rng,
        n=4,
        gamma=1.0,
        mu_sq=1.0,
        alpha_phi=1.0,
        lambda_b=0.5,
        zeta=None,
        peak_prominence=0.04,
        peak_distance=120,
    )
    assert out["count"] == 3
    assert "brake_integral_omitted" in out


def test_interpretive_condition_number_n3_is_one():
    field = np.array([0.2, -0.5, 0.9, -0.1])
    out = metrics.interpretive_condition_number(field, n=3, min_amplitude=1e-10)
    assert out["kappa"] == pytest.approx(1.0)


def test_interpretive_condition_number_n5_dynamic_range_ten():
    field = np.array([0.2, 2.0, 0.2])
    out = metrics.interpretive_condition_number(field, n=5, min_amplitude=1e-10)
    assert out["dynamic_range"] == pytest.approx(10.0)
    assert out["kappa"] == pytest.approx(100.0)


def test_spectral_concentration_uniform_near_one_over_N():
    n_grid = 32
    dx = 1.0 / n_grid
    psi = np.ones(n_grid)
    out = metrics.spectral_concentration_ratio(psi, n=3, gamma=1.0, sigma=0.005, dx=dx)
    expected = 1.0 / n_grid
    assert out["ratio"] == pytest.approx(expected, rel=1e-3, abs=1e-5)
    assert out["N"] == n_grid


def test_spectral_concentration_weakly_nonuniform_still_distributed():
    n_grid = 32
    dx = 1.0 / n_grid
    x = np.arange(n_grid, dtype=float) * dx
    psi = 1.0 + 0.01 * np.sin(2.0 * np.pi * x)
    out = metrics.spectral_concentration_ratio(psi, n=3, gamma=1.0, sigma=0.005, dx=dx)
    expected = 1.0 / n_grid
    rel_err = abs(out["ratio"] - expected) / expected
    assert rel_err < 0.08
    assert out["singular_values"].size <= 10


def test_nonlocal_correction_growth_n3_small_eta():
    n_grid = 64
    l_period = 1.0
    dx = l_period / n_grid
    x = np.arange(n_grid, dtype=float) * dx
    psi = 0.2 * np.sin(2.0 * np.pi * x / l_period)
    out = metrics.nonlocal_correction_growth(
        psi,
        n=3,
        gamma=0.7,
        sigma=0.1,
        dx=dx,
        coarsening_factors=[1.0, 1.2, 1.5, 2.0],
        epsilon=2e-8,
    )
    etas = np.asarray(out["eta_values"], dtype=float)
    assert np.all(np.isfinite(etas))
    assert float(np.max(etas)) < 0.01


def test_nonlocal_correction_growth_n4_exceeds_n3():
    n_grid = 64
    l_period = 1.0
    dx = l_period / n_grid
    x = np.arange(n_grid, dtype=float) * dx
    psi = 0.2 * np.sin(2.0 * np.pi * x / l_period)
    factors = [1.0, 1.2, 1.5, 2.0]
    out3 = metrics.nonlocal_correction_growth(
        psi, n=3, gamma=0.7, sigma=0.1, dx=dx, coarsening_factors=factors, epsilon=2e-8
    )
    out4 = metrics.nonlocal_correction_growth(
        psi, n=4, gamma=0.7, sigma=0.1, dx=dx, coarsening_factors=factors, epsilon=2e-8
    )
    m3 = float(np.nanmax(out3["eta_values"]))
    m4 = float(np.nanmax(out4["eta_values"]))
    assert m4 > m3 + 0.1
