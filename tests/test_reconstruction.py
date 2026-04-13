"""Tests for shared.reconstruction (coarse-graining and numerical inverse)."""

import numpy as np
import pytest

from shared.reconstruction import (
    ReconstructionLUT,
    _c_min_even_branch,
    coarse_grain,
    coarse_grain_derivative,
    is_monotonic_region,
    reconstruct,
)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("gamma", [0.3, 1.0, 1.7])
def test_round_trip_reconstruct_coarse_grain(n: int, gamma: float):
    tol = 1e-10
    if n % 2 == 0:
        psi_c = float(-((1.0 / (n * gamma)) ** (1.0 / (n - 1))))
        lo = psi_c + 0.08
        psi_vals = np.linspace(lo, lo + 2.5, 12)
    else:
        psi_vals = np.linspace(-1.2, 1.2, 15)

    for psi in psi_vals:
        C = coarse_grain(psi, n, gamma)
        psi_back = float(reconstruct(C, n, gamma, tol=tol))
        assert psi_back == pytest.approx(float(psi), rel=0.0, abs=500 * tol)


def test_coarse_grain_derivative_positive_odd_n():
    psi = np.linspace(-3.0, 3.0, 301)
    d = coarse_grain_derivative(psi, n=5, gamma=1.0)
    assert np.all(d > 0)


def test_coarse_grain_derivative_even_n_goes_negative():
    gamma = 1.0
    n = 4
    psi_c = float(-((1.0 / (n * gamma)) ** (1.0 / (n - 1))))
    psi = np.linspace(psi_c - 1.5, psi_c - 0.01, 50)
    d = coarse_grain_derivative(psi, n=n, gamma=gamma)
    assert np.any(d < 0)


def test_is_monotonic_region_flags_n4():
    gamma = 1.0
    n = 4
    psi_c = float(-((1.0 / (n * gamma)) ** (1.0 / (n - 1))))
    psi = np.array([psi_c - 0.5, psi_c + 0.5])
    flags = is_monotonic_region(psi, n=n, gamma=gamma)
    assert not bool(flags[0])
    assert bool(flags[1])


def test_lut_matches_brentq_n3():
    rng = np.random.default_rng(0)
    lut = ReconstructionLUT(3, 1.0, -5.0, 5.0, n_samples=12_000)
    c_try = rng.uniform(-2.0, 2.0, size=1000)
    ref = reconstruct(c_try, 3, 1.0)
    got = lut(c_try)
    err = np.max(np.abs(got - ref))
    assert err < 1e-6


def test_lut_matches_brentq_n4():
    gamma = 1.0
    n = 4
    c_floor = _c_min_even_branch(n, gamma)
    rng = np.random.default_rng(1)
    lo, hi = c_floor + 0.02, 2.0
    c_try = rng.uniform(lo, hi, size=1000)
    lut = ReconstructionLUT(n, gamma, -3.0, 3.0, n_samples=12_000)
    ref = reconstruct(c_try, n, gamma)
    got = lut(c_try)
    err = np.max(np.abs(got - ref))
    assert err < 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lut_even_n_domain_nan_below_floor():
    n, gamma = 4, 1.0
    c_floor = _c_min_even_branch(n, gamma)
    lut = ReconstructionLUT(n, gamma, -3.0, 3.0, n_samples=8000)
    c_bad = c_floor - 0.05
    assert np.isnan(float(reconstruct(np.array([c_bad]), n, gamma)[0]))
    assert np.isnan(float(lut(np.array([c_bad]))[0]))
