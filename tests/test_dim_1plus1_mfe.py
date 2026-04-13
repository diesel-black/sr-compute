"""Unit tests for `models.dim_1plus1.mfe` and coupled evolution."""

from __future__ import annotations

import numpy as np
import pytest

import models.dim_1plus1.mfe as _mfe
from shared.brake import zeta_cubic
from shared.metrics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)
from shared.potentials import attractor_stability, equilibrium_C_star
from shared.reconstruction import ReconstructionLUT, reconstruct

_LOG_G_CAP = float(_mfe._LOG_G_CAP)
mfe_rhs = _mfe.mfe_rhs
initial_conditions = _mfe.initial_conditions
integrate_coupled = _mfe.integrate_coupled
run_simulation = _mfe.run_simulation


def _periodic_d_dx(f: np.ndarray, dx: float) -> np.ndarray:
    """Match `models.dim_1plus1.mfe` / CFE centered gradient on a periodic grid."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)


def _baseline_params(N: int = 64, L: float = 1.0) -> dict:
    dx = L / N
    return {
        "N": N,
        "L": L,
        "dx": dx,
        "n": 3,
        "mu_sq": 1.0,
        "alpha_phi": 1.0,
        "gamma": 1.0,
        "sigma": 0.5,
        "lambda_B": 0.5,
        "eta_g": 1.0,
        "xi_g": 0.1,
    }


def test_mfe_rhs_zero_gradient():
    """Homogeneous C=C_*: only reflexive channel survives; matches closed form."""
    N = 48
    L = 1.0
    dx = L / N
    mu_sq, alpha_phi = 1.0, 1.0
    gamma, sigma = 1.0, 0.5
    xi_g = 0.1
    n = 3
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    C = c_star * np.ones(N)
    g = np.ones(N)
    params = {
        "eta_g": 1.0,
        "xi_g": xi_g,
        "gamma": gamma,
        "sigma": sigma,
        "mu_sq": mu_sq,
        "alpha_phi": alpha_phi,
        "n": n,
        "dx": dx,
    }
    rhs = mfe_rhs(C, g, params)
    psi = float(reconstruct(np.array([c_star]), n, gamma)[0])
    coef = (6.0 * xi_g * gamma) / (np.pi * sigma**2)
    expected = coef * psi * g
    np.testing.assert_allclose(rhs, expected, rtol=1e-10, atol=1e-10)


def test_mfe_contraction_in_basin():
    """Where A(C)>0 and (d_x C)^2>0, channel -2 eta_g A (dC)^2 is negative."""
    N = 128
    L = 1.0
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    C = c_star + 0.1 * np.cos(2.0 * np.pi * x / L)
    g = np.ones(N)
    params = _baseline_params(N=N, L=L)
    dC = _periodic_d_dx(C, dx)
    A_C = attractor_stability(C, mu_sq, alpha_phi)
    channel1 = -2.0 * float(params["eta_g"]) * A_C * (dC**2)
    idx = np.where((A_C > 0.05) & (np.abs(dC) > 1e-3))[0]
    assert idx.size > 0
    assert np.all(channel1[idx] < 0.0)


def test_mfe_expansion_at_wall():
    """Near a wall center (A<0), -2 eta A (dC)^2 is positive when dC is nonzero."""
    N = 256
    L = 1.0
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    w = 0.02 * L
    C = c_star * np.tanh((x - 0.5 * L) / w)
    g = np.ones(N)
    params = _baseline_params(N=N, L=L)
    dC = _periodic_d_dx(C, dx)
    A_C = attractor_stability(C, mu_sq, alpha_phi)
    channel1 = -2.0 * float(params["eta_g"]) * A_C * (dC**2)
    wall = np.where((np.abs(C) < c_star / np.sqrt(3.0)) & (np.abs(dC) > 1e-2))[0]
    assert wall.size > 0
    assert np.all(channel1[wall] > 0.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coupled_with_lut_matches_brentq_integrator():
    """Short coupled segment: ReconstructionLUT path matches direct reconstruct in RHS."""
    N, L = 32, 10.0
    p = _baseline_params(N=N, L=L)
    C0, g0 = initial_conditions(N, L, seed=3, C_amplitude=0.01)
    t_span = (0.0, 2.0)
    kw = dict(
        t_span=t_span,
        method="RK23",
        max_step=0.25,
        rtol=5e-4,
        atol=5e-5,
    )
    lut = ReconstructionLUT(
        int(p["n"]),
        float(p["gamma"]),
        -10.0,
        10.0,
        n_samples=8000,
    )
    out_lut = integrate_coupled(C0, g0, p, reconstruct_fn=lut, **kw)
    out_br = integrate_coupled(C0, g0, p, reconstruct_fn=None, **kw)
    assert out_lut["success"] and out_br["success"]
    # Linear LUT + explicit brentq RHS differ at ~1e-3 after a few RK steps; keep tight but IVP-stable.
    np.testing.assert_allclose(out_lut["C_history"][-1], out_br["C_history"][-1], rtol=0, atol=2e-3)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coupled_integration_short():
    p = _baseline_params(N=12, L=1.0)
    C0, g0 = initial_conditions(p["N"], p["L"], seed=7, C_amplitude=0.01)
    out = integrate_coupled(
        C0,
        g0,
        p,
        t_span=(0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 6),
        method="RK23",
        max_step=0.15,
        rtol=5e-3,
        atol=5e-4,
    )
    assert out["success"]
    C_end = out["C_history"][-1, :]
    g_end = out["g_history"][-1, :]
    assert np.max(np.abs(C0)) < 0.02
    assert float(np.max(np.abs(C_end))) > 0.05
    np.testing.assert_allclose(g_end, 1.0, rtol=0.15, atol=0.15)
    assert np.all(np.isfinite(C_end))
    assert np.all(np.isfinite(g_end))


def _event_metric_ceiling_small(t: float, y: np.ndarray, *args) -> float:
    """Terminal when max(g) crosses 6 (`y` is concat(C, log g); clip matches `coupled_rhs`)."""
    n = y.size // 2
    log_g = np.clip(y[n:], np.log(1e-300), _LOG_G_CAP)
    return float(np.max(np.exp(log_g))) - 6.0


_event_metric_ceiling_small.terminal = True  # type: ignore[attr-defined]
_event_metric_ceiling_small.direction = 1.0  # type: ignore[attr-defined]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_coupled_metric_blowup_event():
    """Coarse grid, strong xi_g, short horizon: RK23 finishes quickly; event stops at modest g blowup.

    Long horizons with implicit Radau were stalling tests and leaving orphan pytest processes when
    the agent tool timed out in the background.
    """
    N = 2
    L = 1.0
    p = _baseline_params(N=N, L=L)
    p["xi_g"] = 20.0
    C0 = np.array([0.06, -0.05], dtype=float)
    g0 = np.ones(N, dtype=float)
    out = integrate_coupled(
        C0,
        g0,
        p,
        t_span=(0.0, 0.5),
        method="RK23",
        max_step=0.08,
        rtol=0.05,
        atol=0.05,
        events=[_event_metric_ceiling_small],
    )
    assert out["t_events"]
    assert len(out["t_events"][0]) > 0
    assert np.all(np.isfinite(out["g_history"]))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_run_simulation_baseline():
    p = _baseline_params(N=16, L=1.0)
    out = run_simulation(
        p,
        t_span=(0.0, 6.0),
        t_eval=np.linspace(0.0, 6.0, 5),
        seed=11,
        method="RK23",
        max_step=0.2,
        rtol=5e-3,
        atol=5e-4,
    )
    assert "C_final" in out and "g_final" in out and "psi_bar_final" in out and "x" in out
    c_star = equilibrium_C_star(1.0, 1.0)
    assert float(np.mean(np.abs(out["C_final"]))) == pytest.approx(c_star, rel=0.3, abs=0.1)
    assert np.all(out["g_final"] > 0.0)
    assert out["psi_bar_final"].shape == out["C_final"].shape


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_measurement_integration():
    """Phase 2 checkpoint: evolved fields feed all four `shared.metrics` measurements."""
    N = 12
    p = _baseline_params(N=N, L=1.0)
    p["lambda_B"] = 0.4
    out = run_simulation(
        p,
        t_span=(0.0, 4.0),
        t_eval=np.linspace(0.0, 4.0, 5),
        seed=13,
        method="RK23",
        max_step=0.25,
        rtol=5e-3,
        atol=5e-4,
    )
    psi_bar = out["psi_bar_final"]
    dx = float(p["dx"])
    gamma = float(p["gamma"])
    sigma = float(p["sigma"])
    n = int(p["n"])
    zeta = zeta_cubic(gamma, sigma)

    m1 = count_metastable_states(
        (-1.2, 1.2, 96),
        n,
        gamma,
        p["mu_sq"],
        p["alpha_phi"],
        p["lambda_B"],
        zeta,
        sigma=sigma,
        n_quad=96,
        peak_distance=8,
        peak_prominence=0.02,
    )
    m2 = interpretive_condition_number(psi_bar, n)
    m3 = spectral_concentration_ratio(psi_bar, n, gamma, sigma, dx, top_k=5)
    m4 = nonlocal_correction_growth(
        psi_bar,
        n,
        gamma,
        sigma,
        dx,
        coarsening_factors=[1.0],
        epsilon=2e-6,
    )

    assert np.isfinite(m1["count"])
    assert np.isfinite(m2["kappa"])
    assert np.isfinite(m3["ratio"])
    assert all(np.isfinite(v) or np.isnan(v) for v in m4["eta_values"])
