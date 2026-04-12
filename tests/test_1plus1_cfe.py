"""Unit tests for `models.1plus1.cfe` (Laplace-Beltrami and CFE RHS / integration)."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from shared.potentials import equilibrium_C_star

_cfe = import_module("models.1plus1.cfe")
laplace_beltrami_1d = _cfe.laplace_beltrami_1d
cfe_rhs = _cfe.cfe_rhs
integrate_cfe = _cfe.integrate_cfe


def test_laplace_beltrami_flat_metric():
    """On g=1, Delta_g C equals the periodic centered stencil on cos(k x).

    The discrete second derivative has eigenvalue 2(cos(k dx)-1)/dx^2 on this mode, not -k^2
    (they agree at O(dx^2) with the continuum -k^2).
    """
    L = 1.0
    N = 256
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    k = 2.0 * np.pi / L
    C = np.cos(k * x)
    g_flat = np.ones(N)
    lap = laplace_beltrami_1d(C, g_flat, dx)
    lam_disc = 2.0 * (np.cos(k * dx) - 1.0) / (dx**2)
    exact = lam_disc * C
    np.testing.assert_allclose(lap, exact, rtol=1e-9, atol=1e-9)


def test_laplace_beltrami_nonflat_metric():
    """Smooth g(x) activates the -(d g)(d C)/(2 g^2) correction vs flat Laplacian."""
    L = 1.0
    N = 256
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    k = 2.0 * np.pi / L
    C = np.cos(k * x)
    g_flat = np.ones(N)
    g_var = 1.0 + 0.3 * np.sin(k * x)
    lb_flat = laplace_beltrami_1d(C, g_flat, dx)
    lb_var = laplace_beltrami_1d(C, g_var, dx)
    assert not np.allclose(lb_flat, lb_var, rtol=1e-6, atol=1e-6)
    assert lb_var.shape == C.shape


def test_cfe_rhs_equilibrium():
    """Homogeneous C_* with flat g: diffusion and V_eff' vanish; lambda_B=0 removes brake forcing."""
    N = 64
    L = 1.0
    dx = L / N
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    C = c_star * np.ones(N)
    g = np.ones(N)
    params = {
        "mu_sq": mu_sq,
        "alpha_phi": alpha_phi,
        "gamma": 1.0,
        "n": 3,
        "sigma": 0.5,
        "lambda_B": 0.0,
        "dx": dx,
    }
    rhs = cfe_rhs(C, g, params)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-9)


def test_cfe_rhs_unstable_origin():
    """Slight positive bulk C: V_eff'(C) ~ mu_sq C > 0 drives growth away from C=0."""
    N = 32
    L = 1.0
    dx = L / N
    eps = 0.02
    C = eps * np.ones(N)
    g = np.ones(N)
    params = {
        "mu_sq": 1.0,
        "alpha_phi": 1.0,
        "gamma": 1.0,
        "n": 3,
        "sigma": 0.5,
        "lambda_B": 0.0,
        "dx": dx,
    }
    rhs = cfe_rhs(C, g, params)
    assert np.all(rhs > 0.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_integrate_cfe_ssb():
    """Fixed flat g: small random IC near zero flows toward |C| ~ C_* (Allen-Cahn-type SSB).

    Uses lambda_B=0 so the CFE omits the brake chain (no per-point `reconstruct` in `cfe_rhs`), which
    keeps this integration test fast while still exercising diffusion plus uphill V_eff' flow.
    """
    N = 32
    L = 1.0
    dx = L / N
    rng = np.random.default_rng(0)
    C0 = rng.uniform(-0.01, 0.01, size=N)
    g = np.ones(N)
    mu_sq, alpha_phi = 1.0, 1.0
    c_star = equilibrium_C_star(mu_sq, alpha_phi)
    params = {
        "mu_sq": mu_sq,
        "alpha_phi": alpha_phi,
        "gamma": 1.0,
        "n": 3,
        "sigma": 0.5,
        "lambda_B": 0.0,
        "dx": dx,
    }
    out = integrate_cfe(
        C0,
        g,
        params,
        t_span=(0.0, 25.0),
        t_eval=np.array([0.0, 25.0]),
        rtol=1e-5,
        atol=1e-7,
    )
    assert out["success"]
    C_final = out["C"][-1, :]
    assert np.all(np.isfinite(C_final))
    np.testing.assert_allclose(np.mean(np.abs(C_final)), c_star, rtol=0.25, atol=0.08)
