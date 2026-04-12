"""1+1 Coherence Field Equation (CFE): Laplace-Beltrami diffusion, reaction, and brake (Appendix A.1.6)."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Union

import numpy as np
from scipy.integrate import solve_ivp

from shared.brake import brake_variation_analytical
from shared.potentials import V_eff_prime
from shared.reconstruction import reconstruct

Params = Union[Mapping[str, Any], MutableMapping[str, Any]]


def laplace_beltrami_1d(C: np.ndarray, g: np.ndarray, dx: float) -> np.ndarray:
    r"""1D Laplace-Beltrami operator on coherence in conformal gauge \(g_{11} = g(x,t)\):

    \Delta_g C = \frac{1}{g}\left[\partial_{xx} C - \frac{(\partial_x g)(\partial_x C)}{2g}\right].

    Periodic centered differences: \(\partial_x f_i = (f_{i+1}-f_{i-1})/(2\,dx)\),
    \(\partial_{xx} f_i = (f_{i+1}-2f_i+f_{i-1})/dx^2\), with index wrap.
    """
    C = np.asarray(C, dtype=float)
    g = np.asarray(g, dtype=float)
    if C.shape != g.shape:
        raise ValueError("C and g must have the same shape.")
    if dx <= 0:
        raise ValueError("dx must be positive.")

    dC = (np.roll(C, -1) - np.roll(C, 1)) / (2.0 * dx)
    d2C = (np.roll(C, -1) - 2.0 * C + np.roll(C, 1)) / (dx**2)
    dg = (np.roll(g, -1) - np.roll(g, 1)) / (2.0 * dx)

    return (1.0 / g) * (d2C - (dg * dC) / (2.0 * g))


def cfe_rhs(C: np.ndarray, g: np.ndarray, params: Params) -> np.ndarray:
    r"""Right-hand side of the 1+1 CFE (Appendix A.1.6):

    \partial_t C = \Delta_g C + V'_{\mathrm{eff}}(C)
        + \lambda_B\, (\delta B/\delta C)_{\mathrm{analytical}}(h(C), n, \gamma, \sigma, g).

    Here \(V'_{\mathrm{eff}}(C) = \mu^2 C - 4\alpha_\Phi C^3\) (uphill flow on \(V_{\mathrm{eff}}\)),
    \(h(C)\) is `reconstruct`, and \(\delta B/\delta C\) is `brake_variation_analytical` with metric factor \(g\).

    When \(\lambda_B = 0\), the brake chain is skipped so `reconstruct` is not called (avoids per-grid-point
    root finding on every RHS evaluation).
    """
    C = np.asarray(C, dtype=float)
    g = np.asarray(g, dtype=float)
    mu_sq = float(params["mu_sq"])
    alpha_phi = float(params["alpha_phi"])
    gamma = float(params["gamma"])
    n = int(params["n"])
    sigma = float(params["sigma"])
    lambda_b = float(params["lambda_B"])
    dx = float(params["dx"])

    diffusion = laplace_beltrami_1d(C, g, dx)
    reaction = V_eff_prime(C, mu_sq, alpha_phi)
    if lambda_b == 0.0:
        return diffusion + reaction
    psi_bar = reconstruct(C, n, gamma)
    brake = brake_variation_analytical(psi_bar, n, gamma, sigma, g_metric=g)
    return diffusion + reaction + lambda_b * brake


def integrate_cfe(
    C0: np.ndarray,
    g: np.ndarray,
    params: Params,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Integrate the CFE with fixed metric g using `scipy.integrate.solve_ivp` (testing harness)."""
    C0 = np.asarray(C0, dtype=float).ravel()
    g = np.asarray(g, dtype=float)
    if C0.shape != g.shape:
        raise ValueError("C0 and g must have the same shape.")

    def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return cfe_rhs(y, g, params)

    sol = solve_ivp(
        _rhs,
        t_span,
        C0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    C_hist = sol.y.T if sol.y.size else np.zeros((0, C0.size))
    return {
        "t": sol.t,
        "C": C_hist,
        "success": bool(sol.success),
        "message": sol.message,
    }
