"""1+1 Metric Field Equation (MFE) and coupled C–g evolution (Appendix A.1.6–A.1.8)."""

from __future__ import annotations

from typing import Any, Mapping, Union

import numpy as np
from scipy.integrate import solve_ivp

from shared.potentials import attractor_stability
from shared.reconstruction import reconstruct

from .cfe import cfe_rhs

Params = Union[Mapping[str, Any], dict[str, Any]]

_LOG_G_FLOOR = np.log(1e-300)
# Keep exp(log_g) finite in float64 (implicit solvers may probe huge intermediate log_g).
_LOG_G_CAP = float(np.nextafter(np.log(np.finfo(np.float64).max), 0.0))


def _centered_gradient(f: np.ndarray, dx: float) -> np.ndarray:
    """Periodic centered first derivative, same stencil as `laplace_beltrami_1d`."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)


def mfe_rhs(C: np.ndarray, g: np.ndarray, params: Params) -> np.ndarray:
    r"""Right-hand side of the 1+1 MFE with \(R_{\mu\nu}\equiv 0\) (no Ricci term):

    \partial_t g = -2\eta_g\, A(C)\, (\partial_x C)^2
        + \frac{6\xi_g\gamma}{\pi\sigma^2}\, h(C)\, g.

    \(A(C)\) is `attractor_stability`; \(h(C)\) is `reconstruct`; \((\partial_x C)^2\) uses the
    periodic centered gradient matching the CFE diffusion stencil.
    """
    C = np.asarray(C, dtype=float)
    g = np.asarray(g, dtype=float)
    eta_g = float(params["eta_g"])
    xi_g = float(params["xi_g"])
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    mu_sq = float(params["mu_sq"])
    alpha_phi = float(params["alpha_phi"])
    n = int(params["n"])
    dx = float(params["dx"])

    if sigma <= 0:
        raise ValueError("sigma must be positive for the MFE reflexive term.")
    if np.any(g <= 0.0):
        raise ValueError("mfe_rhs requires strictly positive metric components g.")

    dC = _centered_gradient(C, dx)
    A_C = attractor_stability(C, mu_sq, alpha_phi)
    psi_bar = reconstruct(C, n, gamma)

    contraction = -2.0 * eta_g * A_C * (dC**2)
    expansion_coef = (6.0 * xi_g * gamma) / (np.pi * sigma**2)
    reflexive = expansion_coef * psi_bar * g

    return contraction + reflexive


def coupled_rhs(t: float, state: np.ndarray, params: Params) -> np.ndarray:
    r"""Combined RHS for the coupled system.

    The IVP state is ``concat(C, \log g)`` (length \(2N\)), not ``(C, g)``. With \(g = \exp(\log g)\)
    strictly positive, \(\Delta_g C\) and the MFE are evaluated at the same metric the solver stores
    up to floating error. The evolution of \(\log g\) is

    \partial_t(\log g) = (\partial_t g) / g,

    with \(\partial_t g\) from `mfe_rhs`.
    """
    state = np.asarray(state, dtype=float).ravel()
    n_grid = state.size // 2
    if 2 * n_grid != state.size:
        raise ValueError("state length must be even (concatenation of C and log g).")

    C = state[:n_grid]
    log_g = np.clip(state[n_grid:], _LOG_G_FLOOR, _LOG_G_CAP)
    g = np.exp(log_g)

    dCdt = cfe_rhs(C, g, params)
    dgdt = mfe_rhs(C, g, params)
    dlogg_dt = dgdt / g
    return np.concatenate([dCdt, dlogg_dt])


def initial_conditions(
    N: int,
    L: float,
    seed: int | None = None,
    C_amplitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Small random \(C\) near zero (SSB seed) and flat metric \(g \equiv 1\) on a periodic grid."""
    rng = np.random.default_rng(seed)
    C0 = rng.uniform(-float(C_amplitude), float(C_amplitude), size=int(N))
    g0 = np.ones(int(N), dtype=float)
    return C0, g0


def _event_metric_floor(t: float, y: np.ndarray, params: Params) -> float:
    n = y.size // 2
    log_g = np.clip(y[n:], _LOG_G_FLOOR, _LOG_G_CAP)
    g_min = float(np.min(np.exp(log_g)))
    return g_min - 1e-6


_event_metric_floor.terminal = True  # type: ignore[attr-defined]
_event_metric_floor.direction = -1.0  # type: ignore[attr-defined]


def _event_metric_ceiling(t: float, y: np.ndarray, params: Params) -> float:
    n = y.size // 2
    log_g = np.clip(y[n:], _LOG_G_FLOOR, _LOG_G_CAP)
    g_max = float(np.max(np.exp(log_g)))
    return g_max - 1e6


_event_metric_ceiling.terminal = True  # type: ignore[attr-defined]
_event_metric_ceiling.direction = 1.0  # type: ignore[attr-defined]


def integrate_coupled(
    C0: np.ndarray,
    g0: np.ndarray,
    params: Params,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    max_step: float | None = None,
    events: Any | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Coupled `solve_ivp` driver; internal state uses `\log g` so `g` stays positive.

    Default terminal events catch metric collapse or blowup (§A.1.8). For stiff diffusion at moderate
    `N`, implicit `Radau` with relaxed tolerances is often far cheaper than `RK45`.
    """
    C0 = np.asarray(C0, dtype=float).ravel()
    g0 = np.asarray(g0, dtype=float).ravel()
    if C0.shape != g0.shape:
        raise ValueError("C0 and g0 must have the same shape.")
    if np.any(g0 <= 0.0):
        raise ValueError("Initial metric g0 must be strictly positive.")

    p: dict[str, Any] = dict(params)

    if events is None:
        events_arg: Any = [_event_metric_floor, _event_metric_ceiling]
    elif events == []:
        events_arg = None
    else:
        events_arg = events

    log_g0 = np.log(g0)
    state0 = np.concatenate([C0, log_g0])
    sol = solve_ivp(
        coupled_rhs,
        t_span,
        state0,
        method=method,
        t_eval=t_eval,
        args=(p,),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events_arg,
    )

    n = C0.size
    if sol.y.size == 0:
        C_hist = np.zeros((0, n))
        g_hist = np.zeros((0, n))
    else:
        C_hist = sol.y[:n, :].T
        log_block = np.clip(sol.y[n:, :], _LOG_G_FLOOR, _LOG_G_CAP)
        g_hist = np.exp(log_block).T

    t_events_out: list[np.ndarray] = []
    if sol.t_events is not None:
        for te in sol.t_events:
            if te is not None and len(te) > 0:
                t_events_out.append(np.asarray(te, dtype=float))

    return {
        "t": sol.t,
        "C_history": C_hist,
        "g_history": g_hist,
        "success": bool(sol.success),
        "message": sol.message,
        "t_events": t_events_out,
    }


def run_simulation(
    params: Params,
    t_span: tuple[float, float] = (0.0, 50.0),
    t_eval: np.ndarray | None = None,
    seed: int = 42,
    method: str = "RK45",
    max_step: float = 0.1,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Initial conditions, coupled integration, spatial grid, and final reconstructed \(\bar\psi = h(C)\)."""
    N = int(params["N"])
    L = float(params["L"])
    dx = float(params["dx"])
    n_poly = int(params["n"])
    gamma = float(params["gamma"])

    x = np.linspace(0.0, L, N, endpoint=False, dtype=float)
    C0, g0 = initial_conditions(N, L, seed=seed)

    out = integrate_coupled(
        C0,
        g0,
        params,
        t_span,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    C_hist = out["C_history"]
    g_hist = out["g_history"]

    if C_hist.shape[0] == 0:
        C_final = C0.copy()
        g_final = g0.copy()
    else:
        C_final = C_hist[-1, :].copy()
        g_final = g_hist[-1, :].copy()

    psi_bar_final = np.asarray(reconstruct(C_final, n_poly, gamma), dtype=float)

    return {
        **out,
        "C_final": C_final,
        "g_final": g_final,
        "psi_bar_final": psi_bar_final,
        "x": x,
    }
