"""IVP outcome labels for Thread 7 experiment reports.

Maps SciPy ``solve_ivp`` success and stopping time to human-readable outcomes:

- ``completed``: integrator reports success and ``t_final`` reached ``t_span[1]`` within tolerance.
- ``terminal``: integration ended before the horizon with stored final fields (metric events, step
  failure, etc.); measurements are still defined at the last time slice.
- ``timeout``: reserved for wallclock-killed runs (subprocess timeout in ``run_single``); not set
  from ``solve_ivp`` alone.
"""

from __future__ import annotations

import math

T_FINAL_COMPLETION_RTOL = 0.01


def t_final_at_horizon(
    t_final: float,
    t_end: float,
    *,
    rtol: float = T_FINAL_COMPLETION_RTOL,
) -> bool:
    """True if ``t_final`` matches the integration end time within relative ``rtol``."""
    if not (math.isfinite(t_final) and math.isfinite(t_end)):
        return False
    span = abs(float(t_end) - float(t_final))
    if abs(t_end) < 1e-14:
        return span < 1e-14
    return span <= float(rtol) * abs(t_end)


def outcome_from_integrator(integrator_success: bool, t_final: float, t_end: float) -> str:
    """``completed`` if the run finished successfully at the horizon; else ``terminal``."""
    if integrator_success and t_final_at_horizon(t_final, t_end):
        return "completed"
    return "terminal"
