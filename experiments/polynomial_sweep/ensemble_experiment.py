"""Multi-seed ensemble at n=4 (plus n=3 control) for Thread 7 terminal-state variability.

``initial_conditions`` in ``models.dim_1plus1.mfe`` uses ``np.random.default_rng(seed)`` (not the
global ``np.random``). Pass each ensemble index as ``run_simulation(..., seed=s)`` so ICs differ
per seed; no changes to ``mfe.py`` are required.

Run from repository root::

    python -m experiments.polynomial_sweep.ensemble_experiment
    python -m experiments.polynomial_sweep.ensemble_experiment --quick

Default report path: ``experiments/polynomial_sweep/results/ensemble_report.txt``.
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np

from experiments.polynomial_sweep.config import (
    BASELINE_PARAMS,
    GRID,
    INTEGRATION,
    INTEGRATION_OVERRIDES_BY_N,
    NONLOCAL,
    QUICK,
    RECONSTRUCTION_LUT,
    RESULTS_DIR,
)
from experiments.polynomial_sweep.outcome_utils import outcome_from_integrator
from models.dim_1plus1.mfe import run_simulation
from shared.metrics import (
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)

# Primary polynomial order; n=3 is algebraic control for kappa.
ENSEMBLE_N_PRIMARY = 4
ENSEMBLE_N_CONTROL = 3

FULL_SEEDS: tuple[int, ...] = tuple(range(20))
QUICK_SEEDS: tuple[int, ...] = tuple(range(5))

KAPPA_N3_TOL = 1e-9
GROWTH_VERDICT_EPS = 1e-6

# Per-integration wallclock (SIGALRM / setitimer). Quick mode uses a shorter cap.
WALLCLOCK_TIMEOUT_S = 90.0
QUICK_WALLCLOCK_TIMEOUT_S = 45.0

_T = TypeVar("_T")

# Unix: real-time timer interrupts the main thread; Windows has no SIGALRM.
_WALLCLOCK_AVAILABLE = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class EnsembleSimulationTimeout(TimeoutError):
    """Raised in the SIGALRM handler when a single ensemble run exceeds its wallclock."""


def _sigalrm_handler(_signum: int, _frame: Any) -> None:
    raise EnsembleSimulationTimeout()


def _run_with_wallclock(seconds: float, fn: Callable[[], _T]) -> _T:
    """Run ``fn`` on the main thread with a ``signal.ITIMER_REAL`` deadline."""
    if seconds <= 0.0 or not _WALLCLOCK_AVAILABLE:
        return fn()
    old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(seconds), 0.0)
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_params_ensemble(n: int, *, quick: bool) -> dict[str, Any]:
    """Baseline SR parameters, grid, and integration for ensemble member ``n`` (3 or 4).

    Merge order matches the main sweep: ``INTEGRATION`` then ``INTEGRATION_OVERRIDES_BY_N[n]``,
    then ``QUICK`` overrides when ``quick`` is True.
    """
    grid = {"N": QUICK["N"], "L": QUICK["L"]} if quick else dict(GRID)
    integ = {**INTEGRATION, **INTEGRATION_OVERRIDES_BY_N.get(int(n), {})}
    if quick:
        integ = {
            **integ,
            "t_span": (float(QUICK["t_span"][0]), float(QUICK["t_span"][1])),
            "max_step": float(QUICK["max_step"]),
        }
    params: dict[str, Any] = {**BASELINE_PARAMS, "n": int(n)}
    params["N"] = int(grid["N"])
    params["L"] = float(grid["L"])
    params["dx"] = params["L"] / params["N"]
    params["t_span"] = (float(integ["t_span"][0]), float(integ["t_span"][1]))
    params["method"] = str(integ["method"])
    params["max_step"] = float(integ["max_step"])
    params["seed"] = int(integ.get("seed", INTEGRATION["seed"]))
    if "rtol" in integ:
        params["rtol"] = float(integ["rtol"])
    if "atol" in integ:
        params["atol"] = float(integ["atol"])
    return params


def _measure_terminal(
    psi_bar: np.ndarray,
    params: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Returns kappa, spectral_ratio, eta_f1, first nonlocal growth rate."""
    n = int(params["n"])
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    dx = float(params["dx"])

    try:
        ic = interpretive_condition_number(psi_bar, n)
        kappa = float(ic["kappa"])
        if not math.isfinite(kappa):
            kappa = float("nan")
    except Exception:
        kappa = float("nan")

    try:
        sc = spectral_concentration_ratio(psi_bar, n, gamma, sigma, dx)
        spectral_ratio = float(sc["ratio"])
        if not math.isfinite(spectral_ratio):
            spectral_ratio = float("nan")
    except Exception:
        spectral_ratio = float("nan")

    try:
        nl = nonlocal_correction_growth(
            psi_bar,
            n,
            gamma,
            sigma,
            dx,
            coarsening_factors=NONLOCAL["coarsening_factors"],
        )
        etas = nl.get("eta_values") or []
        growths = nl.get("growth_rates") or []
        eta1 = float(etas[0]) if etas else float("nan")
        gr0 = float(growths[0]) if growths else float("nan")
        if not math.isfinite(eta1):
            eta1 = float("nan")
        if not math.isfinite(gr0):
            gr0 = float("nan")
    except Exception:
        eta1 = float("nan")
        gr0 = float("nan")

    return kappa, spectral_ratio, eta1, gr0


def _timeout_row(n: int, seed: int) -> dict[str, Any]:
    """Placeholder row when the integrator hits the wallclock (no terminal state)."""
    nan = float("nan")
    return {
        "n": int(n),
        "seed": int(seed),
        "t_final": nan,
        "outcome": "timeout",
        "timeout": True,
        "C_range": nan,
        "kappa": nan,
        "spectral_ratio": nan,
        "eta_f1": nan,
        "growth": nan,
    }


def run_one_seed(
    n: int,
    seed: int,
    *,
    quick: bool,
    wallclock_s: float,
    lut_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single (n, seed): coupled IVP with LUT, then four terminal measurements.

    Honors ``wallclock_s`` via ``signal.ITIMER_REAL`` when available; on timeout returns
    ``_timeout_row`` with NaN measurements and ``timeout=True``.
    """
    params = build_params_ensemble(n, quick=quick)
    params["seed"] = int(seed)
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))
    lut_opts = {**RECONSTRUCTION_LUT, **(lut_cfg or {})}

    sim_kwargs: dict[str, Any] = {
        "t_span": t_span,
        "seed": int(seed),
        "method": str(params["method"]),
        "max_step": float(params["max_step"]),
        "use_reconstruction_lut": True,
        "lut_C_min": float(lut_opts["C_min"]),
        "lut_C_max": float(lut_opts["C_max"]),
        "lut_n_samples": int(lut_opts["n_samples"]),
    }
    if "rtol" in params:
        sim_kwargs["rtol"] = float(params["rtol"])
    if "atol" in params:
        sim_kwargs["atol"] = float(params["atol"])

    def _integrate_and_measure() -> dict[str, Any]:
        sim = run_simulation(params, **sim_kwargs)
        t_arr = np.asarray(sim["t"], dtype=float)
        t_final = float(t_arr[-1]) if t_arr.size else float(t_span[0])
        integrator_success = bool(sim["success"])
        t_end = float(t_span[1])
        outcome = outcome_from_integrator(integrator_success, t_final, t_end)

        C_final = np.asarray(sim["C_final"], dtype=float)
        c_range = float(np.max(C_final) - np.min(C_final)) if C_final.size else float("nan")

        psi_bar_final = np.asarray(sim["psi_bar_final"], dtype=float)
        kappa, spec_r, eta1, growth = _measure_terminal(psi_bar_final, params)

        return {
            "n": int(n),
            "seed": int(seed),
            "t_final": t_final,
            "outcome": outcome,
            "timeout": False,
            "C_range": c_range,
            "kappa": kappa,
            "spectral_ratio": spec_r,
            "eta_f1": eta1,
            "growth": growth,
        }

    use_timer = wallclock_s > 0.0 and _WALLCLOCK_AVAILABLE
    try:
        if use_timer:
            return _run_with_wallclock(wallclock_s, _integrate_and_measure)
        return _integrate_and_measure()
    except EnsembleSimulationTimeout:
        return _timeout_row(n, seed)


def _fmt_cli_num(x: float, *, decimals: int = 2) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{decimals}f}"


def _print_progress_line(
    step_i: int,
    total_steps: int,
    r: dict[str, Any],
    elapsed_s: float,
    wallclock_limit: float,
) -> None:
    w = len(str(total_steps))
    counter = f"({step_i:>{w}}/{total_steps})"
    n = int(r["n"])
    seed = int(r["seed"])
    tag = str(r.get("outcome", "timeout"))
    if r.get("timeout"):
        msg = (
            f"[ensemble] {counter} n={n} seed={seed:>2}  -> {tag:<9}  "
            f"({wallclock_limit:g}s limit)  ({elapsed_s:.1f}s wall)"
        )
        print(msg, flush=True)
        return
    msg = (
        f"[ensemble] {counter} n={n} seed={seed:>2}  -> {tag:<9}  "
        f"t={_fmt_cli_num(float(r['t_final']))}  "
        f"C_range={_fmt_cli_num(float(r['C_range']))}  "
        f"kappa={_fmt_cli_num(float(r['kappa']))}  "
        f"({elapsed_s:.1f}s)"
    )
    print(msg, flush=True)


def _pretty_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0.0 or x == -0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e7 or ax < 1e-4:
        return f"{x:.4e}"
    if ax >= 1.0:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.12f}".rstrip("0").rstrip(".")
    if s in ("-0", "-0."):
        return "0"
    return s if s else "0"


def _format_seed_table(rows: Sequence[dict[str, Any]]) -> str:
    headers = (
        "seed",
        "t_final",
        "outcome",
        "C_range",
        "kappa",
        "spec",
        "eta_f1",
        "growth",
    )
    sep = " | "

    def cells(r: dict[str, Any]) -> list[str]:
        return [
            str(int(r["seed"])),
            _pretty_float(float(r["t_final"])),
            str(r.get("outcome", "")),
            _pretty_float(float(r["C_range"])),
            _pretty_float(float(r["kappa"])),
            _pretty_float(float(r["spectral_ratio"])),
            _pretty_float(float(r["eta_f1"])),
            _pretty_float(float(r["growth"])),
        ]

    matrix: list[list[str]] = [list(headers)] + [cells(r) for r in rows]
    ncol = len(headers)
    widths = [0] * ncol
    for row in matrix:
        for j in range(ncol):
            widths[j] = max(widths[j], len(row[j]))
    lines: list[str] = []
    for i, row in enumerate(matrix):
        lines.append(sep.join(row[j].rjust(widths[j]) for j in range(ncol)))
        if i == 0:
            lines.append("-" * len(lines[-1]))
    return "\n".join(lines)


def _usable_row(r: dict[str, Any]) -> bool:
    """Rows with terminal-state measurements (excludes wallclock timeouts only)."""
    return str(r.get("outcome")) in ("completed", "terminal")


def _stats_lines(
    label: str,
    rows: Sequence[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[str]:
    """Mean, std, min, max over usable runs (``completed`` + ``terminal``) for each key."""
    ok = [r for r in rows if _usable_row(r)]
    lines = [
        f"Statistics ({label}, usable runs (completed + terminal), N={len(ok)}):",
    ]
    if not ok:
        lines.append("  (no usable runs; statistics omitted)")
        return lines

    for key in keys:
        vals = [float(r[key]) for r in ok if math.isfinite(float(r[key]))]
        if not vals:
            lines.append(f"  {key}: no finite values")
            continue
        m = float(statistics.mean(vals))
        s = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        mn = float(min(vals))
        mx = float(max(vals))
        lines.append(
            f"  {key}: mean={m:.10g} std={s:.10g} min={mn:.10g} max={mx:.10g}"
        )
    return lines


def _growth_verdict(rows_n4: Sequence[dict[str, Any]]) -> list[str]:
    """Text verdict on whether terminal growth stays above 1.0 or clusters near 1."""
    ok = [r for r in rows_n4 if _usable_row(r)]
    growths = [float(r["growth"]) for r in ok if math.isfinite(float(r["growth"]))]
    lines = [
        "Robustness verdict (n=4 terminal growth = first eta ratio):",
    ]
    if not growths:
        lines.append("  No usable (non-timeout) runs with finite growth; cannot judge.")
        return lines

    above = sum(1 for g in growths if g > 1.0 + GROWTH_VERDICT_EPS)
    below = sum(1 for g in growths if g < 1.0 - GROWTH_VERDICT_EPS)
    near = len(growths) - above - below
    lines.append(
        f"  Finite growth samples: {len(growths)}; strictly > 1+{GROWTH_VERDICT_EPS:g}: {above}; "
        f"strictly < 1-{GROWTH_VERDICT_EPS:g}: {below}; within band: {near}."
    )
    if above == len(growths):
        lines.append("  Reading: growth is consistently above 1.0 across usable seeds.")
    elif below == len(growths):
        lines.append("  Reading: growth stays below 1.0 across usable seeds.")
    elif near == len(growths):
        lines.append("  Reading: growth clusters near 1.0 (within numerical band) for all usable seeds.")
    else:
        lines.append(
            "  Reading: growth is seed-dependent (mix of above, below, and/or near-1 values)."
        )
    return lines


def _seed_diversity_lines(rows: Sequence[dict[str, Any]], n_report: int) -> list[str]:
    """Unique ``t_final`` at 3 d.p. among usable runs; proxies diversity of trajectories (ICs differ by seed)."""
    usable = [r for r in rows if _usable_row(r)]
    n_ut = len(usable)
    if n_ut == 0:
        return [
            f"Seed diversity check (n={n_report}): no usable (non-timeout) runs; "
            f"cannot assess t_final diversity.",
        ]
    rounded: list[float] = []
    for r in usable:
        tf = float(r["t_final"])
        if math.isfinite(tf):
            rounded.append(round(tf, 3))
    uq = len(set(rounded))
    if uq == n_ut:
        return [
            f"Seed diversity check (n={n_report}): {uq} unique t_final values (3 decimal places) "
            f"out of {n_ut} usable seeds; distinct stopping times (ICs vary by seed).",
        ]
    return [
        f"Seed diversity check (n={n_report}): WARNING only {uq} unique t_final values "
        f"(3 decimal places) among {n_ut} usable seeds; duplicates may indicate weak or broken seeding.",
    ]


def _n3_kappa_check(rows_n3: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"Control check (n=3): interpretive kappa should be 1.0 within {KAPPA_N3_TOL:g} for every seed.",
    ]
    checked = [r for r in rows_n3 if _usable_row(r)]
    if not checked:
        lines.append("  SKIP: no usable (non-timeout) n=3 runs to check.")
        return lines
    bad: list[str] = []
    for r in checked:
        k = float(r["kappa"])
        if not math.isfinite(k) or abs(k - 1.0) > KAPPA_N3_TOL:
            bad.append(f"seed={r['seed']}: kappa={k}")
    if bad:
        lines.append("  BUG SUSPECTED (kappa != 1 at n=3):")
        for b in bad:
            lines.append(f"    {b}")
    else:
        lines.append("  PASS: all seeds satisfy kappa == 1 within tolerance.")
    return lines


def build_ensemble_report_text(
    *,
    seeds: tuple[int, ...],
    quick: bool,
    run_timestamp_utc: str,
    rows_n4: Sequence[dict[str, Any]],
    rows_n3: Sequence[dict[str, Any]],
    wallclock_timeout_s: float,
    partial: bool = False,
    completed_runs: Optional[int] = None,
    total_planned_runs: Optional[int] = None,
) -> str:
    """Assemble the full report body (same structure as ``ensemble_report.txt``)."""
    meas_keys = ("kappa", "spectral_ratio", "eta_f1", "growth", "C_range")

    header_lines = [
        "Thread 7 ensemble experiment (multi-seed terminal variability)",
        f"Run timestamp: {run_timestamp_utc}",
        f"quick_mode: {quick}",
        f"wallclock_timeout_s: {wallclock_timeout_s}",
        f"polynomial_order_primary: n={ENSEMBLE_N_PRIMARY}",
        f"polynomial_order_control: n={ENSEMBLE_N_CONTROL}",
        f"seeds: {list(seeds)} (count={len(seeds)})",
        f"baseline_params: {BASELINE_PARAMS}",
    ]
    if partial and completed_runs is not None and total_planned_runs is not None:
        header_lines.append(
            f"PARTIAL_RUN: interrupted after {completed_runs}/{total_planned_runs} integrations."
        )
    header_lines.extend(
        [
            "",
            "=== n=4 (primary): per-seed terminal state ===",
            "",
        ]
    )
    body_n4 = _format_seed_table(rows_n4)
    diversity_n4 = _seed_diversity_lines(rows_n4, ENSEMBLE_N_PRIMARY)
    stats_n4 = _stats_lines(f"n={ENSEMBLE_N_PRIMARY}", rows_n4, meas_keys)
    verdict = _growth_verdict(rows_n4)

    section_n3 = [
        "",
        "=== n=3 (control): per-seed terminal state ===",
        "",
        _format_seed_table(rows_n3),
        "",
        *_seed_diversity_lines(rows_n3, ENSEMBLE_N_CONTROL),
        "",
        *_n3_kappa_check(rows_n3),
        "",
        *_stats_lines(f"n={ENSEMBLE_N_CONTROL} (informational)", rows_n3, meas_keys),
        "",
    ]

    return (
        "\n".join(header_lines)
        + body_n4
        + "\n\n"
        + "\n".join(diversity_n4)
        + "\n\n"
        + "\n".join(stats_n4)
        + "\n\n"
        + "\n".join(verdict)
        + "\n"
        + "\n".join(section_n3)
    )


def run_ensemble_experiment(
    *,
    quick: bool = False,
    write_disk: bool = True,
    output_dir: Optional[Path] = None,
    wallclock_s: Optional[float] = None,
) -> dict[str, Any]:
    """Run n=4 across seeds, then n=3 control; optionally write ``ensemble_report.txt``.

    Returns dict with keys ``rows_n4``, ``rows_n3``, ``report_text`` (full string),
    and ``interrupted`` (bool) if ``KeyboardInterrupt`` was handled after a partial write.
    """
    root = _repo_root()
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR
    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    seeds: tuple[int, ...] = QUICK_SEEDS if quick else FULL_SEEDS
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    limit = (
        float(wallclock_s)
        if wallclock_s is not None
        else (QUICK_WALLCLOCK_TIMEOUT_S if quick else WALLCLOCK_TIMEOUT_S)
    )

    total_steps = len(seeds) * 2
    rows_n4: list[dict[str, Any]] = []
    rows_n3: list[dict[str, Any]] = []
    interrupted = False
    completed = 0

    print(
        f"[ensemble] n={ENSEMBLE_N_PRIMARY} multi-seed + n={ENSEMBLE_N_CONTROL} control; "
        f"{len(seeds)} seeds {list(seeds)[0]}..{list(seeds)[-1]} => {total_steps} integrations.",
        flush=True,
    )
    if not _WALLCLOCK_AVAILABLE:
        print(
            "[ensemble] Wallclock timeout unavailable on this platform (no SIGALRM); "
            "integrations are not time-capped.",
            flush=True,
        )
    else:
        print(f"[ensemble] Per-run wallclock: {limit:g} s", flush=True)

    def _write_report(partial_run: bool) -> str:
        text = build_ensemble_report_text(
            seeds=seeds,
            quick=quick,
            run_timestamp_utc=ts,
            rows_n4=rows_n4,
            rows_n3=rows_n3,
            wallclock_timeout_s=limit,
            partial=partial_run,
            completed_runs=completed if partial_run else None,
            total_planned_runs=total_steps if partial_run else None,
        )
        if write_disk:
            out_path = out_dir / "ensemble_report.txt"
            out_path.write_text(text, encoding="utf-8")
        return text

    try:
        print(f"\n=== n=4 primary ({len(seeds)} seeds) ===", flush=True)
        for seed in seeds:
            completed += 1
            t0 = time.perf_counter()
            r4 = run_one_seed(ENSEMBLE_N_PRIMARY, seed, quick=quick, wallclock_s=limit)
            rows_n4.append(r4)
            _print_progress_line(completed, total_steps, r4, time.perf_counter() - t0, limit)

        print(f"\n=== n=3 control ({len(seeds)} seeds) ===", flush=True)
        for seed in seeds:
            completed += 1
            t0 = time.perf_counter()
            r3 = run_one_seed(ENSEMBLE_N_CONTROL, seed, quick=quick, wallclock_s=limit)
            rows_n3.append(r3)
            _print_progress_line(completed, total_steps, r3, time.perf_counter() - t0, limit)

    except KeyboardInterrupt:
        interrupted = True
        text = _write_report(partial_run=True)
        if write_disk:
            out_path = out_dir / "ensemble_report.txt"
            print(
                f"[ensemble] Interrupted after {completed}/{total_steps} runs. "
                f"Partial results written to {out_path.relative_to(root)}",
                flush=True,
            )
        else:
            print(
                f"[ensemble] Interrupted after {completed}/{total_steps} runs "
                f"(write_disk=False; report not saved).",
                flush=True,
            )
        return {
            "rows_n4": rows_n4,
            "rows_n3": rows_n3,
            "report_text": text,
            "interrupted": True,
        }

    full_text = _write_report(partial_run=False)
    print(full_text, flush=True)
    if write_disk:
        out_path = out_dir / "ensemble_report.txt"
        print(f"Wrote {out_path.relative_to(root)}", flush=True)

    return {
        "rows_n4": rows_n4,
        "rows_n3": rows_n3,
        "report_text": full_text,
        "interrupted": False,
    }


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*ReconstructionLUT.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser(
        description="Thread 7 ensemble experiment (multi-seed n=4, control n=3)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="N=32, short t_span, seeds 0..4 only",
    )
    args = parser.parse_args()
    out = run_ensemble_experiment(quick=args.quick)
    if out.get("interrupted"):
        sys.exit(130)


if __name__ == "__main__":
    main()
