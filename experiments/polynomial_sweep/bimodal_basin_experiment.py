"""Bimodal basin characterization at n=4: IC amplitude and wavenumber sweeps.

Thread 7 ensemble (20 seeds at baseline) revealed two terminal pathways at n=4:
  Path A (fine-structure death): small final C_range, shorter survival, growth > 1.
  Path B (amplitude death):      large final C_range, longer survival, growth ≈ 1.

Existing data shows a hard gap: Path A seeds have C_range 0.12–0.64, Path B seeds
have C_range 7.57–95.6. This experiment maps the IC structure that determines
path assignment at fixed baseline parameters.

Phase 1 — IC amplitude sweep:
  Uniform-random IC amplitude A swept over N_AMPLITUDE_POINTS log-spaced values
  from A_MIN to A_MAX. N_MICRO_SEEDS independent seeds per amplitude. Path is
  classified by C_range threshold PATH_A_THRESHOLD.

Phase 2 — IC wavenumber sweep:
  Sinusoidal IC: C(x) = A_FIXED * cos(2π k0 x / L), g = 1.
  k0 swept over integers 1..K0_MAX. Single deterministic IC per k0.

Theoretical expectation (swallowtail catastrophe at n=4): the basin boundary
between Path A and Path B should be a smooth codimension-1 surface in IC space.
Phase 1 finds whether the A axis crosses this boundary and where. Phase 2 finds
whether the dominant spatial wavenumber is the orthogonal IC coordinate that
governs path assignment.

Run from repository root::

    python -m experiments.polynomial_sweep.bimodal_basin_experiment
    python -m experiments.polynomial_sweep.bimodal_basin_experiment --quick

Default report: experiments/polynomial_sweep/results/bimodal_basin_report.txt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
from models.dim_1plus1.mfe import integrate_coupled
from shared.metrics import (
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)
from shared.reconstruction import ReconstructionLUT

# Polynomial order under study
BASIN_N = 4

# Path classifier: C_range below this threshold → Path A (fine-structure death).
# Calibrated against 20-seed ensemble: max Path A C_range ≈ 0.64, min Path B ≈ 7.57.
PATH_A_THRESHOLD = 1.0

# Phase 1 — amplitude sweep
N_AMPLITUDE_POINTS = 30
A_MIN = 1e-3
A_MAX = 1e1
N_MICRO_SEEDS = 5

QUICK_N_AMPLITUDE_POINTS = 8
QUICK_N_MICRO_SEEDS = 2

# Phase 2 — wavenumber sweep
# A_FIXED matches initial_conditions default so phase 2 is comparable to existing ensemble.
A_FIXED = 0.01
K0_MAX = 10       # integer wavenumbers 1..K0_MAX on the periodic domain

QUICK_K0_MAX = 5


def classify_path(C_range: float, threshold: float = PATH_A_THRESHOLD) -> str:
    """Path A (fine-structure death): C_range < threshold. Path B (amplitude death): C_range >= threshold."""
    if not math.isfinite(C_range):
        return "unknown"
    return "A" if C_range < threshold else "B"


def build_params_basin(*, quick: bool) -> dict[str, Any]:
    """n=4 params dict; same merge logic as the main sweep."""
    grid = {"N": QUICK["N"], "L": QUICK["L"]} if quick else dict(GRID)
    integ = {**INTEGRATION, **INTEGRATION_OVERRIDES_BY_N.get(BASIN_N, {})}
    if quick:
        integ = {
            **integ,
            "t_span": (float(QUICK["t_span"][0]), float(QUICK["t_span"][1])),
            "max_step": float(QUICK["max_step"]),
        }
    params: dict[str, Any] = {**BASELINE_PARAMS, "n": BASIN_N}
    params["N"] = int(grid["N"])
    params["L"] = float(grid["L"])
    params["dx"] = params["L"] / params["N"]
    params["t_span"] = (float(integ["t_span"][0]), float(integ["t_span"][1]))
    params["method"] = str(integ["method"])
    params["max_step"] = float(integ["max_step"])
    if "rtol" in integ:
        params["rtol"] = float(integ["rtol"])
    if "atol" in integ:
        params["atol"] = float(integ["atol"])
    return params


def _run_probe(
    C0: np.ndarray,
    g0: np.ndarray,
    params: dict[str, Any],
    lut: ReconstructionLUT,
) -> dict[str, Any]:
    """Run one coupled IVP from (C0, g0) and return path classification + measurements."""
    n = int(params["n"])
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    dx = float(params["dx"])
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))

    integrate_kwargs: dict[str, Any] = {
        "method": params["method"],
        "max_step": float(params["max_step"]),
        "reconstruct_fn": lut,
    }
    if "rtol" in params:
        integrate_kwargs["rtol"] = float(params["rtol"])
    if "atol" in params:
        integrate_kwargs["atol"] = float(params["atol"])

    try:
        out = integrate_coupled(C0, g0, params, t_span, **integrate_kwargs)
    except (ValueError, RuntimeError):
        # Even-n LUT has a C_floor; large-amplitude ICs can push C below it, producing NaN
        # in the reconstructed psi_bar, which corrupts the Radau Jacobian. Treat as diverged.
        nan = float("nan")
        return {
            "outcome": "diverged",
            "t_final": nan,
            "C_range": nan,
            "path": "unknown",
            "kappa": nan,
            "spectral_ratio": nan,
            "growth": nan,
        }

    t_arr = np.asarray(out["t"], dtype=float)
    t_final = float(t_arr[-1]) if t_arr.size else float(t_span[0])
    outcome = outcome_from_integrator(bool(out["success"]), t_final, float(t_span[1]))

    C_hist = np.asarray(out["C_history"], dtype=float)
    C_final = C_hist[-1] if C_hist.shape[0] > 0 else C0.copy()
    C_range = float(np.max(C_final) - np.min(C_final))

    psi_final = np.asarray(lut(C_final), dtype=float)

    try:
        ic = interpretive_condition_number(psi_final, n)
        kappa = float(ic["kappa"]) if math.isfinite(float(ic["kappa"])) else float("nan")
    except Exception:
        kappa = float("nan")

    try:
        sc = spectral_concentration_ratio(psi_final, n, gamma, sigma, dx)
        spec_r = float(sc["ratio"]) if math.isfinite(float(sc["ratio"])) else float("nan")
    except Exception:
        spec_r = float("nan")

    try:
        nl = nonlocal_correction_growth(
            psi_final, n, gamma, sigma, dx, NONLOCAL["coarsening_factors"]
        )
        growths = nl.get("growth_rates") or []
        growth = float(growths[0]) if growths and math.isfinite(float(growths[0])) else float("nan")
    except Exception:
        growth = float("nan")

    return {
        "outcome": outcome,
        "t_final": t_final,
        "C_range": C_range,
        "path": classify_path(C_range),
        "kappa": kappa,
        "spectral_ratio": spec_r,
        "growth": growth,
    }


def _build_lut(params: dict[str, Any]) -> ReconstructionLUT:
    lut_cfg = RECONSTRUCTION_LUT
    return ReconstructionLUT(
        int(params["n"]),
        float(params["gamma"]),
        float(lut_cfg["C_min"]),
        float(lut_cfg["C_max"]),
        n_samples=int(lut_cfg["n_samples"]),
    )


def run_amplitude_sweep(
    params: dict[str, Any],
    *,
    n_points: int,
    a_min: float,
    a_max: float,
    micro_seeds: int,
) -> list[dict[str, Any]]:
    """Phase 1: sweep uniform-random IC amplitude A across n_points log-spaced values."""
    N = int(params["N"])
    amplitudes = np.logspace(math.log10(a_min), math.log10(a_max), n_points)
    rows: list[dict[str, Any]] = []
    total = n_points * micro_seeds

    print(
        f"[basin] Phase 1: amplitude sweep  {n_points} points × {micro_seeds} seeds = {total} runs",
        flush=True,
    )

    for i, A in enumerate(amplitudes):
        lut = _build_lut(params)
        for s in range(micro_seeds):
            rng = np.random.default_rng(s)
            C0 = rng.uniform(-float(A), float(A), size=N)
            g0 = np.ones(N, dtype=float)

            t0 = time.perf_counter()
            row = _run_probe(C0, g0, params, lut)
            elapsed = time.perf_counter() - t0

            step = i * micro_seeds + s + 1
            print(
                f"[basin] ({step:>{len(str(total))}}/{total})  A={A:.3e}  seed={s}"
                f"  -> path={row['path']}  C_range={row['C_range']:.4f}"
                f"  outcome={row['outcome']}  ({elapsed:.1f}s)",
                flush=True,
            )
            rows.append({"A": float(A), "seed": s, **row})

    return rows


def run_wavenumber_sweep(
    params: dict[str, Any],
    *,
    a_fixed: float,
    k0_max: int,
) -> list[dict[str, Any]]:
    """Phase 2: sinusoidal IC C(x) = A * cos(2π k0 x / L) swept over k0 = 1..k0_max."""
    N = int(params["N"])
    L = float(params["L"])
    x = np.linspace(0.0, L, N, endpoint=False)
    rows: list[dict[str, Any]] = []

    print(
        f"[basin] Phase 2: wavenumber sweep  k0 = 1..{k0_max}  A_fixed={a_fixed:.3e}",
        flush=True,
    )

    for k0 in range(1, k0_max + 1):
        lut = _build_lut(params)
        C0 = float(a_fixed) * np.cos(2.0 * np.pi * k0 * x / L)
        g0 = np.ones(N, dtype=float)

        t0 = time.perf_counter()
        row = _run_probe(C0, g0, params, lut)
        elapsed = time.perf_counter() - t0

        print(
            f"[basin] k0={k0:>3}  -> path={row['path']}  C_range={row['C_range']:.4f}"
            f"  outcome={row['outcome']}  ({elapsed:.1f}s)",
            flush=True,
        )
        rows.append({"k0": k0, **row})

    return rows


def _fmt(x: float, decimals: int = 4) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{decimals}f}"


def _phase1_table(rows: list[dict[str, Any]]) -> str:
    headers = ("A", "seed", "path", "C_range", "kappa", "spec", "growth", "outcome")
    sep = " | "

    def cells(r: dict[str, Any]) -> list[str]:
        return [
            f"{r['A']:.3e}",
            str(int(r["seed"])),
            str(r["path"]),
            _fmt(float(r["C_range"])),
            _fmt(float(r["kappa"])),
            _fmt(float(r["spectral_ratio"])),
            _fmt(float(r["growth"])),
            str(r["outcome"]),
        ]

    matrix = [list(headers)] + [cells(r) for r in rows]
    ncol = len(headers)
    widths = [max(len(row[j]) for row in matrix) for j in range(ncol)]
    lines: list[str] = []
    for i, row in enumerate(matrix):
        lines.append(sep.join(row[j].rjust(widths[j]) for j in range(ncol)))
        if i == 0:
            lines.append("-" * len(lines[-1]))
    return "\n".join(lines)


def _phase2_table(rows: list[dict[str, Any]]) -> str:
    headers = ("k0", "path", "C_range", "kappa", "spec", "growth", "outcome")
    sep = " | "

    def cells(r: dict[str, Any]) -> list[str]:
        return [
            str(int(r["k0"])),
            str(r["path"]),
            _fmt(float(r["C_range"])),
            _fmt(float(r["kappa"])),
            _fmt(float(r["spectral_ratio"])),
            _fmt(float(r["growth"])),
            str(r["outcome"]),
        ]

    matrix = [list(headers)] + [cells(r) for r in rows]
    ncol = len(headers)
    widths = [max(len(row[j]) for row in matrix) for j in range(ncol)]
    lines: list[str] = []
    for i, row in enumerate(matrix):
        lines.append(sep.join(row[j].rjust(widths[j]) for j in range(ncol)))
        if i == 0:
            lines.append("-" * len(lines[-1]))
    return "\n".join(lines)


def _phase1_summary(rows: list[dict[str, Any]]) -> list[str]:
    """For each unique amplitude, report path counts."""
    by_a: dict[float, list[str]] = {}
    for r in rows:
        a = float(r["A"])
        by_a.setdefault(a, []).append(str(r["path"]))

    lines = ["Amplitude → path assignment (counts per A):"]
    for a in sorted(by_a):
        paths = by_a[a]
        n_a = paths.count("A")
        n_b = paths.count("B")
        n_u = len(paths) - n_a - n_b
        tag = f"A:{n_a} B:{n_b}" + (f" ?:{n_u}" if n_u else "")
        lines.append(f"  A={a:.3e}  {tag}")
    return lines


def _boundary_estimate(rows: list[dict[str, Any]]) -> list[str]:
    """Estimate the amplitude below/above which Path B first appears."""
    by_a: dict[float, list[str]] = {}
    for r in rows:
        a = float(r["A"])
        by_a.setdefault(a, []).append(str(r["path"]))

    sorted_a = sorted(by_a)
    last_all_a: Optional[float] = None
    first_any_b: Optional[float] = None

    for a in sorted_a:
        paths = by_a[a]
        if any(p == "B" for p in paths):
            if first_any_b is None:
                first_any_b = a
        elif all(p == "A" for p in paths):
            last_all_a = a

    lines = ["Basin boundary estimate (amplitude axis):"]
    n_diverged = sum(
        1 for r in rows if str(r.get("path")) == "unknown"
    )
    if n_diverged:
        lines.append(
            f"  Note: {n_diverged} runs diverged (NaN in n=4 even-branch reconstruction)."
            " Large-amplitude ICs push C below the principal-branch floor; these amplitudes"
            " are outside the valid domain of the n=4 coarse-graining map."
        )
    if first_any_b is None:
        lines.append("  No Path B runs observed; all seeds stayed in Path A across the amplitude range.")
        lines.append("  Swallowtail prediction: boundary exists above A_MAX or requires different IC geometry.")
    elif last_all_a is None:
        lines.append(f"  Path B present at lowest tested amplitude (A={sorted_a[0]:.3e}).")
        lines.append("  Boundary is below A_MIN; extend sweep downward.")
    else:
        lines.append(f"  Last amplitude with all Path A seeds: A={last_all_a:.3e}")
        lines.append(f"  First amplitude with any Path B seed: A={first_any_b:.3e}")
        lines.append(
            "  Boundary bracket: [{:.3e}, {:.3e}]".format(last_all_a, first_any_b)
        )
        lines.append("  Refine with denser sampling in this bracket to locate the swallowtail seam.")
    return lines


def _phase2_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["Wavenumber → path assignment:"]
    path_a_k0 = [int(r["k0"]) for r in rows if r["path"] == "A"]
    path_b_k0 = [int(r["k0"]) for r in rows if r["path"] == "B"]
    path_u_k0 = [int(r["k0"]) for r in rows if r["path"] not in ("A", "B")]
    if path_a_k0:
        lines.append(f"  Path A (fine-structure death): k0 = {path_a_k0}")
    if path_b_k0:
        lines.append(f"  Path B (amplitude death):      k0 = {path_b_k0}")
    if path_u_k0:
        lines.append(f"  Unknown path:                  k0 = {path_u_k0}")
    if not path_a_k0 and not path_b_k0:
        lines.append("  No path assignments (all outcomes unknown or no runs).")
    return lines


def build_report(
    *,
    rows_phase1: list[dict[str, Any]],
    rows_phase2: list[dict[str, Any]],
    params: dict[str, Any],
    quick: bool,
    ts: str,
    n_amplitude_points: int,
    micro_seeds: int,
    a_min: float,
    a_max: float,
    a_fixed: float,
    k0_max: int,
) -> str:
    header = "\n".join([
        "Thread 7 bimodal basin characterization (n=4)",
        f"Run timestamp: {ts}",
        f"quick_mode: {quick}",
        f"basin_n: {BASIN_N}",
        f"path_a_threshold (C_range): {PATH_A_THRESHOLD}",
        f"baseline_params: {BASELINE_PARAMS}",
        f"grid: N={params['N']}  L={params['L']}  dx={params['dx']:.6f}",
        f"integration: method={params['method']}  t_span={params['t_span']}  max_step={params['max_step']}",
        "",
        "Classification: Path A = fine-structure death (C_range < threshold, short survival, growth > 1)",
        "                Path B = amplitude death     (C_range >= threshold, long survival, growth ≈ 1)",
        "",
        f"=== Phase 1: IC amplitude sweep  (n_points={n_amplitude_points}  seeds/A={micro_seeds}  "
        f"A_range=[{a_min:.3e}, {a_max:.3e}]) ===",
    ])
    p1_table = _phase1_table(rows_phase1)
    p1_summary = "\n".join([""] + _phase1_summary(rows_phase1))
    p1_boundary = "\n".join([""] + _boundary_estimate(rows_phase1))

    phase2_header = "\n".join([
        "",
        f"=== Phase 2: IC wavenumber sweep  (A_fixed={a_fixed:.3e}  k0_max={k0_max}) ===",
    ])
    p2_table = _phase2_table(rows_phase2)
    p2_summary = "\n".join([""] + _phase2_summary(rows_phase2))

    return "\n".join([
        header, "", p1_table, p1_summary, p1_boundary,
        phase2_header, "", p2_table, p2_summary, "",
    ])


def run_basin_experiment(
    *,
    quick: bool = False,
    write_disk: bool = True,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run both phases and optionally write bimodal_basin_report.txt."""
    root = Path(__file__).resolve().parents[2]
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR
    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    params = build_params_basin(quick=quick)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    n_amp = QUICK_N_AMPLITUDE_POINTS if quick else N_AMPLITUDE_POINTS
    n_seeds = QUICK_N_MICRO_SEEDS if quick else N_MICRO_SEEDS
    k0_max = QUICK_K0_MAX if quick else K0_MAX

    rows_p1 = run_amplitude_sweep(
        params,
        n_points=n_amp,
        a_min=A_MIN,
        a_max=A_MAX,
        micro_seeds=n_seeds,
    )

    print("", flush=True)
    rows_p2 = run_wavenumber_sweep(params, a_fixed=A_FIXED, k0_max=k0_max)

    report = build_report(
        rows_phase1=rows_p1,
        rows_phase2=rows_p2,
        params=params,
        quick=quick,
        ts=ts,
        n_amplitude_points=n_amp,
        micro_seeds=n_seeds,
        a_min=A_MIN,
        a_max=A_MAX,
        a_fixed=A_FIXED,
        k0_max=k0_max,
    )

    print(report, flush=True)
    if write_disk:
        out_path = out_dir / "bimodal_basin_report.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(root)}", flush=True)

    return {"rows_phase1": rows_p1, "rows_phase2": rows_p2, "report_text": report}


def main() -> None:
    warnings.filterwarnings("ignore", message=r".*ReconstructionLUT.*", category=UserWarning)
    parser = argparse.ArgumentParser(
        description="Thread 7 bimodal basin characterization (n=4 IC amplitude + wavenumber sweep)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"N={QUICK['N']}, short t_span, {QUICK_N_AMPLITUDE_POINTS} amplitudes, {QUICK_N_MICRO_SEEDS} seeds, k0_max={QUICK_K0_MAX}",
    )
    args = parser.parse_args()
    run_basin_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
