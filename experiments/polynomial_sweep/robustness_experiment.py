"""Parameter-robustness stress test for Thread 7 qualitative patterns (n = 3, 4, 5).

Each run perturbs exactly one baseline SR parameter, integrates the coupled 1+1 model with the
same per-n solver settings as the main sweep, and records the four sweep measurements on the
final fields plus integration metadata.

Run from repository root::

    python -m experiments.polynomial_sweep.robustness_experiment
    python -m experiments.polynomial_sweep.robustness_experiment --quick

Default report path: ``experiments/polynomial_sweep/results/robustness_report.txt``.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from experiments.polynomial_sweep.config import (
    BASELINE_PARAMS,
    GRID,
    INTEGRATION,
    INTEGRATION_OVERRIDES_BY_N,
    METASTABLE_PSI_RANGE,
    NONLOCAL,
    QUICK,
    RECONSTRUCTION_LUT,
    RESULTS_DIR,
)
from models.dim_1plus1.mfe import run_simulation
from shared.metrics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)

# Polynomial orders bracketing the cubic aperture boundary.
ROBUSTNESS_NS: tuple[int, ...] = (3, 4, 5)

# Single-parameter sweeps (label, one-key override merged into BASELINE_PARAMS).
FULL_PERTURBATIONS: tuple[tuple[str, dict[str, float]], ...] = (
    ("mu_sq=0.5", {"mu_sq": 0.5}),
    ("mu_sq=2.0", {"mu_sq": 2.0}),
    ("gamma=0.5", {"gamma": 0.5}),
    ("gamma=2.0", {"gamma": 2.0}),
    ("sigma=0.3", {"sigma": 0.3}),
    ("sigma=1.0", {"sigma": 1.0}),
    ("lambda_B=0.1", {"lambda_B": 0.1}),
    ("lambda_B=1.0", {"lambda_B": 1.0}),
    ("eta_g=0.5", {"eta_g": 0.5}),
    ("eta_g=2.0", {"eta_g": 2.0}),
    ("xi_g=0.01", {"xi_g": 0.01}),
    ("xi_g=0.5", {"xi_g": 0.5}),
)

QUICK_PERTURBATIONS: tuple[tuple[str, dict[str, float]], ...] = (
    ("mu_sq=2.0", {"mu_sq": 2.0}),
    ("gamma=0.5", {"gamma": 0.5}),
)

# C_range above this threshold counts as a non-trivial structured field for Q2.
STRUCTURED_C_RANGE_EPS = 1e-6

# Integration horizon used in Q3 when comparing to t=30 (full run uses 30.0 from config).
T_LAB_TARGET = 30.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _merge_sr_core(overrides: Mapping[str, float]) -> dict[str, Any]:
    """Baseline Appendix A parameters with optional single-key replacement."""
    return {**BASELINE_PARAMS, **dict(overrides)}


def build_params_for_robustness(
    n: int,
    sr_core: Mapping[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    """Grid, integration, and SR parameters for one (n, perturbation) case.

    Merge order matches ``run_single`` / ``build_params_for_snapshot``: ``INTEGRATION`` then
    ``INTEGRATION_OVERRIDES_BY_N[n]``, then quick overrides last.
    """
    grid = {"N": QUICK["N"], "L": QUICK["L"]} if quick else dict(GRID)
    integ = {**INTEGRATION, **INTEGRATION_OVERRIDES_BY_N.get(int(n), {})}
    if quick:
        integ = {
            **integ,
            "t_span": (float(QUICK["t_span"][0]), float(QUICK["t_span"][1])),
            "max_step": float(QUICK["max_step"]),
        }
    params: dict[str, Any] = {**dict(sr_core), "n": int(n)}
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


def _metastable_count_for_n(params: dict[str, Any]) -> int:
    n = int(params["n"])
    ms = count_metastable_states(
        METASTABLE_PSI_RANGE,
        n,
        float(params["gamma"]),
        float(params["mu_sq"]),
        float(params["alpha_phi"]),
        float(params["lambda_B"]),
        zeta=None,
        sigma=float(params["sigma"]),
        n_quad=512,
        peak_prominence=0.04,
        peak_distance=120,
    )
    return int(ms["count"])


def _measure_final_fields(
    psi_bar: np.ndarray,
    params: dict[str, Any],
) -> tuple[int, float, float, float, float]:
    """Returns metastable_count, kappa, spectral_ratio, eta_f1, first growth rate."""
    n = int(params["n"])
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    dx = float(params["dx"])

    metastable_count = _metastable_count_for_n(params)

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

    return metastable_count, kappa, spectral_ratio, eta1, gr0


def run_one_case(
    perturbation_label: str,
    sr_core: Mapping[str, Any],
    n: int,
    *,
    quick: bool,
    lut_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single (perturbation, n): coupled IVP with LUT, then measurements on final ``psi_bar``."""
    params = build_params_for_robustness(n, sr_core, quick=quick)
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))
    lut_opts = {**RECONSTRUCTION_LUT, **(lut_cfg or {})}

    sim_kwargs: dict[str, Any] = {
        "t_span": t_span,
        "seed": int(params["seed"]),
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

    sim = run_simulation(params, **sim_kwargs)
    t_arr = np.asarray(sim["t"], dtype=float)
    t_final = float(t_arr[-1]) if t_arr.size else float(t_span[0])
    success = bool(sim["success"])

    C_final = np.asarray(sim["C_final"], dtype=float)
    c_range = float(np.max(C_final) - np.min(C_final)) if C_final.size else float("nan")

    psi_bar_final = np.asarray(sim["psi_bar_final"], dtype=float)
    m_count, kappa, spec_r, eta1, growth = _measure_final_fields(psi_bar_final, params)

    return {
        "perturbation_label": str(perturbation_label),
        "n": int(n),
        "t_final": t_final,
        "success": success,
        "C_range": c_range,
        "metastable_count": int(m_count),
        "kappa": kappa,
        "spectral_ratio": spec_r,
        "eta_f1": eta1,
        "growth": growth,
        "t_span_end": float(t_span[1]),
    }


def _pretty_float(x: float) -> str:
    """Human-readable float for text tables (matches snapshot experiment style)."""
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


_TABLE_SEP = " | "
_TABLE_HEADERS = (
    "perturbation",
    "n",
    "t_final",
    "success",
    "C_range",
    "meta",
    "kappa",
    "spec",
    "eta_f1",
    "growth",
)


def _row_cells(r: dict[str, Any]) -> list[str]:
    mc = r["metastable_count"]
    meta_s = str(int(mc)) if isinstance(mc, int) and mc >= 0 else str(mc)
    return [
        str(r["perturbation_label"]),
        str(r["n"]),
        _pretty_float(float(r["t_final"])),
        str(bool(r["success"])),
        _pretty_float(float(r["C_range"])),
        meta_s,
        _pretty_float(float(r["kappa"])),
        _pretty_float(float(r["spectral_ratio"])),
        _pretty_float(float(r["eta_f1"])),
        _pretty_float(float(r["growth"])),
    ]


def _format_table(rows: Sequence[dict[str, Any]]) -> str:
    sep = _TABLE_SEP
    header_cells = list(_TABLE_HEADERS)
    matrix: list[list[str]] = [header_cells] + [_row_cells(r) for r in rows]
    ncol = len(header_cells)
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


def _summarize(rows: list[dict[str, Any]], *, quick: bool) -> str:
    """Qualitative checks Q1–Q3 against collected rows."""

    def rows_for(label: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["perturbation_label"] == label]

    labels = sorted({str(r["perturbation_label"]) for r in rows})
    t_goal = T_LAB_TARGET
    if quick:
        t_goal = min(T_LAB_TARGET, float(rows[0]["t_span_end"]) if rows else T_LAB_TARGET)

    lines: list[str] = [
        "Summary (qualitative patterns)",
        f"  Integration horizon for Q3 reference: t_end >= {t_goal} (full config targets t={T_LAB_TARGET}; "
        f"quick mode uses shorter t_span from config.QUICK).",
        "",
        "Q1: kappa = 1.0 at n=3 (exponent n-3 vanishes in interpretive condition number)",
    ]

    q1_ok = True
    for lab in labels:
        for r in rows_for(lab):
            if int(r["n"]) != 3:
                continue
            k = float(r["kappa"])
            if not math.isfinite(k):
                q1_ok = False
                lines.append(f"  FAIL {lab} n=3: kappa is non-finite ({k}).")
            elif abs(k - 1.0) > 1e-6:
                q1_ok = False
                lines.append(f"  FAIL {lab} n=3: kappa={k} (expected 1.0).")
    if q1_ok:
        lines.append("  PASS: all n=3 rows have kappa == 1 within 1e-6.")

    lines.extend(
        [
            "",
            f"Q2: at n=4, kappa > 1 when the field is structured (C_range > {STRUCTURED_C_RANGE_EPS:g})",
        ]
    )
    q2_ok = True
    for lab in labels:
        r4 = next((x for x in rows_for(lab) if int(x["n"]) == 4), None)
        if r4 is None:
            continue
        cr = float(r4["C_range"])
        k4 = float(r4["kappa"])
        if cr > STRUCTURED_C_RANGE_EPS:
            if not (math.isfinite(k4) and k4 > 1.0):
                q2_ok = False
                lines.append(
                    f"  FAIL {lab} n=4: structured (C_range={cr:g}) but kappa={k4} (expected > 1)."
                )
        else:
            lines.append(
                f"  SKIP {lab} n=4: nearly flat field (C_range={cr:g}); kappa threshold not applied."
            )
    if q2_ok:
        lines.append("  PASS: every structured n=4 case has kappa > 1 (or n=4 was flat / skipped above).")

    lines.extend(
        [
            "",
            "Q3: n=5 reaches the integration horizon or outlasts n=4 (t_final comparison per perturbation)",
        ]
    )
    q3_ok = True
    for lab in labels:
        r4 = next((x for x in rows_for(lab) if int(x["n"]) == 4), None)
        r5 = next((x for x in rows_for(lab) if int(x["n"]) == 5), None)
        if r4 is None or r5 is None:
            q3_ok = False
            lines.append(f"  FAIL {lab}: missing n=4 or n=5 row.")
            continue
        t4 = float(r4["t_final"])
        t5 = float(r5["t_final"])
        reached_horizon = t5 >= t_goal - 1e-6
        outlasts = t5 >= t4 - 1e-9
        if not (reached_horizon or outlasts):
            q3_ok = False
            lines.append(
                f"  FAIL {lab}: t_final n=5={t5:g} < horizon {t_goal:g} and < n=4 final {t4:g}."
            )
    if q3_ok:
        lines.append("  PASS: for each perturbation, n=5 survives to t_end or matches/exceeds n=4.")

    return "\n".join(lines)


def run_robustness_experiment(
    *,
    quick: bool = False,
    write_disk: bool = True,
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Run baseline + perturbations for n in (3,4,5); optionally write ``robustness_report.txt``.

    Returns row dicts (tests and scripting); each dict includes ``t_span_end`` for summaries.
    """
    root = _repo_root()
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR
    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    perturbations: tuple[tuple[str, dict[str, float]], ...] = QUICK_PERTURBATIONS if quick else FULL_PERTURBATIONS
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cases: list[tuple[str, dict[str, Any]]] = [("baseline", _merge_sr_core({}))]
    cases.extend((lab, _merge_sr_core(ov)) for lab, ov in perturbations)

    n_cases = len(cases)
    n_orders = len(ROBUSTNESS_NS)
    total_steps = n_cases * n_orders
    print(
        f"[robustness] {n_cases} parameter sets (baseline + perturbations) x "
        f"{n_orders} orders n={list(ROBUSTNESS_NS)} => {total_steps} integrations.",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    step_i = 0
    for label, sr_core in cases:
        for n in ROBUSTNESS_NS:
            step_i += 1
            print(
                f"[robustness] ({step_i}/{total_steps}) start  perturbation={label} n={n}",
                flush=True,
            )
            row = run_one_case(label, sr_core, n, quick=quick)
            rows.append(row)
            print(
                f"[robustness] ({step_i}/{total_steps}) done   perturbation={label} n={n} "
                f"success={row['success']} t_final={row['t_final']}",
                flush=True,
            )

    header_lines = [
        "Thread 7 robustness experiment (parameter perturbations, n = 3, 4, 5)",
        f"Run timestamp: {ts}",
        f"quick_mode: {quick}",
        f"baseline_params: {BASELINE_PARAMS}",
        f"polynomial_orders: {list(ROBUSTNESS_NS)}",
        "",
        "Perturbations tested (single-parameter replacements relative to baseline):",
    ]
    if quick:
        header_lines.append(f"  [quick] {list(QUICK_PERTURBATIONS)}")
    else:
        header_lines.append(f"  {list(FULL_PERTURBATIONS)}")
    header_lines.extend(
        [
            "",
            "Columns: meta = metastable_count; spec = spectral_ratio; eta_f1 = eta(coarsening 1.0); "
            "growth = first ratio eta(f_{i+1})/eta(f_i).",
            "",
        ]
    )

    body = _format_table(rows)
    summary = _summarize(rows, quick=quick)
    full_text = "\n".join(header_lines) + body + "\n\n" + summary + "\n"

    print(full_text)
    if write_disk:
        out_path = out_dir / "robustness_report.txt"
        out_path.write_text(full_text, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(root)}", flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thread 7 robustness experiment (parameter perturbations, n=3,4,5)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="N=32, short t_span, only mu_sq=2.0 and gamma=0.5 perturbations",
    )
    args = parser.parse_args()
    run_robustness_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
