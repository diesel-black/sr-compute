"""Coupling scale experiment: seed spectral analysis and sigma-sweep of k0_crit (n=4).

Two-phase follow-up to the bimodal basin characterization experiment.

Phase 0 — seed spectral analysis (no simulation):
  For each micro seed (0..4) used in the bimodal basin amplitude sweep, compute the
  power spectrum of the initial condition C0 and report the dominant wavenumber and the
  power split below and above the k0=7 basin threshold (last Path B) from Phase 2.
  Tests whether seed-to-path partition (0,3,4 -> B; 1,2 -> A) correlates with dominant
  IC spectral content relative to the k0~7.5 threshold.

Phase 1 — sigma sweep:
  Run the wavenumber sweep (sinusoidal IC, k0 = 1..K0_MAX, same A_FIXED) at sigma in
  SIGMA_VALUES. For each sigma, find k0_crit (last k0 in Path B before the first Path A
  transition) and compute k0_crit * sigma. If the coupling kernel's resolution scale is
  the basin discriminator, this product should be approximately constant.

  Note: sigma enters the dynamics through two scalar prefactors only:
    - CFE brake: zeta_cubic(gamma, sigma) = 9*gamma^2 / (pi^{3/2} * sigma^3)
    - MFE expansion: (6*xi_g*gamma) / (pi * sigma^2)
  The Gaussian kernel G_sigma does NOT appear as a spatial filter in the IVP itself;
  it appears only in the interpretive_condition_number and spectral_concentration_ratio
  metrics. Varying sigma therefore changes brake strength and MFE expansion, not spatial
  coupling range. Whether k0_crit * sigma is invariant under these scalar changes is a
  falsifiable test of the coupling-scale hypothesis.

Run from repository root::

    python -m experiments.polynomial_sweep.coupling_scale_experiment
    python -m experiments.polynomial_sweep.coupling_scale_experiment --quick
"""

from __future__ import annotations

import argparse
import math
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

BASIN_N = 4
PATH_A_THRESHOLD = 1.0      # C_range below this -> Path A (fine-structure death)
A_FIXED = 0.01              # IC amplitude for both phases; matches bimodal basin Phase 2
N_SEEDS = 5                 # seeds 0..4
K0_THRESHOLD = 7            # last k0 in Path B from bimodal basin Phase 2 (at sigma=0.5)

# Path assignments from bimodal basin Phase 1 (amplitude-independent, confirmed at all A in [1e-3, 0.42])
SEED_PATH_KNOWN: dict[int, str] = {0: "B", 1: "A", 2: "A", 3: "B", 4: "B"}

SIGMA_VALUES = [0.3, 0.5, 1.0]

# k0_max per sigma: prediction is k0_crit ~ K0_THRESHOLD * (0.5 / sigma); add margin
# sigma=0.3 -> k0_crit ~ 12.5, sigma=0.5 -> ~7.5, sigma=1.0 -> ~3.75
K0_MAX = 18
K0_MAX_QUICK = 8


def _classify_path(C_range: float) -> str:
    if not math.isfinite(C_range):
        return "unknown"
    return "A" if C_range < PATH_A_THRESHOLD else "B"


def _build_lut(params: dict[str, Any]) -> ReconstructionLUT:
    lut_cfg = RECONSTRUCTION_LUT
    return ReconstructionLUT(
        int(params["n"]),
        float(params["gamma"]),
        float(lut_cfg["C_min"]),
        float(lut_cfg["C_max"]),
        n_samples=int(lut_cfg["n_samples"]),
    )


def _build_params(*, sigma: float, quick: bool) -> dict[str, Any]:
    grid = {"N": QUICK["N"], "L": QUICK["L"]} if quick else dict(GRID)
    integ = {**INTEGRATION, **INTEGRATION_OVERRIDES_BY_N.get(BASIN_N, {})}
    if quick:
        integ = {
            **integ,
            "t_span": (float(QUICK["t_span"][0]), float(QUICK["t_span"][1])),
            "max_step": float(QUICK["max_step"]),
        }
    params: dict[str, Any] = {**BASELINE_PARAMS, "n": BASIN_N}
    params["sigma"] = sigma
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
    """Run one coupled IVP from (C0, g0) and return path classification and measurements."""
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
        "path": _classify_path(C_range),
        "kappa": kappa,
        "spectral_ratio": spec_r,
        "growth": growth,
    }


# ---------------------------------------------------------------------------
# Phase 0: seed spectral analysis
# ---------------------------------------------------------------------------

def run_seed_spectral_analysis(N: int) -> list[dict[str, Any]]:
    """Phase 0: power spectrum of the Phase 1 uniform-random ICs for seeds 0..4.

    Computes rfft magnitude spectrum and reports dominant non-DC wavenumber index
    and power fractions below and above K0_THRESHOLD for each seed.
    """
    rows: list[dict[str, Any]] = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        C0 = rng.uniform(-A_FIXED, A_FIXED, size=N)
        mag = np.abs(np.fft.rfft(C0))

        dominant_k = int(np.argmax(mag[1:]) + 1)

        # Power in k=1..K0_THRESHOLD (below threshold, last Path B wavenumber)
        power_low = float(np.sum(mag[1 : K0_THRESHOLD + 1] ** 2))
        # Power in k=K0_THRESHOLD+1..Nyquist (above threshold, first Path A wavenumber)
        power_high = float(np.sum(mag[K0_THRESHOLD + 1 :] ** 2))
        power_total = float(np.sum(mag[1:] ** 2))

        rows.append({
            "seed": seed,
            "dominant_k": dominant_k,
            "power_low_frac": power_low / power_total if power_total > 0 else float("nan"),
            "power_high_frac": power_high / power_total if power_total > 0 else float("nan"),
            "known_path": SEED_PATH_KNOWN[seed],
        })
    return rows


# ---------------------------------------------------------------------------
# Phase 1: sigma sweep
# ---------------------------------------------------------------------------

def run_sigma_wavenumber_sweep(
    sigma: float,
    *,
    quick: bool,
    k0_max: int,
) -> list[dict[str, Any]]:
    """Phase 1: sinusoidal IC wavenumber sweep at a given sigma."""
    params = _build_params(sigma=sigma, quick=quick)
    N = int(params["N"])
    L = float(params["L"])
    x = np.linspace(0.0, L, N, endpoint=False)
    lut = _build_lut(params)
    rows: list[dict[str, Any]] = []

    for k0 in range(1, k0_max + 1):
        C0 = A_FIXED * np.cos(2.0 * np.pi * k0 * x / L)
        g0 = np.ones(N, dtype=float)

        t0 = time.perf_counter()
        row = _run_probe(C0, g0, params, lut)
        elapsed = time.perf_counter() - t0

        print(
            f"[coupling] sigma={sigma:.1f}  k0={k0:>3}  -> path={row['path']}"
            f"  C_range={row['C_range']:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )
        rows.append({"sigma": sigma, "k0": k0, **row})

    return rows


def _find_k0_crit(rows: list[dict[str, Any]]) -> Optional[float]:
    """Return midpoint between last Path B k0 and first Path A k0, or None if ambiguous.

    Returns None if no transition is found (all B or all A, or non-monotone).
    """
    last_b: Optional[int] = None
    first_a: Optional[int] = None
    for r in sorted(rows, key=lambda x: int(x["k0"])):
        k = int(r["k0"])
        p = str(r["path"])
        if p == "B":
            last_b = k
        elif p == "A" and first_a is None:
            first_a = k

    if last_b is None or first_a is None:
        return None
    return 0.5 * (last_b + first_a)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _fmt(x: float, decimals: int = 4) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{decimals}f}"


def _phase0_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'seed':>5}  {'dominant_k':>11}  {'power_low_frac':>14}  "
        f"{'power_high_frac':>15}  {'known_path':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['seed']:>5}  {r['dominant_k']:>11}  "
            f"{_fmt(r['power_low_frac']):>14}  {_fmt(r['power_high_frac']):>15}  "
            f"{r['known_path']:>10}"
        )
    return "\n".join(lines)


def _phase1_tables(all_rows: list[list[dict[str, Any]]]) -> str:
    sections: list[str] = []
    invariants: list[str] = []

    for rows in all_rows:
        if not rows:
            continue
        sigma = float(rows[0]["sigma"])
        k0_crit = _find_k0_crit(rows)

        header = (
            f"{'k0':>4}  {'path':>5}  {'C_range':>8}  {'kappa':>10}  "
            f"{'spec':>6}  {'growth':>7}  {'outcome':>9}"
        )
        sep = "-" * len(header)
        table_lines = [header, sep]
        for r in sorted(rows, key=lambda x: int(x["k0"])):
            table_lines.append(
                f"{r['k0']:>4}  {r['path']:>5}  {_fmt(r['C_range']):>8}  "
                f"{_fmt(r['kappa']):>10}  {_fmt(r['spectral_ratio']):>6}  "
                f"{_fmt(r['growth']):>7}  {r['outcome']:>9}"
            )

        crit_str = f"{k0_crit:.1f}" if k0_crit is not None else "none"
        lam_str = f"{float('nan'):.4f}" if k0_crit is None else _fmt(10.0 / k0_crit)
        prod_str = (
            _fmt(k0_crit * sigma) if k0_crit is not None else "nan"
        )

        sections.append(
            f"sigma={sigma:.1f}  k0_crit={crit_str}"
            f"  lambda_crit={lam_str}  k0_crit*sigma={prod_str}\n"
            + "\n".join(table_lines)
        )
        invariants.append(f"  sigma={sigma:.1f}  k0_crit={crit_str}  k0_crit*sigma={prod_str}")

    return "\n\n".join(sections) + "\n\nInvariant check:\n" + "\n".join(invariants)


def build_report(
    rows_phase0: list[dict[str, Any]],
    rows_phase1_all: list[list[dict[str, Any]]],
    *,
    quick: bool,
    ts: str,
) -> str:
    header = "\n".join([
        "Thread 7 coupling scale experiment (n=4)",
        f"Run timestamp: {ts}",
        f"quick_mode: {quick}",
        f"basin_n: {BASIN_N}",
        f"sigma_values: {SIGMA_VALUES}",
        f"A_fixed: {A_FIXED}",
        f"k0_threshold_reference: {K0_THRESHOLD}  (last Path B at sigma=0.5 from bimodal basin)",
        f"baseline_params: {BASELINE_PARAMS}",
        "",
        "Hypothesis: basin discriminator is the coupling kernel resolution scale.",
        "  sigma enters dynamics as scalar prefactors only:",
        "    brake: zeta_cubic = 9*gamma^2 / (pi^(3/2) * sigma^3)",
        "    MFE:   expansion_coef = 6*xi_g*gamma / (pi * sigma^2)",
        "  G_sigma does NOT appear as a spatial filter in the IVP.",
        "  Prediction: k0_crit * sigma = const (approx 3.75 at sigma=0.5).",
        "",
    ])

    phase0_section = "\n".join([
        f"=== Phase 0: seed spectral analysis  (N={BASIN_N}  A_fixed={A_FIXED})  ===",
        "",
        "Power split relative to k0_threshold=7 (last Path B wavenumber at sigma=0.5).",
        "power_low_frac: fraction of non-DC power at k=1..7",
        "power_high_frac: fraction at k=8..Nyquist",
        "",
        _phase0_table(rows_phase0),
        "",
    ])

    phase1_section = "\n".join([
        "=== Phase 1: sigma sweep  (sinusoidal IC, k0=1..K0_MAX) ===",
        "",
        _phase1_tables(rows_phase1_all),
    ])

    return "\n".join([header, phase0_section, phase1_section]) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_coupling_scale_experiment(
    *,
    quick: bool = False,
    write_disk: bool = True,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Phase 0 (spectral analysis) and Phase 1 (sigma sweep)."""
    root = Path(__file__).resolve().parents[2]
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR
    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    k0_max = K0_MAX_QUICK if quick else K0_MAX

    # Phase 0
    base_params = _build_params(sigma=0.5, quick=quick)
    N = int(base_params["N"])
    print(f"[coupling] Phase 0: seed spectral analysis  N={N}", flush=True)
    rows_phase0 = run_seed_spectral_analysis(N)
    for r in rows_phase0:
        print(
            f"  seed={r['seed']}  dominant_k={r['dominant_k']:>3}"
            f"  low={r['power_low_frac']:.3f}  high={r['power_high_frac']:.3f}"
            f"  known_path={r['known_path']}",
            flush=True,
        )

    # Phase 1
    rows_phase1_all: list[list[dict[str, Any]]] = []
    for sigma in SIGMA_VALUES:
        print(f"\n[coupling] Phase 1: sigma={sigma:.1f}  k0_max={k0_max}", flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            rows = run_sigma_wavenumber_sweep(sigma, quick=quick, k0_max=k0_max)
        rows_phase1_all.append(rows)
        k0_crit = _find_k0_crit(rows)
        prod = _fmt(k0_crit * sigma) if k0_crit is not None else "none"
        print(f"[coupling] sigma={sigma:.1f}  k0_crit={k0_crit}  k0_crit*sigma={prod}", flush=True)

    report = build_report(rows_phase0, rows_phase1_all, quick=quick, ts=ts)
    print("\n" + report, flush=True)

    if write_disk:
        out_path = out_dir / "coupling_scale_report.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(root)}", flush=True)

    return {
        "rows_phase0": rows_phase0,
        "rows_phase1_all": rows_phase1_all,
        "report_text": report,
    }


def main() -> None:
    warnings.filterwarnings("ignore", message=r".*ReconstructionLUT.*", category=UserWarning)
    parser = argparse.ArgumentParser(
        description="Thread 7 coupling scale experiment (seed spectral analysis + sigma sweep)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"N={QUICK['N']}, short t_span, k0_max={K0_MAX_QUICK}",
    )
    args = parser.parse_args()
    run_coupling_scale_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
