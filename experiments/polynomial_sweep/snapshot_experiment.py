"""Intermediate-time snapshot experiment for n = 4, 5, 6 (control: n = 4).

Evolves the coupled 1+1 system with the same per-n integration settings as the main sweep,
records fields along a dense time grid, then evaluates the four Thread 7 measurements and
simple spatial ranges at target times (or the final time if the run ends earlier).

Run from repository root::

    python -m experiments.polynomial_sweep.snapshot_experiment
    python -m experiments.polynomial_sweep.snapshot_experiment --quick

Default report path: ``experiments/polynomial_sweep/results/snapshot_report.txt`` (alongside ``analysis.txt``).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

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
from shared.reconstruction import ReconstructionLUT

# Polynomial orders for this diagnostic (n = 4 is matched-time control vs n = 5, 6).
SNAPSHOT_NS: tuple[int, ...] = (4, 5, 6)

# Target lab times; each maps to the nearest stored time, or the terminal frame if t_target > t_final.
SNAPSHOT_TARGET_TIMES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_params_for_snapshot(n: int, *, quick: bool) -> dict[str, Any]:
    """Baseline SR params plus grid and integration, mirroring ``run_single`` merge order."""
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


def _build_t_eval(t_span: tuple[float, float], *, quick: bool) -> np.ndarray:
    """Dense output times so nearest-time lookup is accurate; always includes target lab times in range."""
    t0, t1 = float(t_span[0]), float(t_span[1])
    n_lin = 120 if quick else 600
    lin = np.linspace(t0, t1, num=max(2, n_lin), dtype=float)
    extras = [t for t in SNAPSHOT_TARGET_TIMES if t0 < t <= t1]
    merged = np.sort(np.unique(np.concatenate([lin, np.asarray(extras, dtype=float)])))
    return merged


def nearest_history_index(t_hist: np.ndarray, t_target: float) -> tuple[int, float]:
    """Index into history for a lab target time; use the last frame if the run ended before t_target."""
    if t_hist.size == 0:
        raise ValueError("empty time history")
    t_final = float(t_hist[-1])
    if t_target > t_final:
        return t_hist.size - 1, t_final
    idx = int(np.argmin(np.abs(t_hist - float(t_target))))
    return idx, float(t_hist[idx])


@dataclass(frozen=True)
class SnapshotRow:
    n: int
    t_target: float
    t_snapshot: float
    C_range: float
    g_range: float
    metastable_count: int
    kappa: float
    spectral_ratio: float
    eta_at_f1: float
    growth_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "t_target": self.t_target,
            "t_snapshot": self.t_snapshot,
            "C_range": self.C_range,
            "g_range": self.g_range,
            "metastable_count": self.metastable_count,
            "kappa": self.kappa,
            "spectral_ratio": self.spectral_ratio,
            "eta_at_f1": self.eta_at_f1,
            "growth_rate": self.growth_rate,
        }


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


def _measure_snapshot_fields(
    psi_bar: np.ndarray,
    params: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Returns kappa, spectral_ratio, eta_at_f1, growth_rate (first coarsening growth).

    ``nonlocal_correction_growth`` uses ``psi_bar`` and grid spacing only; ``g`` is not required.
    """
    n = int(params["n"])
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    dx = float(params["dx"])

    ic = interpretive_condition_number(psi_bar, n)
    kappa = float(ic["kappa"])

    sc = spectral_concentration_ratio(psi_bar, n, gamma, sigma, dx)
    spectral_ratio = float(sc["ratio"])

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

    if not math.isfinite(kappa):
        kappa = float("nan")
    if not math.isfinite(spectral_ratio):
        spectral_ratio = float("nan")
    if not math.isfinite(eta1):
        eta1 = float("nan")
    if not math.isfinite(gr0):
        gr0 = float("nan")

    return kappa, spectral_ratio, eta1, gr0


def run_snapshot_for_n(
    n: int,
    *,
    quick: bool,
    lut_cfg: Optional[dict[str, Any]] = None,
) -> tuple[list[SnapshotRow], dict[str, Any]]:
    """Single polynomial order: integrate, then one row per target lab time."""
    params = build_params_for_snapshot(n, quick=quick)
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))
    t_eval = _build_t_eval(t_span, quick=quick)
    lut_opts = {**RECONSTRUCTION_LUT, **(lut_cfg or {})}

    sim_kwargs: dict[str, Any] = {
        "t_span": t_span,
        "t_eval": t_eval,
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
    t_hist = np.asarray(sim["t"], dtype=float)
    c_hist = np.asarray(sim["C_history"], dtype=float)
    g_hist = np.asarray(sim["g_history"], dtype=float)

    meta = {
        "n": n,
        "success": bool(sim["success"]),
        "message": str(sim["message"]),
        "t_final": float(t_hist[-1]) if t_hist.size else float("nan"),
        "method": params["method"],
        "t_span": list(t_span),
    }

    if t_hist.size == 0:
        return [], meta

    lut = ReconstructionLUT(
        int(params["n"]),
        float(params["gamma"]),
        float(lut_opts["C_min"]),
        float(lut_opts["C_max"]),
        n_samples=int(lut_opts["n_samples"]),
    )
    m_count = _metastable_count_for_n(params)

    rows: list[SnapshotRow] = []
    for t_target in SNAPSHOT_TARGET_TIMES:
        idx, t_snap = nearest_history_index(t_hist, float(t_target))
        c_row = c_hist[idx]
        g_row = g_hist[idx]
        c_range = float(np.max(c_row) - np.min(c_row))
        g_range = float(np.max(g_row) - np.min(g_row))
        psi_bar = np.asarray(lut(c_row), dtype=float)
        kappa, spec_r, eta1, gr = _measure_snapshot_fields(psi_bar, params)
        rows.append(
            SnapshotRow(
                n=n,
                t_target=float(t_target),
                t_snapshot=t_snap,
                C_range=c_range,
                g_range=g_range,
                metastable_count=m_count,
                kappa=kappa,
                spectral_ratio=spec_r,
                eta_at_f1=eta1,
                growth_rate=gr,
            )
        )

    return rows, meta


_SNAPSHOT_TABLE_SEP = " | "
_SNAPSHOT_TABLE_HEADERS = (
    "n",
    "t_tgt",
    "t_snap",
    "C_range",
    "g_range",
    "meta",
    "kappa",
    "spec",
    "eta_f1",
    "growth",
)


def _pretty_float(x: float) -> str:
    """Format a float for text tables: fixed-point when readable, scientific only when needed.

    Scientific form is used for very large magnitude (>= 1e7) or very small non-zero (< 1e-4).
    Otherwise use fixed decimals with trailing zeros stripped. NaN and inf are literal strings.
    """
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


def _snapshot_row_cells(r: SnapshotRow) -> list[str]:
    """Unpadded cell strings for one data row (same order as ``_SNAPSHOT_TABLE_HEADERS``)."""
    return [
        str(r.n),
        f"{r.t_target:.2f}",
        f"{r.t_snapshot:.4f}",
        _pretty_float(r.C_range),
        _pretty_float(r.g_range),
        str(r.metastable_count),
        _pretty_float(r.kappa),
        _pretty_float(r.spectral_ratio),
        _pretty_float(r.eta_at_f1),
        _pretty_float(r.growth_rate),
    ]


def _format_table(rows: Sequence[SnapshotRow]) -> str:
    """Build a pipe table: adaptive number formatting, columns padded to a common width."""
    sep = _SNAPSHOT_TABLE_SEP
    header_cells = list(_SNAPSHOT_TABLE_HEADERS)
    matrix: list[list[str]] = [header_cells] + [_snapshot_row_cells(r) for r in rows]
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


def run_snapshot_experiment(
    *,
    quick: bool = False,
    write_disk: bool = True,
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Run n = 4, 5, 6 snapshots; optionally write ``snapshot_report.txt`` under ``results/``.

    Returns flat list of row dicts (for tests and scripting).
    """
    root = _repo_root()
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR
    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_rows: list[SnapshotRow] = []
    run_metas: list[dict[str, Any]] = []

    for n in SNAPSHOT_NS:
        rows, meta = run_snapshot_for_n(n, quick=quick)
        all_rows.extend(rows)
        run_metas.append(meta)

    header_lines = [
        "Thread 7 snapshot experiment (intermediate-time fields)",
        f"Run timestamp: {ts}",
        f"quick_mode: {quick}",
        f"baseline_params: {BASELINE_PARAMS}",
        f"polynomial_orders: {list(SNAPSHOT_NS)}",
        f"lab_target_times: {list(SNAPSHOT_TARGET_TIMES)}",
        "",
        "Per-n integration outcome:",
    ]
    for m in run_metas:
        header_lines.append(
            f"  n={m['n']}: success={m['success']}, t_final={m['t_final']}, "
            f"method={m['method']}, message={m['message']!r}"
        )
    header_lines.extend(["", "Columns: meta = metastable_count (landscape-only, constant in n).", ""])

    body = _format_table(all_rows)
    full_text = "\n".join(header_lines) + body + "\n"

    print(full_text)
    if write_disk:
        out_path = out_dir / "snapshot_report.txt"
        out_path.write_text(full_text, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(root)}", flush=True)

    return [r.as_dict() for r in all_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Intermediate-time snapshot experiment (n=4,5,6)")
    parser.add_argument("--quick", action="store_true", help="N=32, short t_span (for tests / smoke)")
    args = parser.parse_args()
    run_snapshot_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
