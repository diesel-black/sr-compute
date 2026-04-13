"""Solver-parity experiment: disentangle implicit vs explicit integration for Thread 7 sweep.

Runs A through D call ``run_single`` with explicit ``integration`` overrides so
``INTEGRATION_OVERRIDES_BY_N`` does not dictate the solver when we override it
(last-merge wins inside ``run_single``).

Do not rely on this module from library code; it is an experiment driver only.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from experiments.polynomial_sweep.config import RECONSTRUCTION_LUT, RESULTS_DIR
from experiments.polynomial_sweep.run import _to_jsonable, run_single

# Radau tolerances aligned with high-n sweep overrides (config INTEGRATION_OVERRIDES_BY_N).
_RADAU_TOLS: dict[str, float] = {"rtol": 1e-6, "atol": 1e-9}

# Wall-clock cap for n=5 RK45 (Run C): spawn subprocess cannot recover partial SciPy state on kill.
_RUN_C_WALLCLOCK_SEC = 300.0

_PARITY_RUNS: tuple[dict[str, Any], ...] = (
    {
        "parity_run": "A",
        "label": "n=3 Radau",
        "n": 3,
        "integration": {
            "t_span": (0.0, 30.0),
            "method": "Radau",
            "max_step": 2.0,
            **_RADAU_TOLS,
        },
        "max_wallclock": None,
    },
    {
        "parity_run": "B",
        "label": "n=3 RK45",
        "n": 3,
        "integration": {
            "t_span": (0.0, 30.0),
            "method": "RK45",
            "max_step": 0.1,
        },
        "max_wallclock": None,
    },
    {
        "parity_run": "C",
        "label": "n=5 RK45",
        "n": 5,
        "integration": {
            "t_span": (0.0, 30.0),
            "method": "RK45",
            "max_step": 0.01,
        },
        "max_wallclock": _RUN_C_WALLCLOCK_SEC,
    },
    {
        "parity_run": "D",
        "label": "n=4 RK45",
        "n": 4,
        "integration": {
            "t_span": (0.0, 30.0),
            "method": "RK45",
            "max_step": 0.1,
        },
        "max_wallclock": None,
    },
)


def _repo_root() -> Path:
    """Repository root (parent of ``experiments/``)."""
    return Path(__file__).resolve().parents[2]


def _first_eta(nonlocal_growth: Optional[Mapping[str, Any]]) -> Optional[float]:
    """First eta(f) at unit coarsening, or None if missing or skipped."""
    if not nonlocal_growth or nonlocal_growth.get("skipped"):
        return None
    etas = nonlocal_growth.get("eta_values") or []
    if not etas:
        return None
    v = etas[0]
    if v is None:
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _save_parity_row(
    out_dir: Path,
    row: dict[str, Any],
    fields: Mapping[str, np.ndarray],
) -> None:
    """Write ``parity_{tag}.npz`` and ``parity_{tag}.json``."""
    tag = str(row["parity_run"])
    stem = f"parity_{tag}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        C_final=fields["C_final"],
        g_final=fields["g_final"],
        psi_bar_final=fields["psi_bar_final"],
        x=fields["x"],
    )
    record = {
        "parity_run": row["parity_run"],
        "parity_label": row.get("label", ""),
        "n": row["n"],
        "solver": row["params"].get("method"),
        "t_final": row["t_final"],
        "outcome": row["outcome"],
        "message": row["message"],
        "hit_blowup": row["hit_blowup"],
        "measurements": _to_jsonable(row["measurements"]),
        "params": _to_jsonable(row["params"]),
        "fields_file": f"{stem}.npz",
    }
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, allow_nan=False)


def _fmt_cli_num(x: float, *, decimals: int = 2) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{decimals}f}"


def _fmt_metric_cell(val: Any, *, decimals: int = 3) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and not math.isfinite(val):
        return ""
    return _fmt_cli_num(float(val), decimals=decimals)


def _print_parity_progress_line(
    step_i: int,
    total_steps: int,
    tag: str,
    label: str,
    row: dict[str, Any],
    elapsed_s: float,
    *,
    wallclock_limited: bool,
) -> None:
    w = len(str(total_steps))
    counter = f"({step_i:>{w}}/{total_steps})"
    head = f"{tag}: {label}"
    outcome = str(row["outcome"])
    if outcome == "timeout":
        if wallclock_limited:
            print(
                f"[parity] {counter} {head:<18}  -> timeout   (wallclock limit)  ({elapsed_s:.1f}s wall)",
                flush=True,
            )
        else:
            print(
                f"[parity] {counter} {head:<18}  -> timeout   ({elapsed_s:.1f}s wall)",
                flush=True,
            )
        return
    m = row.get("measurements") or {}
    kappa = m.get("condition_number")
    spec = m.get("spectral_ratio")
    eta = _first_eta(m.get("nonlocal_growth"))
    kappa_s = _fmt_metric_cell(kappa, decimals=2)
    spec_s = _fmt_metric_cell(spec, decimals=3)
    eta_s = _fmt_metric_cell(eta, decimals=1) if eta is not None else ""
    parts = [
        f"[parity] {counter} {head:<18}  -> {outcome:<10}",
        f"t={_fmt_cli_num(float(row['t_final']), decimals=3)}",
    ]
    if kappa_s:
        parts.append(f"kappa={kappa_s}")
    if spec_s:
        parts.append(f"spec={spec_s}")
    if eta_s:
        parts.append(f"eta={eta_s}")
    parts.append(f"({elapsed_s:.1f}s)")
    print("  ".join(parts), flush=True)


def _print_parity_summary(rows: list[dict[str, Any]]) -> None:
    """Row-per-run summary (same information as analysis parity section, readable layout)."""
    print("Parity Experiment Summary", flush=True)
    sep = " | "
    headers = ("Run", "Label", "solver", "t_final", "outcome", "kappa", "spectral", "eta")
    print(sep.join(f"{h:<14}" for h in headers))
    print(sep.join("-" * 14 for _ in headers))
    for r in rows:
        m = r.get("measurements") or {}
        kappa = m.get("condition_number")
        spec = m.get("spectral_ratio")
        eta = _first_eta(m.get("nonlocal_growth"))
        tf = float(r["t_final"])
        t_cell = _fmt_metric_cell(tf, decimals=6) if math.isfinite(tf) else ""
        k_cell = _fmt_metric_cell(kappa, decimals=4) if kappa is not None else ""
        s_cell = _fmt_metric_cell(spec, decimals=4) if spec is not None else ""
        e_cell = _fmt_metric_cell(eta, decimals=4) if eta is not None else ""
        print(
            sep.join(
                [
                    f"{str(r['parity_run']):<14}",
                    f"{str(r['label']):<14}",
                    f"{str(r['params'].get('method', '')):<14}",
                    f"{t_cell:<14}",
                    f"{str(r['outcome']):<14}",
                    f"{k_cell:<14}",
                    f"{s_cell:<14}",
                    f"{e_cell:<14}",
                ]
            ),
            flush=True,
        )


def run_parity_experiment(
    results_subdir: str = "parity",
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Execute runs A through D, save under ``RESULTS_DIR / results_subdir``, return row dicts."""
    root = _repo_root()
    out_dir = root / RESULTS_DIR / results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    total_steps = len(_PARITY_RUNS)

    for step_i, spec in enumerate(_PARITY_RUNS, start=1):
        tag = spec["parity_run"]
        n = int(spec["n"])
        label = str(spec["label"])
        wall = spec.get("max_wallclock")
        wall_f = float(wall) if wall is not None else None
        t0 = time.perf_counter()
        row = run_single(
            n,
            integration=dict(spec["integration"]),
            seed=seed,
            reconstruction_lut=RECONSTRUCTION_LUT,
            max_wallclock=wall_f,
        )
        elapsed = time.perf_counter() - t0
        flat = {
            "parity_run": tag,
            "label": label,
            "n": n,
            "t_final": row["t_final"],
            "outcome": row["outcome"],
            "message": row["message"],
            "hit_blowup": row["hit_blowup"],
            "measurements": row["measurements"],
            "params": row["params"],
        }
        fields = row["fields"]
        _save_parity_row(out_dir, flat, fields)
        rows_out.append(flat)
        _print_parity_progress_line(
            step_i,
            total_steps,
            str(tag),
            label,
            row,
            elapsed,
            wallclock_limited=wall_f is not None,
        )

    _print_parity_summary(rows_out)
    return rows_out


def main() -> None:
    warnings.filterwarnings("ignore", message=r".*ReconstructionLUT.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"\[n=\d+\] Simulation reported success=False, message=.*; using final state for measurements\.",
        category=UserWarning,
    )
    run_parity_experiment()


if __name__ == "__main__":
    main()
