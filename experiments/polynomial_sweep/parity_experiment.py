"""Solver-parity experiment: disentangle implicit vs explicit integration for Thread 7 sweep.

Runs A through D call ``run_single`` with explicit ``integration`` overrides so
``INTEGRATION_OVERRIDES_BY_N`` does not dictate the solver when we override it
(last-merge wins inside ``run_single``).

Do not rely on this module from library code; it is an experiment driver only.
"""

from __future__ import annotations

import json
import math
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


def _format_cell(x: Any, width: int, fmt: str = ".4g") -> str:
    if x is None:
        s = ""
    elif isinstance(x, float):
        if not math.isfinite(x):
            s = str(x)
        else:
            s = format(x, fmt)
    else:
        s = str(x)
    return s[:width].ljust(width)


def _print_parity_table(by_label: dict[str, dict[str, Any]]) -> None:
    """Print the four-column comparison (column order fixed)."""
    cols = ("n=3 Radau", "n=3 RK45", "n=5 RK45", "n=4 RK45")
    rows_spec = (
        ("t_final", "t_final", ".6g"),
        ("kappa", "condition_number", ".4g"),
        ("spectral", "spectral_ratio", ".4g"),
        ("eta", "_eta_first", ".4g"),
    )
    w_label = 9
    w_col = 14
    sep = " | "

    print("Parity Experiment Results")
    print("=" * 57)
    head = _format_cell("", w_label) + sep + sep.join(_format_cell(c, w_col) for c in cols)
    print(head)
    for row_name, key, fmt in rows_spec:
        cells = []
        for lab in cols:
            r = by_label.get(lab, {})
            if key == "_eta_first":
                m = r.get("measurements") or {}
                ng = m.get("nonlocal_growth")
                val = _first_eta(ng)
            elif key == "condition_number":
                m = r.get("measurements") or {}
                val = m.get("condition_number")
            else:
                val = r.get(key)
            cells.append(_format_cell(val, w_col, fmt))
        print(_format_cell(row_name, w_label) + sep + sep.join(cells))
    print("=" * 57)


def run_parity_experiment(
    results_subdir: str = "parity",
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Execute runs A through D, save under ``RESULTS_DIR / results_subdir``, return row dicts."""
    root = _repo_root()
    out_dir = root / RESULTS_DIR / results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}

    for spec in _PARITY_RUNS:
        tag = spec["parity_run"]
        n = int(spec["n"])
        label = str(spec["label"])
        print(f"[parity {tag}: {label}] starting...", flush=True)
        wall = spec.get("max_wallclock")
        row = run_single(
            n,
            integration=dict(spec["integration"]),
            seed=seed,
            reconstruction_lut=RECONSTRUCTION_LUT,
            max_wallclock=float(wall) if wall is not None else None,
        )
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
        by_label[label] = flat
        print(
            f"[parity {tag}] done. solver={row['params'].get('method')} t_final={row['t_final']!s} "
            f"outcome={row['outcome']}",
            flush=True,
        )

    _print_parity_table(by_label)
    return rows_out


def main() -> None:
    run_parity_experiment()


if __name__ == "__main__":
    main()
