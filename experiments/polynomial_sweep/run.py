"""Thread 7 sweep driver: run_simulation at each n, collect four measurements, optional save."""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from models.dim_1plus1.mfe import run_simulation
from shared.metrics import (
    count_metastable_states,
    interpretive_condition_number,
    nonlocal_correction_growth,
    spectral_concentration_ratio,
)

from experiments.polynomial_sweep.config import (
    BASELINE_PARAMS,
    GRID,
    INTEGRATION,
    METASTABLE_PSI_RANGE,
    NONLOCAL,
    N_VALUES,
    QUICK,
    RESULTS_DIR,
)

Params = dict[str, Any]


def build_params(
    n: int,
    grid: Optional[Mapping[str, Any]] = None,
    integration: Optional[Mapping[str, Any]] = None,
) -> Params:
    """Merge baseline SR parameters with polynomial order n, grid, and integration snapshot.

    Returns the dict passed to ``run_simulation`` plus integration keys ``t_span``, ``method``,
    ``max_step``, and ``seed`` so saved metadata round-trips the full numerical setup.
    """
    g = dict(GRID if grid is None else grid)
    integ = {**INTEGRATION, **(integration or {})}
    params: Params = {**BASELINE_PARAMS, "n": int(n)}
    params["N"] = int(g["N"])
    params["L"] = float(g["L"])
    params["dx"] = params["L"] / params["N"]
    params["t_span"] = (float(integ["t_span"][0]), float(integ["t_span"][1]))
    params["method"] = str(integ["method"])
    params["max_step"] = float(integ["max_step"])
    params["seed"] = int(integ["seed"])
    return params


def _infer_hit_blowup(g_final: np.ndarray) -> bool:
    """True if final metric suggests the MFE ceiling event (g_max crossing ~1e6)."""
    g_max = float(np.max(g_final))
    return g_max >= 0.99e6


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable Python natives (NaN/Inf -> None)."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            out[key] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def _collect_measurements(
    n: int,
    params: Params,
    psi_bar_final: np.ndarray,
) -> dict[str, Any]:
    """Run the four Thread 7 measurements; failures become None with stderr-style prints."""
    gamma = float(params["gamma"])
    sigma = float(params["sigma"])
    dx = float(params["dx"])
    mu_sq = float(params["mu_sq"])
    alpha_phi = float(params["alpha_phi"])
    lambda_b = float(params["lambda_B"])

    measurements: dict[str, Any] = {
        "metastable_count": None,
        "condition_number": None,
        "spectral_ratio": None,
        "nonlocal_growth": None,
    }

    try:
        ms = count_metastable_states(
            METASTABLE_PSI_RANGE,
            n,
            gamma,
            mu_sq,
            alpha_phi,
            lambda_b,
            zeta=None,
            sigma=sigma,
            n_quad=512,
            peak_prominence=0.04,
            peak_distance=120,
        )
        measurements["metastable_count"] = int(ms["count"])
    except Exception as exc:  # noqa: BLE001 — sweep must survive a bad measurement
        print(f"[n={n}] metastable_count failed: {exc}", flush=True)

    try:
        ic = interpretive_condition_number(psi_bar_final, n)
        kappa = float(ic["kappa"])
        measurements["condition_number"] = kappa if math.isfinite(kappa) else None
    except Exception as exc:
        print(f"[n={n}] condition_number failed: {exc}", flush=True)

    try:
        sc = spectral_concentration_ratio(psi_bar_final, n, gamma, sigma, dx)
        ratio = float(sc["ratio"])
        measurements["spectral_ratio"] = ratio if math.isfinite(ratio) else None
    except Exception as exc:
        print(f"[n={n}] spectral_ratio failed: {exc}", flush=True)

    try:
        nl = nonlocal_correction_growth(
            psi_bar_final,
            n,
            gamma,
            sigma,
            dx,
            coarsening_factors=NONLOCAL["coarsening_factors"],
        )
        measurements["nonlocal_growth"] = nl
    except Exception as exc:
        print(f"[n={n}] nonlocal_growth failed: {exc}", flush=True)

    return measurements


def run_single(
    n: int,
    grid: Optional[Mapping[str, Any]] = None,
    integration: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """One simulation at polynomial order ``n`` and four measurements on the final fields."""
    integ = {**INTEGRATION, **(integration or {})}
    params = build_params(n, grid=grid, integration=integ)
    seed_eff = int(seed) if seed is not None else int(params["seed"])
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))
    method = str(params["method"])
    max_step = float(params["max_step"])

    sim = run_simulation(
        params,
        t_span=t_span,
        seed=seed_eff,
        method=method,
        max_step=max_step,
    )

    success = bool(sim["success"])
    message = str(sim["message"])
    if not success or sim.get("t_events"):
        warnings.warn(
            f"[n={n}] Simulation reported success={success!r}, message={message!r}; "
            "using final state for measurements.",
            UserWarning,
            stacklevel=1,
        )

    t_arr = np.asarray(sim["t"], dtype=float)
    t_final = float(t_arr[-1]) if t_arr.size else t_span[0]

    C_final = np.asarray(sim["C_final"], dtype=float)
    g_final = np.asarray(sim["g_final"], dtype=float)
    psi_bar_final = np.asarray(sim["psi_bar_final"], dtype=float)
    x = np.asarray(sim["x"], dtype=float)

    measurements = _collect_measurements(n, params, psi_bar_final)

    return {
        "n": int(n),
        "success": success,
        "message": message,
        "t_final": t_final,
        "hit_blowup": _infer_hit_blowup(g_final),
        "measurements": measurements,
        "fields": {
            "C_final": C_final,
            "g_final": g_final,
            "psi_bar_final": psi_bar_final,
            "x": x,
        },
        "params": params,
    }


def run_sweep(
    n_values: Optional[list[int]] = None,
    grid: Optional[Mapping[str, Any]] = None,
    integration: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    save: bool = True,
    results_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Run ``run_single`` for each ``n`` in ``n_values``; optionally persist npz + json."""
    ns = list(N_VALUES if n_values is None else n_values)
    out_dir = Path(results_dir or RESULTS_DIR)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    for n in ns:
        print(f"[n={n}] Starting simulation...", flush=True)
        row = run_single(n, grid=grid, integration=integration, seed=seed)
        results.append(row)
        mc = row["measurements"]["metastable_count"]
        print(
            f"[n={n}] Done. t_final={row['t_final']}, blowup={row['hit_blowup']}, "
            f"metastable_count={mc}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    print(f"Sweep finished in {elapsed:.1f} s ({elapsed / 60.0:.2f} min).", flush=True)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}
        for row in results:
            n = row["n"]
            stem = f"n{n}"
            fields = row["fields"]
            np.savez_compressed(
                out_dir / f"{stem}.npz",
                C_final=fields["C_final"],
                g_final=fields["g_final"],
                psi_bar_final=fields["psi_bar_final"],
                x=fields["x"],
            )
            meta = {
                "n": n,
                "success": row["success"],
                "message": row["message"],
                "t_final": row["t_final"],
                "hit_blowup": row["hit_blowup"],
                "measurements": _to_jsonable(row["measurements"]),
                "params": _to_jsonable(row["params"]),
            }
            with (out_dir / f"{stem}_measurements.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, allow_nan=False)
            summary[str(n)] = _to_jsonable(row["measurements"])
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thread 7 polynomial sweep")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small grid for fast testing",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help="Specific n values to run",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )
    args = parser.parse_args()

    grid_cfg = QUICK if args.quick else GRID
    integration_cfg = {**INTEGRATION}
    if args.quick:
        integration_cfg["t_span"] = QUICK["t_span"]
        integration_cfg["max_step"] = QUICK["max_step"]

    run_sweep(
        n_values=args.n or N_VALUES,
        grid={"N": grid_cfg["N"], "L": grid_cfg["L"]},
        integration=integration_cfg,
        save=not args.no_save,
    )
