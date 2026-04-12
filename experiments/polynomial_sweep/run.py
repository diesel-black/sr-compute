"""Thread 7 sweep driver: run_simulation at each n, collect four measurements, optional save."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
import warnings
from pathlib import Path
from queue import Empty
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
    INTEGRATION_OVERRIDES_BY_N,
    METASTABLE_PSI_RANGE,
    NONLOCAL,
    N_VALUES,
    QUICK,
    RECONSTRUCTION_LUT,
    RESULTS_DIR,
    SWEEP_DEFAULT_WALLCLOCK_SEC,
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
    if "rtol" in integ:
        params["rtol"] = float(integ["rtol"])
    if "atol" in integ:
        params["atol"] = float(integ["atol"])
    return params


def _sweep_child_run_simulation(queue: mp.Queue, params: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Spawn target: run ``run_simulation`` and return result or traceback via ``queue``."""
    try:
        from models.dim_1plus1.mfe import run_simulation as _rs

        out = _rs(params, **kwargs)
        queue.put(("ok", out))
    except Exception:
        import traceback

        queue.put(("err", traceback.format_exc()))


def _wallclock_timeout_snapshot(params: Params) -> dict[str, Any]:
    """Placeholder IVP output after subprocess termination (no partial SciPy state available)."""
    n_grid = int(params["N"])
    length = float(params["L"])
    x = np.linspace(0.0, length, n_grid, endpoint=False, dtype=float)
    nan_f = np.full(n_grid, np.nan, dtype=float)
    return {
        "t": np.array([], dtype=float),
        "C_history": np.zeros((0, n_grid), dtype=float),
        "g_history": np.zeros((0, n_grid), dtype=float),
        "success": False,
        "message": "Wallclock timeout (subprocess terminated)",
        "t_events": [],
        "C_final": nan_f.copy(),
        "g_final": nan_f.copy(),
        "psi_bar_final": nan_f.copy(),
        "x": x,
    }


def _run_simulation_wallclock(
    max_wallclock: float,
    params: Params,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run ``run_simulation`` in a spawn child; kill after ``max_wallclock`` seconds if still alive."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_sweep_child_run_simulation, args=(q, params, call_kwargs))
    proc.start()
    proc.join(timeout=float(max_wallclock))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        return _wallclock_timeout_snapshot(params)

    try:
        kind, payload = q.get(timeout=2.0)
    except Empty:
        return {
            **_wallclock_timeout_snapshot(params),
            "message": "Subprocess finished without returning a result (queue empty).",
        }

    if kind == "err":
        warnings.warn(
            f"run_simulation subprocess failed:\n{payload}",
            UserWarning,
            stacklevel=2,
        )
        return {
            **_wallclock_timeout_snapshot(params),
            "message": f"Subprocess error: {payload[:500]}",
        }
    return payload


def _infer_hit_blowup(g_final: np.ndarray) -> bool:
    """True if final metric suggests the MFE ceiling event (g_max crossing ~1e6)."""
    g_max = float(np.max(g_final))
    if not math.isfinite(g_max):
        return False
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
    reconstruction_lut: Optional[Mapping[str, Any]] = None,
    max_wallclock: Optional[float] = None,
) -> dict[str, Any]:
    """One simulation at polynomial order ``n`` and four measurements on the final fields.

    ``max_wallclock``: if positive, run ``run_simulation`` in a spawn subprocess and terminate
    after that many seconds (no partial SciPy state; fields become NaN on timeout).
    """
    integ = {
        **INTEGRATION,
        **INTEGRATION_OVERRIDES_BY_N.get(int(n), {}),
        **(integration or {}),
    }
    lut_cfg = {**RECONSTRUCTION_LUT, **(reconstruction_lut or {})}
    params = build_params(n, grid=grid, integration=integ)
    seed_eff = int(seed) if seed is not None else int(params["seed"])
    t_span = (float(params["t_span"][0]), float(params["t_span"][1]))
    method = str(params["method"])
    max_step = float(params["max_step"])

    call_kwargs: dict[str, Any] = {
        "t_span": t_span,
        "seed": seed_eff,
        "method": method,
        "max_step": max_step,
        "lut_C_min": float(lut_cfg["C_min"]),
        "lut_C_max": float(lut_cfg["C_max"]),
        "lut_n_samples": int(lut_cfg["n_samples"]),
    }
    if "rtol" in integ:
        call_kwargs["rtol"] = float(integ["rtol"])
    if "atol" in integ:
        call_kwargs["atol"] = float(integ["atol"])

    if max_wallclock is not None and max_wallclock > 0:
        sim = _run_simulation_wallclock(float(max_wallclock), params, call_kwargs)
    else:
        sim = run_simulation(params, **call_kwargs)

    success = bool(sim["success"])
    message = str(sim["message"])
    if not success:
        warnings.warn(
            f"[n={n}] Simulation reported success=False, message={message!r}; "
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
    reconstruction_lut: Optional[Mapping[str, Any]] = None,
    max_wallclock: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Run ``run_single`` for each ``n`` in ``n_values``; optionally persist npz + json.

    Pass ``max_wallclock=None`` (default) to run in-process. The CLI uses
    ``SWEEP_DEFAULT_WALLCLOCK_SEC`` when set, or ``--wallclock SEC``; ``--quick`` forces in-process.

    The ``integration`` mapping should list only *overrides* (e.g. ``t_span``, ``max_step``), not a
    full copy of ``INTEGRATION``, or per-``n`` defaults from ``INTEGRATION_OVERRIDES_BY_N`` are lost.
    """
    ns = list(N_VALUES if n_values is None else n_values)
    out_dir = Path(results_dir or RESULTS_DIR)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    for n in ns:
        print(f"[n={n}] Starting simulation...", flush=True)
        row = run_single(
            n,
            grid=grid,
            integration=integration,
            seed=seed,
            reconstruction_lut=reconstruction_lut,
            max_wallclock=max_wallclock,
        )
        results.append(row)
        mc = row["measurements"]["metastable_count"]
        print(
            f"[n={n}] Done. t_final={row['t_final']}, blowup={row['hit_blowup']}, "
            f"metastable_count={mc}",
            flush=True,
        )
        if "Wallclock timeout" in str(row["message"]):
            print(
                f"[n={n}] Hint: integrator hit wallclock; use --wallclock 0 to retry this n, "
                "or check INTEGRATION_OVERRIDES_BY_N / Radau settings.",
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
    parser.add_argument(
        "--wallclock",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Max wall seconds per n via spawn subprocess (0 = in-process unlimited). "
            "Omitted: use SWEEP_DEFAULT_WALLCLOCK_SEC from config (None = always in-process). "
            "Example: --wallclock 900"
        ),
    )
    args = parser.parse_args()

    grid_cfg = QUICK if args.quick else GRID
    # Do not pass ``{**INTEGRATION}`` here: ``run_single`` merges ``(integration or {})`` *last*, so a
    # full INTEGRATION dict would overwrite ``INTEGRATION_OVERRIDES_BY_N`` (e.g. Radau for n>=4).
    if args.quick:
        integration_cfg: Optional[dict[str, Any]] = {
            "t_span": QUICK["t_span"],
            "max_step": QUICK["max_step"],
        }
    else:
        integration_cfg = None

    if args.quick:
        wallclock_sec: Optional[float] = None
    elif args.wallclock is not None:
        wallclock_sec = None if args.wallclock <= 0 else float(args.wallclock)
    elif SWEEP_DEFAULT_WALLCLOCK_SEC is not None and SWEEP_DEFAULT_WALLCLOCK_SEC > 0:
        wallclock_sec = float(SWEEP_DEFAULT_WALLCLOCK_SEC)
    else:
        wallclock_sec = None

    run_sweep(
        n_values=args.n or N_VALUES,
        grid={"N": grid_cfg["N"], "L": grid_cfg["L"]},
        integration=integration_cfg,
        save=not args.no_save,
        max_wallclock=wallclock_sec,
    )
