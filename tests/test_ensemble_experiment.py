"""Quick-mode checks for ``experiments.polynomial_sweep.ensemble_experiment``."""

from __future__ import annotations

import math

from experiments.polynomial_sweep import ensemble_experiment as ee


def _assert_row_measurements(r: dict) -> None:
    assert r["outcome"] in ("completed", "terminal", "timeout")
    if r.get("timeout"):
        assert r["outcome"] == "timeout"
        assert math.isnan(float(r["t_final"]))
        for key in ("C_range", "kappa", "spectral_ratio", "eta_f1", "growth"):
            assert math.isnan(float(r[key])), f"timeout row {key} should be nan: {r}"
        return
    assert r["outcome"] in ("completed", "terminal")
    assert math.isfinite(float(r["t_final"]))
    assert math.isfinite(float(r["C_range"]))
    for key in ("kappa", "spectral_ratio", "eta_f1", "growth"):
        assert math.isfinite(float(r[key])), f"n={r['n']} seed={r['seed']} {key} not finite"


def test_ensemble_experiment_quick_rows_stats_and_finite():
    out = ee.run_ensemble_experiment(quick=True, write_disk=False)
    rows_n4 = out["rows_n4"]
    rows_n3 = out["rows_n3"]
    text = out["report_text"]

    assert len(rows_n4) == len(ee.QUICK_SEEDS)
    assert len(rows_n3) == len(ee.QUICK_SEEDS)
    assert {int(r["seed"]) for r in rows_n4} == set(ee.QUICK_SEEDS)
    assert {int(r["seed"]) for r in rows_n3} == set(ee.QUICK_SEEDS)

    assert "Statistics (n=4" in text
    assert "usable runs (completed + terminal)" in text
    assert "mean=" in text
    assert "Robustness verdict" in text
    assert "wallclock_timeout_s:" in text
    assert "Seed diversity check (n=4)" in text
    assert "Seed diversity check (n=3)" in text
    assert "no successful runs" not in text.lower()

    n4_usable = sum(1 for r in rows_n4 if r["outcome"] in ("completed", "terminal"))
    assert n4_usable > 0
    assert f"N={n4_usable}" in text

    for r in rows_n4 + rows_n3:
        assert int(r["n"]) in (ee.ENSEMBLE_N_PRIMARY, ee.ENSEMBLE_N_CONTROL)
        assert "timeout" in r
        assert "outcome" in r
        _assert_row_measurements(r)

    for r in rows_n3:
        if r["outcome"] == "timeout":
            continue
        assert abs(float(r["kappa"]) - 1.0) < ee.KAPPA_N3_TOL


def test_ensemble_timeout_row_shape():
    """Immediate wallclock forces a timeout row with NaN measurements (Unix timer only)."""
    if not ee._WALLCLOCK_AVAILABLE:
        return
    r = ee.run_one_seed(4, 0, quick=True, wallclock_s=1e-9)
    assert r.get("timeout") is True
    assert r["outcome"] == "timeout"
    assert math.isnan(float(r["t_final"]))
    for key in ("C_range", "kappa", "spectral_ratio", "eta_f1", "growth"):
        assert math.isnan(float(r[key]))
