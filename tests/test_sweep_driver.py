"""End-to-end tests for the Thread 7 polynomial sweep driver."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from experiments.polynomial_sweep.config import QUICK
from experiments.polynomial_sweep.run import run_single, run_sweep


def _expected_keys() -> set[str]:
    return {
        "n",
        "success",
        "message",
        "t_final",
        "hit_blowup",
        "measurements",
        "fields",
        "params",
    }


def _field_keys() -> set[str]:
    return {"C_final", "g_final", "psi_bar_final", "x"}


def test_run_single_quick_n3():
    row = run_single(3, grid=QUICK, integration=QUICK)
    assert set(row.keys()) == _expected_keys()
    assert set(row["fields"].keys()) == _field_keys()
    assert row["fields"]["C_final"].size == QUICK["N"]
    assert row["success"] is True
    m = row["measurements"]
    assert m["metastable_count"] == 2
    assert m["condition_number"] == pytest.approx(1.0)
    assert m["spectral_ratio"] is not None
    assert np.isfinite(m["spectral_ratio"])
    assert m["spectral_ratio"] > 0.0
    nl = m["nonlocal_growth"]
    assert isinstance(nl, dict)
    etas = np.asarray(nl["eta_values"], dtype=float)
    assert np.all(np.isfinite(etas))


def test_run_single_quick_n4():
    row = run_single(4, grid=QUICK, integration=QUICK)
    assert set(row.keys()) == _expected_keys()
    m = row["measurements"]
    assert m["metastable_count"] == 3
    assert m["condition_number"] is not None
    assert m["condition_number"] > 1.0
    assert m["spectral_ratio"] is not None
    assert np.isfinite(m["spectral_ratio"])
    assert isinstance(m["nonlocal_growth"], dict)


def test_run_single_n2_degenerate():
    with pytest.warns(UserWarning, match="n=2"):
        row = run_single(2, grid=QUICK, integration=QUICK)
    assert set(row.keys()) == _expected_keys()
    m = row["measurements"]
    nl = m["nonlocal_growth"]
    assert isinstance(nl, dict)
    assert nl.get("skipped") is True
    assert m["metastable_count"] is not None
    assert m["spectral_ratio"] is not None


def test_run_sweep_quick():
    results = run_sweep(
        n_values=[3, 4],
        grid=QUICK,
        integration=QUICK,
        save=False,
        max_wallclock=None,
    )
    assert len(results) == 2
    c3 = results[0]["measurements"]["metastable_count"]
    c4 = results[1]["measurements"]["metastable_count"]
    assert c3 == 2 and c4 == 3
    assert results[0]["measurements"]["condition_number"] is not None
    assert results[1]["measurements"]["condition_number"] is not None


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        run_sweep(
            n_values=[3],
            grid=QUICK,
            integration=QUICK,
            save=True,
            results_dir=tmp,
            max_wallclock=None,
        )
        d = Path(tmp)
        assert (d / "n3.npz").is_file()
        assert (d / "n3_measurements.json").is_file()
        assert (d / "summary.json").is_file()

        data = np.load(d / "n3.npz")
        for key in ("C_final", "g_final", "psi_bar_final", "x"):
            assert key in data.files

        with (d / "n3_measurements.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("metastable_count", "condition_number", "spectral_ratio", "nonlocal_growth"):
            assert key in meta["measurements"]

        with (d / "summary.json").open(encoding="utf-8") as f:
            summary = json.load(f)
        assert "3" in summary
        assert "metastable_count" in summary["3"]
