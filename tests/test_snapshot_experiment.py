"""Quick-mode checks for ``experiments.polynomial_sweep.snapshot_experiment``."""

from __future__ import annotations

import math

from experiments.polynomial_sweep import snapshot_experiment as se


def test_snapshot_experiment_quick_shape_and_finite():
    rows = se.run_snapshot_experiment(quick=True, write_disk=False)
    assert len(rows) == len(se.SNAPSHOT_NS) * len(se.SNAPSHOT_TARGET_TIMES)
    assert {r["n"] for r in rows} == set(se.SNAPSHOT_NS)

    for n in se.SNAPSHOT_NS:
        targets = sorted({r["t_target"] for r in rows if r["n"] == n})
        assert targets == list(se.SNAPSHOT_TARGET_TIMES)

    for r in rows:
        assert math.isfinite(r["t_snapshot"])
        assert math.isfinite(r["C_range"])
        assert math.isfinite(r["g_range"])
        assert isinstance(r["metastable_count"], int)
        for key in ("kappa", "spectral_ratio", "eta_at_f1"):
            assert math.isfinite(r[key]), f"{key} not finite: {r}"
        gr = r["growth_rate"]
        assert not math.isinf(gr), f"growth_rate infinite: {r}"
