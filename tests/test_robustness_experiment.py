"""Quick-mode checks for ``experiments.polynomial_sweep.robustness_experiment``."""

from __future__ import annotations

import math

from experiments.polynomial_sweep import robustness_experiment as re


def test_robustness_experiment_quick_rows_and_finite_measurements():
    rows = re.run_robustness_experiment(quick=True, write_disk=False)

    expected_labels = ("baseline",) + tuple(lab for lab, _ in re.QUICK_PERTURBATIONS)
    expected_pairs = {(lab, n) for lab in expected_labels for n in re.ROBUSTNESS_NS}
    actual_pairs = {(str(r["perturbation_label"]), int(r["n"])) for r in rows}
    assert actual_pairs == expected_pairs
    assert len(rows) == len(expected_pairs)

    for r in rows:
        n = int(r["n"])
        assert n in re.ROBUSTNESS_NS
        assert math.isfinite(float(r["t_final"]))
        assert isinstance(r["success"], bool)
        assert math.isfinite(float(r["C_range"]))
        assert isinstance(r["metastable_count"], int)
        for key in ("kappa", "spectral_ratio", "eta_f1", "growth"):
            assert math.isfinite(float(r[key])), f"n={n} {key} not finite: {r}"
        assert not math.isinf(float(r["growth"]))
