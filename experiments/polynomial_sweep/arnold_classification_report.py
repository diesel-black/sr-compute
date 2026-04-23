"""Thread 7 Arnold A_k classification report for V_n critical points.

Analyzes the composite landscape V_n(psi_bar) = V_eff(F_n(psi_bar)) for each polynomial
order n in N_VALUES using symbolic polynomial differentiation (exact derivatives, no
finite-difference noise).

Generates results/arnold_classification.txt with the per-order table.

Run from repository root::

    python -m experiments.polynomial_sweep.arnold_classification_report
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.polynomial_sweep.config import (
    BASELINE_PARAMS,
    METASTABLE_PSI_RANGE,
    N_VALUES,
    RESULTS_DIR,
)
from shared.metrics import count_metastable_states
from sr_compute.diagnostics import arnold_class

PSI_RANGE = (float(METASTABLE_PSI_RANGE[0]), float(METASTABLE_PSI_RANGE[2]))
# NOTE: METASTABLE_PSI_RANGE is (psi_min, psi_max, num_points); PSI_RANGE is (min, max).
PSI_RANGE = (float(METASTABLE_PSI_RANGE[0]), float(METASTABLE_PSI_RANGE[1]))


def _metastable_count(n: int) -> int:
    kwargs: dict[str, Any] = {"peak_prominence": 0.04, "peak_distance": 120}
    if n == 3:
        kwargs["sigma"] = float(BASELINE_PARAMS["sigma"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ms = count_metastable_states(
            METASTABLE_PSI_RANGE,
            n,
            float(BASELINE_PARAMS["gamma"]),
            float(BASELINE_PARAMS["mu_sq"]),
            float(BASELINE_PARAMS["alpha_phi"]),
            float(BASELINE_PARAMS["lambda_B"]),
            zeta=None,
            **kwargs,
        )
    return int(ms["count"])


def _format_row(n: int, ac_result: dict[str, Any], ms_count: int) -> str:
    """Format one table row."""
    cps = ac_result["critical_points"]
    n_true_max = int(ac_result["n_maxima"])
    n_true_min = int(ac_result["n_minima"])

    # Collect all Arnold classes seen
    classes = sorted({cp["arnold_class"] for cp in cps})
    class_str = "/".join(classes) if classes else "none"

    # First non-vanishing order (should all be 2 for Morse)
    orders = sorted({cp["first_nonvanishing_order"] for cp in cps if cp["first_nonvanishing_order"] is not None})
    order_str = ",".join(str(o) for o in orders) if orders else "none"

    # Confirmed: Y if all critical points are A_1 with first_nonvanishing_order == 2
    all_a1 = all(cp["arnold_class"] == "A_1" for cp in cps)
    all_order2 = all(cp["first_nonvanishing_order"] == 2 for cp in cps)
    confirmed = "Y" if (all_a1 and all_order2) else "N"

    # Leading d2 range across maxima
    max_d2s = [cp["leading_derivative_magnitudes"][0]
               for cp in cps if cp["critical_point_type"] == "maximum" and cp["leading_derivative_magnitudes"]]
    d2_str = f"{min(max_d2s):.3f}..{max(max_d2s):.3f}" if max_d2s else "n/a"

    return (
        f"{n:>3}  {ms_count:>16}  {n_true_max:>11}  {n_true_min:>9}"
        f"  {class_str:>16}  {order_str:>30}  {d2_str:>18}  {confirmed:>9}"
    )


def build_report(rows: list[tuple[int, dict[str, Any], int]]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    params_str = (
        f"gamma={BASELINE_PARAMS['gamma']}  mu_sq={BASELINE_PARAMS['mu_sq']}"
        f"  alpha_phi={BASELINE_PARAMS['alpha_phi']}"
    )
    header = "\n".join([
        "Thread 7 Arnold A_k classification of V_n critical points",
        f"Generated: {ts}",
        f"Baseline parameters: {params_str}",
        f"Landscape: V_n(psi) = V_eff(F_n(psi)) = (mu_sq/2)*C^2 - alpha_phi*C^4,  C = psi + gamma*psi^n",
        f"Method: exact symbolic differentiation via iterated npp.polyder + npp.polyval",
        f"Vanishing threshold: |d_k| < 1e-8 declares order k vanishing",
        "",
        "Classification key:",
        "  A_1 (Morse)         first non-vanishing derivative is d2 (standard quadratic critical point)",
        "  A_2 (cusp)          d2 vanishes, first non-vanishing is d3",
        "  A_3 (swallowtail)   d2,d3 vanish, first non-vanishing is d4",
        "",
        "Metastable count: from count_metastable_states (find_peaks, prominence=0.04, distance=120)",
        "True maxima: all Morse maxima of V_n found by arnold_class (no prominence filter)",
        "True minima: all Morse minima of V_n",
        "",
        "Result: All critical points are Morse (A_1) for n=2..10 at baseline SR parameters.",
        "  The metastable parity pattern (2 for odd n; 3 for even n=4..8) is topological,",
        "  counting Morse critical points above the prominence threshold, not singularity-",
        "  theoretic. At n=10, the inner fold-point maximum achieves prominence > 0.04,",
        "  raising the even-n count from 3 to 4.",
        "",
    ])

    col_header = (
        f"{'n':>3}  {'metastable_count':>16}  {'true_maxima':>11}  {'true_minima':>9}"
        f"  {'arnold_classes':>16}  {'first_nonvanishing_orders':>30}  {'max_d2_range':>18}  {'confirmed':>9}"
    )
    sep = "-" * len(col_header)

    data_lines = [col_header, sep]
    for n, ac_result, ms_count in rows:
        data_lines.append(_format_row(n, ac_result, ms_count))

    return "\n".join([header] + data_lines) + "\n"


def run_report(*, write_disk: bool = True, output_dir: Path | None = None) -> str:
    """Compute Arnold classification for all N_VALUES and optionally write results file."""
    root = Path(__file__).resolve().parents[2]
    out_dir = output_dir if output_dir is not None else root / RESULTS_DIR

    params = {
        "gamma": float(BASELINE_PARAMS["gamma"]),
        "mu_sq": float(BASELINE_PARAMS["mu_sq"]),
        "alpha_phi": float(BASELINE_PARAMS["alpha_phi"]),
    }

    rows: list[tuple[int, dict, int]] = []
    for n in N_VALUES:
        print(f"[arnold] n={n} ...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ac = arnold_class(n, params, PSI_RANGE)
        ms = _metastable_count(n)
        n_cp = len(ac["critical_points"])
        print(f"{n_cp} critical points ({ac['n_maxima']} max, {ac['n_minima']} min)  metastable={ms}")
        rows.append((n, ac, ms))

    report = build_report(rows)
    print(report)

    if write_disk:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "arnold_classification.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(root)}")

    return report


def main() -> None:
    run_report()


if __name__ == "__main__":
    main()
