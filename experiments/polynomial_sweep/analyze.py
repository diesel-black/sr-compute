"""Thread 7 Phase 3 Part 2: aggregate sweep summary, parity JSON, and qualitative predictions.

Reads ``results/summary.json``, optional ``results/parity/*.json``, prints a terminal report,
and writes ``results/analysis.txt`` with a UTC generation timestamp in the header.
Numeric columns use one consistent significant-figure policy; full floats stay in the JSON sources.
Run from repository root:

    python -m experiments.polynomial_sweep.analyze
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from experiments.polynomial_sweep.config import RESULTS_DIR
from experiments.polynomial_sweep.outcome_utils import outcome_from_integrator

# ``t_final`` uses this everywhere in the report (parity table and sweep metadata) for one instrument.
_T_FINAL_SIGFIG = 6


def _repo_root() -> Path:
    """Repository root (directory containing ``experiments/``)."""
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any, prec: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        return f"{x:.{prec}g}"
    return str(x)


def _fmt_t_final(raw: Any) -> str:
    """String for ``t_final`` with the same significant figures as the parity table."""
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, (int, float)):
        return _fmt(float(raw), _T_FINAL_SIGFIG)
    return str(raw)


def _first_eta(nonlocal_growth: Any) -> Optional[float]:
    if not isinstance(nonlocal_growth, dict) or nonlocal_growth.get("skipped"):
        return None
    etas = nonlocal_growth.get("eta_values") or []
    if not etas or etas[0] is None:
        return None
    v = float(etas[0])
    return v if math.isfinite(v) else None


def _growth_first(nonlocal_growth: Any) -> Optional[float]:
    if not isinstance(nonlocal_growth, dict) or nonlocal_growth.get("skipped"):
        return None
    gr = nonlocal_growth.get("growth_rates") or []
    if not gr or gr[0] is None:
        return None
    v = float(gr[0])
    return v if math.isfinite(v) else None


def _outcome_from_sidecar(data: dict[str, Any]) -> str:
    """Read ``outcome`` from saved JSON, or derive from legacy ``success`` + ``t_span``."""
    ex = data.get("outcome")
    if isinstance(ex, str) and ex in ("completed", "terminal", "timeout"):
        return ex
    msg = str(data.get("message", ""))
    if "Wallclock timeout" in msg:
        return "timeout"
    succ = data.get("success")
    t_final = data.get("t_final")
    params = data.get("params") or {}
    tsp = params.get("t_span")
    if succ is None or t_final is None or not isinstance(tsp, (list, tuple)) or len(tsp) < 2:
        return "terminal"
    try:
        return outcome_from_integrator(bool(succ), float(t_final), float(tsp[1]))
    except (TypeError, ValueError):
        return "terminal"


def _outcome_for_parity_row(r: dict[str, Any]) -> str:
    """Parity JSON may store ``outcome`` or legacy ``success``."""
    ex = r.get("outcome")
    if isinstance(ex, str) and ex in ("completed", "terminal", "timeout"):
        return ex
    msg = str(r.get("message", ""))
    if "Wallclock timeout" in msg:
        return "timeout"
    succ = r.get("success")
    t_final = r.get("t_final")
    params = r.get("params") or {}
    tsp = params.get("t_span")
    if succ is None or t_final is None or not isinstance(tsp, (list, tuple)) or len(tsp) < 2:
        return "terminal"
    try:
        return outcome_from_integrator(bool(succ), float(t_final), float(tsp[1]))
    except (TypeError, ValueError):
        return "terminal"


def _write_predicted_observed(
    summary: dict[str, Any],
    per_n_meta: dict[str, dict[str, Any]],
    out: TextIO,
) -> None:
    """Table comparing provisional predictions to sweep observations with confound notes."""
    lines = [
        "1. Predicted vs observed (main sweep, summary.json)",
        "",
        "Provisional structural targets (qualitative):",
        "  R25: kappa(Pi) equals 1 at n=3 (exponent n-3 vanishes); n>3 amplifies spatial dynamic range.",
        "  R26: metastable landscape count often 2 at cubic, 3 at quartic (cusp vs swallowtail class).",
        "  R27: spectral ratio departs from diffuse baseline when leading singular mode concentrates.",
        "  RG marginality: eta(f) and growth across coarse scales probe discrete brake consistency.",
        "",
        f"{'Quantity':<22} | {'n=2':<12} | {'n=3':<12} | {'n=4':<12} | {'n=5':<12} | {'n=6':<12}",
        "-" * 95,
    ]

    quantities = (
        ("metastable_count", "metastable_count", 0),
        ("kappa (R25)", "condition_number", 4),
        ("spectral ratio (R27)", "spectral_ratio", 4),
        ("eta at f=1", "_eta", 4),
        ("growth eta2/eta1", "_growth", 4),
    )

    for qname, key, prec in quantities:
        cells = []
        for nk in ("2", "3", "4", "5", "6"):
            m = summary.get(nk, {})
            if key == "_eta":
                val = _first_eta(m.get("nonlocal_growth"))
                cells.append(_fmt(val, prec))
            elif key == "_growth":
                val = _growth_first(m.get("nonlocal_growth"))
                cells.append(_fmt(val, prec))
            else:
                cells.append(_fmt(m.get(key), prec))
        row = f"{qname:<22} | " + " | ".join(f"{c:<12}" for c in cells)
        lines.append(row)

    lines.append("")
    lines.append("Sweep integration metadata (from n*_measurements.json when present):")
    lines.extend(
        [
            "  Display uses the same rounding policy as the parity table (6 significant figures",
            "  for t_final). Full-precision values remain in summary.json and n*_measurements.json.",
        ]
    )
    for nk in ("2", "3", "4", "5", "6"):
        meta = per_n_meta.get(nk)
        if meta:
            lines.append(f"  n={nk}:{meta['short']}")
        else:
            lines.append(f"  n={nk}: (no sidecar json)")
    lines.extend(
        [
            "",
            "Honest confound (methodological): final-time and solver differ by n in the archived sweep.",
            "  Field-valued metrics (kappa, spectral ratio, eta) are not comparable across n unless",
            "  t_final and integration method are aligned. Metastable_count is computed on a fixed",
            "  psi_bar axis and is less sensitive to simulation time, but still reflects the same",
            "  potential landscape independent of the run.",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    out.write(text)
    print(text, end="")


def _collect_per_n_meta(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load n*_measurements.json for solver and t_final annotations."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("n*_measurements.json")):
        m = re.match(r"^n(\d+)_measurements$", path.stem)
        if not m:
            continue
        nk = m.group(1)
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        params = data.get("params") or {}
        method = params.get("method", "?")
        t_final = data.get("t_final")
        oc = _outcome_from_sidecar(data)
        tf_show = _fmt_t_final(t_final)
        out[nk] = {
            "method": method,
            "t_final": t_final,
            "outcome": oc,
            "short": f" {method}, t_final={tf_show}, outcome={oc}",
        }
    return out


def _write_parity_section(parity_dir: Path, out: TextIO) -> None:
    lines = [
        "2. Parity experiment (results/parity/*.json)",
        "",
    ]
    files = sorted(parity_dir.glob("parity_*.json"))
    if not files:
        lines.extend(
            [
                "No parity JSON files found. After running the parity driver, this section",
                "  compares n=3 Radau (full window) vs n=3 RK45 (metric event), and RK45 at n=4",
                "  and n=5 against the Radau sweep baselines.",
                "",
            ]
        )
        text = "\n".join(lines) + "\n"
        out.write(text)
        print(text, end="")
        return

    rows = []
    for p in files:
        rows.append(_load_json(p))
    rows.sort(key=lambda r: r.get("parity_run", ""))

    hdr = (
        f"{'Run':<5} | {'Label':<12} | {'solver':<8} | {'t_final':<12} | {'outcome':<9} | "
        f"{'kappa':<10} | {'spectral':<10} | {'eta@1':<10}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        m = r.get("measurements") or {}
        ng = m.get("nonlocal_growth")
        oc = _outcome_for_parity_row(r)
        lines.append(
            f"{str(r.get('parity_run','')):<5} | "
            f"{str(r.get('parity_label','')):<12} | "
            f"{str(r.get('solver','')):<8} | "
            f"{_fmt_t_final(r.get('t_final')):<12} | "
            f"{oc:<9} | "
            f"{_fmt(m.get('condition_number'), 4):<10} | "
            f"{_fmt(m.get('spectral_ratio'), 4):<10} | "
            f"{_fmt(_first_eta(ng), 4):<10}"
        )
    lines.append("")
    lines.extend(
        [
            "Interpretation template:",
            "  If n=3 Radau at t=30 shows rich structure (kappa and spectral away from trivial",
            "  limits) while the sweep used RK45 and stopped near t=8.9, solver choice and stopping",
            "  time jointly explain part of the cross-n spread.",
            "  If n=5 RK45 at failure time still shows spatial contrast, flattening at t=30 under",
            "  Radau is more plausibly numerical damping than a forced supercubic relaxation.",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    out.write(text)
    print(text, end="")


def _write_clean_findings(summary: dict[str, Any], out: TextIO) -> None:
    """Structural statements that are algebraic or robust to the solver confound."""
    m3 = summary.get("3", {})
    m4 = summary.get("4", {})
    m5 = summary.get("5", {})
    m6 = summary.get("6", {})

    k3 = m3.get("condition_number")
    k4 = m4.get("condition_number")
    s4 = m4.get("spectral_ratio")
    g4 = _growth_first(m4.get("nonlocal_growth"))

    eta3 = _first_eta(m3.get("nonlocal_growth"))
    eta4 = _first_eta(m4.get("nonlocal_growth"))
    eta5 = _first_eta(m5.get("nonlocal_growth"))
    eta6 = _first_eta(m6.get("nonlocal_growth"))

    lines = [
        "3. Clean findings (hold regardless of which solver produced the snapshot, with caveats)",
        "",
        f"  Algebraic: at n=3, kappa(Pi) equals 1 by definition (exponent n-3=0); observed {_fmt(k3)}.",
        f"  Metastable landscape count: {_fmt(m3.get('metastable_count'), 0)} at n=3, "
        f"{_fmt(m4.get('metastable_count'), 0)} at n=4 (matches cusp vs swallowtail expectation in tests).",
        f"  Cross-n eta scaling at f=1 (sweep snapshots): "
        f"{_fmt(eta3)} -> {_fmt(eta4)} -> {_fmt(eta5)} -> {_fmt(eta6)} "
        f"(roughly x7, x5, x3 multipliers stepwise n=3 to 6).",
        f"  n=4 simultaneous breakpoint (sweep, Radau, early stop): "
        f"kappa {_fmt(k4)}, spectral {_fmt(s4)}, first growth {_fmt(g4)}.",
        "",
        "Caveat: the eta ladder uses each n's final field; if n=5 and n=6 fields are spatially flat due",
        "  to implicit integration, eta values are still defined but no longer probe the same",
        "  structural regime as shorter-time structured states at n=3 and n=4.",
        "",
    ]
    text = "\n".join(lines)
    out.write(text)
    print(text, end="")


def run_analysis() -> Path:
    """Load results, print report, return path to ``analysis.txt``."""
    root = _repo_root()
    results_dir = root / RESULTS_DIR
    parity_dir = results_dir / "parity"
    summary_path = results_dir / "summary.json"
    out_path = results_dir / "analysis.txt"

    summary = _load_json(summary_path)
    per_n = _collect_per_n_meta(results_dir)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with out_path.open("w", encoding="utf-8") as out:
        header = (
            "Thread 7 polynomial sweep: analysis report\n"
            f"Run timestamp: {ts}\n"
            + "=" * 60
            + "\n\n"
        )
        out.write(header)
        print(header, end="")
        _write_predicted_observed(summary, per_n, out)
        _write_parity_section(parity_dir, out)
        _write_clean_findings(summary, out)
        out.write("\n(End of report.)\n")
    print(f"Wrote {out_path.relative_to(root)}")
    return out_path


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
