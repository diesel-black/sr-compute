"""Figure 1: eta_n vs n with secondary local-exponent axis.

Data source: experiments/polynomial_sweep/results/analysis.txt
Regenerate upstream data:
  python -m experiments.polynomial_sweep.run
  python -m experiments.polynomial_sweep.analyze
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from shared.visualization import COLORS, FIGSIZE, annotate_commit, apply_style, save

_RESULTS = Path(__file__).parent.parent / "results" / "analysis.txt"


def _parse_eta_ladder(path: Path) -> tuple[list[int], list[float]]:
    """Extract (n_values, eta_values) from the analysis report table.

    Reads the "eta at f=1" row; blank cells (n=2) are skipped.
    """
    text = path.read_text()
    header_match = re.search(r"Quantity\s+\|(.*?)\n", text)
    eta_match = re.search(r"eta at f=1\s+\|(.*?)\n", text)
    if not header_match or not eta_match:
        raise ValueError(f"Could not parse eta ladder from {path}")

    header_cols = [c.strip() for c in header_match.group(1).split("|")]
    eta_cols = [c.strip() for c in eta_match.group(1).split("|")]

    n_vals: list[int] = []
    eta_vals: list[float] = []
    for hdr, val in zip(header_cols, eta_cols):
        m = re.match(r"n=(\d+)", hdr)
        if m and val:
            try:
                n_vals.append(int(m.group(1)))
                eta_vals.append(float(val))
            except ValueError:
                pass
    return n_vals, eta_vals


def make_eta_ladder() -> plt.Figure:
    """Build and return the eta ladder figure.

    Primary axis: log(eta_n) vs n.
    Secondary axis: local exponent p_n = log(eta_n / eta_{n-1}) / log(n / (n-1)).
    """
    apply_style()

    ns_list, etas_list = _parse_eta_ladder(_RESULTS)
    ns = np.array(ns_list, dtype=float)
    etas = np.array(etas_list, dtype=float)

    p_ns = ns[1:]
    p_vals = np.log(etas[1:] / etas[:-1]) / np.log(ns[1:] / ns[:-1])

    fig, ax1 = plt.subplots(figsize=FIGSIZE["single_column"])
    ax2 = ax1.twinx()

    ax1.semilogy(ns, etas, "o-", color=COLORS["primary"], lw=1.5, ms=5,
                 label=r"$\eta_n$")
    ax1.set_xlabel("polynomial order $n$")
    ax1.set_ylabel(r"$\eta_n$ (steps to stabilize)")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.NullLocator())

    ax2.plot(p_ns, p_vals, "s--", color=COLORS["secondary"], lw=1.0, ms=4,
             label="$p_n$")
    ax2.axhline(5.0, color=COLORS["muted"], lw=0.8, ls=":")
    ax2.text(max(ns) - 0.1, 5.15, "asymptote", ha="right", va="bottom",
             fontsize=7, color=COLORS["muted"])
    ax2.set_ylabel("local exponent $p_n$")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    idx3 = int(np.where(ns == 3.0)[0][0])
    ax1.annotate(
        "cubic aperture",
        xy=(3, etas[idx3]),
        xytext=(3.6, etas[idx3] * 4),
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=0.8),
        fontsize=7,
        color=COLORS["accent"],
    )

    annotate_commit(fig)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_eta_ladder()
    save(fig, "eta_ladder", experiment="polynomial_sweep")
    plt.close(fig)
