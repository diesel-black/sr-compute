"""Figure 2: metastable count sequence, n=2..10.

Data source: experiments/polynomial_sweep/results/arnold_classification.txt
Regenerate upstream data:
  python -m experiments.polynomial_sweep.arnold_classification_report
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from shared.visualization import COLORS, FIGSIZE, annotate_commit, apply_style, save

_RESULTS = Path(__file__).parent.parent / "results" / "arnold_classification.txt"


def _parse_count_sequence(path: Path) -> tuple[list[int], list[int]]:
    """Extract (n_values, count_values) from the Arnold classification report."""
    text = path.read_text()
    ns: list[int] = []
    counts: list[int] = []
    for line in text.splitlines():
        m = re.match(r"\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+", line)
        if m:
            ns.append(int(m.group(1)))
            counts.append(int(m.group(2)))
    return ns, counts


def make_count_sequence() -> plt.Figure:
    """Build and return the metastable count sequence figure.

    Stem plot with n=10 highlighted to mark the parity break.
    """
    apply_style()

    ns_list, counts_list = _parse_count_sequence(_RESULTS)
    ns = np.array(ns_list)
    counts = np.array(counts_list)

    fig, ax = plt.subplots(figsize=FIGSIZE["single_column"])

    # Lollipop profile: restrained geometry highlights the parity rhythm cleanly.
    ax.vlines(ns, 0, counts, color=COLORS["muted"], lw=1.0, zorder=1)
    ax.plot(ns, counts, "-", color=COLORS["muted"], lw=0.9, zorder=2)
    ax.plot(
        ns,
        counts,
        "o",
        markerfacecolor="white",
        markeredgecolor=COLORS["primary"],
        markeredgewidth=1.5,
        ms=6.5,
        zorder=3,
    )

    idx10 = int(np.where(ns == 10)[0][0])
    ax.plot(
        ns[idx10],
        counts[idx10],
        "o",
        markerfacecolor=COLORS["accent"],
        markeredgecolor=COLORS["accent"],
        ms=7.2,
        zorder=4,
    )
    ax.annotate(
        "parity break at $n=10$",
        xy=(10, counts[idx10]),
        xytext=(8.3, counts[idx10] + 0.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=0.8),
        fontsize=7,
        color=COLORS["accent"],
    )

    ax.set_xlabel("$n$")
    ax.set_ylabel("metastable equilibria")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(0, int(max(counts)) + 1)

    annotate_commit(fig)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_count_sequence()
    save(fig, "count_sequence", experiment="polynomial_sweep")
    plt.close(fig)
