"""Figure 3: (k_0, sigma) outcome grid for n=4.

Data source: experiments/polynomial_sweep/results/coupling_scale_report.txt
Regenerate upstream data:
  python -m experiments.polynomial_sweep.coupling_scale_experiment
"""

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from shared.visualization import COLORS, FIGSIZE, annotate_commit, apply_style, save

_RESULTS = Path(__file__).parent.parent / "results" / "coupling_scale_report.txt"

_OUTCOME_LABEL = {
    "A": "Path A (fine-structure death)",
    "B": "Path B (amplitude death)",
}


def _parse_sigma_window(path: Path) -> dict[float, dict[int, str]]:
    """Parse sigma-keyed k0/path tables from the coupling scale report.

    Returns {sigma: {k0: path_letter}}.
    Only matches Phase 1 sigma headers (unindented lines).
    """
    text = path.read_text()
    result: dict[float, dict[int, str]] = {}
    current_sigma: float | None = None

    for line in text.splitlines():
        sigma_m = re.match(r"sigma=([\d.]+)\s+k0_crit=", line)
        if sigma_m:
            current_sigma = float(sigma_m.group(1))
            result[current_sigma] = {}
            continue
        if current_sigma is not None:
            row_m = re.match(r"\s+(\d+)\s+([AB])\s+", line)
            if row_m:
                result[current_sigma][int(row_m.group(1))] = row_m.group(2)

    return result


def _boundary_positions(row_data: dict[int, str]) -> list[float]:
    """Return k0 midpoints where adjacent outcomes switch label."""
    ordered = sorted(row_data.items())
    boundaries: list[float] = []
    for (k_prev, label_prev), (k_curr, label_curr) in zip(ordered[:-1], ordered[1:]):
        if label_prev != label_curr:
            boundaries.append(0.5 * (k_prev + k_curr))
    return boundaries


def make_sigma_window() -> plt.Figure:
    """Build and return the (k_0, sigma) outcome grid figure.

    Three stacked panels (one per sigma), cells colored by path outcome.
    """
    apply_style()

    data = _parse_sigma_window(_RESULTS)
    sigmas = sorted(data.keys())
    all_k0 = sorted({k for d in data.values() for k in d.keys()})

    color_map = {"A": COLORS["accent"], "B": COLORS["primary"]}

    fig, axes = plt.subplots(
        len(sigmas), 1,
        figsize=FIGSIZE["double_column"],
        sharex=True,
    )
    if len(sigmas) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigmas):
        row_data = data[sigma]
        for k0 in all_k0:
            color = color_map.get(row_data.get(k0, ""), COLORS["muted"])
            ax.bar(k0, 1, color=color, width=0.85, linewidth=0)

        ax.set_yticks([])
        ax.set_ylabel(f"$\\sigma={sigma}$", rotation=0, ha="right", va="center",
                      labelpad=38, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        for x_boundary in _boundary_positions(row_data):
            ax.axvline(x_boundary, color=COLORS["muted"], lw=0.8, zorder=5)

    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel("$k_0$ (initial coupling intensity)")
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[-1].xaxis.set_minor_locator(ticker.NullLocator())
    axes[-1].set_xlim(min(all_k0) - 0.5, max(all_k0) + 0.5)

    patches = [
        mpatches.Patch(color=color_map["B"], label=_OUTCOME_LABEL["B"]),
        mpatches.Patch(color=color_map["A"], label=_OUTCOME_LABEL["A"]),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14), fontsize=8, frameon=False)

    fig.subplots_adjust(hspace=0.08)
    annotate_commit(fig)
    return fig


if __name__ == "__main__":
    fig = make_sigma_window()
    save(fig, "sigma_window", experiment="polynomial_sweep")
    plt.close(fig)
