"""Shared matplotlib style module for sr-compute figure scripts."""

import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
COLORS: dict[str, str] = {
    "primary":   "#0072B2",  # blue
    "secondary": "#E69F00",  # orange/amber
    "accent":    "#D55E00",  # vermillion
    "muted":     "#999999",  # gray
}

FIGSIZE: dict[str, tuple[float, float]] = {
    "single_column": (3.5, 2.6),
    "double_column": (7.0, 3.0),
    "square":        (3.5, 3.5),
}


def apply_style() -> None:
    """Set rcParams for publication-quality figures.

    Serif font family, Type 42 fonts (journal-submission grade),
    no top/right spines, minor ticks on, grid off.
    """
    plt.rcParams.update({
        "font.family":          "serif",
        "font.size":            10,
        "axes.labelsize":       11,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            False,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "savefig.bbox":         "tight",
        "savefig.dpi":          150,
    })


def save(fig: plt.Figure, name: str, *, experiment: str) -> None:
    """Write fig to docs/figures/<experiment>/<name>.{png,pdf}.

    Creates the output directory if it does not exist and prints paths.
    """
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "docs" / "figures" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path.relative_to(repo_root)}")


def annotate_commit(fig: plt.Figure, *, fontsize: int = 6) -> None:
    """Draw short git commit hash in the lower-right corner of fig.

    No-op if git is unavailable or HEAD cannot be resolved.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent.parent,
        )
        sha = result.stdout.strip()
    except Exception:
        return
    fig.text(
        0.99, 0.01, sha,
        ha="right", va="bottom",
        fontsize=fontsize,
        color=COLORS["muted"],
        transform=fig.transFigure,
    )
