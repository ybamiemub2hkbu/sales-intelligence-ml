"""One visual system for every figure in the project.

Colors come from a validated, colour-vision-deficiency-checked categorical
palette and are assigned to *entities* (a category is always the same colour
in every chart), never to rank.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# ---- tokens --------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"   # de-emphasised marks

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32626"])

CATEGORIES = ["Electronics", "Home & Kitchen", "Outdoor & Camping", "Apparel",
              "Sports & Fitness", "Beauty & Care", "Toys & Games", "Gourmet Grocery"]
CATEGORY_COLORS = dict(zip(CATEGORIES, PALETTE))
CHANNEL_COLORS = dict(zip(["In-Store", "Online", "Mobile App"], PALETTE[:3]))
REGION_COLORS = dict(zip(["North", "South", "East", "West"], PALETTE[:4]))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 140,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlesize": 12.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def money_axis(ax, axis: str = "y") -> None:
    def fmt(x, _):
        a = abs(x)
        if a >= 1e6:
            return f"${x / 1e6:.1f}M"
        if a >= 1e3:
            return f"${x / 1e3:.0f}K"
        return f"${x:.0f}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(fmt))


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    f = FuncFormatter(lambda x, _: f"{x * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def titled(ax, title: str, sub: str | None = None) -> None:
    """Title plus an optional secondary line beneath it, in secondary ink."""
    ax.set_title(title, pad=24 if sub else 12)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                    textcoords="offset points", fontsize=8.8, color=INK_2,
                    va="bottom", ha="left")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


apply_style()
