"""Shared plot theme so every figure in the repo looks deliberate."""
import matplotlib as mpl

# Node colors used consistently across all temperature plots
NODE_COLORS = {
    "T_die": "#d1495b",
    "T_ihs": "#e8853a",
    "T_coldplate": "#edae49",
    "T_loop": "#3a86b8",
    "T_fac_return": "#1b3a5b",
}
ACCENT = "#2a9d8f"
WARN = "#d1495b"
GRID = "#d9d9d9"


def apply():
    """Apply the ThermaLoop matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.family": "sans-serif",
        "figure.dpi": 130,
    })
