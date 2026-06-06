"""UQ ensemble plot suite.

Distributions and time-band envelopes for the ensemble runner in
`thermaloop.scenarios.ensemble`. Shares the theme from `viz.style`.
"""
import numpy as np
import matplotlib.pyplot as plt

from thermaloop.viz import style

style.apply()


def margin_distribution(res):
    """Histogram of min-margin-K across samples with percentile markers."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    m = res["distributions"]["min_margin_K"]
    ax.hist(m, bins=24, color=style.ACCENT, alpha=0.75, edgecolor="white")
    pct = res["percentiles"]["min_margin_K"]
    ax.axvline(pct[5],  color=style.WARN, ls=":",  lw=1.2,
               label=f"P5: {pct[5]:.1f} K")
    ax.axvline(pct[50], color="#444",     ls="--", lw=1.2,
               label=f"P50: {pct[50]:.1f} K")
    ax.axvline(pct[95], color="#444",     ls=":",  lw=1.2,
               label=f"P95: {pct[95]:.1f} K")
    ax.axvline(0, color=style.WARN, lw=1.6,
               label="throttle threshold (margin = 0)")
    ax.set_xlabel("min margin to T_limit (K)")
    ax.set_ylabel("samples")
    ax.set_title(f"Minimum margin distribution — N={res['n_samples']} "
                 f"({res['sampler'].upper()})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def peak_die_distribution(res):
    """Histogram of peak T_die across samples with the limit overlaid."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = res["distributions"]["peak_T_die"]
    ax.hist(t, bins=24, color=style.NODE_COLORS["T_die"], alpha=0.75,
            edgecolor="white")
    pct = res["percentiles"]["peak_T_die"]
    ax.axvline(pct[5],  color="#444", ls=":",  lw=1.2,
               label=f"P5: {pct[5]:.1f} C")
    ax.axvline(pct[50], color="#444", ls="--", lw=1.2,
               label=f"P50: {pct[50]:.1f} C")
    ax.axvline(pct[95], color="#444", ls=":",  lw=1.2,
               label=f"P95: {pct[95]:.1f} C")
    ax.axvline(res["T_limit"], color=style.WARN, lw=1.6,
               label=f"T_limit: {res['T_limit']:.0f} C")
    ax.set_xlabel("peak T_die (C)")
    ax.set_ylabel("samples")
    ax.set_title(f"Peak die-temperature distribution — "
                 f"throttle prob = {res['throttle_prob']*100:.1f} %")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def margin_envelope_timeline(res):
    """Median and 5-95 % band of T_die over time with T_limit overlay."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    t = res["t"]
    grid = res["T_die_grid"]   # (N, len(t))
    p05 = np.percentile(grid,  5, axis=0)
    p50 = np.percentile(grid, 50, axis=0)
    p95 = np.percentile(grid, 95, axis=0)
    ax.fill_between(t, p05, p95, color=style.NODE_COLORS["T_die"], alpha=0.20,
                    label="5-95 % band")
    ax.plot(t, p50, color=style.NODE_COLORS["T_die"], lw=1.8, label="median")
    ax.axhline(res["T_limit"], ls="--", color=style.WARN, alpha=0.85,
               label=f"T_limit {res['T_limit']:.0f} C")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("T_die (C)")
    ax.set_title(f"Die-temperature envelope over time — N={res['n_samples']}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
