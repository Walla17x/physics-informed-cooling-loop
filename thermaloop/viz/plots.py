"""Plot suite for ThermaLoop.

Each function returns a matplotlib Figure so the report generator can embed it.
All plots share the theme in `style.py`.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

from thermaloop.viz import style
from thermaloop.thermal import rc_network, loop_1d
from thermaloop import safety as safety_mod

style.apply()


def scenario_overview(result):
    """Power, 5-node temperatures with safety band, and pump power for a run."""
    t = result["t"]
    T_limit = result["safety"]["T_limit"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2.2, 1]))

    ax[0].plot(t, result["power_per_gpu"], color=style.WARN, linewidth=1.0)
    ax[0].set_ylabel("per-GPU\npower (W)")
    ax[0].set_title(f"Scenario: {result['name']}")

    for key in ["T_die", "T_ihs", "T_coldplate", "T_loop", "T_fac_return"]:
        ax[1].plot(t, result[key], label=key, color=style.NODE_COLORS[key])
    ax[1].axhline(T_limit, ls="--", color=style.WARN, alpha=0.7,
                  label=f"limit {T_limit:.0f}C")
    ax[1].axhspan(T_limit, max(T_limit + 5, result["T_die"].max() + 2),
                  color=style.WARN, alpha=0.06)
    ax[1].set_ylabel("temperature (C)")
    ax[1].legend(ncol=3, loc="upper right")
    s = result["safety"]
    if s["throttled"]:
        ax[1].axvline(s["time_to_throttle_s"], color=style.WARN, ls=":")
        ax[1].text(s["time_to_throttle_s"], T_limit + 1,
                   f" throttle @ {s['time_to_throttle_s']:.0f}s",
                   color=style.WARN, fontsize=8)

    ax[2].plot(t, result["pump_power_t"], color=style.ACCENT)
    ax[2].set_ylabel("pump\npower (W)")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def safety_margin_timeline(result):
    """Margin-to-limit over time with the danger band shaded."""
    t = result["t"]
    T_limit = result["safety"]["T_limit"]
    margin = T_limit - result["T_die"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, margin, color=style.NODE_COLORS["T_die"])
    ax.axhline(0, color=style.WARN, ls="--")
    ax.axhspan(margin.min() - 2 if margin.min() < 0 else -2, 0,
               color=style.WARN, alpha=0.08)
    ax.axhspan(0, 5, color=style.WARN, alpha=0.04)
    ax.set_ylabel("margin to limit (K)")
    ax.set_xlabel("time (s)")
    ax.set_title(f"Safety margin - {result['name']}")
    fig.tight_layout()
    return fig


def sweep_curve(sweep):
    """Parameter vs die temperature and pump power (twin axis)."""
    rows = sweep["rows"]
    x = [r["sweep_value"] for r in rows]
    T_die = [r["T_die"] for r in rows]
    pump = [r["pump_power_W"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xlabel = (f"{sweep['param']} (x design)" if sweep["axis_is_factor"]
              else sweep["param"])
    ax.plot(x, T_die, "o-", color=style.NODE_COLORS["T_die"], label="T_die")
    ax.axhline(sweep["T_limit"], ls="--", color=style.WARN, alpha=0.7,
               label=f"limit {sweep['T_limit']:.0f}C")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("T_die (C)", color=style.NODE_COLORS["T_die"])
    ax2 = ax.twinx()
    ax2.plot(x, pump, "s--", color=style.ACCENT, label="pump power")
    ax2.set_ylabel("pump power (W)", color=style.ACCENT)
    ax2.grid(False)
    ax.set_title(f"Sweep: {sweep['name']}")
    fig.tight_layout()
    return fig


def sweep_pareto(sweep):
    """Pump power vs thermal margin Pareto front."""
    rows = sweep["rows"]
    pump = [r["pump_power_W"] for r in rows]
    margin = [r["margin_K"] for r in rows]
    labels = [r["sweep_value"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pump, margin, "-", color=style.GRID, zorder=1)
    sc = ax.scatter(pump, margin, c=margin, cmap="RdYlGn", s=90,
                    edgecolor="#333", zorder=2, vmin=-5,
                    vmax=max(margin) if margin else 1)
    ax.axhline(0, color=style.WARN, ls="--", alpha=0.7)
    for p, m, lab in zip(pump, margin, labels):
        ax.annotate(f"{lab}", (p, m), fontsize=7,
                    textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel("pump power (W)")
    ax.set_ylabel("thermal margin (K)")
    ax.set_title(f"Pareto: margin vs pump power - {sweep['name']}")
    plt.colorbar(sc, ax=ax, label="margin (K)")
    fig.tight_layout()
    return fig


def thermal_envelope(params=None, P_range=(300, 900), flow_frac=(0.25, 1.2),
                     n=13):
    """T_die contour over the per-GPU power x flow-fraction plane."""
    params = params or rc_network.default_params()
    P_grid = np.linspace(*P_range, n)
    f_grid = np.linspace(*flow_frac, n)
    Tj = np.zeros((n, n))
    for i, P in enumerate(P_grid):
        for j, f in enumerate(f_grid):
            p = dict(params, m_dot=params["m_dot"] * f)
            Tj[i, j] = rc_network.steady_state(p, P)["T_die"]
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(f_grid, P_grid, Tj, levels=18, cmap="inferno")
    cs = ax.contour(f_grid, P_grid, Tj, levels=[70, 80, 90],
                    colors="white", linewidths=1.0)
    ax.clabel(cs, fmt="%.0f C", fontsize=8)
    plt.colorbar(cf, ax=ax, label="steady-state T_die (C)")
    ax.set_xlabel("coolant flow (x design)")
    ax.set_ylabel("per-GPU power (W)")
    ax.set_title("Thermal envelope (white iso: 70/80/90 C)")
    fig.tight_layout()
    return fig


def heat_path_sankey(result=None):
    """Heat-flow Sankey: GPU die -> loop -> CDU -> facility, plus pump input."""
    if result is None:
        from thermaloop import system
        result = system.run_server(T_horizon=300.0)
    q_die = result["mean_gpu_power_W"] * result["params"]["n_gpus"] / 1000.0  # kW
    pump_kw = result.get("pump_power_W",
                         result.get("pump_power_t", [0])[0]) / 1000.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    sankey = Sankey(ax=ax, unit=" kW", scale=1.0 / max(q_die, 1e-6),
                    format="%.1f", gap=0.4)
    sankey.add(flows=[q_die, pump_kw, -(q_die + pump_kw)],
               labels=["GPU heat", "pump work", "to facility water"],
               orientations=[0, 1, 0],
               pathlengths=[0.6, 0.4, 0.6],
               facecolor=style.ACCENT)
    sankey.finish()
    ax.set_title("Heat path: silicon to facility water "
                 f"({q_die:.1f} kW server load)")
    fig.tight_layout()
    return fig


def loop_1d_heatmap(P_const=700.0, T_horizon=240.0):
    """Spatial-temporal coolant temperature along the 1D loop."""
    params = loop_1d.default_params_1d()
    N = params["geom"]["N"]
    t_axis = np.arange(0, T_horizon, 1.0)
    P = np.full_like(t_axis, P_const, dtype=float)
    sol = loop_1d.simulate_1d(t_axis, P, params)
    T_loop = sol.y[3:3 + N, :]
    x = (np.arange(N) + 0.5) / N
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.pcolormesh(t_axis, x, T_loop, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax, label="coolant T (C)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("loop position s/L")
    ax.set_title("1-D loop: coolant temperature along the loop")
    fig.tight_layout()
    return fig
