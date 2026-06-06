"""ThermaLoop command-line interface.

    python -m thermaloop run   configs/baseline.yaml
    python -m thermaloop run   configs/faults/pump_degradation.yaml
    python -m thermaloop sweep configs/sweeps/pump_speed.yaml
    python -m thermaloop envelope

Each command writes a self-contained HTML engineering report to reports/.
"""
import argparse
import os

from thermaloop.scenarios import engine, sweeps, ensemble
from thermaloop.viz import plots, report, uq_plots


def _cmd_run(args):
    r = engine.run_scenario(args.config)
    s = r["safety"]
    summary = [
        ("Mean per-GPU power", f"{r['mean_gpu_power_W']:.0f} W"),
        ("Peak T_die", f"{s['peak_T_die']:.1f} C"),
        ("Minimum margin", f"{s['min_margin_K']:.1f} K"),
        ("Die-temp limit", f"{s['T_limit']:.0f} C"),
        ("Throttled", "yes" if s["throttled"] else "no"),
        ("Pump energy", f"{r['pump_energy_Wh']:.1f} Wh"),
    ]
    if s["throttled"]:
        summary.append(("Time to throttle", f"{s['time_to_throttle_s']:.0f} s"))
        verdict = (f"THROTTLE: die temperature reached the {s['T_limit']:.0f} C "
                   f"limit at t={s['time_to_throttle_s']:.0f} s.")
    else:
        verdict = (f"SAFE: die stayed {s['min_margin_K']:.1f} K below the "
                   f"{s['T_limit']:.0f} C limit throughout.")
    sections = [
        ("Scenario overview", plots.scenario_overview(r), None),
        ("Safety margin timeline", plots.safety_margin_timeline(r), None),
    ]
    out = os.path.join("reports", r["name"], "report.html")
    report.write_report(out, title=r["name"], description=r["description"],
                        summary=summary, sections=sections,
                        verdict=verdict, verdict_ok=not s["throttled"])
    print(f"peak T_die {s['peak_T_die']:.1f}C  min margin {s['min_margin_K']:.1f}K  "
          f"throttled={s['throttled']}")
    print(f"report -> {out}")


def _cmd_sweep(args):
    sw = sweeps.run_sweep(args.config)
    best = max(sw["rows"], key=lambda r: r["margin_K"] if r["margin_K"] > 0
               else -1e9)
    summary = [
        ("Swept parameter", sw["param"]),
        ("Points", str(len(sw["rows"]))),
        ("T_die range",
         f"{min(r['T_die'] for r in sw['rows']):.1f}–"
         f"{max(r['T_die'] for r in sw['rows']):.1f} C"),
        ("Pump power range",
         f"{min(r['pump_power_W'] for r in sw['rows']):.0f}–"
         f"{max(r['pump_power_W'] for r in sw['rows']):.0f} W"),
    ]
    verdict = (f"Most efficient setting with positive margin: "
               f"{sw['param']} = {best['sweep_value']} "
               f"({best['margin_K']:.1f} K margin, "
               f"{best['pump_power_W']:.0f} W pump).")
    sections = [
        ("Tradeoff curve", plots.sweep_curve(sw), None),
        ("Pareto front", plots.sweep_pareto(sw), None),
    ]
    out = os.path.join("reports", sw["name"], "report.html")
    report.write_report(out, title=sw["name"], description=sw["description"],
                        summary=summary, sections=sections,
                        verdict=verdict, verdict_ok=True)
    print(f"swept {sw['param']} over {len(sw['rows'])} points")
    print(f"report -> {out}")


def _cmd_ensemble(args):
    res = ensemble.run_ensemble(args.config)
    pct = res["percentiles"]
    summary = [
        ("Scenario", res["name"]),
        ("Sampler / N", f"{res['sampler'].upper()} / {res['n_samples']}"),
        ("Uncertain params", ", ".join(res["param_names"])),
        ("Min-margin P5 / P50 / P95",
         f"{pct['min_margin_K'][5]:.1f} / "
         f"{pct['min_margin_K'][50]:.1f} / "
         f"{pct['min_margin_K'][95]:.1f} K"),
        ("Peak T_die P5 / P50 / P95",
         f"{pct['peak_T_die'][5]:.1f} / "
         f"{pct['peak_T_die'][50]:.1f} / "
         f"{pct['peak_T_die'][95]:.1f} C"),
        ("Throttle probability", f"{res['throttle_prob']*100:.1f} %"),
    ]
    verdict_ok = res["throttle_prob"] < 0.05
    if verdict_ok:
        verdict = (f"SAFE: across {res['n_samples']} parametric samples, "
                   f"throttle probability {res['throttle_prob']*100:.1f} % "
                   f"(< 5 %).")
    else:
        verdict = (f"AT RISK: across {res['n_samples']} parametric samples, "
                   f"throttle probability {res['throttle_prob']*100:.1f} % "
                   f"— design margin does not survive realistic parameter "
                   f"uncertainty.")
    sections = [
        ("Margin distribution", uq_plots.margin_distribution(res), None),
        ("Peak T_die distribution",
         uq_plots.peak_die_distribution(res), None),
        ("Die-temperature envelope over time",
         uq_plots.margin_envelope_timeline(res), None),
    ]
    out = os.path.join("reports", res["name"], "report.html")
    report.write_report(out, title=res["name"],
                        description=res["description"],
                        summary=summary, sections=sections,
                        verdict=verdict, verdict_ok=verdict_ok)
    print(f"throttle prob {res['throttle_prob']*100:.1f}%  "
          f"P50 margin {pct['min_margin_K'][50]:.1f}K  "
          f"P5 margin {pct['min_margin_K'][5]:.1f}K")
    print(f"report -> {out}")


def _cmd_envelope(args):
    sections = [
        ("Thermal envelope", plots.thermal_envelope(), None),
        ("1-D loop field", plots.loop_1d_heatmap(),
         "Coolant temperature along the loop; hottest at cold-plate exit."),
        ("Heat path", plots.heat_path_sankey(),
         "Server heat plus pump work, rejected to facility water."),
    ]
    out = os.path.join("reports", "envelope", "report.html")
    report.write_report(out, title="operating envelope",
                        description="Design-space maps for the reference server.",
                        summary=[("Reference", "8-GPU H100-class, 30 C facility")],
                        sections=sections,
                        verdict="Reference operating maps generated.",
                        verdict_ok=True)
    print(f"report -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="thermaloop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a scenario config")
    p_run.add_argument("config")
    p_run.set_defaults(func=_cmd_run)
    p_sw = sub.add_parser("sweep", help="run an optimization sweep config")
    p_sw.add_argument("config")
    p_sw.set_defaults(func=_cmd_sweep)
    p_uq = sub.add_parser("ensemble", help="run a UQ ensemble scenario config")
    p_uq.add_argument("config")
    p_uq.set_defaults(func=_cmd_ensemble)
    p_env = sub.add_parser("envelope", help="generate reference design-space maps")
    p_env.set_defaults(func=_cmd_envelope)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
