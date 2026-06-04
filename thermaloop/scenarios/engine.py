"""Scenario engine.

Runs a YAML-defined scenario: a workload, optional static parameter overrides,
and a perturbation timeline that varies parameters over time (step or ramp).
The thermal physics is the validated `rc_network.odes` right-hand side,
evaluated with time-varying parameters — no physics is duplicated here.

A scenario config looks like:

    name: pump_degradation
    description: Coolant flow degrades to 40% over the run.
    workload:
      type: synthetic        # or: azure
      T_horizon: 600
      seed: 0
    overrides: {}            # static overrides on default thermal params
    perturbations:
      - param: m_dot         # m_dot | UA_hx | T_fac_in | power
        kind: ramp           # step | ramp
        start_s: 200
        end_s: 400
        to_factor: 0.4       # multiplicative; or to_value for absolute
    safety:
      T_limit: 90
"""
import os
import numpy as np
import yaml
from scipy.integrate import solve_ivp

from thermaloop.thermal import rc_network
from thermaloop.power.gpu_power import GPUPowerModel
from thermaloop.workload.synthetic import synthetic_workload
from thermaloop.hydraulics.pump import Pump
from thermaloop import safety

_FACTOR_PARAMS = {"m_dot", "UA_hx"}          # perturbed multiplicatively
_ABS_PARAMS = {"T_fac_in"}                    # perturbed by absolute value


def _schedule_value(base, pert, t):
    """Value of one perturbed parameter at time t."""
    start, end = pert["start_s"], pert.get("end_s", pert["start_s"])
    if "to_factor" in pert:
        target = base * pert["to_factor"]
        origin = base * pert.get("from_factor", 1.0)
    else:
        target = pert["to_value"]
        origin = pert.get("from_value", base)
    if t <= start:
        return origin
    if t >= end or pert["kind"] == "step":
        return target if t >= start else origin
    frac = (t - start) / (end - start)
    return origin + frac * (target - origin)


def _build_params_at(base_params, perturbations):
    """Return params_at(t): base params with time-varying perturbations applied."""
    therm_perts = [p for p in perturbations if p["param"] != "power"]

    def params_at(t):
        p = dict(base_params)
        for pert in therm_perts:
            p[pert["param"]] = _schedule_value(base_params[pert["param"]],
                                               pert, t)
        return p
    return params_at


def _build_power_func(t_axis, P_base, perturbations):
    """Return P_func(t) with any 'power' perturbation (e.g. workload spike) applied."""
    power_perts = [p for p in perturbations if p["param"] == "power"]
    dt = t_axis[1] - t_axis[0]

    def P_func(t):
        i = min(int(t / dt), len(P_base) - 1)
        val = P_base[i]
        for pert in power_perts:
            start, end = pert["start_s"], pert.get("end_s", t_axis[-1])
            if start <= t <= end:
                if "to_value" in pert:
                    val = max(val, pert["to_value"])
                elif "to_factor" in pert:
                    val = val * pert["to_factor"]
        return val
    return P_func


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_scenario(config, dt=1.0):
    """Run a scenario dict (or path). Returns a result dict with trajectories."""
    if isinstance(config, str):
        config = load_config(config)

    n_gpus = config.get("n_gpus", 8)
    base_params = rc_network.default_params(n_gpus=n_gpus)
    base_params.update(config.get("overrides", {}) or {})

    wl = config.get("workload", {}) or {}
    T_horizon = wl.get("T_horizon", 600.0)
    if wl.get("type", "synthetic") == "azure":
        from thermaloop.workload.azure import load_azure_trace
        arrivals, durations, n_pf, n_dec = load_azure_trace(
            wl["path"], t_horizon=T_horizon)
    else:
        arrivals, durations, n_pf, n_dec = synthetic_workload(
            T_horizon=T_horizon, seed=wl.get("seed", 0),
            base_rate=wl.get("base_rate", 2.0),
            burst_amplitude=wl.get("burst_amplitude", 3.0),
            short_frac=wl.get("short_frac", 0.7))

    gpm = GPUPowerModel(P_tdp=config.get("P_tdp", 700.0))
    t_axis, P_base = gpm.power_trace(arrivals, n_pf, n_dec, durations,
                                     dt=dt, T_horizon=T_horizon)

    perts = config.get("perturbations", []) or []
    params_at = _build_params_at(base_params, perts)
    P_func = _build_power_func(t_axis, P_base, perts)

    def rhs(t, T):
        return rc_network.odes(t, T, P_func, params_at(t))

    T0 = [base_params["T_fac_in"] + 25, base_params["T_fac_in"] + 20,
          base_params["T_fac_in"] + 15, base_params["T_fac_in"] + 8,
          base_params["T_fac_in"] + 4]
    sol = solve_ivp(rhs, (t_axis[0], t_axis[-1]), T0, t_eval=t_axis,
                    method="LSODA", rtol=1e-6, atol=1e-8, max_step=2.0)
    T_die, T_ihs, T_cp, T_loop, T_fac = sol.y

    # Pump energy over the run (flow may vary)
    pump = Pump(m_dot_ref=n_gpus * base_params["m_dot_ref"])
    m_dot_t = np.array([params_at(t)["m_dot"] for t in t_axis])
    pump_power_t = np.array([pump.power(m) for m in m_dot_t])
    pump_energy_Wh = float(np.trapezoid(pump_power_t, t_axis) / 3600.0)

    T_limit = (config.get("safety", {}) or {}).get("T_limit",
                                                   safety.DEFAULT_T_LIMIT)
    safety_summary = safety.evaluate(t_axis, T_die, T_limit=T_limit)

    # Effective per-GPU power actually applied (after power perturbations)
    P_applied = np.array([P_func(t) for t in t_axis])

    return dict(
        name=config.get("name", "scenario"),
        description=config.get("description", ""),
        t=t_axis,
        power_per_gpu=P_applied,
        T_die=T_die, T_ihs=T_ihs, T_coldplate=T_cp,
        T_loop=T_loop, T_fac_return=T_fac,
        m_dot_t=m_dot_t, pump_power_t=pump_power_t,
        pump_energy_Wh=pump_energy_Wh,
        mean_gpu_power_W=float(P_applied.mean()),
        safety=safety_summary,
        params=base_params, config=config,
    )
