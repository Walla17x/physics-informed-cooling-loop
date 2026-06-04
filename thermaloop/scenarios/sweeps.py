"""Parametric optimization sweeps.

Runs the steady-state model across a grid of one parameter and collects the
metrics that define the engineering tradeoff (die temperature, pump power,
safety margin). Used for pump-speed, flow-vs-margin, and CDU-setpoint studies.
"""
import numpy as np
import yaml

from thermaloop.thermal import rc_network
from thermaloop.hydraulics.pump import Pump
from thermaloop import safety


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_sweep(config):
    """Run a sweep dict (or path). Returns grid + collected metrics."""
    if isinstance(config, str):
        config = load_config(config)

    n_gpus = config.get("n_gpus", 8)
    base = rc_network.default_params(n_gpus=n_gpus)
    base.update(config.get("overrides", {}) or {})
    from thermaloop.fluids import apply_fluid
    base, _fluid = apply_fluid(base, config.get("fluid", "water"))
    P_const = config.get("P_per_gpu", 700.0)
    T_limit = (config.get("safety", {}) or {}).get("T_limit",
                                                   safety.DEFAULT_T_LIMIT)

    sweep = config["sweep"]
    param = sweep["param"]
    if "values_factor" in sweep:
        values = [base[param] * f for f in sweep["values_factor"]]
        axis_label_vals = sweep["values_factor"]
        axis_is_factor = True
    else:
        values = list(sweep["values"])
        axis_label_vals = values
        axis_is_factor = False

    pump = Pump(m_dot_ref=n_gpus * base["m_dot_ref"])
    rows = []
    for v, axv in zip(values, axis_label_vals):
        p = dict(base)
        p[param] = v
        ss = rc_network.steady_state(p, P_const)
        pump_power = pump.power(p["m_dot"])
        rows.append(dict(
            sweep_value=axv,
            raw_value=v,
            T_die=ss["T_die"],
            T_loop=ss["T_loop"],
            epsilon=ss["epsilon"],
            margin_K=T_limit - ss["T_die"],
            pump_power_W=pump_power,
        ))
    return dict(
        name=config.get("name", "sweep"),
        description=config.get("description", ""),
        param=param,
        axis_is_factor=axis_is_factor,
        T_limit=T_limit,
        P_per_gpu=P_const,
        rows=rows,
        config=config,
    )
