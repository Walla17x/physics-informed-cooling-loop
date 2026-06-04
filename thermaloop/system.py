"""Full-server simulation: compose workload, power, thermal, CDU, pump, safety.

This is the convenience layer that assembles the subsystems into a single run.
Each subsystem lives in its own module and can be swapped independently; this
file only wires them together and adds pump-energy and safety bookkeeping.
"""
import numpy as np

from thermaloop.workload.synthetic import synthetic_workload
from thermaloop.power.gpu_power import GPUPowerModel
from thermaloop.thermal import rc_network
from thermaloop.hydraulics.pump import Pump
from thermaloop import safety


def run_server(params=None, workload=None, power_model=None,
               dt=1.0, T_horizon=900.0, T_limit=safety.DEFAULT_T_LIMIT,
               seed=0):
    """Run a single-server simulation end to end.

    Parameters
    ----------
    params : dict or None
        Thermal parameters (defaults to the validated 8-GPU server).
    workload : tuple or None
        (arrivals, durations, prefill_tokens, decode_tokens). If None, a
        synthetic workload is generated.
    power_model : GPUPowerModel or None
    dt, T_horizon : float
    T_limit : float
        Die-temperature limit for safety evaluation (C).

    Returns
    -------
    result : dict with time axis, power trace, node temperatures, pump power,
             and a safety summary.
    """
    params = params or rc_network.default_params()
    power_model = power_model or GPUPowerModel(P_tdp=700.0)

    if workload is None:
        workload = synthetic_workload(T_horizon=T_horizon, seed=seed)
    arrivals, durations, n_pf, n_dec = workload

    t_axis, P_per_gpu = power_model.power_trace(
        arrivals, n_pf, n_dec, durations, dt=dt, T_horizon=T_horizon)

    sol = rc_network.simulate(t_axis, P_per_gpu, params)
    T_die, T_ihs, T_cp, T_loop, T_fac = sol.y

    pump = Pump(m_dot_ref=params['n_gpus'] * params['m_dot_ref'])
    pump_power = pump.power(params['m_dot'])

    safety_summary = safety.evaluate(t_axis, T_die, T_limit=T_limit)

    return dict(
        t=t_axis,
        power_per_gpu=P_per_gpu,
        T_die=T_die, T_ihs=T_ihs, T_coldplate=T_cp,
        T_loop=T_loop, T_fac_return=T_fac,
        pump_power_W=pump_power,
        mean_gpu_power_W=float(P_per_gpu.mean()),
        safety=safety_summary,
        params=params,
    )


if __name__ == '__main__':
    r = run_server(T_horizon=300.0)
    print(f"mean GPU power : {r['mean_gpu_power_W']:.1f} W")
    print(f"peak T_die     : {r['safety']['peak_T_die']:.1f} C")
    print(f"min margin     : {r['safety']['min_margin_K']:.1f} K")
    print(f"pump power     : {r['pump_power_W']:.1f} W")
    print(f"throttled      : {r['safety']['throttled']}")
