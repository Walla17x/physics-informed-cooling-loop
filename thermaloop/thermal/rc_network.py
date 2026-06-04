"""Lumped 5-node RC thermal network for a D2C-cooled GPU server.

State vector: [T_die, T_ihs, T_coldplate, T_loop, T_facility_return]

Validated steady state (8 GPUs, 700 W each, design flow, 30 C facility supply):
    CDU effectiveness  ~ 0.82   (Khalili et al. 2024 anchor: 0.82-0.83)
    T_die              ~ 74 C
    loop dT vs facility~ 6.8 K
    heat balance closure 100 %

See docs/VALIDATION.md and docs/ASSUMPTIONS.md.
"""
import numpy as np
from scipy.integrate import solve_ivp

from thermaloop.cdu.epsilon_ntu import epsilon_ntu_counterflow


def default_params(n_gpus=8):
    """Validated default parameters for an 8-GPU H100-class server."""
    return dict(
        n_gpus=n_gpus,
        # Conductive resistances (K/W per GPU); sum ~0.045 -> typical H100 R_jc
        R_j_ihs=0.025,
        R_ihs_cp=0.020,
        # Cold-plate convection (microchannel), flow-scaled via Dittus-Boelter
        h0=30000.0,            # W/m^2/K reference convection coefficient
        A_cp=0.004,            # m^2 wetted area per cold plate
        m_dot_ref=0.03,        # kg/s reference flow per GPU
        # Capacitances (J/K, per server)
        C_die=5.0 * n_gpus,
        C_ihs=80.0 * n_gpus,
        C_cp=900.0 * n_gpus,
        C_loop=12000.0,
        C_fac=80000.0,
        # Flows (per server)
        m_dot=0.03 * n_gpus,       # secondary loop, kg/s (1.5 GPM/GPU)
        m_dot_fac=0.06 * n_gpus,   # facility/primary side, kg/s
        cp_water=4186.0,
        # CDU conductance calibrated so eps ~ 0.82 at design flow
        UA_hx=300.0 * n_gpus,
        # Facility supply temperature (warm-water cooling)
        T_fac_in=30.0,
    )


def _odes(t, T, P_func, p):
    T_j, T_ihs, T_cp, T_loop, T_fac = T
    P_total = P_func(t) * p['n_gpus']

    q_j_ihs = (T_j - T_ihs) / (p['R_j_ihs'] / p['n_gpus'])
    q_ihs_cp = (T_ihs - T_cp) / (p['R_ihs_cp'] / p['n_gpus'])
    h_eff = (p['h0'] * (p['m_dot'] / (p['n_gpus'] * p['m_dot_ref'])) ** 0.8
             * p.get('h_property_factor', 1.0))
    q_cp_loop = h_eff * (p['A_cp'] * p['n_gpus']) * (T_cp - T_loop)

    eps, C_min = epsilon_ntu_counterflow(
        p['m_dot'], p['m_dot_fac'], p['cp_water'], p['cp_water'], p['UA_hx'])
    Q_hx = eps * C_min * (T_loop - p['T_fac_in'])
    q_fac_out = p['m_dot_fac'] * p['cp_water'] * (T_fac - p['T_fac_in'])

    return [
        (P_total - q_j_ihs) / p['C_die'],
        (q_j_ihs - q_ihs_cp) / p['C_ihs'],
        (q_ihs_cp - q_cp_loop) / p['C_cp'],
        (q_cp_loop - Q_hx) / p['C_loop'],
        (Q_hx - q_fac_out) / p['C_fac'],
    ]


def simulate(t_axis, P_per_gpu, params, T0=None):
    """Integrate the lumped model over a per-GPU power trace.

    `P_per_gpu` may be an array aligned to `t_axis` or a callable P(t).
    Returns the scipy solve_ivp solution (sol.y rows are the 5 states).
    """
    if T0 is None:
        T0 = [params['T_fac_in'] + 25, params['T_fac_in'] + 20,
              params['T_fac_in'] + 15, params['T_fac_in'] + 8,
              params['T_fac_in'] + 4]

    if callable(P_per_gpu):
        P_func = P_per_gpu
    else:
        dt = t_axis[1] - t_axis[0]
        P_arr = np.asarray(P_per_gpu)

        def P_func(t):
            return P_arr[min(int(t / dt), len(P_arr) - 1)]

    return solve_ivp(_odes, (t_axis[0], t_axis[-1]), T0,
                     args=(P_func, params), t_eval=t_axis,
                     method='LSODA', rtol=1e-6, atol=1e-8, max_step=1.0)


def steady_state_closed_form(params, P_per_gpu_const=700.0):
    """Analytic steady state (no integration).

    This is the exact closed form the in-browser explorer (docs/explorer.html)
    reimplements in JavaScript. A test asserts it matches `steady_state`
    (solve_ivp) so the browser demo provably runs the validated physics.
    """
    n = params['n_gpus']
    cp = params['cp_water']
    hf = params.get('h_property_factor', 1.0)
    Q = P_per_gpu_const * n
    eps, C_min = epsilon_ntu_counterflow(
        params['m_dot'], params['m_dot_fac'], cp, cp, params['UA_hx'])
    h_eff = params['h0'] * (params['m_dot'] / (n * params['m_dot_ref'])) ** 0.8 * hf
    T_loop = params['T_fac_in'] + Q / (eps * C_min)
    T_cp = T_loop + Q / (h_eff * params['A_cp'] * n)
    T_ihs = T_cp + Q * (params['R_ihs_cp'] / n)
    T_die = T_ihs + Q * (params['R_j_ihs'] / n)
    return dict(T_die=T_die, T_ihs=T_ihs, T_coldplate=T_cp, T_loop=T_loop,
                epsilon=eps, Q_die_W=Q)


def steady_state(params, P_per_gpu_const=700.0, settle_s=2400.0):
    """Drive at constant power to steady state; return node temps + heat balance."""
    t_axis = np.arange(0, settle_s, 1.0)
    P = np.full_like(t_axis, P_per_gpu_const, dtype=float)
    sol = simulate(t_axis, P, params)
    T_j, T_ihs, T_cp, T_loop, T_fac = sol.y[:, -1]
    eps, C_min = epsilon_ntu_counterflow(
        params['m_dot'], params['m_dot_fac'],
        params['cp_water'], params['cp_water'], params['UA_hx'])
    Q_hx = eps * C_min * (T_loop - params['T_fac_in'])
    Q_die = P_per_gpu_const * params['n_gpus']
    return dict(T_die=T_j, T_ihs=T_ihs, T_coldplate=T_cp, T_loop=T_loop,
                T_fac_return=T_fac, Q_die_W=Q_die, Q_hx_W=Q_hx, epsilon=eps,
                deltaT_loop=T_loop - params['T_fac_in'],
                closure=Q_hx / Q_die if Q_die else float('nan'))


# Public alias so the scenario engine can reuse the validated RHS
# with time-varying parameters without duplicating the physics.
odes = _odes
