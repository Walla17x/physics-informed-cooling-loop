"""
1D finite-volume discretization of the D2C secondary loop.

Replaces the lumped T_loop node from the original simulator with N cells
arranged around a closed loop. Cold plates inject heat over a contiguous
range of cells; the CDU removes heat over another range. Coolant advects
between cells (upwind scheme).

State vector layout:
    [T_die, T_ihs, T_cp, T_cell_0, ..., T_cell_{N-1}, T_fac_return]

Cell index convention (default geometry):
    0 .. N_cp-1               cold plate cells (heat input)
    N_cp .. N_cp+N_hot-1      hot leg (adiabatic transit)
    N_cp+N_hot ..             CDU cells (heat output)
        ... + N_cdu-1
    remaining                 cold leg back to cold plate

Steady-state heat balance: integral of cold-plate heat input equals integral
of CDU heat removal, calibrated to match the lumped-model effectiveness at
design flow.
"""
import numpy as np
from scipy.integrate import solve_ivp

from thermaloop.cdu.epsilon_ntu import epsilon_ntu_counterflow as _eps_ntu


def make_loop_geometry(N=50, frac_cp=0.20, frac_cdu=0.20,
                       L_loop=4.0, V_loop=2.87e-3):
    """
    Define the discretized loop. V_loop is total coolant volume; gives a
    lumped C_loop matching the original lumped model (~12000 J/K of water).
    """
    N_cp = max(1, int(round(N * frac_cp)))
    N_cdu = max(1, int(round(N * frac_cdu)))
    # Cells 0..N_cp-1 are CP, then a gap, then CDU, then return leg
    idx_cp = np.arange(0, N_cp)
    gap_after_cp = int(round((N - N_cp - N_cdu) * 0.4))
    idx_cdu = np.arange(N_cp + gap_after_cp,
                        N_cp + gap_after_cp + N_cdu)
    # Per-cell water mass (kg) -- distributes total loop volume evenly
    rho = 1000.0
    m_cell = rho * V_loop / N
    dx = L_loop / N
    return dict(N=N, N_cp=N_cp, N_cdu=N_cdu,
                idx_cp=idx_cp, idx_cdu=idx_cdu,
                m_cell=m_cell, dx=dx, L_loop=L_loop)


def default_params_1d(n_gpus=8, geom=None):
    if geom is None:
        geom = make_loop_geometry()
    return dict(
        n_gpus=n_gpus,
        geom=geom,
        # Resistances (K/W) -- same as lumped
        R_j_ihs=0.025,
        R_ihs_cp=0.020,
        # Cold plate convection (modern microchannel); flow_exponent is
        # regime-dependent -- see thermal/convection.py (default 0.8 kept for
        # backward compatibility with the calibrated anchor point).
        h0=30000.0,
        A_cp=0.004,
        m_dot_ref=0.03,
        flow_exponent=0.8,
        # Capacitances (J/K)
        C_die=5.0 * n_gpus,
        C_ihs=80.0 * n_gpus,
        C_cp=900.0 * n_gpus,
        C_fac=80000.0,
        # Flow
        m_dot=0.03 * n_gpus,
        m_dot_fac=0.06 * n_gpus,
        cp_water=4186.0,
        # CDU: distributed UA matched to lumped-model design point
        # (lumped UA_hx=300 W/K/GPU gave eps=0.82; for 1D we use the same
        # total UA distributed across the CDU cells)
        UA_hx_total=300.0 * n_gpus,
        T_fac_in=30.0,
    )


def thermal_odes_1d(t, T, P_func, p):
    """
    Right-hand side for the 1D loop model.
    State: [T_die, T_ihs, T_cp, T_loop_0..T_loop_{N-1}, T_fac_return]
    """
    g = p['geom']
    N = g['N']
    T_die, T_ihs, T_cp = T[0], T[1], T[2]
    T_loop = T[3:3 + N]
    T_fac = T[3 + N]

    # Conductive path: die -> IHS -> cold plate
    q_j_ihs = (T_die - T_ihs) / (p['R_j_ihs'] / p['n_gpus'])
    q_ihs_cp = (T_ihs - T_cp) / (p['R_ihs_cp'] / p['n_gpus'])

    # Cold plate -> coolant convection, distributed over CP cells
    h_eff = (p['h0'] * (p['m_dot'] / (p['n_gpus'] * p['m_dot_ref']))
             ** p.get('flow_exponent', 0.8)
             * p.get('h_property_factor', 1.0))
    A_total = p['A_cp'] * p['n_gpus']
    A_per_cp_cell = A_total / g['N_cp']
    q_cp_cells = h_eff * A_per_cp_cell * (T_cp - T_loop[g['idx_cp']])  # array
    q_cp_total = q_cp_cells.sum()

    # CDU -> facility, distributed over CDU cells
    UA_per_cdu_cell = p['UA_hx_total'] / g['N_cdu']
    q_cdu_cells = UA_per_cdu_cell * (T_loop[g['idx_cdu']] - p['T_fac_in'])
    q_cdu_total = q_cdu_cells.sum()

    # Advection: upwind, closed loop
    # m_dot * cp * (T_{i-1} - T_i)
    m_cp = p['m_dot'] * p['cp_water']
    T_upstream = np.empty(N)
    T_upstream[1:] = T_loop[:-1]
    T_upstream[0] = T_loop[-1]   # closed loop
    advection = m_cp * (T_upstream - T_loop)

    # Per-cell capacitance
    C_cell = g['m_cell'] * p['cp_water']

    # Sources at each cell
    source = np.zeros(N)
    source[g['idx_cp']] = q_cp_cells
    source[g['idx_cdu']] = -q_cdu_cells

    # Loop cell ODEs
    dT_loop = (advection + source) / C_cell

    # Lumped node ODEs
    P_total = P_func(t) * p['n_gpus']
    dT_die = (P_total - q_j_ihs) / p['C_die']
    dT_ihs = (q_j_ihs - q_ihs_cp) / p['C_ihs']
    dT_cp = (q_ihs_cp - q_cp_total) / p['C_cp']
    dT_fac = (q_cdu_total - p['m_dot_fac'] * p['cp_water']
              * (T_fac - p['T_fac_in'])) / p['C_fac']

    return np.concatenate(([dT_die, dT_ihs, dT_cp],
                           dT_loop,
                           [dT_fac]))


def simulate_1d(t_axis, P_per_gpu, params, T0=None):
    g = params['geom']
    N = g['N']
    if T0 is None:
        # Initialize with a smooth gradient: hot just after CP, cool just after CDU
        T_loop0 = np.full(N, params['T_fac_in'] + 6.0)
        T_loop0[g['idx_cp']] = params['T_fac_in'] + 7.0
        T_loop0[g['idx_cdu']] = params['T_fac_in'] + 3.0
        T0 = np.concatenate((
            [params['T_fac_in'] + 35,
             params['T_fac_in'] + 25,
             params['T_fac_in'] + 12],
            T_loop0,
            [params['T_fac_in'] + 2]
        ))

    dt = t_axis[1] - t_axis[0]

    def P_func(t):
        i = min(int(t / dt), len(P_per_gpu) - 1)
        return P_per_gpu[i]

    sol = solve_ivp(thermal_odes_1d,
                    (t_axis[0], t_axis[-1]), T0,
                    args=(P_func, params), t_eval=t_axis,
                    method='LSODA', rtol=1e-6, atol=1e-7,
                    max_step=1.0)
    return sol


def steady_state_1d(params, P_const=700.0):
    """Drive at constant power until steady state, report node values."""
    t_axis = np.arange(0, 2400, 1.0)
    P = np.full_like(t_axis, P_const, dtype=float)
    sol = simulate_1d(t_axis, P, params)
    g = params['geom']
    N = g['N']
    T_die, T_ihs, T_cp = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
    T_loop = sol.y[3:3 + N, -1]
    T_fac = sol.y[3 + N, -1]
    # Heat balance check
    h_eff = (params['h0'] * (params['m_dot'] / (params['n_gpus'] * params['m_dot_ref']))
             ** params.get('flow_exponent', 0.8)
             * params.get('h_property_factor', 1.0))
    A_per_cp = params['A_cp'] * params['n_gpus'] / g['N_cp']
    UA_per_cdu = params['UA_hx_total'] / g['N_cdu']
    Q_cp = (h_eff * A_per_cp * (T_cp - T_loop[g['idx_cp']])).sum()
    Q_cdu = (UA_per_cdu * (T_loop[g['idx_cdu']] - params['T_fac_in'])).sum()
    Q_die = P_const * params['n_gpus']
    eps_eff = Q_cdu / (params['m_dot'] * params['cp_water']
                       * (T_loop[g['idx_cdu']].mean() - params['T_fac_in']))
    return dict(T_die=T_die, T_ihs=T_ihs, T_cp=T_cp,
                T_loop_mean=T_loop.mean(),
                T_loop_max=T_loop.max(), T_loop_min=T_loop.min(),
                T_loop_profile=T_loop, T_fac=T_fac,
                Q_die_W=Q_die, Q_cp_W=Q_cp, Q_cdu_W=Q_cdu,
                eps_effective=eps_eff)
