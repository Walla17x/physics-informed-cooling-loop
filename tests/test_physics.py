"""ThermaLoop test suite.

These tests encode the physics invariants and the published-anchor validation.
They are what let a reviewer trust the model without reading every line:
energy must balance, the steady state must match the Khalili anchor, and
perturbations must move temperatures in the physically correct direction.
"""
import numpy as np
import pytest

from thermaloop.thermal import rc_network, loop_1d
from thermaloop.cdu.epsilon_ntu import epsilon_ntu_counterflow
from thermaloop.hydraulics.pump import Pump
from thermaloop import system, safety


# --------------------------------------------------------------------------
# Anchor validation (Khalili et al. 2024: eps ~ 0.82-0.83)
# --------------------------------------------------------------------------
def test_steady_state_matches_anchor():
    ss = rc_network.steady_state(rc_network.default_params(), 700.0)
    assert 0.80 <= ss['epsilon'] <= 0.84, ss['epsilon']
    assert 68.0 <= ss['T_die'] <= 80.0, ss['T_die']
    assert 5.0 <= ss['deltaT_loop'] <= 10.0, ss['deltaT_loop']


def test_energy_balance_closes():
    ss = rc_network.steady_state(rc_network.default_params(), 700.0)
    assert abs(ss['closure'] - 1.0) < 0.02, ss['closure']


def test_energy_balance_closes_1d():
    ss = loop_1d.steady_state_1d(loop_1d.default_params_1d(), 700.0)
    closure = ss['Q_cdu_W'] / ss['Q_die_W']
    assert abs(closure - 1.0) < 0.02, closure


# --------------------------------------------------------------------------
# Monotonicity gates (physical-direction sanity)
# --------------------------------------------------------------------------
def test_more_power_hotter_die():
    p = rc_network.default_params()
    t_lo = rc_network.steady_state(p, 400.0)['T_die']
    t_hi = rc_network.steady_state(p, 700.0)['T_die']
    assert t_hi > t_lo


def test_less_flow_hotter_die():
    p = rc_network.default_params()
    base = rc_network.steady_state(p, 700.0)['T_die']
    p_low = dict(p, m_dot=p['m_dot'] * 0.5)
    low_flow = rc_network.steady_state(p_low, 700.0)['T_die']
    assert low_flow > base


def test_warmer_facility_hotter_die():
    p = rc_network.default_params()
    base = rc_network.steady_state(p, 700.0)['T_die']
    p_warm = dict(p, T_fac_in=p['T_fac_in'] + 8.0)
    warm = rc_network.steady_state(p_warm, 700.0)['T_die']
    assert warm > base


def test_cdu_fouling_hotter_die():
    p = rc_network.default_params()
    base = rc_network.steady_state(p, 700.0)['T_die']
    p_foul = dict(p, UA_hx=p['UA_hx'] * 0.6)   # degraded conductance
    foul = rc_network.steady_state(p_foul, 700.0)['T_die']
    assert foul > base


# --------------------------------------------------------------------------
# Cross-model consistency
# --------------------------------------------------------------------------
def test_lumped_and_1d_agree_at_steady_state():
    t_lump = rc_network.steady_state(rc_network.default_params(), 700.0)['T_die']
    t_1d = loop_1d.steady_state_1d(loop_1d.default_params_1d(), 700.0)['T_die']
    # 1D resolves spatial structure so it runs slightly cooler; within a few K
    assert abs(t_lump - t_1d) < 5.0, (t_lump, t_1d)


# --------------------------------------------------------------------------
# CDU effectiveness bounds
# --------------------------------------------------------------------------
def test_epsilon_in_unit_interval():
    for f in np.linspace(0.1, 2.0, 20):
        eps, _ = epsilon_ntu_counterflow(0.24 * f, 0.48, 4186, 4186, 2400)
        assert 0.0 <= eps <= 1.0, (f, eps)


# --------------------------------------------------------------------------
# Pump affinity law (P ~ flow^3)
# --------------------------------------------------------------------------
def test_pump_affinity_cubic():
    pump = Pump(m_dot_ref=0.24, P_ref=250.0, P_parasitic=0.0)
    p_ref = pump.power(0.24)
    p_half = pump.power(0.12)
    # halving flow -> 1/8 the dynamic power
    assert abs(p_half / p_ref - 0.125) < 1e-6, (p_ref, p_half)


def test_pump_power_monotonic_in_flow():
    pump = Pump(m_dot_ref=0.24)
    flows = np.linspace(0.05, 0.40, 10)
    powers = [pump.power(f) for f in flows]
    assert all(b > a for a, b in zip(powers, powers[1:]))


# --------------------------------------------------------------------------
# Safety bookkeeping
# --------------------------------------------------------------------------
def test_time_to_throttle_detects_crossing():
    t = np.arange(0, 10, 1.0)
    T = np.linspace(80, 100, 10)        # crosses 90 partway
    ttt = safety.time_to_throttle(t, T, T_limit=90.0)
    assert ttt is not None and 4.0 <= ttt <= 5.0, ttt


def test_no_throttle_when_cool():
    t = np.arange(0, 10, 1.0)
    T = np.full(10, 70.0)
    assert safety.time_to_throttle(t, T, T_limit=90.0) is None


# --------------------------------------------------------------------------
# Reproducibility and numerical health
# --------------------------------------------------------------------------
def test_seeded_run_is_reproducible():
    a = system.run_server(T_horizon=120.0, seed=42)
    b = system.run_server(T_horizon=120.0, seed=42)
    assert np.allclose(a['T_die'], b['T_die'])


def test_no_nans_in_baseline_run():
    r = system.run_server(T_horizon=300.0, seed=1)
    for key in ['T_die', 'T_loop', 'T_coldplate']:
        assert np.isfinite(r[key]).all(), key


# --------------------------------------------------------------------------
# Closed-form == solver  (the contract the browser explorer relies on)
# --------------------------------------------------------------------------
def test_closed_form_matches_solver():
    from thermaloop.fluids import apply_fluid
    cases = [
        (700, 1.0, 30, "water", 1.0), (500, 1.0, 30, "water", 1.0),
        (700, 0.6, 30, "water", 1.0), (700, 1.0, 42, "water", 1.0),
        (700, 1.0, 30, "pg25", 1.0),  (700, 1.0, 30, "water", 0.6),
        (900, 1.2, 22, "pg25", 0.8),
    ]
    for P_gpu, flow, Tfac, fluid, ua in cases:
        p = rc_network.default_params()
        p, _ = apply_fluid(p, fluid, T_ref=Tfac + 7.0)
        p["m_dot"] *= flow
        p["UA_hx"] *= ua
        p["T_fac_in"] = Tfac
        cf = rc_network.steady_state_closed_form(p, P_gpu)["T_die"]
        ivp = rc_network.steady_state(p, P_gpu)["T_die"]
        assert abs(cf - ivp) < 0.05, (P_gpu, flow, Tfac, fluid, ua, cf, ivp)
