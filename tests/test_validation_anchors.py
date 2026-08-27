"""Validation-anchor CI locks: Villanova cold-plate anchor (Martinez et al.
2024, JEP 146(4) 041118).

Three locked properties:

1. REGIME LOCK — the anchor's operating envelope is laminar (Re ~ 28-171)
   when computed with ThermaLoop's own PG25 property model. If a fluids
   change moves this materially, the anchor framing must be revisited.

2. LAMINAR FIT — a flow-independent convection coefficient (laminar
   fully-developed, flow_exponent = 0) reproduces the measured R_Fo(flow)
   sweep within 5% RMS (measurement uncertainty ~10%).

3. FALSIFICATION GUARD — the turbulent Dittus-Boelter exponent (0.8, the
   historical hard-coded value) misfits the same measured sweep by >20% RMS.
   This documents, permanently and executably, why flow_exponent became a
   regime-dependent parameter. If this test ever fails, the guard has been
   silently weakened — investigate before touching the tolerance.

Anchor model: R_Fo(m) = R_fixed + 1/(eps(m)*m*cp), UA ~ m^n, calibrated at
2 LPM. R_fixed absorbs TIM + conduction (the measured R_Fo is a
cold-plate/TIM SYSTEM resistance — see the YAML measure-check notes).
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

from thermaloop.fluids import get_fluid
from thermaloop.thermal import convection

CFG = Path(__file__).parent.parent / "configs" / "validation" / \
    "villanova_coldplate.yaml"


@pytest.fixture(scope="module")
def anchor():
    with open(CFG) as f:
        return yaml.safe_load(f)


def _fit(anchor, flow_exponent):
    """Return (measured, predicted) R_Fo arrays for a given scaling."""
    pg = get_fluid(anchor["fluid"])
    tin = anchor["inlet_temp_C"]
    rho, cp = pg.rho(tin), pg.cp(tin)
    data = anchor["measured"]["R_Fo_vs_flow"]
    flows = np.array(sorted(float(k) for k in data))
    r_meas = np.array([data[k] for k in sorted(data)])
    mdot = flows / 1000.0 / 60.0 * rho

    cal_lpm = anchor["calibration"]["calibrate_at_LPM"]
    icals = np.where(flows == cal_lpm)[0]
    assert icals.size == 1, "calibration flow must be in the measured set"
    ical = icals[0]

    ua_cal = 1.0 / anchor["measured"]["R_LMTD_KperW"]
    ntu = ua_cal / (mdot[ical] * cp)
    eps = 1.0 - np.exp(-ntu)
    r_fixed = r_meas[ical] - 1.0 / (eps * mdot[ical] * cp)
    assert r_fixed > 0, "TIM+conduction residual must be physical"

    ua = ua_cal * (mdot / mdot[ical]) ** flow_exponent
    ntu = ua / (mdot * cp)
    eps = 1.0 - np.exp(-ntu)
    r_pred = r_fixed + 1.0 / (eps * mdot * cp)
    return flows, r_meas, r_pred, ical


def _rms(r_meas, r_pred, skip):
    idx = [i for i in range(len(r_meas)) if i != skip]
    rel = (r_pred[idx] - r_meas[idx]) / r_meas[idx]
    return float(np.sqrt(np.mean(rel ** 2)))


def test_anchor_regime_is_laminar(anchor):
    pg = get_fluid(anchor["fluid"])
    g = anchor["geometry"]
    data = anchor["measured"]["R_Fo_vs_flow"]
    for lpm in (min(data), max(data)):
        for t in (22.0, 32.0, 42.0):
            mdot = float(lpm) / 1000.0 / 60.0 * pg.rho(t)
            re = convection.channel_reynolds(
                mdot, g["n_channels"], g["channel_width_m"],
                g["channel_height_m"], pg.rho(t), pg.mu(t))
            assert anchor["regime"]["Re_min_expected"] < re < \
                anchor["regime"]["Re_max_expected"]
            assert convection.classify_regime(re) == "laminar"
            ok, _ = convection.check_regime(re, 0.8)
            assert not ok, "regime check must flag 0.8 as invalid here"
            ok, _ = convection.check_regime(re, 0.0)
            assert ok


def test_laminar_scaling_fits_measured_sweep(anchor):
    _, r_meas, r_pred, ical = _fit(anchor, flow_exponent=0.0)
    rms = _rms(r_meas, r_pred, skip=ical)
    assert rms <= anchor["tolerances"]["laminar_fit_rms_max"], \
        f"laminar fit RMS {rms:.3f} exceeds lock"


def test_turbulent_scaling_falsified_by_measured_sweep(anchor):
    _, r_meas, r_pred, ical = _fit(anchor, flow_exponent=0.8)
    rms = _rms(r_meas, r_pred, skip=ical)
    assert rms >= anchor["tolerances"]["turbulent_misfit_rms_min"], \
        f"falsification guard weakened: 0.8 misfit RMS only {rms:.3f}"


def test_flow_exponent_plumbing_lumped_closed_form():
    """flow_exponent must actually reach the closed-form solver."""
    from thermaloop.thermal import rc_network
    p8 = rc_network.default_params()
    p0 = rc_network.default_params()
    p0["flow_exponent"] = 0.0
    # at design flow (m_dot == n*m_dot_ref) exponent is irrelevant:
    t8 = rc_network.steady_state_closed_form(p8)["T_die"]
    t0 = rc_network.steady_state_closed_form(p0)["T_die"]
    assert abs(t8 - t0) < 1e-9
    # off design (half flow) laminar h is flow-independent -> cooler die
    p8["m_dot"] *= 0.5
    p0["m_dot"] *= 0.5
    t8 = rc_network.steady_state_closed_form(p8)["T_die"]
    t0 = rc_network.steady_state_closed_form(p0)["T_die"]
    assert t0 < t8, "laminar (n=0) must not degrade h at reduced flow"
    # monotonicity preserved under laminar scaling: less flow still hotter
    p_ref = rc_network.default_params()
    p_ref["flow_exponent"] = 0.0
    t_ref = rc_network.steady_state_closed_form(p_ref)["T_die"]
    assert t0 > t_ref, "caloric term must keep less-flow->hotter direction"
