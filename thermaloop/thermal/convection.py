"""Cold-plate convection flow-scaling: regime-aware exponent selection.

ThermaLoop's cold-plate convection term is

    h_eff = h0 * (m_dot / m_dot_ref) ** flow_exponent * h_property_factor

Historically the exponent was hard-coded to 0.8 (Dittus-Boelter, turbulent,
valid for Re >~ 1e4). Measured cold-plate data falsifies that exponent in the
laminar regime that real single-phase D2C cold plates occupy at a few L/min:

  * Martinez, Caceres & Ortega (2024), J. Electron. Packag. 146(4) 041118 —
    microchannel cold plate "S" (120 channels, 0.2 x 4 x 43 mm, D_h = 0.381 mm),
    PG25, 1-4 LPM, 1000 W: channel Re = 28-171 (deeply laminar). Measured
    R_Fo(flow) is reproduced within measurement uncertainty (RMS ~2%) by a
    flow-independent convection coefficient (laminar fully-developed,
    Nu ~ const -> exponent 0), while exponent 0.8 misfits by ~29% RMS
    (+40% at 1 LPM). See tests/test_validation_anchors.py.

Regime guidance (smooth channels, engineering approximations):
    laminar fully developed   Re < ~2300, L >> L_th : exponent ~ 0.0
    laminar thermally developing (Graetz/Shah-London), L <~ L_th :
                              exponent ~ 1/3
    transitional              2300 < Re < ~4000     : ill-defined; warn
    turbulent (Dittus-Boelter) Re > ~1e4            : exponent 0.8

Thermal entry length: L_th ~ 0.05 * Re * Pr * D_h. For PG25 (Pr ~ 15) at
Re ~ 100, D_h ~ 0.4 mm: L_th ~ 30 mm — comparable to typical channel lengths,
so real plates often sit between exponent 0 and 1/3.

The default remains 0.8 for backward compatibility with the calibrated
Heydari anchor operating point (the calibration point itself is unaffected by
the exponent; only off-design flow sensitivity changes). New configs SHOULD
set `flow_exponent` deliberately, or provide channel geometry so
`check_regime` can flag a mismatch at runtime.
"""

LAMINAR_RE = 2300.0
TURBULENT_RE = 1.0e4

# Named exponent presets
EXPONENTS = {
    "laminar_developed": 0.0,
    "laminar_developing": 1.0 / 3.0,
    "turbulent_dittus_boelter": 0.8,
}


def channel_reynolds(m_dot_total, n_channels, chan_w, chan_h, rho, mu):
    """Reynolds number in one rectangular channel of a cold plate.

    m_dot_total : kg/s delivered to the plate (split evenly across channels)
    chan_w, chan_h : channel cross-section, m
    """
    d_h = 2.0 * chan_w * chan_h / (chan_w + chan_h)
    a_c = chan_w * chan_h
    v = m_dot_total / n_channels / (rho * a_c)
    return rho * v * d_h / mu


def classify_regime(re):
    if re < LAMINAR_RE:
        return "laminar"
    if re < TURBULENT_RE:
        return "transitional"
    return "turbulent"


def recommended_exponent(re, developing=True):
    """Flow exponent appropriate to the regime at Reynolds number `re`."""
    regime = classify_regime(re)
    if regime == "laminar":
        return EXPONENTS["laminar_developing"] if developing \
            else EXPONENTS["laminar_developed"]
    if regime == "turbulent":
        return EXPONENTS["turbulent_dittus_boelter"]
    # transitional: no reliable correlation; prefer the conservative laminar
    # developing value and let check_regime warn.
    return EXPONENTS["laminar_developing"]


def check_regime(re, flow_exponent):
    """Validity check: does the configured exponent match the flow regime?

    Returns (ok, message). `ok` is False when the configured exponent's
    validity range excludes the operating Reynolds number — e.g. the
    Dittus-Boelter exponent 0.8 applied at laminar Re.
    """
    regime = classify_regime(re)
    if regime == "laminar" and flow_exponent > 0.5:
        return False, (
            f"Re={re:.0f} is laminar but flow_exponent={flow_exponent:.2f} "
            f"is a turbulent (Dittus-Boelter) scaling; measured laminar "
            f"cold-plate data rejects this (Martinez et al. 2024 anchor, "
            f"~29% RMS misfit). Use ~0-0.33.")
    if regime == "turbulent" and flow_exponent < 0.5:
        return False, (
            f"Re={re:.0f} is turbulent but flow_exponent="
            f"{flow_exponent:.2f} is a laminar scaling; use ~0.8.")
    if regime == "transitional":
        return False, (
            f"Re={re:.0f} is transitional (2300-1e4): no reliable scaling; "
            f"results carry extra uncertainty in flow sensitivity.")
    return True, f"Re={re:.0f} {regime}: flow_exponent={flow_exponent:.2f} ok"
