"""Coolant Distribution Unit (CDU) heat-exchanger model.

Counterflow effectiveness-NTU. UA is calibrated so that effectiveness lands
at ~0.82 at design flow, matching the experimental anchor in Khalili et al.,
Applied Thermal Engineering 240 (2024).
"""
import numpy as np


def epsilon_ntu_counterflow(m_hot, m_cold, cp_hot, cp_cold, UA):
    """Counterflow heat-exchanger effectiveness and minimum heat-capacity rate.

    Parameters
    ----------
    m_hot, m_cold : float
        Mass flow rates (kg/s) of the hot (secondary loop) and cold
        (facility) streams.
    cp_hot, cp_cold : float
        Specific heats (J/kg/K).
    UA : float
        Overall conductance (W/K).

    Returns
    -------
    eps : float
        Effectiveness in [0, 1].
    C_min : float
        Minimum heat-capacity rate (W/K).
    """
    C_hot = m_hot * cp_hot
    C_cold = m_cold * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    Cr = C_min / C_max if C_max > 0 else 0.0
    NTU = UA / C_min if C_min > 0 else 0.0
    if Cr < 0.999:
        eps = (1.0 - np.exp(-NTU * (1.0 - Cr))) / \
              (1.0 - Cr * np.exp(-NTU * (1.0 - Cr)))
    else:
        eps = NTU / (1.0 + NTU)
    return eps, C_min


def heat_rejected(T_loop, T_fac_in, m_loop, m_fac, cp_water, UA):
    """Heat removed from the secondary loop by the CDU (W)."""
    eps, C_min = epsilon_ntu_counterflow(m_loop, m_fac, cp_water, cp_water, UA)
    return eps * C_min * (T_loop - T_fac_in)
