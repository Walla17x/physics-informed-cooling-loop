"""Coolant pump model.

A deliberately simple hydraulic model: flow is treated as a control input and
pump electrical power is estimated from the affinity laws rather than solved
from a full network hydraulic balance. This is enough to give flow an *energy
cost*, which is what makes the pump-speed / thermal-margin tradeoffs in the
optimization sweeps meaningful. See docs/ASSUMPTIONS.md for the limits of
this model.

Affinity laws (centrifugal pump, fixed system curve):
    Q   ~ N          (flow proportional to speed)
    H   ~ N^2        (head proportional to speed^2)
    P   ~ N^3        (shaft power proportional to speed^3)

So at fixed system resistance, electrical power scales with the cube of the
flow ratio.
"""


class Pump:
    """Affinity-law pump power model.

    Parameters
    ----------
    m_dot_ref : float
        Reference (design-point) coolant mass flow, kg/s.
    P_ref : float
        Electrical power drawn at the reference flow, W.
    P_parasitic : float
        Constant electronics/controls draw independent of flow, W.
    """

    def __init__(self, m_dot_ref, P_ref=250.0, P_parasitic=15.0):
        self.m_dot_ref = m_dot_ref
        self.P_ref = P_ref
        self.P_parasitic = P_parasitic

    def power(self, m_dot):
        """Electrical power (W) to sustain coolant mass flow `m_dot` (kg/s)."""
        ratio = m_dot / self.m_dot_ref if self.m_dot_ref > 0 else 0.0
        return self.P_parasitic + self.P_ref * ratio ** 3

    def speed_fraction(self, m_dot):
        """Pump speed as a fraction of reference speed (= flow ratio)."""
        return m_dot / self.m_dot_ref if self.m_dot_ref > 0 else 0.0
