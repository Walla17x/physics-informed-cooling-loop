"""Coolant fluid properties.

Temperature-dependent correlations for the coolants used in D2C loops. This
replaces the earlier blanket "constant fluid properties" assumption with a
selectable fluid whose specific heat, density, and viscosity follow real
correlations. Water is the default and is calibrated so cp at the nominal loop
temperature matches the value the validated baseline was anchored with (4186
J/kg/K), so existing validation is preserved.

Within a single run the solver evaluates properties at a representative loop
temperature (constant for that run). Full per-timestep property variation is a
documented future refinement; see docs/ASSUMPTIONS.md.

Correlations are engineering-grade fits over ~20-60 C, adequate for
system-level behavior, not a substitute for a property database.
"""


class Fluid:
    name = "fluid"

    def cp(self, T):       # J/kg/K
        raise NotImplementedError

    def rho(self, T):      # kg/m^3
        raise NotImplementedError

    def mu(self, T):       # Pa.s
        raise NotImplementedError


class Water(Fluid):
    """Pure water. cp tuned so cp(37 C) = 4186 J/kg/K to preserve the baseline
    anchor; variation across the operating range is small (<0.3%), as in reality.
    """
    name = "water"

    def cp(self, T):
        return 4186.0 + 0.10 * (T - 37.0) - 0.003 * (T - 37.0) ** 2

    def rho(self, T):
        return 1000.6 - 0.0128 * T - 0.0035 * T ** 2

    def mu(self, T):
        # Vogel-type, T in Celsius -> Kelvin
        Tk = T + 273.15
        return 2.414e-5 * 10 ** (247.8 / (Tk - 140.0))


class PropyleneGlycolWater(Fluid):
    """Propylene-glycol / water mixture (default 25% PG by mass), a common D2C
    coolant chosen for freeze protection and material compatibility. Lower
    specific heat and higher viscosity than water, so the same heat load
    produces a larger loop temperature rise.
    """
    name = "pg25"

    def __init__(self, pg_fraction=0.25):
        self.pg_fraction = pg_fraction

    def cp(self, T):
        # ~3910 J/kg/K at 37 C for 25% PG, rising mildly with T
        base = 4186.0 - 1100.0 * self.pg_fraction
        return base + 3.0 * (T - 37.0)

    def rho(self, T):
        return (1000.6 - 0.0128 * T - 0.0035 * T ** 2) + 85.0 * self.pg_fraction

    def mu(self, T):
        Tk = T + 273.15
        water_mu = 2.414e-5 * 10 ** (247.8 / (Tk - 140.0))
        # PG raises viscosity substantially
        return water_mu * (1.0 + 6.0 * self.pg_fraction)


_REGISTRY = {
    "water": Water,
    "pg25": lambda: PropyleneGlycolWater(0.25),
    "pg": lambda: PropyleneGlycolWater(0.25),
}


def get_fluid(name="water"):
    """Return a Fluid instance by name ('water', 'pg25')."""
    key = (name or "water").lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown fluid '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def apply_fluid(params, fluid_name="water", T_ref=None):
    """Set fluid-dependent params from the chosen fluid at a representative loop
    temperature. Returns (params, fluid). Defaults to water; T_ref defaults to
    facility supply + 7 K (a typical loop temperature).

    Sets:
      cp_water            specific heat at T_ref (J/kg/K)
      h_property_factor   cold-plate convection correction relative to water,
                          from a Dittus-Boelter property ratio
                          (mu_ref/mu)^0.4 * (cp/cp_ref)^0.4; exactly 1.0 for
                          water so the validated water anchor is preserved.
    """
    fluid = get_fluid(fluid_name)
    if T_ref is None:
        T_ref = params.get("T_fac_in", 30.0) + 7.0
    water_ref = Water()
    mu_ref, cp_ref = water_ref.mu(T_ref), water_ref.cp(T_ref)
    mu_f, cp_f = fluid.mu(T_ref), fluid.cp(T_ref)
    h_factor = (mu_ref / mu_f) ** 0.4 * (cp_f / cp_ref) ** 0.4
    p = dict(params)
    p["cp_water"] = cp_f
    p["h_property_factor"] = h_factor
    p["fluid_name"] = fluid.name
    return p, fluid
