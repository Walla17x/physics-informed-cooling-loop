# ADR-003: Temperature-dependent, selectable fluid properties

Status: accepted

## Context

v0.1 held fluid properties constant (cp = 4186 J/kg/K). This hid a real effect:
glycol/water coolants, used in practice for freeze protection, have lower
specific heat and substantially higher viscosity than water, which raises die
temperature. "Constant properties regardless of fluid" was an honest but
limiting assumption.

## Decision

Add a `fluids` module with temperature-dependent correlations for `water` and
`pg25` (25% propylene-glycol/water), selectable per scenario via a `fluid:` key.
`apply_fluid` sets `cp_water` from the fluid and a `h_property_factor` that
corrects cold-plate convection by a Dittus-Boelter property ratio
`(mu_ref/mu)^0.4 * (cp/cp_ref)^0.4`.

Water is calibrated so `cp(37 C) = 4186` and `h_property_factor = 1.0` exactly,
so the validated water anchor and all existing tests are unchanged. Properties
are evaluated at a representative loop temperature and constant within a run.

## Consequences

The "constant properties" assumption is replaced by a fluid model that is real
physics, while water behavior is bit-for-bit preserved. Running `pg_coolant.yaml`
shows die temperature rising ~2.5 K and margin dropping, driven mostly by the
viscosity-degraded convection (h factor ~0.67). The remaining simplification —
constant properties *within* a run — is stated in ASSUMPTIONS.md as future work.
