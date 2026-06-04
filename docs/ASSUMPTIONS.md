# Assumptions and limitations

ThermaLoop is a teaching-grade simulation lab, not a CFD tool or a validated
production simulator. Every result is only as good as the assumptions below.
They are listed so you always know what the model can and cannot tell you.

## Thermal model

- **Lumped / 1-D, never 3-D.** Heat transfer is modeled as a small RC network
  (5 nodes) or a 1-D advecting loop (N cells). There is no spatial field inside
  the die, the cold plate, or across a manifold. For 3-D gradients, hot spots
  on the die, or detailed cold-plate channel design, you need CFD (Ansys,
  Cadence). ThermaLoop is for system-level behavior, not chip-level detail.
- **Fluid properties from a selectable fluid model.** Specific heat, density,
  and viscosity come from temperature-dependent correlations for the chosen
  coolant (`water` by default, `pg25` for a 25% propylene-glycol/water mixture).
  Viscosity feeds the cold-plate convection via a Dittus-Boelter property ratio,
  so a more viscous coolant correctly runs hotter. Properties are evaluated at a
  representative loop temperature and held constant *within* a run; full
  per-timestep property variation is a documented future refinement, not yet
  modeled. Water is calibrated so the validated anchor (cp = 4186 J/kg/K at the
  nominal loop temperature) is preserved exactly.
- **Single phase only.** No boiling, no two-phase cold plates, no immersion.
- **Resistances and capacitances are representative, not measured.** They are
  set to reproduce published steady-state behavior for H100-class hardware, not
  fitted to a specific server. Treat absolute temperatures as plausible, not
  exact.

## CDU model

- **Single-zone effectiveness-NTU.** The CDU is one counterflow heat exchanger
  with one UA value, calibrated so effectiveness is ~0.82 at design flow
  (Khalili et al. 2024). It does not resolve the internal temperature profile of
  the exchanger and does not enforce a dew-point / condensation guard.

## Hydraulics

- **Flow is an input, not a solved variable.** There is no hydraulic network
  balance. Pump electrical power is estimated from the affinity laws
  (`P ~ flow^3`), which captures the *energy cost* of flow but not pressure
  drop, cavitation, or pump-curve operating points.

## Power model

- **Three-state power machine.** GPU power is prefill / decode / idle with
  per-state values anchored to published H100 measurements, not a hardware
  power model. Decode power especially is a representative fraction of TDP.

## Workload

- **Synthetic by default.** The built-in generator is statistical, not a real
  trace. Real Azure LLM inference traces are supported (see `data/`) but are an
  optional input, not a validation of the workload model itself.

## Scope

- **Single server / rack.** No multi-rack, row-level, or facility-scale
  topology. No telemetry/BMS integration. No control loop, no optimization
  beyond the parametric sweeps. This is a sandbox for understanding behavior,
  not an operations tool.
