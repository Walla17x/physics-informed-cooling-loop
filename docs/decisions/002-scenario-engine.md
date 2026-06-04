# ADR-002: Scenario engine via time-varying parameters

Status: accepted

## Context

Faults (pump degradation, CDU fouling, warm facility water) and the workload
spike all require parameters to change *during* a run. The validated steady and
transient physics in `rc_network` takes static parameters.

## Decision

Scenarios are YAML files with a perturbation timeline (step/ramp on `m_dot`,
`UA_hx`, `T_fac_in`, or `power`). The engine evaluates a `params_at(t)`
function at each integration step and feeds it to the *same* validated
right-hand side (`rc_network.odes`). No physics is duplicated; the only new code
is the schedule that varies parameters over time.

Sweeps are handled separately (`sweeps.py`) as steady-state evaluations over a
parameter grid, since they ask a design question rather than a transient one.

## Consequences

Adding a fault or a sweep is a config file, not code. The physics stays in one
place and stays the version the test suite validates. Power perturbations are
applied as an overlay on the workload-derived power trace so a lightly loaded
server can be hit with a spike without changing the workload model.
