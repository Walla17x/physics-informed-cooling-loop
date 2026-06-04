# ADR-000: Charter — scope and non-goals

Status: accepted

## Context

ThermaLoop began as a single-file direct-to-chip cooling simulator. The risk
with this kind of project is scope drift — toward a CFD competitor, an
"enterprise digital twin," or a general infrastructure-physics platform — each
of which is either already owned by well-resourced incumbents (Cadence, Ansys,
NVIDIA Omniverse) or a multi-year research effort. This charter fixes what
ThermaLoop is and, more importantly, what it is not, before features accrete.

## Decision

ThermaLoop is an **open, physics-informed simulation and analysis lab for
direct-to-chip liquid cooling loops in AI data centers.** It is a teaching and
methodology tool: transparent, modular, validated against published anchors,
and honest about its assumptions.

### In scope

- System-level physics: workload → power → thermal (lumped + 1-D) → CDU →
  facility, with pump energy and safety margins.
- Scenario-driven faults and parametric optimization sweeps.
- Clear visualization and reproducible, tested runs.

### Explicit non-goals

- **Not CFD.** No 3-D fields, meshing, or chip-level hot spots.
- **Not an enterprise digital twin.** No live telemetry, BMS integration, or
  state synchronization against real hardware.
- **Not a new scientific primitive.** The methods (RC networks, ε-NTU,
  affinity-law pumps) are standard. The value is implementation, framing,
  validation, and clarity — not novelty.
- **No RL or learned control in core.** A surrogate exists only under
  `experimental/`, clearly labeled and unsupported.
- **No multi-domain platform claims.** This is about cooling, period.

## Consequences

Contributors and reviewers can rely on the boundary: a feature request that
pushes toward CFD fidelity, production operations, or a "platform" is out of
scope by charter, not by case-by-case judgment. The honesty docs
(`ASSUMPTIONS.md`, `VALIDATION.md`) are part of the deliverable, not optional.
