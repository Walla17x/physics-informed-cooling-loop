# ADR-004: Parametric uncertainty quantification via ensemble scenario runs

Status: accepted

## Context

The scenario engine reports a single point estimate of die temperature,
margin, and time-to-throttle for a given parameter set. Operators do not face
point estimates — real values of cold-plate convection coefficient, CDU
conductance, TIM resistance, and facility supply temperature carry uncertainty
on the order of 10-20 % around any specified design point. A margin reported
as "8.2 K" with no width is operationally less actionable than the same
margin reported as a P5/P50/P95 of "3.4 / 8.2 / 12.1 K", and it is weaker
evidence to a reviewer that the validated point estimate is robust.

## Decision

Add an ensemble runner that draws uncertain parameter values from declared
distributions and runs N scenarios. Each sample is a scenario run with the
draw merged into `overrides` — the same mechanism the existing engine already
uses to apply static parameter overrides. No new right-hand side, no
duplicated physics. Same philosophy as ADR-002.

Spec format extends scenario YAML with an `ensemble` block:

    ensemble:
      n_samples: 200
      seed: 1
      sampler: lhs                       # lhs | mc
      uncertain:
        - { param: h0,    dist: lognormal, mean: 30000.0, cv: 0.15 }
        - { param: UA_hx, dist: normal,    mean_factor: 1.0, cv: 0.10 }
        - { param: R_ihs_cp, dist: normal, mean: 0.020, cv: 0.10, clip_min: 0.005 }
        - { param: T_fac_in, dist: uniform, low: 28.0, high: 32.0 }

Supported distributions for v0: `normal`, `lognormal`, `uniform`,
`triangular`. `mean_factor` × default is the natural counterpart to
`to_factor` in the existing perturbation schema. Optional `clip_min` /
`clip_max` for physical floors and ceilings. Default sampler is Latin
hypercube; Monte Carlo is available with `sampler: mc`.

Per-sample output: scenario summary (min margin, peak T_die, throttle,
time-to-throttle, pump energy) plus the draws used. `T_die(t)` is kept per
sample so the report can render a 5-95 % envelope over time. The other four
node trajectories are dropped to bound memory at typical N.

Aggregate output: per-quantity percentiles {5, 25, 50, 75, 95}, throttle
probability across samples, and the `T_die(t)` sample grid.

CLI: `python -m thermaloop ensemble <config>`. The HTML report renders three
plots: margin distribution, peak-T_die distribution, and the
die-temperature envelope over time.

## Consequences

- The headline operational quantity (margin) becomes a distribution, not a
  scalar. The CLI verdict flags throttle probability > 5 % as at-risk.
- The reduced-order solver is unchanged. The Khalili validation anchor and
  the closed-form/JS equivalence test stay intact.
- Independence is assumed across uncertain parameters in v0. Some pairs
  (cold-plate convection and TIM resistance, both driven by manufacturing
  tolerance) are physically correlated. Correlation matrices are deferred
  to a follow-up, documented in `ASSUMPTIONS.md`.
- Sampling is single-process in v0. At N=200 with `T_horizon=600` s the
  serial ensemble runs in ~1-2 minutes; multiprocessing is a follow-up if
  larger N becomes routine.
- Sensitivity decomposition (Sobol / Morris) is a separate concern and a
  separate ADR; this ADR is scope-limited to propagating uncertainty, not
  attributing it.

## Honest framing

The contribution is the validated D2C cooling base plus the application of
UQ to it — not UQ itself. Published material describes this as "ThermaLoop
wraps the validated solver in parametric UQ so margin reports as a
distribution, with realistic CVs propagated from the anchor to fault
scenarios," not as a methodological advance in UQ.
