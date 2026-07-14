# ThermaLoop: An Open, Physics-Informed Simulation Lab for Direct-to-Chip Liquid Cooling in AI Data Centers

**Paper version 0.2 · July 1, 2026**
**Corresponds to ThermaLoop release v0.3.0**
**Author:** Travis Walla
**Repository:** [github.com/Walla17x/thermaloop](https://github.com/Walla17x/thermaloop)
**Interactive demo:** [walla17x.github.io/thermaloop/explorer.html](https://walla17x.github.io/thermaloop/explorer.html)
**License:** MIT

---

## Abstract

ThermaLoop is an open-source simulation lab for single-rack, direct-to-chip (D2C) liquid cooling loops in AI data centers. The package provides a reduced-order physics stack — a five-node lumped RC thermal network, a 1D finite-volume fluid loop, an ε-NTU coolant distribution unit, and an affinity-law pump — together with a YAML-driven scenario engine for transient fault analysis, a parametric uncertainty quantification (UQ) ensemble runner, and an in-browser interactive explorer whose closed-form steady-state solver is continuously verified in CI to match the Python reference implementation to 0.000 °C. ThermaLoop is calibrated against one published experimental anchor — Heydari et al., *Applied Thermal Engineering* 239 (2024) — matching the anchor's measured CDU effectiveness of 0.82, with predicted die temperature of 74.1 °C inside the H100-class operational envelope and 100 % heat-balance closure at the reference operating point. The ensemble runner converts the package's headline safety output — margin to throttle — from a scalar point estimate into a distribution, surfacing operational risk that point-estimate analysis can miss: for the pump-degradation fault, the same nominal margin of 2.5 K corresponds to a 17 % throttle probability under realistic 8–15 % parameter uncertainty. ThermaLoop is not a CFD replacement, an enterprise digital twin, or a new methodological primitive; it is offered as the open, transparent middle ground between heavyweight CFD and back-of-envelope spreadsheets that the D2C operational community currently lacks. This paper describes the model, the current validation state, the scenario engine, the UQ runner, the explorer, the known limitations, and the planned multi-source validation expansion.

---

## 1. The Cooling Problem at AI Density

AI training and inference workloads have pushed data-center rack power well past the limits of conventional air cooling. The Uptime Institute's 2024 *Global Data Center Survey* reports continued increases in deployments exceeding 15 kW per rack, with leading-edge AI deployments now operating in the 75–120 kW per rack range and vendor and industry-analyst roadmaps projecting continued growth into the several-hundred-kilowatt-per-rack regime within the next two to three years (NVIDIA Vera Rubin generation and successors). ASHRAE Technical Committee 9.9 has updated its data-center thermal guidelines repeatedly through 2024–2026 to bring liquid cooling, CDU design, transient modeling, and coolant quality monitoring into mainstream practice.

Direct-to-chip (D2C) liquid cooling has become the dominant response. D2C places cold plates directly on CPUs, GPUs, and AI accelerators; a coolant loop transports heat to a coolant distribution unit (CDU), which rejects to the facility loop. The architecture is straightforward in concept. The operational regime is not. Power transients from idle to multi-kilowatt within seconds, partial pump degradation, gradual CDU fouling, facility temperature excursions during chiller faults, and slow coolant loss from manifold leaks all couple into chip die temperature on time scales that matter operationally. The questions that operators and designers need answered are concrete: how much thermal margin does the chip have right now, and how does that margin behave when something goes wrong — and how confident are we in that answer given that the underlying parameters are never known exactly?

The tooling available to answer those questions is bimodal. At one end, CFD packages (ANSYS Icepak, COMSOL Multiphysics, OpenFOAM) provide high-fidelity field solutions at the cost of meshing effort, licensing budgets, and runtimes that are incompatible with rapid scenario exploration or operational decision support. At the other end, sizing spreadsheets and vendor demonstration tools provide first-pass estimates whose assumptions are usually opaque, often oversimplified, and rarely independently verifiable. The middle ground — a transparent, reduced-order, physics-informed model that operators and researchers can run, modify, validate, and trust — is sparsely populated and dominated by closed in-house tooling at hyperscale operators and cooling vendors. Neither end of the spectrum reports operational uncertainty in a useful way: the CFD tools produce single high-fidelity runs that take too long to repeat at sample size, and the spreadsheets produce point estimates whose uncertainty bands are typically absent or hand-waved.

ThermaLoop is a contribution to that middle. It is open-source, version-controlled, continuously tested, and validated against published experimental data. Its physics are textbook and that is the point. Its scope is deliberately bounded and that is also the point. Its interactive in-browser explorer puts the same validated solver into any web browser, removing the deployment friction that has historically kept reduced-order thermal models inside individual engineering teams. Its ensemble UQ runner makes the validated point estimate's robustness to realistic parameter uncertainty an explicit, reproducible output.

---

## 2. Scope: What ThermaLoop Is — and Isn't

ThermaLoop adopts an explicit scope charter, set early in the project and held against the natural pressure to sprawl.

**ThermaLoop is:**

- A reduced-order, physics-informed simulation lab for single-server / single-rack D2C cooling loops.
- A tool for exploring steady-state operating points, transient response to fault scenarios, and the propagation of parameter uncertainty through both.
- An open-source, reproducible reference for the underlying physics, the assumptions, and the validation state.
- An interactive in-browser explorer that lets non-developers inspect the same model that the Python solver runs.

**ThermaLoop is not:**

- A CFD replacement. ThermaLoop models lumped thermal nodes and 1D fluid flow. It does not solve Navier–Stokes, does not resolve flow fields, does not predict hot-spot geometry, and does not capture boundary-layer phenomena.
- An enterprise digital twin. ThermaLoop does not ingest live telemetry, does not provide control-system integration, and is not designed for closed-loop production decision-making without human supervision.
- A multi-rack or facility-scale model. The current scope is one server / one rack with one CDU. Multi-rack and shared-CDU topologies are explicit roadmap items, held under deliberate scope discipline pending the validation expansion described in §4.4.
- A new methodological primitive. The physics layers — lumped capacitance, 1D fluid transport, ε-NTU heat exchanger formulation, affinity laws — are standard. The uncertainty-propagation layer — Latin hypercube sampling with quantile reporting — is also standard. The contribution is the *integration, transparency, and interactive reproducibility*, not new modeling theory.

The honest positioning matters. Reviewers, vendors, and operators evaluating any new model are right to ask "what's the contribution?" For ThermaLoop the answer is operational and methodological clarity, applied to D2C cooling on a validated open-source base. Claiming otherwise would forfeit the credibility this paper is written to establish.

---

## 3. Model Architecture

ThermaLoop couples four physics layers under a single solver, with an uncertainty-propagation layer that wraps the solver. The architecture is intentionally modest in component count to keep each piece inspectable.

### 3.1 Lumped Thermal Network (Five-Node RC)

The chip-to-coolant heat path is modeled as a five-node RC network. Power dissipation at the die *P*<sub>die</sub>(*t*) drives the network; the energy balance at each node *i* takes the standard form:

```
C_i · dT_i/dt = sum_j (T_j - T_i) / R_ij + sources_i
```

Capacitances *C*<sub>i</sub> and conductances 1/*R*<sub>ij</sub> are specified per node and per coupling. The cold-plate-to-fluid resistance is modeled with a convective heat-transfer coefficient *h* that depends on fluid properties and local mass flow; this is the coupling between the thermal network and the fluid loop. Time constants τ<sub>ij</sub> = *R*<sub>ij</sub> · *C*<sub>i</sub> govern the transient response.

The lumped formulation is appropriate when the time scales of interest — seconds to minutes for fault transients, milliseconds for the explorer's interactive feedback — are longer than the internal conduction time scales within each lumped node. For chip-to-cold-plate paths in D2C architectures, where TIM and copper conduction are fast relative to fluid transport, this assumption holds at the fidelity needed for operational margin analysis.

### 3.2 1D Finite-Volume Fluid Loop

The coolant loop is discretized as a 1D series of finite-volume cells. Each cell evolves fluid enthalpy under mass flow ṁ (set by the pump curve and total loop resistance, §3.4), heat input from cold-plate convection, and heat removal at the CDU (§3.3).

The 1D formulation captures transport delay — heat absorbed at the cold plate takes finite time to reach the CDU and return as cooled coolant. This delay is essential for transient analysis and is not represented in pure steady-state lumped models. Fluid properties (ρ, *c*<sub>p</sub>, μ) are evaluated per cell from selectable fluid models: water as default, with PG25 (25 % propylene-glycol / water) added in v0.2.0 to support fluids common in real D2C deployments. Per-timestep property variation is implemented internally but held as an optional scenario feature; current runs use scenario-averaged properties.

### 3.3 ε-NTU CDU

The coolant distribution unit is modeled as a single-pass, single-phase liquid-to-liquid heat exchanger using the ε-NTU formulation:

```
ε  = (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in)    [for C_hot = C_min]
Q  = ε · C_min · (T_hot_in - T_cold_in)
```

Effectiveness ε is treated as a parameter of the CDU design, calibrated against the validation anchor described in §4. Facility-loop coolant inlet temperature *T*<sub>cold_in</sub> is a scenario input. CDU fouling is modeled as a parametric reduction in ε to simulate degradation faults; this is a simplification — real fouling alters heat-exchanger geometry and pressure drop in ways the lumped ε does not capture — but is adequate for the operational margin questions ThermaLoop targets.

### 3.4 Affinity-Law Pump

Pump head *H* and mass flow ṁ at speed *N* follow standard affinity scaling against a reference operating point (*H*<sub>0</sub>, ṁ<sub>0</sub>, *N*<sub>0</sub>):

```
ṁ / ṁ_0      = N / N_0
H  / H_0     = (N / N_0)²
P_pump / P_pump_0 = (N / N_0)³
```

Pump speed becomes a scenario control input; pump degradation faults are modeled as effective-*N* reduction. Off-design pump curves are not modeled; the affinity laws are accurate near the design point and degrade as operation moves away from it.

### 3.5 Safety Outputs

Two derived safety quantities are reported at every solver step and are the primary operational outputs of the package:

- **Margin to throttle:** Δ*T*<sub>margin</sub> = *T*<sub>throttle</sub> − *T*<sub>die</sub>. Positive values indicate operational headroom.
- **Time-to-throttle:** under a continued perturbation, the projected time at which *T*<sub>die</sub> crosses *T*<sub>throttle</sub>. Computed from current Δ*T*<sub>margin</sub> and observed d*T*<sub>die</sub>/d*t*; conservative for monotonic transients, indicative for oscillatory ones.

These metric choices align with the operational framing in ASHRAE TC 9.9's 2024 technical bulletin *Liquid Cooling: Resiliency Guidance for Cold Plate Deployments*, which uses the minimum server time-to-throttle under worst-case failure of the resilient design as a key timeframe constraint for load-migration and resiliency planning in liquid-cooled IT, and recommends transient modeling to verify the performance of systems for which empirical data is unavailable. ThermaLoop's safety outputs and scenario engine are intended to support this kind of analysis. The same problem framing has since been institutionalized as funded research: ASHRAE research project 1972-TRP, *Data Center Direct-to-Chip Liquid Cooling Resiliency — Failure Modes and IT Throttling Impacts; Liquid Cooling Energy Use Metrics and Modeling* (proposals closed December 2025, scheduled start April 2026), targets failure time constants, pump partial/full failures, and throttling durations as open modeling questions — the same quantities ThermaLoop's fault library and safety outputs report today.

These two quantities translate the raw thermal solution into operational language. They are the headline outputs of every point-estimate scenario run. The ensemble UQ runner (§3.7) extends these from scalars to distributions, reporting margin percentiles and throttle probability across realistic parameter uncertainty.

### 3.6 Closed-Form Steady-State

In the limit d*T*<sub>i</sub>/d*t* = 0 across all nodes with constant inputs, the system reduces to a linear algebraic problem in the node temperatures. ThermaLoop provides a closed-form steady-state solver (`rc_network.steady_state_closed_form`) that bypasses time-stepping. This is the engine behind the interactive in-browser explorer (§6): the JavaScript reimplementation of the closed-form solver is continuously CI-verified to match the Python reference to 0.000 °C across the parameter envelope. The equivalence is a hard test, not a tolerance — both implementations must agree to numerical precision or CI fails.

### 3.7 Parametric Uncertainty Quantification

Operators face values, not point estimates. Cold-plate convection coefficients, CDU effectiveness, TIM thermal resistances, and facility-loop supply temperatures all carry uncertainty on the order of 8–15 % around any specified design point, driven by manufacturing tolerances, fouling state, and operational variability. ThermaLoop's ensemble UQ runner propagates this uncertainty through the validated solver and reports its impact on the headline safety outputs.

Each ensemble sample is a complete scenario run with uncertain parameters drawn from declared distributions and merged into the scenario's parameter overrides. The validated right-hand-side function is reused without modification; no physics is duplicated. The uncertain-parameter declaration is a YAML extension to the existing scenario schema, so any fault or sweep can be propagated through parametric uncertainty without writing code:

```yaml
ensemble:
  n_samples: 200
  seed: 1
  sampler: lhs                       # lhs | mc
  uncertain:
    - { param: h0,       dist: lognormal, mean: 30000, cv: 0.15 }
    - { param: UA_hx,    dist: normal,    mean_factor: 1.0, cv: 0.10 }
    - { param: R_ihs_cp, dist: normal,    mean: 0.020, cv: 0.10, clip_min: 0.005 }
    - { param: T_fac_in, dist: uniform,   low: 28.0, high: 32.0 }
```

Supported distributions for v0 are normal, lognormal, uniform, and triangular. Latin hypercube sampling (LHS) is the default; Monte Carlo is available for sanity checks. Per-sample outputs are the same safety quantities the point-estimate runner produces: minimum margin to throttle, peak die temperature, time-to-throttle, and pump energy. The runner aggregates these into distributions and reports percentiles (P5, P25, P50, P75, P95), throttle probability — the fraction of samples that cross the die-temperature limit — and a 5–95 % envelope of *T*<sub>die</sub>(*t*) across the full transient.

The methodological content of this layer is not novel. Latin hypercube sampling, ensemble propagation, and quantile reporting are standard techniques. The contribution is applying them to the validated D2C cooling base in a reproducible, open-source form, with explicit and modest assumptions documented in ADR-004 and `ASSUMPTIONS.md`: parametric uncertainty only (model-form uncertainty is not propagated), independent uncertain parameters (correlation matrices are a planned follow-up), and user-supplied distribution shapes (no automated distribution-fitting from data). The honest framing is that the ensemble makes the validated point estimate's robustness inspectable, not that it changes the underlying model.

---

## 4. Validation

### 4.1 Reference Anchor

ThermaLoop is currently calibrated and validated against one published experimental reference:

> Heydari, A., Gharaibeh, A. R., Tradat, M., Soud, Q., Manaserh, Y., Radmard, V., Eslami, B., Rodriguez, J., & Sammakia, B. (2024). Experimental evaluation of direct-to-chip cold plate liquid cooling for high-heat-density data centers. *Applied Thermal Engineering*, **239**, 122122.

The anchor provides:

- CDU effectiveness ε ≈ 0.82–0.83 at high heat load (two CDUs serving three racks at a combined 128 kW, measured at 0.83 and 0.82 respectively).
- Measured loop operating conditions (heat load, flow, fluid, supply temperatures) and thermal-test-vehicle case temperatures at the reference state.
- Sufficient operating-point metadata to construct a consistent ThermaLoop input scenario.

Two transfers from the anchor to the model are made explicit here. First, the anchor's effectiveness values were measured on liquid-to-air CDUs; ThermaLoop models a liquid-to-liquid CDU and adopts ε as a calibrated design parameter, treating effectiveness as transferable at the ε-NTU level of abstraction. Second, the anchor instruments thermal test vehicles (case temperatures via thermocouple), not GPU dies; ThermaLoop's predicted die temperature is therefore checked against the published operational envelope for H100-class hardware rather than against a die temperature measured in the anchor experiment.

### 4.2 Result at the Reference State

Running ThermaLoop at its reference scenario — an 8-GPU server at 700 W per GPU, design-flow secondary loop, warm-water supply at 30 °C, constructed to be consistent with the anchor's loop conditions — returns:

- **Lumped five-node model:** predicted *T*<sub>die</sub> = 74.1 °C, CDU effectiveness 0.822, loop temperature rise above facility 6.8 K. The CDU effectiveness matches the anchor's measured 0.82–0.83 directly; the *T*<sub>die</sub> sits inside the operational envelope for H100-class hardware (70–85 °C).
- **1D finite-volume model:** predicted *T*<sub>die</sub> = 71.6 °C, spatial loop temperature rise 5.6 K. The 1D model runs slightly cooler than the lumped model because it resolves the cold-plate-to-CDU spatial gradient explicitly rather than averaging.
- **Heat-balance closure: 100 % in both models.** The sum of power entering the loop equals the sum of power leaving the loop and the CDU to within numerical precision at every solver step.
- Fluid property calibration: *c*<sub>p</sub>(water, 37 °C) = 4186 J/kg·K used throughout, consistent with the anchor's working fluid.

Heat-balance closure is a baseline self-consistency check. It is not a validation by itself — a model can be perfectly self-consistent and quantitatively wrong. The validation content at this anchor is the direct match to the anchor's measured CDU effectiveness, together with a predicted die temperature that lands inside the published operational envelope for the reference hardware class; the closure is the proof that the implementation is not silently leaking energy. Because the anchor instruments thermal test vehicles rather than GPU dies, no die-temperature residual against a measured value is claimed. That is a deliberate limit of the current validation state, and one the multi-source expansion in §4.4 is designed to tighten.

### 4.3 Honest Single-Anchor Disclosure

ThermaLoop is calibrated against one published experimental anchor. This is sufficient to demonstrate that the implementation is internally consistent and matches one well-instrumented reference state at the CDU-effectiveness level. It is **not sufficient to claim broad operational generality across the D2C parameter envelope.** A reader or reviewer should treat ThermaLoop's quantitative predictions as well-anchored at the Heydari operating point and as carrying increasing uncertainty as scenarios depart from that anchor — different power ranges, different flow ranges, different working fluids, different CDU architectures.

This is explicit in the codebase and the repo documentation, and it is explicit here. The point of saying it openly is to make the validation expansion plan (§4.4) load-bearing rather than aspirational.

### 4.4 Multi-Source Validation Roadmap

The planned validation expansion targets two additional independent anchors, prioritized by physics-layer complementarity and laboratory independence from the current anchor:

1. **Cold-plate-level anchor (different physics layer).** Samal et al. (2025) report a combined numerical and experimental study of a distributed inlet–outlet jet impingement cold plate (DIOJIC-CP) using PG25 fluid, with reported thermal resistance as low as *R*<sub>th</sub> ≈ 0.0224 °C/W and demonstrated TDP capability above 3500 W. The work is from the National Yang Ming Chiao Tung University mechanical-engineering group, laboratory-independent of the current Heydari anchor. Anchoring at this layer would validate the cold-plate convection model independently of the full-loop integration and would specifically exercise the PG25 fluid model added in v0.2.0. Whether the *R*<sub>th</sub> figure is taken from the experimental measurements or the CFD result will be confirmed against the source paper at integration time.

2. **Real-world deployment anchor.** California Energy Commission report CEC-500-2024-061, *Demonstration of Low-Cost Data Center Liquid Cooling*, documenting an Asetek-based D2C system in operational deployment. This anchor is weaker on precise instrumentation but stronger on credibility narrative — it ties ThermaLoop's predictions to real hardware in real use.

A third candidate, Heydari et al.'s ITherm 2023 single-phase CDU paper, was considered and dropped: the authorship overlaps with the current anchor (both originate from the Binghamton ES2 Center / Sammakia laboratory cluster), so adding it would not provide laboratory-independent validation. The decision to omit same-lab anchors is documented and is part of the project's commitment not to inflate the validation footprint with non-independent sources.

Each new anchor will be added under the same rigor as the current Heydari anchor: full operating-point specification, predicted-versus-measured comparison with honest residual reporting, and a CI test that locks the new anchor's outputs to tolerance against future code changes.

### 4.5 Ensemble Robustness at the Anchor

The ensemble UQ runner (§3.7) provides a second-order check on the point-estimate validation. Running 200 LHS samples at the Heydari anchor with realistic parameter uncertainty (lognormal CV 12 % on cold-plate convection, normal CV 8 % on CDU UA, normal CV 10 % on TIM thermal resistance) returns:

- **Peak *T*<sub>die</sub> distribution:** the 74.1 °C lumped point estimate lies strictly inside the predicted 5–95 % envelope.
- **Throttle probability at the anchor: 0 %.** No sample crosses the 90 °C die-temperature limit, consistent with the safe operating margin at the design point.
- **Bracketing test:** the ensemble's minimum *T*<sub>die</sub> is below the anchor's value, and the maximum is above. This is locked as a CI test (`tests/test_uq.py::test_ensemble_brackets_validated_anchor`).

This is a self-consistency property of the validation, not an independent validation. But it makes the validated point estimate's robustness to realistic parameter uncertainty inspectable, which is the operational question.

---

## 5. Scenario Engine

ThermaLoop's transient capability is exposed through a YAML-driven scenario engine. A scenario specifies the base operating point (power, flow, facility-loop temperature, fluid), time-varying perturbations (step changes, ramps, oscillations on any input), an optional ensemble specification (§3.7), and solver settings (time step, end time, output rate). The scenario engine reuses the same validated right-hand-side function as the steady-state and transient solvers — there is no separate "scenario physics" branch — so any validation effort that anchors the core solver simultaneously anchors every scenario built on it.

### 5.1 Fault Library

Five canonical fault scenarios ship with the package:

| Fault | Mechanism | Operational analog |
|---|---|---|
| `workload_spike` | Step change in *P*<sub>die</sub> | LLM inference burst, model load step |
| `pump_degradation` | Linear reduction in effective pump *N* | Bearing wear, cavitation onset |
| `cdu_fouling` | Linear reduction in CDU ε | Particulate accumulation, biofilm |
| `warm_facility` | Step rise in facility-loop *T*<sub>cold_in</sub> | Chiller outage, summer peak event |
| `coolant_loss` | Step reduction in loop mass flow ṁ | Leak, manifold valve failure |

Each fault produces a transient trajectory in *T*<sub>die</sub>, Δ*T*<sub>margin</sub>, and time-to-throttle, which the operator or designer can inspect through the report HTML or interrogate programmatically.

### 5.2 Real Workload Traces

Two scenarios built from the public AzureLLMInferenceDataset2023 (Patel et al., 2024) — released by Microsoft Azure Research under CC-BY — are included as alternatives to synthetic step-and-hold workloads:

- `azure_conv`: conversational inference workload at approximately 696 W average die power, with realistic temporal variation.
- `azure_code`: code-completion inference workload at approximately 262 W average die power, with bursty load profile.

These traces let the user ask the more honest version of the design question: not "how does my loop behave under an artificial 1 kW step?" but "how does my loop behave under what the chip actually does in production?"

### 5.3 Parametric Sweeps

Three parameter sweeps are shipped, intended to build the envelope plots designers use to bound operating windows:

| Sweep | Varies | Reports |
|---|---|---|
| `pump_speed` | Pump *N* | Steady *T*<sub>die</sub>, margin, pump power |
| `flow_vs_margin` | Mass flow ṁ | Δ*T*<sub>margin</sub>, time-to-throttle |
| `cdu_setpoint` | Facility *T*<sub>cold_in</sub> | Steady *T*<sub>die</sub>, sensitivity to ε |

### 5.4 Ensemble UQ Scenarios

The fault scenarios in §5.1 produce a point estimate of margin and throttle status under a single set of nominal parameters. The ensemble UQ runner reruns any of those faults under a declared distribution of parameter values, producing a margin distribution and a throttle probability. Two canonical ensemble configurations ship with the package:

- `anchor_uq.yaml`: parametric uncertainty propagated through the anchor operating point with no fault. Used as the robustness check in §4.5.
- `pump_degradation_uq.yaml`: the pump-degradation fault from §5.1 propagated through realistic parameter uncertainty (cold-plate convection CV 15 %, CDU effectiveness CV 10 %, TIM thermal resistance CV 10 %, facility supply temperature uniform on [28 °C, 32 °C]).

The pump-degradation ensemble result is the headline motivating example for this paper. The point-estimate version of the fault reports a minimum margin of 2.5 K and **no throttle event** — a result that a sizing-spreadsheet workflow would record as "safe." The ensemble version of the same fault, at N = 200 samples and the parameter CVs above, returns:

| Metric | Point estimate | Ensemble (N = 200) |
|---|---|---|
| Minimum margin | 2.5 K | P50 2.5 K, **P5 −1.4 K** |
| Throttle outcome | No throttle | **17 % throttle probability** |

The point estimate and the ensemble median agree, as they should. The ensemble's P5 tail and throttle probability tell the operational story the point estimate cannot: in roughly one out of six realistic parameter combinations, the same fault crosses the throttle limit. This is not a methodological breakthrough — it is the expected result of propagating realistic parameter uncertainty through a sensitive nonlinear system, applied to the right problem. The point estimate is not wrong; it is incomplete. ThermaLoop's contribution is making that calculation reproducible and open, anchored against published data, for D2C cooling loops specifically.

---

## 6. The Interactive Explorer

The differentiating artifact of ThermaLoop is its interactive in-browser explorer at `docs/explorer.html`. The explorer reimplements the validated Python closed-form steady-state solver in vanilla JavaScript, with no framework dependencies, and is continuously verified in CI to return identical outputs to the Python reference to 0.000 °C across the full input envelope.

The user interface provides drag-and-explore controls for chip power, mass flow, facility temperature, CDU effectiveness (health), and coolant type. As the user manipulates any control, the explorer reports live *T*<sub>die</sub>, Δ*T*<sub>margin</sub>, the operating envelope position, and the steady-state temperature ladder across the five thermal nodes. Animated fault transients are driven by the same scenario engine logic — steady-state baseline, perturbation injection, response trajectory — so the visual behavior the user sees in the browser is the same behavior the Python solver produces in a batch run.

This is an engineering choice with operational consequences. Reduced-order thermal models are most useful when many people across an organization — operators, designers, vendor representatives, customer engineers — can interrogate them with the same physics in front of them. Historically that has meant training each person on a Python or MATLAB environment, or building bespoke web tools per organization. The ThermaLoop explorer is a single self-contained HTML/JS artifact that any of those audiences can open from a link and inspect, with the technical guarantee that the result is CI-locked equal to the Python solver to numerical precision.

---

## 7. Reproducibility and Software Quality

ThermaLoop is built as a pip-installable Python package with the following quality posture:

- **Source:** [github.com/Walla17x/thermaloop](https://github.com/Walla17x/thermaloop), MIT license.
- **Live demo:** [walla17x.github.io/thermaloop/explorer.html](https://walla17x.github.io/thermaloop/explorer.html).
- **Tests:** 47 passing pytest cases covering the validation anchor (steady state, heat-balance closure, lumped-vs-1D agreement, physical-direction monotonicity), the JavaScript-versus-Python closed-form equivalence test, per-fault smoke tests for each scenario in §5.1, fluid-model property checks, Azure-trace runs, and the new UQ ensemble test suite (reproducibility, sampler coverage, anchor bracketing).
- **Continuous integration:** GitHub Actions runs the full test suite on each push. The badge in the repo reflects current main-branch status.
- **Dependencies:** numpy, scipy, matplotlib, pandas, pyyaml, jinja2 for the core solver, ensemble runner, and reporting; vanilla HTML and JavaScript for the explorer. No proprietary or paid dependencies, no compiled extensions, no hardware-specific code paths.
- **Documentation:** `ASSUMPTIONS.md`, `VALIDATION.md`, and architecture decision records (ADRs 000–004) under `docs/decisions/`.
- **Output artifacts:** scenario, sweep, and ensemble runs produce self-contained HTML reports. The CLI exposes `thermaloop run`, `thermaloop sweep`, `thermaloop ensemble`, and `thermaloop envelope`.

Every figure intended to accompany this paper can be regenerated from the repository by running the documented scenarios at the tagged commit. The version of ThermaLoop described here is v0.2.0 plus the ensemble UQ runner (release tag v0.3.0 pending); subsequent revisions of this paper will be tagged against subsequent versions of the code, with the validation table in §4 extended as new anchors land.

---

## 8. Limitations and Roadmap

### 8.1 Known Limitations

- **Single-anchor validation.** See §4.3. This is the largest current credibility gap and the highest-priority near-term work.
- **Single-rack scope.** Multi-rack and shared-CDU topologies are not modeled. Cross-rack interactions — manifold dynamics, shared facility-loop temperature drift, control-loop coupling between racks — are not captured.
- **Lumped thermal model.** No spatial resolution within the die, TIM, or cold plate. Hot-spot prediction requires CFD; ThermaLoop reports a single representative die temperature, not a temperature field.
- **Scenario-averaged fluid properties.** Per-timestep ρ(*T*), *c*<sub>p</sub>(*T*), μ(*T*) variation is implemented internally but is not yet exposed as a scenario option; current runs use scenario-averaged properties.
- **No explicit control system.** The model assumes whatever pump speed and CDU setpoint the scenario specifies. Closed-loop control system dynamics — PID gains on pump speed against die temperature, fan-speed control on the facility side — are out of scope and a meaningful source of real-world transient behavior that ThermaLoop does not reproduce.
- **Parametric uncertainty only.** The ensemble runner (§3.7) propagates uncertainty in parameter *values*, not in model *structure* (lumped vs distributed, single-phase vs two-phase, single-zone vs multi-zone CDU). Structural uncertainty is generally larger than parametric uncertainty and is outside v1 UQ scope.
- **Independent uncertain parameters.** The ensemble assumes uncertain parameters are statistically independent. Some pairs (cold-plate convection and TIM resistance, both driven by manufacturing tolerance) are physically correlated; treating them as independent likely overstates the breadth of the predicted distribution. Correlation matrices are a planned follow-up.

### 8.2 Roadmap

- **Validation expansion** (§4.4) is the highest-priority near-term effort.
- **Multi-rack / shared-CDU topology.** Explicitly held pending the validation expansion. Will be reopened once multi-source validation is in place.
- **Correlated UQ.** Correlation matrices on uncertain parameters; second-order ensemble extensions.
- **Sensitivity decomposition.** Sobol indices and/or Morris screening as a natural complement to the UQ propagation already in place; would answer "which parameters dominate margin?" rather than "what is the distribution of margin?"
- **Per-timestep fluid property variation.** Optional feature, parked behind a scenario flag.
- **FNO-trunk surrogate.** An experimental neural surrogate for accelerated scenario evaluation was prototyped early in development and has been demoted to `experimental/` — kept for provenance and labeled unsupported. Not on the active roadmap.
- **Versioned paper releases.** Each major code release will be accompanied by a versioned update of this paper, with the validation table extended as new anchors land. Versioning the documentation against the code is treated as a project commitment.

---

## 9. Conclusion

ThermaLoop is not a category-defining contribution. The physics it implements has been standard for decades; the uncertainty-propagation techniques are textbook. What it offers is a tool — transparent, validated against one published anchor with an explicit plan to expand, reproducible from source, and capable of reporting operational margin as a distribution rather than as a point estimate — for a problem (D2C cooling design and operational margin analysis under realistic faults) where the current options are either too heavy for rapid iteration (CFD) or too opaque for independent trust (vendor digital twins, ad-hoc spreadsheets). The pump-degradation case in §5.4 is the central motivating example: the same fault, the same nominal parameters, evaluated as a point estimate looks safe at 2.5 K margin and reports no throttle; evaluated under realistic 8–15 % parameter uncertainty throttles in 17 % of samples. The point estimate is not wrong; it is incomplete. Making that gap inspectable is the operational contribution.

The intended readers are data-center operators making cooling-design decisions, ML infrastructure engineers planning capacity, cooling-equipment vendors validating internal sizing tools, and researchers who want a reduced-order reference they can run today and modify if needed. Feedback, validation-anchor proposals, and pull requests are welcome through the repository.

---

## References

1. Heydari, A., Gharaibeh, A. R., Tradat, M., Soud, Q., Manaserh, Y., Radmard, V., Eslami, B., Rodriguez, J., & Sammakia, B. (2024). Experimental evaluation of direct-to-chip cold plate liquid cooling for high-heat-density data centers. *Applied Thermal Engineering*, **239**, 122122. doi:10.1016/j.applthermaleng.2023.122122
2. Samal, S. K., Chang, H. C., Fulpagare, Y., & Wang, C. C. (2025). Thermal management of data centers: Chip-scale cooling using novel distributed inlet–outlet jet impingement liquid cold plate. *Applied Thermal Engineering*, **271**, 126360. doi:10.1016/j.applthermaleng.2025.126360
3. Branton, S., Greenberg, S., Earni, S., Park, B., & Ferguson, S. (2024). *Demonstration of Low-Cost Data Center Liquid Cooling.* CEC-500-2024-061. California Energy Commission, Energy Research and Development Division. https://www.energy.ca.gov/publications/2024/demonstration-low-cost-data-center-liquid-cooling
4. ASHRAE Technical Committee 9.9 (2024). *ASHRAE TC 9.9 Datacom Encyclopedia.* ASHRAE, Atlanta, GA. https://datacom.ashrae.org/
5. ASHRAE Technical Committee 9.9 (2024). *Liquid Cooling: Resiliency Guidance for Cold Plate Deployments.* Technical Bulletin, September 2024. ASHRAE, Atlanta, GA. https://tpc.ashrae.org/FileDownload?idx=0c61b286-059f-46f1-9858-9af91968cf89
6. ASHRAE Technical Committee 9.9 (2026). *TCS Coolant Integrity and System Readiness Best Practices.* Technical Alert. ASHRAE, Atlanta, GA. https://tpc.ashrae.org/FileDownload?idx=09cb747c-22a9-4bd1-b397-2f77be04b530
7. Uptime Institute (2024). *Global Data Center Survey.* Uptime Institute Research Report.
8. Microsoft Azure Research (2023). *AzureLLMInferenceDataset2023 — Code and Conversation Traces.* Azure Public Dataset (CC-BY). https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
9. Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R. (2024). Splitwise: Efficient generative LLM inference using phase splitting. In *Proceedings of the International Symposium on Computer Architecture (ISCA 2024).* ACM, Buenos Aires, Argentina. (Paper of record for the AzureLLMInferenceDataset2023 traces used by the `azure_conv` and `azure_code` scenarios.)
10. ASHRAE (2025). *1972-TRP: Data Center Direct-to-Chip Liquid Cooling Resiliency — Failure Modes and IT Throttling Impacts; Liquid Cooling Energy Use Metrics and Modeling.* Request for Proposal, sponsored by TC 9.9; co-sponsored by TC 4.10 and TC 7.6. ASHRAE, Atlanta, GA. https://www.ashrae.org/file%20library/technical%20resources/research/research%20project%20bidding/1972-rfp.pdf

---

## Appendix A: Reproducing the Reference Validation

```bash
# Clone and install (editable, with dev dependencies)
git clone https://github.com/Walla17x/thermaloop.git
cd thermaloop
pip install -e ".[dev]"

# Run the full test suite (47 tests, ~1 minute on CPU).
# The Heydari validation anchor is enforced by tests/test_physics.py:
#   - CDU effectiveness 0.82 at design flow
#   - T_die at TDP within H100-class envelope
#   - 100 % heat-balance closure in both lumped and 1D models
# The closed-form / Python equivalence is also locked in tests/test_physics.py
# (the same test the JavaScript explorer's correctness depends on).
pytest -q

# Reproduce the point-estimate baseline scenario
python3 -m thermaloop run configs/baseline.yaml

# Reproduce the canonical ensemble UQ runs
python3 -m thermaloop ensemble configs/uq/anchor_uq.yaml
python3 -m thermaloop ensemble configs/uq/pump_degradation_uq.yaml
```

Each run writes a self-contained HTML report to `reports/<scenario_name>/report.html` with embedded base64 figures. The interactive explorer at `docs/explorer.html` is a static page; no build step is required.

---

## Appendix B: Architecture Decision Records

The ADRs referenced throughout this paper are tracked under `docs/decisions/`:

- **ADR 000** — Project scope and charter. Documents the cooling-only, single-rack scope decision and the explicit non-goals.
- **ADR 001** — Reduced-order model choice (lumped RC + 1D FV + ε-NTU + affinity laws) over CFD or alternative reduced-order formulations.
- **ADR 002** — Scenario engine via time-varying parameters. Faults and sweeps use the same validated right-hand side; no duplicated physics.
- **ADR 003** — Fluid properties. Selectable fluid model (water default, PG25 alternative) with temperature-dependent properties; water calibrated to preserve the validation anchor.
- **ADR 004** — Parametric uncertainty quantification via ensemble scenario runs. Each sample is a scenario run with draws merged into the engine's `overrides` mechanism, reusing the validated RHS; specifies the YAML extension for declaring uncertain parameter distributions and the v0 sampler/distribution coverage.
