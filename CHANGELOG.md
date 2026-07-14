# Changelog

All notable changes to ThermaLoop. Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is semantic. Each release of the white paper (`docs/whitepaper.md`) is tagged against the code release it describes.

## [0.3.0] — 2026-07-01

### Added
- **Parametric uncertainty-quantification ensemble runner** (ADR-004):
  `thermaloop ensemble` propagates declared parameter distributions (normal,
  lognormal, uniform, triangular; LHS default, MC available) through the
  validated solver. Reports margin percentiles (P5–P95), throttle
  probability, and 5–95 % transient die-temperature envelopes.
- Canonical UQ configs: `configs/uq/anchor_uq.yaml`,
  `configs/uq/pump_degradation_uq.yaml`. Headline result: the
  pump-degradation fault's 2.5 K point-estimate margin carries a 17 %
  throttle probability under realistic 8–15 % parameter uncertainty.
- UQ test suite (12 test functions): sampler reproducibility and coverage,
  ensemble-brackets-anchor CI lock. Suite total: 35 → 47 tests.
- **White paper** at `docs/whitepaper.md` (paper v0.2, July 1 2026): model
  architecture, validation state, scenario engine, UQ methodology,
  limitations, roadmap.

### Changed
- **Validation anchor framing corrected** (paper §4.1–§4.3, README,
  VALIDATION.md): the Heydari et al. 2024 anchor provides liquid-to-air CDU
  effectiveness (0.82/0.83, three racks at 128 kW combined) and
  thermal-test-vehicle case temperatures — not measured GPU die
  temperatures. The 8×700 W H100-class scenario is ThermaLoop's constructed
  reference state, checked against the published H100 operational envelope.
  Both transfer assumptions (L2A→L2L effectiveness; TTV case temps →
  envelope check) are now stated explicitly. No model behavior changed.
- ASHRAE citation corrected to the September 2024 Technical Bulletin
  *Liquid Cooling: Resiliency Guidance for Cold Plate Deployments*.
- README refreshed: 47-test count, `thermaloop ensemble` quick-start, UQ
  scenario docs, roadmap reordered around validation expansion (whitepaper
  §4.4).
- `pyproject.toml`: added `[tool.pytest.ini_options] testpaths = ["tests"]`
  to pin the pytest rootdir to the repository.

## [0.2.0] — 2026-06-04

### Added
- YAML scenario engine with perturbation timelines (ADR-002); five canonical
  fault scenarios; three parametric sweeps.
- Temperature-dependent, selectable fluid model — water default, PG25
  alternative (ADR-003).
- Real-workload scenarios from AzureLLMInferenceDataset2023 (CC-BY):
  `azure_conv`, `azure_code`.
- Interactive in-browser explorer (`docs/explorer.html`) reimplementing the
  closed-form steady-state solver in vanilla JS, CI-locked to match the
  Python reference to 0.000 °C.
- Self-contained HTML reports; plot suite; gallery.

## [0.1.0] — 2026-06-04

### Added
- Initial release: five-node lumped RC thermal network, 1-D finite-volume
  fluid loop, ε-NTU CDU, affinity-law pump, safety outputs (margin to
  throttle, time-to-throttle), closed-form steady-state solver, Heydari 2024
  calibration anchor, CI with physics-invariant test suite.
