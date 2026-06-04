# ThermaLoop

**An open, physics-informed simulation lab for direct-to-chip liquid cooling in AI data centers.**

ThermaLoop models the full path heat takes — from an LLM-inference workload,
through the GPU package and cold plate, around the coolant loop, across the CDU,
and out to facility water — as a transparent, modular system you can run,
perturb, and learn from in seconds. It is deliberately *not* a CFD replacement,
an enterprise digital twin, or a commercial tool. It is a teaching-grade
engineering sandbox for understanding how liquid-cooled AI infrastructure
behaves under real workloads, faults, and design tradeoffs — with every
assumption written down and every steady-state number traceable to a published
anchor.

Here "physics-informed" means physics-based modeling (conservation laws, ε-NTU,
affinity-law pumps), stated plainly — not a machine-learning method claim.

## What it does

- Simulates an 8-GPU H100-class server cooling loop end to end.
- Lumped 5-node RC thermal model **and** a 1-D finite-volume loop that resolves
  the spatial gradient between cold-plate exit and CDU exit.
- ε-NTU CDU calibrated to a published experimental anchor.
- Affinity-law pump model so flow has a real energy cost.
- Safety margins and time-to-throttle under transients.
- Runs with **no external data** (synthetic workload by default); optionally
  driven by real Azure LLM inference traces.

## What it does **not** do

- No CFD, 3-D fields, meshing, or chip-level hot spots — use Ansys/Cadence.
- No two-phase or immersion cooling.
- No enterprise digital twin, telemetry, or BMS integration.
- No RL or learned control. (A DeepONet surrogate lives under `experimental/`,
  clearly labeled and unsupported.)
- No multi-rack or facility-scale topology.

See [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) for the full list of
simplifications and [`docs/decisions/000-charter.md`](docs/decisions/000-charter.md)
for the scope charter.

## Quick start

```bash
pip install -e .
python -m thermaloop.system        # runs a baseline server simulation
pytest -q                          # physics invariants + anchor validation
```

## Validation

Reference scenario: one 8-GPU server, 700 W/GPU sustained, 30 °C facility
supply, design flow.

| Quantity                | ThermaLoop | Anchor                      |
|-------------------------|-----------:|-----------------------------|
| CDU effectiveness       | 0.822      | 0.82–0.83 (Khalili 2024)    |
| T_die at TDP            | 74.1 °C    | 70–85 °C, H100-class        |
| Loop ΔT above facility  | 6.8 K      | 5–10 K, warm-water D2C      |
| Heat-balance closure    | 100 %      | conservation                |

These numbers are regenerated from the model and enforced by the test suite on
every push. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Repository layout

```
thermaloop/
  workload/   synthetic generator + Azure trace loader
  power/      GPU 3-state power model
  thermal/    rc_network.py (lumped 5-node), loop_1d.py (1-D loop)
  cdu/        ε-NTU heat exchanger
  hydraulics/ affinity-law pump
  safety.py   margin + time-to-throttle
  system.py   composes the full server simulation
experimental/ DeepONet surrogate (unsupported; see its README)
data/         optional Azure traces (CC-BY)
docs/         ASSUMPTIONS, VALIDATION, decision records
tests/        physics invariants + anchor validation
```

## Roadmap

- Scenario engine (YAML-driven baseline / fault / sweep runs)
- Fault library: workload spike, pump degradation, CDU fouling, warm facility
  water, coolant-loss approximation
- Optimization sweeps: pump speed vs die temperature, flow vs thermal margin,
  CDU setpoint vs energy
- Auto-generated HTML engineering report and a polished plot suite
- (Experimental) operator-learning surrogate with a Fourier-neural-operator trunk

## Author

**Travis Walla** — Consulting engineer, San Antonio / New Braunfels, Texas
GitHub: [@Walla17x](https://github.com/Walla17x)

## License

MIT for code and documentation. Azure trace data under `data/` is CC-BY per its
original distribution; see [`data/README.md`](data/README.md).
