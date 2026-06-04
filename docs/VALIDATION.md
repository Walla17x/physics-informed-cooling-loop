# Validation

ThermaLoop's steady-state behavior is calibrated against a published
experimental anchor and checked by the test suite on every run.

## Anchor

Khalili et al., *Experimental evaluation of direct-to-chip cold plate liquid
cooling for high-heat-density data centers*, Applied Thermal Engineering 240
(2024): CDU effectiveness 0.82-0.83 at design flow.

## Reference scenario

One 8-GPU H100-class server, 700 W per GPU sustained, warm-water facility
supply at 30 C, design-point flow.

## Lumped 5-node model

| Quantity                | ThermaLoop | Expected / anchor          |
|-------------------------|-----------:|----------------------------|
| CDU effectiveness       | 0.822      | 0.82-0.83 (Khalili 2024)   |
| T_die at TDP            | 74.1 C    | 70-85 C, H100-class        |
| Loop dT above facility  | 6.8 K     | 5-10 K, warm-water D2C     |
| Heat-balance closure    | 100.0 %   | 100 % (conservation)       |

## 1-D loop model

| Quantity                | ThermaLoop | Note                       |
|-------------------------|-----------:|----------------------------|
| T_die at TDP            | 71.6 C    | runs cooler; resolves gradient |
| Heat-balance closure    | 100.0 %   | 100 %                      |
| Spatial loop dT         | 5.6 K     | cold-plate exit vs CDU exit |

## How this is enforced

`tests/test_physics.py` asserts the anchor bounds, heat-balance closure for
both models, lumped-vs-1D agreement, and physical-direction monotonicity
(less flow -> hotter; warmer facility -> hotter; fouling -> hotter; more power
-> hotter). CI runs them on every push.

These numbers are regenerated from the model, not hand-written.
