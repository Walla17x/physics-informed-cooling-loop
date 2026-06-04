"""Run ThermaLoop on a real Azure LLM inference trace and print a summary.

    python examples/run_azure.py

The CSV traces ship in data/ (CC-BY); no download needed. This is the same
path the CLI uses (`thermaloop run configs/azure_conv.yaml`), shown here as a
script for clarity.
"""
from thermaloop.scenarios import engine

for cfg in ("configs/azure_conv.yaml", "configs/azure_code.yaml"):
    r = engine.run_scenario(cfg)
    s = r["safety"]
    print(f"{r['name']:12s}  mean {r['mean_gpu_power_W']:.0f} W/GPU  "
          f"peak T_die {s['peak_T_die']:.1f} C  "
          f"min margin {s['min_margin_K']:.1f} K  "
          f"throttled={s['throttled']}")
