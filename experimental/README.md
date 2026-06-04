# Experimental

Code in this directory is **not part of the supported ThermaLoop core.** It is
research-grade, less rigorously validated, and may change or be removed. Nothing
in the core simulation depends on it.

## DeepONet surrogate (`surrogate.py`, `train_surrogate.py`)

An operator-learning surrogate that maps a per-GPU power trace `P(t)` to the
1D loop temperature field `T(x, t)` in a single forward pass.

**Honest status.** This is a demonstration, not a validated tool:

- Best validation RMSE plateaued at ~0.5 K and did not improve with further
  training — that is the architecture's capacity ceiling on this dataset, not
  under-training. A Fourier-neural-operator trunk would likely do better; it
  has not been tried here.
- On a held-out Azure conversation trace the surrogate reached MAE ~0.25 K,
  but this was single-trace evaluation, not a distribution over conditions.
- Retraining requires regenerating the rollout dataset first
  (`surrogate.generate_dataset(...)`); the cached dataset is not committed.

It is kept here because the work is real and the operator-learning pattern is
worth showing, but it is deliberately outside the core because the core's value
is a transparent, validated physics simulator — not an ML model. Treat any
number from the surrogate as indicative, not authoritative.

Requires the optional `torch` dependency: `pip install -e ".[experimental]"`.
