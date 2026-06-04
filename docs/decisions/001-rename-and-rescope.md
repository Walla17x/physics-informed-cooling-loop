# ADR-001: Rename and honest re-scoping

Status: accepted

## Context

The project's first public version was named `physics-informed-cooling-loop`
and led with a DeepONet surrogate. Two problems:

1. The name claimed "physics-informed" in the machine-learning sense
   (physics residual terms in a loss), but the surrogate was a plain regression
   model with no physics-informed loss. The name overclaimed.
2. The surrogate was the headline despite being the least validated component
   (accuracy plateaued at ~0.5 K, single-trace evaluation, retraining not
   reproducible from a clean clone).

## Decision

- **Rename to `thermaloop`.** The repository is positioned as a *simulation
  lab*, and the name should reflect that without overclaiming a method it does
  not implement. GitHub redirects the old URL automatically.
- **Demote the surrogate to `experimental/`.** It is preserved for provenance
  and because the operator-learning pattern is worth showing, but it is removed
  from the headline and clearly labeled as unsupported. The core's value is a
  transparent, validated physics simulator, not an ML model.

## Consequences

The repository can be trusted at face value: the name matches the contents, the
validated physics is the product, and the experimental work is fenced and
honestly described. "Physics-informed" now means physics-based modeling, stated
plainly in the README, not a method claim the code doesn't back up.
