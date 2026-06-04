"""Thermal safety margin and time-to-throttle.

Operators run GPUs against a junction-temperature limit. This module turns a
die-temperature trajectory into the quantities that matter operationally:
instantaneous margin to the limit, and — under a transient — how long until
the limit is reached (time-to-throttle).
"""
import numpy as np

DEFAULT_T_LIMIT = 90.0   # C, representative throttle/spec limit for H100-class


def margin(T_die, T_limit=DEFAULT_T_LIMIT):
    """Instantaneous margin to the limit (K). Positive = safe headroom."""
    return T_limit - np.asarray(T_die)


def min_margin(T_die, T_limit=DEFAULT_T_LIMIT):
    """Worst-case (smallest) margin over a trajectory (K)."""
    return float(np.min(margin(T_die, T_limit)))


def time_to_throttle(t_axis, T_die, T_limit=DEFAULT_T_LIMIT):
    """First time (s) the die temperature reaches the limit, else None.

    Linearly interpolates the crossing for sub-step resolution.
    """
    T_die = np.asarray(T_die)
    t_axis = np.asarray(t_axis)
    over = T_die >= T_limit
    if not over.any():
        return None
    k = int(np.argmax(over))
    if k == 0:
        return float(t_axis[0])
    t0, t1 = t_axis[k - 1], t_axis[k]
    y0, y1 = T_die[k - 1], T_die[k]
    if y1 == y0:
        return float(t1)
    frac = (T_limit - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


def evaluate(t_axis, T_die, T_limit=DEFAULT_T_LIMIT):
    """Summarize safety for a die-temperature trajectory."""
    ttt = time_to_throttle(t_axis, T_die, T_limit)
    return dict(
        T_limit=T_limit,
        min_margin_K=min_margin(T_die, T_limit),
        peak_T_die=float(np.max(T_die)),
        throttled=ttt is not None,
        time_to_throttle_s=ttt,
        fraction_time_within_5K=float(
            np.mean(margin(T_die, T_limit) < 5.0)),
    )
