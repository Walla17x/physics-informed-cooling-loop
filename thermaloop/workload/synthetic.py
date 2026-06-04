"""Synthetic LLM-inference workload generator.

Produces a request stream with Poisson arrivals modulated by a diurnal
envelope and a short/long request mix, matching the qualitative structure of
production inference traces. Runs with no external data, so the repository is
fully reproducible from a clean clone.
"""
import numpy as np


def synthetic_workload(T_horizon=900.0,
                       base_rate=2.0,
                       burst_amplitude=3.0,
                       burst_period=300.0,
                       short_frac=0.7,
                       seed=0):
    """Generate a synthetic inference workload.

    Returns
    -------
    arrivals : ndarray   request arrival times (s)
    durations : ndarray  request wall-clock durations (s)
    prefill_tokens : ndarray
    decode_tokens : ndarray
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    arrivals, durations, n_pf, n_dec = [], [], [], []
    while t < T_horizon:
        rate = base_rate * (1.0 + 0.5 * burst_amplitude *
                            (np.sin(2 * np.pi * t / burst_period) ** 2))
        t += rng.exponential(1.0 / rate)
        if t >= T_horizon:
            break
        if rng.random() < short_frac:
            pf = int(rng.integers(50, 400))
            dec = int(rng.integers(20, 200))
        else:
            pf = int(rng.integers(1000, 8000))
            dec = int(rng.integers(200, 1500))
        durations.append(pf / 8000.0 + dec / 40.0)
        arrivals.append(t)
        n_pf.append(pf)
        n_dec.append(dec)
    return (np.array(arrivals), np.array(durations),
            np.array(n_pf), np.array(n_dec))
