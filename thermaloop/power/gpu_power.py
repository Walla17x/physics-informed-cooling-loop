"""GPU power model.

Maps an LLM-inference request stream to a per-GPU power trace using a
three-state machine: prefill (compute-bound, near TDP), decode (memory-bound,
a fraction of TDP), and idle. Per-state values are anchored to published
H100-class measurements; see docs/ASSUMPTIONS.md.
"""
import numpy as np


class GPUPowerModel:
    """Three-state per-GPU power model.

    Parameters
    ----------
    P_tdp : float
        Thermal design power, W (700 for H100-class).
    P_decode_frac : float
        Decode-phase power as a fraction of TDP.
    P_idle_frac : float
        Idle power as a fraction of TDP.
    prefill_token_rate : float
        Tokens/s processed during prefill (sets prefill burst duration).
    """

    def __init__(self, P_tdp=700.0, P_decode_frac=0.55, P_idle_frac=0.12,
                 prefill_token_rate=8000.0):
        self.P_tdp = P_tdp
        self.P_prefill = P_tdp
        self.P_decode = P_decode_frac * P_tdp
        self.P_idle = P_idle_frac * P_tdp
        self.prefill_rate = prefill_token_rate

    def power_trace(self, arrivals, prefill_tokens, decode_tokens, durations,
                    dt=1.0, T_horizon=900.0):
        """Build a per-GPU power trace (W) over a uniform time grid.

        Returns
        -------
        t_axis : ndarray
        P : ndarray
            Per-GPU power, W, same length as t_axis.
        """
        n_steps = int(np.ceil(T_horizon / dt))
        P = np.full(n_steps, self.P_idle, dtype=float)
        for a, npf, ndec, dur in zip(arrivals, prefill_tokens,
                                     decode_tokens, durations):
            i0 = int(a / dt)
            if i0 >= n_steps:
                continue
            t_prefill = npf / self.prefill_rate
            i_pf_end = min(i0 + max(1, int(np.ceil(t_prefill / dt))), n_steps)
            i_dec_end = min(i0 + int(np.ceil(dur / dt)), n_steps)
            P[i0:i_pf_end] = np.maximum(P[i0:i_pf_end], self.P_prefill)
            P[i_pf_end:i_dec_end] = np.maximum(P[i_pf_end:i_dec_end],
                                               self.P_decode)
        t_axis = np.arange(n_steps) * dt
        return t_axis, P
