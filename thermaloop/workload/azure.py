"""
Azure LLM Inference trace loader.

Converts the public Microsoft Azure LLM Inference Dataset (CSV) into the
same (arrivals, durations, prefill_tokens, decode_tokens) tuple that the
synthetic workload generator returns. Drop-in replacement.

Dataset:  https://github.com/Azure/AzurePublicDataset
Schema:   TIMESTAMP, ContextTokens, GeneratedTokens
License:  CC-BY 4.0 (Patel et al., ISCA 2024; Stojkovic et al., HPCA 2025).
          The Azure trace CSV files distributed under data/ remain under
          their original CC-BY license, separate from this repo's MIT.
"""
import os
import pandas as pd
import numpy as np


# Time-to-first-token and per-decoded-token throughput assumptions
# used to convert raw token counts to wall-clock request durations.
# These match the rough rates Wilkins et al. and Patel et al. report for
# H100-class serving with mixed batch sizes.
PREFILL_TOKEN_RATE = 8000.0   # tokens/s during prefill
DECODE_TOKEN_RATE = 40.0      # tokens/s during decode


def load_azure_trace(csv_path, t_start=0.0, t_horizon=None,
                     time_compression=1.0):
    """
    Load an Azure LLM Inference CSV and return the standard workload tuple.

    Parameters
    ----------
    csv_path : str
        Path to AzureLLMInferenceTrace_conv.csv or _code.csv
    t_start : float
        Start offset (s) relative to the trace's first arrival
    t_horizon : float or None
        Max duration to keep (s). None keeps the full trace.
    time_compression : float
        Multiplier on inter-arrival times. <1.0 = more requests per second,
        used to drive a single-server sim with a trace originally aggregated
        over many servers.

    Returns
    -------
    arrivals_s    : array of request arrival times (s, relative to t_start)
    durations_s   : array of request wall-clock durations
    prefill_toks  : array of input context token counts
    decode_toks   : array of generated token counts
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    t0 = df['TIMESTAMP'].iloc[0]
    arrivals = (df['TIMESTAMP'] - t0).dt.total_seconds().to_numpy()
    arrivals = arrivals * time_compression
    if t_horizon is not None:
        mask = (arrivals >= t_start) & (arrivals < t_start + t_horizon)
    else:
        mask = arrivals >= t_start
    df = df.loc[mask].reset_index(drop=True)
    arrivals = arrivals[mask] - t_start

    prefill_toks = df['ContextTokens'].to_numpy()
    decode_toks = df['GeneratedTokens'].to_numpy()
    durations = (prefill_toks / PREFILL_TOKEN_RATE +
                 decode_toks / DECODE_TOKEN_RATE)
    return arrivals, durations, prefill_toks, decode_toks


def trace_stats(arrivals, prefill_toks, decode_toks, durations):
    """Print summary statistics for a loaded trace."""
    span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
    print(f"  {len(arrivals):,} requests over {span:.1f} s "
          f"({len(arrivals)/span:.2f} req/s mean)")
    print(f"  Prefill tokens: median {np.median(prefill_toks):.0f}, "
          f"p95 {np.quantile(prefill_toks, 0.95):.0f}, "
          f"max {prefill_toks.max()}")
    print(f"  Decode tokens:  median {np.median(decode_toks):.0f}, "
          f"p95 {np.quantile(decode_toks, 0.95):.0f}, "
          f"max {decode_toks.max()}")
    print(f"  Duration:       median {np.median(durations):.2f} s, "
          f"p95 {np.quantile(durations, 0.95):.2f} s")
