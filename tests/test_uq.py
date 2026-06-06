"""Tests for the parametric UQ ensemble runner.

Verifies shape, reproducibility, sampler behavior, clipping, and that the
ensemble brackets the validated point estimate at the anchor.
"""
import numpy as np
import pytest

from thermaloop.scenarios import engine, ensemble


def _baseline_uq_config(n=12, seed=1, sampler="lhs"):
    """Small UQ config on the baseline scenario for fast tests."""
    return {
        "name": "test_uq",
        "description": "small UQ smoke test",
        "workload": {"type": "synthetic", "T_horizon": 120, "seed": 0},
        "perturbations": [],
        "ensemble": {
            "n_samples": n,
            "seed": seed,
            "sampler": sampler,
            "uncertain": [
                {"param": "h0", "dist": "lognormal",
                 "mean": 30000.0, "cv": 0.10},
                {"param": "UA_hx", "dist": "normal",
                 "mean_factor": 1.0, "cv": 0.08},
                {"param": "T_fac_in", "dist": "uniform",
                 "low": 28.0, "high": 32.0},
            ],
        },
        "safety": {"T_limit": 90.0},
    }


def test_ensemble_runs_and_returns_expected_shape():
    res = ensemble.run_ensemble(_baseline_uq_config(n=10))
    assert res["n_samples"] == 10
    assert len(res["samples"]) == 10
    assert res["T_die_grid"].shape[0] == 10
    assert np.isfinite(res["T_die_grid"]).all()
    for s in res["samples"]:
        assert np.isfinite(s["min_margin_K"])
        assert np.isfinite(s["peak_T_die"])
        assert set(s["draws"].keys()) == {"h0", "UA_hx", "T_fac_in"}
    for key in ("min_margin_K", "peak_T_die", "pump_energy_Wh"):
        assert key in res["percentiles"]
        assert set(res["percentiles"][key].keys()) == {5, 25, 50, 75, 95}


def test_ensemble_seed_is_reproducible():
    res_a = ensemble.run_ensemble(_baseline_uq_config(n=8, seed=42))
    res_b = ensemble.run_ensemble(_baseline_uq_config(n=8, seed=42))
    a = np.array([list(s["draws"].values()) for s in res_a["samples"]])
    b = np.array([list(s["draws"].values()) for s in res_b["samples"]])
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(res_a["distributions"]["min_margin_K"],
                               res_b["distributions"]["min_margin_K"])


def test_ensemble_different_seeds_diverge():
    res_a = ensemble.run_ensemble(_baseline_uq_config(n=8, seed=1))
    res_b = ensemble.run_ensemble(_baseline_uq_config(n=8, seed=2))
    a = np.array([list(s["draws"].values()) for s in res_a["samples"]])
    b = np.array([list(s["draws"].values()) for s in res_b["samples"]])
    assert not np.allclose(a, b)


def test_mc_sampler_runs():
    res = ensemble.run_ensemble(_baseline_uq_config(n=8, sampler="mc"))
    assert res["sampler"] == "mc"
    assert res["n_samples"] == 8
    assert np.isfinite(res["T_die_grid"]).all()


def test_percentiles_are_ordered():
    res = ensemble.run_ensemble(_baseline_uq_config(n=20))
    for key, pct in res["percentiles"].items():
        ordered = [pct[p] for p in [5, 25, 50, 75, 95]]
        assert ordered == sorted(ordered), (key, ordered)


def test_lhs_covers_uniform_interval():
    """LHS should fully span the configured uniform interval at modest N."""
    res = ensemble.run_ensemble(_baseline_uq_config(n=24))
    t_draws = np.array([s["draws"]["T_fac_in"] for s in res["samples"]])
    assert t_draws.max() - t_draws.min() >= 0.85 * (32.0 - 28.0)
    assert (t_draws >= 28.0).all() and (t_draws <= 32.0).all()


def test_clip_min_is_honored():
    cfg = _baseline_uq_config(n=24)
    # Default UA_hx for n_gpus=8 is 2400; clip floor at 2500 binds ~half of draws.
    cfg["ensemble"]["uncertain"][1]["clip_min"] = 2500.0
    res = ensemble.run_ensemble(cfg)
    ua = np.array([s["draws"]["UA_hx"] for s in res["samples"]])
    assert (ua >= 2499.999).all()
    assert (ua == 2500.0).any(), "expected at least one draw clipped to floor"


def test_perturbation_compatible_with_ensemble():
    """Ensemble must respect existing perturbation timeline (faults still fire)."""
    cfg = _baseline_uq_config(n=8)
    cfg["perturbations"] = [
        {"param": "m_dot", "kind": "ramp",
         "start_s": 30, "end_s": 100, "to_factor": 0.4},
    ]
    res = ensemble.run_ensemble(cfg)
    # With degraded flow on a short horizon, peak T_die should exceed the
    # no-fault peak from the equivalent ensemble.
    res_clean = ensemble.run_ensemble(_baseline_uq_config(n=8))
    assert res["percentiles"]["peak_T_die"][50] > \
        res_clean["percentiles"]["peak_T_die"][50]


def test_ensemble_brackets_validated_anchor():
    """At the validated operating point, ensemble T_die distribution should
    bracket the point-estimate steady state (74.1 C in VALIDATION.md)."""
    cfg = {
        "name": "anchor_bracket_test",
        "workload": {"type": "synthetic", "T_horizon": 300, "seed": 0},
        "ensemble": {
            "n_samples": 40,
            "seed": 3,
            "sampler": "lhs",
            "uncertain": [
                {"param": "h0", "dist": "lognormal",
                 "mean": 30000.0, "cv": 0.12},
                {"param": "UA_hx", "dist": "normal",
                 "mean_factor": 1.0, "cv": 0.08},
                {"param": "R_ihs_cp", "dist": "normal",
                 "mean": 0.020, "cv": 0.10, "clip_min": 0.005},
            ],
        },
        "safety": {"T_limit": 90.0},
    }
    res = ensemble.run_ensemble(cfg)
    peaks = res["distributions"]["peak_T_die"]
    # The validated anchor lives strictly inside the predicted band.
    assert peaks.min() < 74.1 < peaks.max(), (peaks.min(), peaks.max())


def test_unsupported_dist_raises():
    cfg = _baseline_uq_config(n=4)
    cfg["ensemble"]["uncertain"][0]["dist"] = "weibull"
    with pytest.raises(ValueError, match="unsupported dist"):
        ensemble.run_ensemble(cfg)


def test_missing_ensemble_block_raises():
    cfg = _baseline_uq_config(n=4)
    cfg.pop("ensemble")
    with pytest.raises(ValueError, match="no 'ensemble' block"):
        ensemble.run_ensemble(cfg)


def test_canonical_uq_config_runs():
    res = ensemble.run_ensemble("configs/uq/anchor_uq.yaml")
    assert res["n_samples"] == 200
    assert 0.0 <= res["throttle_prob"] <= 1.0
    # The anchor scenario has no fault; throttle prob under realistic
    # uncertainty must be effectively zero at the design point.
    assert res["throttle_prob"] < 0.05
