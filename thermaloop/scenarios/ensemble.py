"""Parametric uncertainty quantification via ensemble scenario runs.

Wraps the validated scenario engine (`engine.run_scenario`) in an ensemble
runner that draws uncertain parameter values from declared distributions and
reports operationally meaningful outcomes (min margin, peak T_die, time to
throttle) as distributions rather than point estimates.

Each sample is a scenario run with the draw merged into the scenario's
``overrides`` — the same mechanism the engine already uses to apply static
parameter overrides. No new RHS path; no duplicated physics. See ADR-004.

A UQ scenario looks like::

    name: pump_degradation_uq
    workload: { type: synthetic, T_horizon: 600, seed: 0 }
    perturbations:
      - { param: m_dot, kind: ramp, start_s: 200, end_s: 450, to_factor: 0.4 }
    ensemble:
      n_samples: 200
      seed: 1
      sampler: lhs              # lhs | mc
      uncertain:
        - { param: h0,       dist: lognormal, mean: 30000.0, cv: 0.15 }
        - { param: UA_hx,    dist: normal,    mean_factor: 1.0, cv: 0.10 }
        - { param: R_ihs_cp, dist: normal,    mean: 0.020, cv: 0.10, clip_min: 0.005 }
        - { param: T_fac_in, dist: uniform,   low: 28.0, high: 32.0 }
    safety: { T_limit: 90 }
"""
import copy
import numpy as np
import yaml
from scipy.stats import qmc, norm, lognorm, uniform, triang

from thermaloop.scenarios import engine
from thermaloop.thermal import rc_network
from thermaloop.fluids import apply_fluid


_SUPPORTED_DISTS = {"normal", "lognormal", "uniform", "triangular"}
_DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _resolve_base_params(config):
    """Build the post-overrides, post-fluid base param dict.

    Mirrors the opening of `engine.run_scenario` so `mean_factor` resolves
    against the same base the solver sees. Kept inline (not refactored into
    a shared helper) until a third caller needs it.
    """
    n_gpus = config.get("n_gpus", 8)
    base = rc_network.default_params(n_gpus=n_gpus)
    base.update(config.get("overrides", {}) or {})
    base, _fluid = apply_fluid(base, config.get("fluid", "water"))
    return base


def _spec_mean(spec, base_params):
    if "mean" in spec:
        return float(spec["mean"])
    if "mean_factor" in spec:
        return float(spec["mean_factor"]) * float(base_params[spec["param"]])
    raise ValueError(
        f"distribution spec for {spec['param']!r} needs 'mean' or 'mean_factor'")


def _spec_std(spec, mean):
    if "std" in spec:
        return float(spec["std"])
    if "cv" in spec:
        return abs(float(spec["cv"]) * mean)
    raise ValueError(
        f"distribution spec for {spec['param']!r} needs 'std' or 'cv'")


def _build_distributions(uncertain, base_params):
    """For each uncertain-parameter spec, return a frozen scipy distribution."""
    dists = []
    for spec in uncertain:
        kind = str(spec["dist"]).lower()
        if kind not in _SUPPORTED_DISTS:
            raise ValueError(
                f"unsupported dist {kind!r}; supported: {sorted(_SUPPORTED_DISTS)}")
        if kind == "normal":
            mu = _spec_mean(spec, base_params)
            sd = _spec_std(spec, mu)
            dists.append(norm(loc=mu, scale=sd))
        elif kind == "lognormal":
            mu = _spec_mean(spec, base_params)
            sd = _spec_std(spec, mu)
            if mu <= 0:
                raise ValueError(
                    f"lognormal requires mean > 0; got {mu} for {spec['param']!r}")
            # Match arithmetic mean and std to underlying log-normal parameters.
            s2 = float(np.log(1.0 + (sd / mu) ** 2))
            s = float(np.sqrt(s2))
            scale = float(np.exp(np.log(mu) - 0.5 * s2))
            dists.append(lognorm(s=s, scale=scale))
        elif kind == "uniform":
            lo, hi = float(spec["low"]), float(spec["high"])
            dists.append(uniform(loc=lo, scale=hi - lo))
        elif kind == "triangular":
            lo, hi = float(spec["low"]), float(spec["high"])
            mode = float(spec.get("mode", 0.5 * (lo + hi)))
            dists.append(triang(c=(mode - lo) / (hi - lo),
                                loc=lo, scale=hi - lo))
    return dists


def _draw_samples(uncertain, base_params, n_samples, seed, sampler):
    """Return an (N, k) draws array and the parameter-name list."""
    param_names = [s["param"] for s in uncertain]
    dists = _build_distributions(uncertain, base_params)
    k = len(uncertain)
    if sampler == "lhs":
        u = qmc.LatinHypercube(d=k, seed=seed).random(n=n_samples)
    elif sampler == "mc":
        u = np.random.default_rng(seed).random((n_samples, k))
    else:
        raise ValueError(f"unknown sampler {sampler!r}; use 'lhs' or 'mc'")
    draws = np.empty_like(u)
    for j, d in enumerate(dists):
        # Clamp uniforms slightly off the boundaries; norm.ppf(0) = -inf.
        u_j = np.clip(u[:, j], 1e-9, 1 - 1e-9)
        draws[:, j] = d.ppf(u_j)
    for j, spec in enumerate(uncertain):
        if "clip_min" in spec:
            draws[:, j] = np.maximum(draws[:, j], float(spec["clip_min"]))
        if "clip_max" in spec:
            draws[:, j] = np.minimum(draws[:, j], float(spec["clip_max"]))
    return draws, param_names


def run_ensemble(config, dt=1.0, progress=None):
    """Run N samples of a scenario with uncertain parameters from spec.

    Returns a dict with per-sample summaries, parameter draws, T_die
    trajectories, and pre-computed percentile / throttle-probability summaries.
    """
    if isinstance(config, str):
        config = load_config(config)
    ensemble_cfg = config.get("ensemble")
    if not ensemble_cfg:
        raise ValueError(
            "scenario config has no 'ensemble' block; use engine.run_scenario "
            "for point estimates")
    n_samples = int(ensemble_cfg.get("n_samples", 100))
    seed = int(ensemble_cfg.get("seed", 0))
    sampler = str(ensemble_cfg.get("sampler", "lhs")).lower()
    uncertain = ensemble_cfg["uncertain"]
    if not uncertain:
        raise ValueError("'ensemble.uncertain' must list >= 1 parameter spec")

    base_params = _resolve_base_params(config)
    draws, param_names = _draw_samples(uncertain, base_params, n_samples,
                                       seed, sampler)

    samples = []
    t_axis = None
    T_die_grid = []
    for i in range(n_samples):
        sample_cfg = copy.deepcopy(config)
        overrides = dict(sample_cfg.get("overrides") or {})
        draw_record = {}
        for j, name in enumerate(param_names):
            v = float(draws[i, j])
            overrides[name] = v
            draw_record[name] = v
        sample_cfg["overrides"] = overrides
        sample_cfg.pop("ensemble", None)   # engine must not see the block
        r = engine.run_scenario(sample_cfg, dt=dt)
        s = r["safety"]
        samples.append(dict(
            sample_id=i,
            draws=draw_record,
            min_margin_K=float(s["min_margin_K"]),
            peak_T_die=float(s["peak_T_die"]),
            throttled=bool(s["throttled"]),
            time_to_throttle_s=s["time_to_throttle_s"],
            pump_energy_Wh=float(r["pump_energy_Wh"]),
        ))
        if t_axis is None:
            t_axis = np.asarray(r["t"])
        T_die_grid.append(np.asarray(r["T_die"]))
        if progress is not None:
            progress(i + 1, n_samples)

    T_die_grid = np.vstack(T_die_grid)

    arrs = dict(
        min_margin_K=np.array([s["min_margin_K"] for s in samples]),
        peak_T_die=np.array([s["peak_T_die"] for s in samples]),
        pump_energy_Wh=np.array([s["pump_energy_Wh"] for s in samples]),
    )
    percentiles = {
        key: {p: float(np.percentile(arr, p)) for p in _DEFAULT_PERCENTILES}
        for key, arr in arrs.items()
    }
    throttle_prob = float(np.mean([s["throttled"] for s in samples]))

    return dict(
        name=config.get("name", "ensemble"),
        description=config.get("description", ""),
        n_samples=n_samples,
        sampler=sampler,
        seed=seed,
        param_names=param_names,
        samples=samples,
        distributions=arrs,
        percentiles=percentiles,
        throttle_prob=throttle_prob,
        t=t_axis,
        T_die_grid=T_die_grid,
        T_limit=(config.get("safety") or {}).get("T_limit", 90.0),
        config=config,
    )
