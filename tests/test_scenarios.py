"""Tests for the scenario engine and optimization sweeps."""
import glob
import numpy as np
import pytest

from thermaloop.scenarios import engine, sweeps

FAULTS = sorted(glob.glob("configs/faults/*.yaml"))
SWEEPS = sorted(glob.glob("configs/sweeps/*.yaml"))


def test_baseline_runs_and_is_safe():
    r = engine.run_scenario("configs/baseline.yaml")
    assert not r["safety"]["throttled"]
    assert np.isfinite(r["T_die"]).all()


@pytest.mark.parametrize("cfg", FAULTS)
def test_every_fault_runs_clean(cfg):
    r = engine.run_scenario(cfg)
    assert np.isfinite(r["T_die"]).all(), cfg
    assert r["safety"]["peak_T_die"] >= r["T_die"][0]


def test_reduced_flow_raises_die_temp_vs_baseline():
    base = engine.run_scenario("configs/baseline.yaml")
    deg = engine.run_scenario("configs/faults/pump_degradation.yaml")
    assert deg["safety"]["peak_T_die"] > base["safety"]["peak_T_die"]


def test_coolant_loss_is_more_severe_than_pump_degradation():
    deg = engine.run_scenario("configs/faults/pump_degradation.yaml")
    loss = engine.run_scenario("configs/faults/coolant_loss.yaml")
    assert loss["safety"]["peak_T_die"] > deg["safety"]["peak_T_die"]


def test_pump_degradation_lowers_pump_energy():
    base = engine.run_scenario("configs/baseline.yaml")
    deg = engine.run_scenario("configs/faults/pump_degradation.yaml")
    # less flow -> less pump energy (the tradeoff against thermal margin)
    assert deg["pump_energy_Wh"] < base["pump_energy_Wh"]


@pytest.mark.parametrize("cfg", SWEEPS)
def test_every_sweep_runs(cfg):
    sw = sweeps.run_sweep(cfg)
    assert len(sw["rows"]) >= 3
    for row in sw["rows"]:
        assert np.isfinite(row["T_die"])
        assert 0.0 <= row["epsilon"] <= 1.0


def test_flow_sweep_is_monotonic_in_die_temp():
    sw = sweeps.run_sweep("configs/sweeps/pump_speed.yaml")
    T = [r["T_die"] for r in sw["rows"]]   # ascending flow
    assert all(b <= a + 1e-6 for a, b in zip(T, T[1:])), T


def test_pump_speed_sweep_shows_cubic_cost():
    sw = sweeps.run_sweep("configs/sweeps/pump_speed.yaml")
    P = [r["pump_power_W"] for r in sw["rows"]]
    assert P[-1] > P[0] * 4   # cubic growth across the flow range


# --------------------------------------------------------------------------
# Fluid properties (Phase 3)
# --------------------------------------------------------------------------
def test_water_fluid_preserves_anchor_cp():
    from thermaloop.fluids import get_fluid, apply_fluid
    from thermaloop.thermal import rc_network
    assert abs(get_fluid("water").cp(37.0) - 4186.0) < 0.5
    p, _ = apply_fluid(rc_network.default_params(), "water")
    assert abs(p["cp_water"] - 4186.0) < 1.0
    assert abs(p["h_property_factor"] - 1.0) < 1e-9   # water => no correction


def test_pg_coolant_runs_hotter_than_water():
    base = engine.run_scenario("configs/baseline.yaml")
    pg = engine.run_scenario("configs/pg_coolant.yaml")
    # lower cp + higher viscosity -> higher die temp, lower margin
    assert pg["safety"]["peak_T_die"] > base["safety"]["peak_T_die"] + 1.0
    assert pg["safety"]["min_margin_K"] < base["safety"]["min_margin_K"]
    assert pg["params"]["h_property_factor"] < 1.0


def test_pg_has_lower_cp_and_higher_viscosity():
    from thermaloop.fluids import get_fluid
    w, pg = get_fluid("water"), get_fluid("pg25")
    assert pg.cp(37) < w.cp(37)
    assert pg.mu(37) > w.mu(37)


@pytest.mark.parametrize("cfg", ["configs/azure_conv.yaml", "configs/azure_code.yaml"])
def test_azure_traces_run_clean(cfg):
    r = engine.run_scenario(cfg)
    assert np.isfinite(r["T_die"]).all()
    assert 40.0 < r["safety"]["peak_T_die"] < 120.0
