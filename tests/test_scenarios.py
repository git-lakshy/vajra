"""Tests for Indian regional scenario switching and domain configuration."""

from vajra.ingest.domains import SCENARIOS, apply_scenario
from vajra.engine import VajraEngine
from vajra.config import GRID, POIS


def test_scenario_catalog():
    assert "uttarakhand_cloudburst" in SCENARIOS
    assert "delhi_squall" in SCENARIOS
    assert "kolkata_kalbaishakhi" in SCENARIOS
    assert "nagpur_vidarbha" in SCENARIOS


def test_apply_scenario_uttarakhand():
    sc = apply_scenario("uttarakhand_cloudburst")
    assert sc["id"] == "uttarakhand_cloudburst"
    assert "Dehradun" in POIS
    assert "Rishikesh" in POIS
    assert round(GRID["center_lat"], 2) == 30.32


def test_apply_scenario_delhi():
    sc = apply_scenario("delhi_squall")
    assert sc["id"] == "delhi_squall"
    assert "New Delhi" in POIS
    assert "Noida" in POIS
    assert round(GRID["center_lat"], 2) == 28.61


def test_apply_scenario_kolkata():
    sc = apply_scenario("kolkata_kalbaishakhi")
    assert sc["id"] == "kolkata_kalbaishakhi"
    assert "Kolkata" in POIS
    assert "Howrah" in POIS
    assert round(GRID["center_lat"], 2) == 22.57


def test_engine_scenario_lifecycle():
    eng = VajraEngine(source="uttarakhand_cloudburst")
    state = eng.run_cycle()
    assert state.scenario_id == "uttarakhand_cloudburst"
    assert state.frame == 1

    # Switch to Delhi squall
    meta = eng.switch_scenario("delhi_squall")
    assert meta["id"] == "delhi_squall"
    state2 = eng.run_cycle()
    assert state2.scenario_id == "delhi_squall"
    assert "New Delhi" in POIS
