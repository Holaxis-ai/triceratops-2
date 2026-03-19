"""Tests for triceratops.domain.result."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID


def _make_scenario_result(scenario_id: ScenarioID, ln_evidence: float = -100.0) -> ScenarioResult:
    n = 5
    zeros = np.zeros(n)
    return ScenarioResult(
        scenario_id=scenario_id,
        host_star_tic_id=12345678,
        ln_evidence=ln_evidence,
        host_mass_msun=np.ones(n),
        host_radius_rsun=np.ones(n),
        host_u1=zeros,
        host_u2=zeros,
        period_days=np.full(n, 5.0),
        inclination_deg=np.full(n, 89.0),
        impact_parameter=zeros,
        eccentricity=zeros,
        arg_periastron_deg=zeros,
        planet_radius_rearth=np.full(n, 2.0),
        eb_mass_msun=zeros,
        eb_radius_rsun=zeros,
        flux_ratio_eb_tess=zeros,
        companion_mass_msun=zeros,
        companion_radius_rsun=zeros,
        flux_ratio_companion_tess=zeros,
    )


class TestValidationResult:
    def test_validation_result_fpp_property(self) -> None:
        vr = ValidationResult(
            target_id=100,
            false_positive_probability=0.05,
            nearby_false_positive_probability=0.01,
            scenario_results=[],
        )
        assert vr.fpp == pytest.approx(0.05)
        assert vr.nfpp == pytest.approx(0.01)

    def test_validation_result_get_scenario_found(self) -> None:
        sr_tp = _make_scenario_result(ScenarioID.TP)
        sr_eb = _make_scenario_result(ScenarioID.EB)
        vr = ValidationResult(
            target_id=100,
            false_positive_probability=0.05,
            nearby_false_positive_probability=0.01,
            scenario_results=[sr_tp, sr_eb],
        )
        result = vr.get_scenario(ScenarioID.TP)
        assert result is not None
        assert result.scenario_id == ScenarioID.TP

    def test_validation_result_get_scenario_not_found(self) -> None:
        vr = ValidationResult(
            target_id=100,
            false_positive_probability=0.05,
            nearby_false_positive_probability=0.01,
            scenario_results=[],
        )
        assert vr.get_scenario(ScenarioID.NTP) is None

    def test_validation_result_get_scenarios_returns_all_matches(self) -> None:
        sr_one = _make_scenario_result(ScenarioID.NTP)
        sr_two = _make_scenario_result(ScenarioID.NTP)
        sr_two.host_star_tic_id = 87654321
        vr = ValidationResult(
            target_id=100,
            false_positive_probability=0.05,
            nearby_false_positive_probability=0.01,
            scenario_results=[sr_one, sr_two],
        )

        assert vr.get_scenarios(ScenarioID.NTP) == [sr_one, sr_two]

    def test_validation_result_get_scenario_warns_for_duplicate_rows(self) -> None:
        sr_one = _make_scenario_result(ScenarioID.NTP)
        sr_two = _make_scenario_result(ScenarioID.NTP)
        sr_two.host_star_tic_id = 87654321
        vr = ValidationResult(
            target_id=100,
            false_positive_probability=0.05,
            nearby_false_positive_probability=0.01,
            scenario_results=[sr_one, sr_two],
        )

        with pytest.warns(UserWarning, match="Use get_scenarios"):
            result = vr.get_scenario(ScenarioID.NTP)

        assert result is sr_one
