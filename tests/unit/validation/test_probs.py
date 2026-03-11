"""Tests for tutorial-style probabilities table export."""
from __future__ import annotations

import numpy as np

from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.validation import probs_dataframe


def _scenario_result() -> ScenarioResult:
    n = 4
    return ScenarioResult(
        scenario_id=ScenarioID.TP,
        host_star_tic_id=12345678,
        ln_evidence=-5.0,
        host_mass_msun=np.ones(n),
        host_radius_rsun=np.ones(n),
        host_u1=np.full(n, 0.3),
        host_u2=np.full(n, 0.2),
        period_days=np.full(n, 5.0),
        inclination_deg=np.full(n, 88.0),
        impact_parameter=np.full(n, 0.1),
        eccentricity=np.zeros(n),
        arg_periastron_deg=np.full(n, 90.0),
        planet_radius_rearth=np.full(n, 2.0),
        eb_mass_msun=np.zeros(n),
        eb_radius_rsun=np.zeros(n),
        flux_ratio_eb_tess=np.zeros(n),
        companion_mass_msun=np.zeros(n),
        companion_radius_rsun=np.zeros(n),
        flux_ratio_companion_tess=np.zeros(n),
    )


def test_probs_dataframe_matches_tutorial_columns() -> None:
    df = probs_dataframe(
        ValidationResult(
            target_id=12345678,
            false_positive_probability=0.1,
            nearby_false_positive_probability=0.02,
            scenario_results=[_scenario_result()],
        )
    )

    assert list(df.columns) == [
        "ID",
        "scenario",
        "M_s",
        "R_s",
        "P_orb",
        "inc",
        "b",
        "ecc",
        "w",
        "R_p",
        "M_EB",
        "R_EB",
        "M_comp",
        "R_comp",
        "flux_ratio_comp_T",
        "prob",
        "lnZ",
    ]
    assert df.iloc[0]["ID"] == 12345678
    assert df.iloc[0]["scenario"] == "TP"
    assert df.iloc[0]["lnZ"] == -5.0
