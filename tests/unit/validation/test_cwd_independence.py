"""Test that compute_prepared() does not depend on the current working directory.

This is the P8 boundary guarantee: no filesystem IO occurs during compute.
If any code inside compute_prepared() opens a file by relative path or
depends on CWD, this test will fail when run from a non-existent directory.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.molusc import MoluscData
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.scenarios.registry import ScenarioRegistry
from triceratops.validation.engine import ValidationEngine
from triceratops.validation.job import PreparedValidationInputs


@dataclass
class _FakeScenario:
    """Minimal fake scenario for engine tests."""
    _scenario_id: ScenarioID
    _result: ScenarioResult

    @property
    def scenario_id(self) -> ScenarioID:
        return self._scenario_id

    @property
    def is_eb(self) -> bool:
        return False

    @property
    def returns_twin(self) -> bool:
        return False

    def compute(self, **kwargs: object) -> ScenarioResult:
        return self._result


def _make_result(sid: ScenarioID) -> ScenarioResult:
    n = 10
    return ScenarioResult(
        scenario_id=sid,
        host_star_tic_id=0,
        ln_evidence=-5.0,
        host_mass_msun=np.ones(n),
        host_radius_rsun=np.ones(n),
        host_u1=np.full(n, 0.4),
        host_u2=np.full(n, 0.2),
        period_days=np.full(n, 5.0),
        inclination_deg=np.full(n, 87.0),
        impact_parameter=np.full(n, 0.3),
        eccentricity=np.zeros(n),
        arg_periastron_deg=np.full(n, 90.0),
        planet_radius_rearth=np.ones(n),
        eb_mass_msun=np.zeros(n),
        eb_radius_rsun=np.zeros(n),
        flux_ratio_eb_tess=np.zeros(n),
        companion_mass_msun=np.zeros(n),
        companion_radius_rsun=np.zeros(n),
        flux_ratio_companion_tess=np.zeros(n),
    )


def test_compute_prepared_cwd_independent() -> None:
    """compute_prepared() must produce identical results regardless of CWD."""
    star = Star(
        tic_id=12345,
        ra_deg=100.0, dec_deg=20.0,
        tmag=11.0, jmag=10.3, hmag=10.1, kmag=10.0,
        bmag=11.5, vmag=11.2,
        stellar_params=StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
            logg=4.4, metallicity_dex=0.0, parallax_mas=12.0,
        ),
        flux_ratio=1.0,
        transit_depth_required=0.01,
    )
    sf = StellarField(
        target_id=12345, mission="TESS",
        search_radius_pixels=10, stars=[star],
    )
    t = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    lc = LightCurve(time_days=t, flux=flux, flux_err=0.001)
    cfg = Config(n_mc_samples=100, n_best_samples=10, n_workers=0)

    molusc_data = MoluscData(
        semi_major_axis_au=np.array([20.0, 30.0, 50.0]),
        eccentricity=np.array([0.0, 0.1, 0.2]),
        mass_ratio=np.array([0.4, 0.5, 0.6]),
    )

    result = _make_result(ScenarioID.TP)
    fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=result)
    registry = ScenarioRegistry()
    registry.register(fake)

    engine = ValidationEngine(registry=registry)
    prepared = PreparedValidationInputs(
        target_id=12345,
        stellar_field=sf,
        light_curve=lc,
        config=cfg,
        period_days=5.0,
        molusc_data=molusc_data,
    )

    # Run from the real CWD
    result_1 = engine.compute_prepared(prepared)

    # Run from a temporary directory that contains no project files
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result_2 = engine.compute_prepared(prepared)
        finally:
            os.chdir(original_cwd)

    assert result_1.fpp == result_2.fpp
    assert result_1.nfpp == result_2.nfpp
    assert len(result_1.scenario_results) == len(result_2.scenario_results)
