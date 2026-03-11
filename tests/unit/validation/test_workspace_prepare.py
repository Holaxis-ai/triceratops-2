"""Tests for the explicit I/O/compute split: prepare() and compute_prepared()."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.scenarios.registry import ScenarioRegistry
from triceratops.validation.job import PreparedValidationInputs
from triceratops.validation.workspace import ValidationWorkspace


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubCatalogProvider:
    """Returns a pre-built StellarField without network calls."""

    def __init__(self, field: StellarField | None = None) -> None:
        self._field = field or _default_field()

    def query_nearby_stars(
        self, tic_id: int, search_radius_px: int, mission: str,
    ) -> StellarField:
        return self._field


def _default_star() -> Star:
    return Star(
        tic_id=12345678,
        ra_deg=83.82,
        dec_deg=-5.39,
        tmag=10.5,
        jmag=9.8,
        hmag=9.5,
        kmag=9.4,
        bmag=11.2,
        vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        ),
    )


def _default_field() -> StellarField:
    return StellarField(
        target_id=12345678,
        mission="TESS",
        search_radius_pixels=10,
        stars=[_default_star()],
    )


def _make_result(sid: ScenarioID, lnZ: float, n: int = 10) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        host_star_tic_id=0,
        ln_evidence=lnZ,
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


@dataclass
class _FakeScenario:
    _scenario_id: ScenarioID
    _is_eb: bool
    _result: ScenarioResult

    @property
    def scenario_id(self) -> ScenarioID:
        return self._scenario_id

    @property
    def is_eb(self) -> bool:
        return self._is_eb

    @property
    def returns_twin(self) -> bool:
        return self._is_eb

    def compute(
        self,
        light_curve: object = None,
        stellar_params: object = None,
        period_days: object = None,
        config: object = None,
        external_lcs: object = None,
        **kwargs: object,
    ) -> ScenarioResult:
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def transit_lc() -> LightCurve:
    time = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def workspace_with_fake_tp() -> ValidationWorkspace:
    """Workspace with a single fake TP scenario for deterministic tests."""
    tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
    fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
    registry = ScenarioRegistry()
    registry.register(fake_tp)

    ws = ValidationWorkspace(
        tic_id=12345678,
        sectors=np.array([1]),
        catalog_provider=_StubCatalogProvider(),
        config=Config(n_mc_samples=100, n_best_samples=10),
    )
    ws._engine._registry = registry
    return ws


# ---------------------------------------------------------------------------
# prepare() tests
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_returns_prepared_inputs(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        """prepare() returns a PreparedValidationInputs."""
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        assert isinstance(prepared, PreparedValidationInputs)

    def test_prepare_carries_target_id(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        assert prepared.target_id == workspace_with_fake_tp.tic_id

    def test_prepare_carries_light_curve(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        assert prepared.light_curve is transit_lc

    def test_prepare_carries_period(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=3.5,
        )
        assert prepared.period_days == 3.5

    def test_prepare_does_not_cache_result(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        """prepare() should NOT cache a ValidationResult."""
        workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        assert workspace_with_fake_tp.results is None

    def test_prepare_carries_stellar_field(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        assert prepared.stellar_field is not None
        assert prepared.stellar_field.target_id == 12345678


# ---------------------------------------------------------------------------
# compute_prepared() tests
# ---------------------------------------------------------------------------


class TestComputePrepared:
    def test_compute_prepared_returns_validation_result(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        result = workspace_with_fake_tp.compute_prepared(prepared)
        assert isinstance(result, ValidationResult)

    def test_compute_prepared_caches_result(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        result = workspace_with_fake_tp.compute_prepared(prepared)
        assert workspace_with_fake_tp.results is result

    def test_compute_prepared_stores_light_curve_for_plot(
        self, workspace_with_fake_tp: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        """compute_prepared() stores light_curve so plot_fits() can use it."""
        prepared = workspace_with_fake_tp.prepare(
            light_curve=transit_lc, period_days=5.0,
        )
        workspace_with_fake_tp.compute_prepared(prepared)
        assert workspace_with_fake_tp._last_light_curve is transit_lc


# ---------------------------------------------------------------------------
# Two-step API: prepare -> compute_prepared matches compute_probs
# ---------------------------------------------------------------------------


class TestTwoStepEquivalence:
    def test_two_step_matches_compute_probs(
        self, transit_lc: LightCurve,
    ) -> None:
        """prepare() + compute_prepared() gives same FPP as compute_probs()."""
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
        registry = ScenarioRegistry()
        registry.register(fake_tp)

        # Two-step path
        ws1 = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_StubCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        ws1._engine._registry = registry
        prepared = ws1.prepare(light_curve=transit_lc, period_days=5.0)
        result_two_step = ws1.compute_prepared(prepared)

        # One-step path (fresh workspace to avoid cached state)
        tp_result2 = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp2 = _FakeScenario(ScenarioID.TP, False, tp_result2)
        registry2 = ScenarioRegistry()
        registry2.register(fake_tp2)

        ws2 = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_StubCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        ws2._engine._registry = registry2
        result_one_step = ws2.compute_probs(transit_lc, period_days=5.0)

        assert result_two_step.fpp == pytest.approx(result_one_step.fpp, abs=1e-10)
        assert result_two_step.nfpp == pytest.approx(result_one_step.nfpp, abs=1e-10)
        assert len(result_two_step.scenario_results) == len(result_one_step.scenario_results)

    def test_compute_probs_delegates_to_prepare_and_compute_prepared(
        self, transit_lc: LightCurve,
    ) -> None:
        """compute_probs() should be equivalent to prepare() + compute_prepared()."""
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
        registry = ScenarioRegistry()
        registry.register(fake_tp)

        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_StubCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        ws._engine._registry = registry

        result = ws.compute_probs(transit_lc, period_days=5.0)

        # compute_probs should cache the result
        assert isinstance(result, ValidationResult)
        assert ws.results is result
        assert 0.0 <= ws.fpp <= 1.0
