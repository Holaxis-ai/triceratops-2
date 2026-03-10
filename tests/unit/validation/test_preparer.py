"""Tests for ValidationPreparer (Phase 3) and remote-style compute."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from triceratops.catalog.protocols import StarCatalogProvider
from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.population.protocols import PopulationSynthesisProvider, TRILEGALResult
from triceratops.scenarios.registry import ScenarioRegistry
from triceratops.validation.engine import ValidationEngine
from triceratops.validation.job import PreparedValidationInputs
from triceratops.validation.preparer import ValidationPreparer


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_star(tic_id: int = 11111) -> Star:
    return Star(
        tic_id=tic_id,
        ra_deg=50.0, dec_deg=15.0,
        tmag=10.5, jmag=9.8, hmag=9.5, kmag=9.4,
        bmag=11.2, vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        ),
        flux_ratio=1.0,
        transit_depth_required=0.01,
    )


def _make_stellar_field(tic_id: int = 11111) -> StellarField:
    return StellarField(
        target_id=tic_id,
        mission="TESS",
        search_radius_pixels=10,
        stars=[_make_star(tic_id)],
    )


def _make_lc() -> LightCurve:
    t = np.linspace(-0.1, 0.1, 40)
    flux = np.ones(40)
    flux[18:22] = 0.999
    return LightCurve(time_days=t, flux=flux, flux_err=0.001)


def _make_cfg() -> Config:
    return Config(n_mc_samples=100, n_best_samples=10)


def _make_trilegal() -> TRILEGALResult:
    return TRILEGALResult(
        tmags=np.array([14.0]),
        masses=np.array([0.5]),
        loggs=np.array([4.5]),
        teffs=np.array([4000.0]),
        metallicities=np.array([0.0]),
        jmags=np.array([13.0]),
        hmags=np.array([12.8]),
        kmags=np.array([12.7]),
        gmags=np.array([15.0]),
        rmags=np.array([14.5]),
        imags=np.array([14.2]),
        zmags=np.array([14.0]),
    )


class _StubCatalogProvider:
    """Returns a fixed stellar field for any query."""

    def __init__(self, stellar_field: StellarField) -> None:
        self._field = stellar_field
        self.call_count = 0

    def query_nearby_stars(
        self, tic_id: int, search_radius_px: int, mission: str
    ) -> StellarField:
        self.call_count += 1
        return self._field


class _StubPopulationProvider:
    """Returns a fixed TRILEGALResult for any query."""

    def __init__(self, result: TRILEGALResult) -> None:
        self._result = result
        self.call_count = 0

    def query(self, ra_deg: float, dec_deg: float, target_tmag: float, cache_path=None) -> TRILEGALResult:
        self.call_count += 1
        return self._result


class _BoomProvider:
    """Raises if called — used to verify something is not called."""

    def query_nearby_stars(self, **kwargs: object) -> object:
        raise AssertionError("catalog provider must not be called here")

    def query(self, **kwargs: object) -> object:
        raise AssertionError("population provider must not be called here")


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


def _make_scenario_result(sid: ScenarioID, lnZ: float = 0.0, n: int = 10) -> ScenarioResult:
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


# ---------------------------------------------------------------------------
# ValidationPreparer construction tests
# ---------------------------------------------------------------------------


class TestValidationPreparerConstruction:
    def test_constructs_with_catalog_provider_only(self) -> None:
        """Can construct with only a catalog provider."""
        sf = _make_stellar_field()
        preparer = ValidationPreparer(catalog_provider=_StubCatalogProvider(sf))
        assert preparer is not None

    def test_constructs_with_all_providers(self) -> None:
        """Can construct with all providers."""
        sf = _make_stellar_field()
        pop = _StubPopulationProvider(_make_trilegal())
        preparer = ValidationPreparer(
            catalog_provider=_StubCatalogProvider(sf),
            population_provider=pop,
            aperture_provider=None,
        )
        assert preparer is not None

    def test_population_provider_defaults_to_none(self) -> None:
        """population_provider defaults to None."""
        sf = _make_stellar_field()
        preparer = ValidationPreparer(catalog_provider=_StubCatalogProvider(sf))
        assert preparer._population is None


# ---------------------------------------------------------------------------
# ValidationPreparer.prepare() tests
# ---------------------------------------------------------------------------


class TestValidationPreparerPrepare:
    def test_prepare_returns_prepared_validation_inputs(self) -> None:
        """prepare() returns a PreparedValidationInputs."""
        sf = _make_stellar_field()
        catalog = _StubCatalogProvider(sf)
        preparer = ValidationPreparer(catalog_provider=catalog)

        result = preparer.prepare(
            target_id=11111,
            sectors=np.array([1, 2]),
            light_curve=_make_lc(),
            config=_make_cfg(),
            period_days=5.0,
        )
        assert isinstance(result, PreparedValidationInputs)

    def test_prepare_populates_all_required_fields(self) -> None:
        """prepare() populates target_id, stellar_field, light_curve, config, period_days."""
        sf = _make_stellar_field(99999)
        catalog = _StubCatalogProvider(sf)
        lc = _make_lc()
        cfg = _make_cfg()

        preparer = ValidationPreparer(catalog_provider=catalog)
        pvi = preparer.prepare(
            target_id=99999,
            sectors=np.array([5]),
            light_curve=lc,
            config=cfg,
            period_days=14.2,
        )
        assert pvi.target_id == 99999
        assert pvi.stellar_field is sf
        assert pvi.light_curve is lc
        assert pvi.config is cfg
        assert pvi.period_days == pytest.approx(14.2)

    def test_prepare_calls_catalog_provider(self) -> None:
        """prepare() calls the catalog provider exactly once."""
        sf = _make_stellar_field()
        catalog = _StubCatalogProvider(sf)
        preparer = ValidationPreparer(catalog_provider=catalog)

        preparer.prepare(
            target_id=11111,
            sectors=np.array([1]),
            light_curve=_make_lc(),
            config=_make_cfg(),
            period_days=5.0,
        )
        assert catalog.call_count == 1

    def test_prepare_fetches_trilegal_when_provider_given(self) -> None:
        """prepare() fetches TRILEGAL population when population_provider is given."""
        sf = _make_stellar_field()
        catalog = _StubCatalogProvider(sf)
        pop = _StubPopulationProvider(_make_trilegal())
        preparer = ValidationPreparer(catalog_provider=catalog, population_provider=pop)

        pvi = preparer.prepare(
            target_id=11111,
            sectors=np.array([1]),
            light_curve=_make_lc(),
            config=_make_cfg(),
            period_days=5.0,
        )
        assert pop.call_count == 1
        assert pvi.trilegal_population is not None
        assert isinstance(pvi.trilegal_population, TRILEGALResult)

    def test_prepare_trilegal_none_when_no_provider(self) -> None:
        """prepare() leaves trilegal_population=None when no population_provider."""
        sf = _make_stellar_field()
        catalog = _StubCatalogProvider(sf)
        preparer = ValidationPreparer(catalog_provider=catalog, population_provider=None)

        pvi = preparer.prepare(
            target_id=11111,
            sectors=np.array([1]),
            light_curve=_make_lc(),
            config=_make_cfg(),
            period_days=5.0,
        )
        assert pvi.trilegal_population is None

    def test_prepare_optional_fields_default_to_none(self) -> None:
        """Optional fields are None when no files/data provided."""
        sf = _make_stellar_field()
        catalog = _StubCatalogProvider(sf)
        preparer = ValidationPreparer(catalog_provider=catalog)

        pvi = preparer.prepare(
            target_id=11111,
            sectors=np.array([1]),
            light_curve=_make_lc(),
            config=_make_cfg(),
            period_days=5.0,
        )
        assert pvi.external_lcs is None
        assert pvi.contrast_curve is None
        assert pvi.molusc_file is None


# ---------------------------------------------------------------------------
# ValidationWorkspace constructor IO test
# ---------------------------------------------------------------------------


class TestWorkspaceConstructorIO:
    def test_workspace_constructor_calls_catalog_provider(self) -> None:
        """ValidationWorkspace constructor calls catalog_provider once (catalog query)."""
        sf = _make_stellar_field(12345)
        catalog = _StubCatalogProvider(sf)

        from triceratops.validation.workspace import ValidationWorkspace
        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            catalog_provider=catalog,
        )
        assert catalog.call_count == 1

    def test_workspace_population_provider_not_called_during_construction(self) -> None:
        """Population provider must NOT be called during workspace construction."""
        sf = _make_stellar_field(12345)
        catalog = _StubCatalogProvider(sf)
        boom_pop = _BoomProvider()

        from triceratops.validation.workspace import ValidationWorkspace
        # Constructing the workspace with a boom population provider must not raise.
        # The provider should only be called during compute_probs().
        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            catalog_provider=catalog,
            population_provider=boom_pop,  # type: ignore[arg-type]
        )
        # If we reach here, no provider call happened during __init__
        assert ws is not None


# ---------------------------------------------------------------------------
# Remote-style no-provider compute test
# ---------------------------------------------------------------------------


class TestRemoteStyleCompute:
    """Construct PreparedValidationInputs directly (no preparer) and call compute_prepared().

    This is the remote-compute canary: verifies the engine works from a
    manually-assembled payload with no providers anywhere in the call chain.
    """

    def test_remote_style_compute_produces_result_with_fpp_in_range(self) -> None:
        """Direct PreparedValidationInputs → compute_prepared() → FPP in [0, 1]."""
        sf = _make_stellar_field(77777)
        lc = _make_lc()
        cfg = _make_cfg()

        sr = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=sr)
        registry = ScenarioRegistry()
        registry.register(fake)

        # Engine has no providers
        engine = ValidationEngine(registry=registry)

        # PreparedValidationInputs built without any provider
        pvi = PreparedValidationInputs(
            target_id=77777,
            stellar_field=sf,
            light_curve=lc,
            config=cfg,
            period_days=7.0,
            trilegal_population=None,
            external_lcs=None,
            contrast_curve=None,
        )

        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)
        assert 0.0 <= vr.fpp <= 1.0

    def test_remote_style_compute_with_trilegal_produces_result(self) -> None:
        """Remote compute with pre-materialised TRILEGAL population works correctly."""
        sf = _make_stellar_field(88888)
        lc = _make_lc()
        cfg = _make_cfg()
        trilegal = _make_trilegal()

        sr = _make_scenario_result(ScenarioID.DTP, lnZ=-1.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.DTP, _result=sr)
        registry = ScenarioRegistry()
        registry.register(fake)

        # Engine has no providers
        engine = ValidationEngine(registry=registry)

        pvi = PreparedValidationInputs(
            target_id=88888,
            stellar_field=sf,
            light_curve=lc,
            config=cfg,
            period_days=10.0,
            trilegal_population=trilegal,
        )

        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)
        assert 0.0 <= vr.fpp <= 1.0
        assert 0.0 <= vr.nfpp <= 1.0
