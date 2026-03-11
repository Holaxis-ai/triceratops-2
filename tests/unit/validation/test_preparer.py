"""Tests for ValidationPreparer and remote-style compute."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from triceratops.assembly.inputs import AssembledInputs
from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.population.protocols import TRILEGALResult
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

    def query(
        self, ra_deg: float, dec_deg: float, target_tmag: float, cache_path=None,
    ) -> TRILEGALResult:
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

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list | None = None,
        **kwargs: object,
    ) -> ScenarioResult:
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


def _make_assembled_inputs(
    tic_id: int = 11111,
    mission: str = "TESS",
    with_lc: bool = True,
    with_trilegal: bool = False,
    stellar_params: StellarParameters | None = None,
) -> AssembledInputs:
    """Build a minimal AssembledInputs for testing prepare()."""
    from triceratops.assembly.inputs import AssembledInputs
    from triceratops.lightcurve.ephemeris import ResolvedTarget

    sf = _make_stellar_field(tic_id)
    sf.mission = mission

    if stellar_params is not None:
        sf.stars[0].stellar_params = stellar_params

    rt = ResolvedTarget(target_ref=f"TIC {tic_id}", tic_id=tic_id)
    trilegal = _make_trilegal() if with_trilegal else None
    lc = _make_lc() if with_lc else None

    return AssembledInputs(
        resolved_target=rt,
        stellar_field=sf,
        light_curve=lc,
        trilegal_population=trilegal,
    )


# ---------------------------------------------------------------------------
# ValidationPreparer construction tests
# ---------------------------------------------------------------------------


class TestValidationPreparerConstruction:
    def test_constructs_with_defaults(self) -> None:
        """Can construct with no arguments (uses default registry)."""
        preparer = ValidationPreparer()
        assert preparer is not None

    def test_constructs_with_registry(self) -> None:
        """Can construct with a custom registry."""
        registry = ScenarioRegistry()
        preparer = ValidationPreparer(registry=registry)
        assert preparer is not None


# ---------------------------------------------------------------------------
# ValidationPreparer.prepare() tests (formerly prepare_from_assembled)
# ---------------------------------------------------------------------------


class TestPrepare:
    """Tests for ValidationPreparer.prepare()."""

    def test_returns_prepared_validation_inputs(self) -> None:
        """Valid inputs produce a PreparedValidationInputs with correct fields."""
        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        result = preparer.prepare(assembled, cfg, period_days=5.0)

        assert isinstance(result, PreparedValidationInputs)
        assert result.target_id == 11111
        assert result.stellar_field is assembled.stellar_field
        assert result.light_curve is assembled.light_curve
        assert result.config is cfg
        assert result.period_days == pytest.approx(5.0)

    def test_does_not_call_any_provider(self) -> None:
        """prepare() must not call any provider (pure validation)."""
        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        result = preparer.prepare(assembled, cfg, period_days=5.0)
        assert isinstance(result, PreparedValidationInputs)

    def test_passes_scenario_ids_through(self) -> None:
        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        result = preparer.prepare(
            assembled, cfg, period_days=5.0, scenario_ids=[ScenarioID.TP],
        )
        assert result.scenario_ids == [ScenarioID.TP]

    def test_carries_trilegal_population(self) -> None:
        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs(with_trilegal=True)
        cfg = _make_cfg()

        result = preparer.prepare(assembled, cfg, period_days=5.0)
        assert result.trilegal_population is assembled.trilegal_population
        assert result.trilegal_population is not None

    def test_mission_gate_rejects_non_tess(self) -> None:
        from triceratops.validation.errors import UnsupportedComputeModeError

        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs(mission="Kepler")
        cfg = _make_cfg()

        with pytest.raises(UnsupportedComputeModeError, match="TESS"):
            preparer.prepare(assembled, cfg, period_days=5.0)

    def test_rejects_missing_stellar_params(self) -> None:
        from triceratops.validation.errors import PreparedInputIncompleteError

        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        assembled.stellar_field.stars[0].stellar_params = None
        cfg = _make_cfg()

        with pytest.raises(PreparedInputIncompleteError, match="stellar_params"):
            preparer.prepare(assembled, cfg, period_days=5.0)

    def test_rejects_empty_light_curve(self) -> None:
        from triceratops.assembly.inputs import AssembledInputs
        from triceratops.lightcurve.ephemeris import ResolvedTarget
        from triceratops.validation.errors import ValidationInputError

        preparer = ValidationPreparer()
        empty_lc = LightCurve(
            time_days=np.array([]),
            flux=np.array([]),
            flux_err=0.001,
        )
        assembled = AssembledInputs(
            resolved_target=ResolvedTarget(target_ref="TIC 11111", tic_id=11111),
            stellar_field=_make_stellar_field(),
            light_curve=empty_lc,
        )
        cfg = _make_cfg()

        with pytest.raises(ValidationInputError, match="empty"):
            preparer.prepare(assembled, cfg, period_days=5.0)

    def test_rejects_none_light_curve(self) -> None:
        from triceratops.assembly.inputs import AssembledInputs
        from triceratops.lightcurve.ephemeris import ResolvedTarget
        from triceratops.validation.errors import ValidationInputError

        preparer = ValidationPreparer()
        assembled = AssembledInputs(
            resolved_target=ResolvedTarget(target_ref="TIC 11111", tic_id=11111),
            stellar_field=_make_stellar_field(),
            light_curve=None,
        )
        cfg = _make_cfg()

        with pytest.raises(ValidationInputError, match="None"):
            preparer.prepare(assembled, cfg, period_days=5.0)

    def test_rejects_negative_period(self) -> None:
        from triceratops.validation.errors import ValidationInputError

        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        with pytest.raises(ValidationInputError, match="positive"):
            preparer.prepare(assembled, cfg, period_days=-1.0)

    def test_rejects_nonfinite_period(self) -> None:
        from triceratops.validation.errors import ValidationInputError

        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        with pytest.raises(ValidationInputError, match="finite"):
            preparer.prepare(assembled, cfg, period_days=float("inf"))

    def test_rejects_unregistered_scenario_ids(self) -> None:
        empty_registry = ScenarioRegistry()
        preparer = ValidationPreparer(registry=empty_registry)
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        with pytest.raises(ValueError, match="not registered"):
            preparer.prepare(
                assembled, cfg, period_days=5.0,
                scenario_ids=[ScenarioID.TP],
            )

    def test_end_to_end_orchestrator_to_engine(self) -> None:
        """assemble() -> prepare() -> compute_prepared() chain."""
        from triceratops.assembly.config import AssemblyConfig
        from triceratops.assembly.orchestrator import DataAssemblyOrchestrator
        from triceratops.lightcurve.ephemeris import ResolvedTarget

        sf = _make_stellar_field(77777)
        catalog = _StubCatalogProvider(sf)

        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = ResolvedTarget(target_ref="TIC 77777", tic_id=77777)
        asm_config = AssemblyConfig(include_light_curve=False)
        assembled = orch.assemble(target, asm_config)

        from dataclasses import replace
        assembled = replace(assembled, light_curve=_make_lc())

        sr = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=sr)
        registry = ScenarioRegistry()
        registry.register(fake)

        preparer = ValidationPreparer(registry=registry)
        cfg = _make_cfg()
        pvi = preparer.prepare(
            assembled, cfg, period_days=5.0,
            scenario_ids=[ScenarioID.TP],
        )

        engine = ValidationEngine(registry=registry)
        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)
        assert 0.0 <= vr.fpp <= 1.0

    def test_period_range_validation(self) -> None:
        """Period range [min, max] is validated."""
        from triceratops.validation.errors import ValidationInputError

        preparer = ValidationPreparer()
        assembled = _make_assembled_inputs()
        cfg = _make_cfg()

        # min >= max
        with pytest.raises(ValidationInputError, match="min < max"):
            preparer.prepare(assembled, cfg, period_days=[5.0, 3.0])

        # valid range works
        result = preparer.prepare(assembled, cfg, period_days=[3.0, 5.0])
        assert isinstance(result, PreparedValidationInputs)


# ---------------------------------------------------------------------------
# ValidationWorkspace constructor IO test
# ---------------------------------------------------------------------------


class TestWorkspaceConstructorIO:
    def test_workspace_constructor_defers_catalog_query(self) -> None:
        """ValidationWorkspace constructor does NOT query catalog eagerly."""
        sf = _make_stellar_field(12345)
        catalog = _StubCatalogProvider(sf)

        from triceratops.validation.workspace import ValidationWorkspace
        ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            catalog_provider=catalog,
        )
        assert catalog.call_count == 0

    def test_workspace_catalog_queried_on_first_stars_access(self) -> None:
        """Catalog is queried lazily on first .stars access."""
        sf = _make_stellar_field(12345)
        catalog = _StubCatalogProvider(sf)

        from triceratops.validation.workspace import ValidationWorkspace
        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            catalog_provider=catalog,
        )
        _ = ws.stars
        assert catalog.call_count == 1

    def test_workspace_population_provider_not_called_during_construction(self) -> None:
        """Population provider must NOT be called during workspace construction."""
        sf = _make_stellar_field(12345)
        catalog = _StubCatalogProvider(sf)
        boom_pop = _BoomProvider()

        from triceratops.validation.workspace import ValidationWorkspace
        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            catalog_provider=catalog,
            population_provider=boom_pop,  # type: ignore[arg-type]
        )
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
        """Direct PreparedValidationInputs -> compute_prepared() -> FPP in [0, 1]."""
        sf = _make_stellar_field(77777)
        lc = _make_lc()
        cfg = _make_cfg()

        sr = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=sr)
        registry = ScenarioRegistry()
        registry.register(fake)

        engine = ValidationEngine(registry=registry)

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
