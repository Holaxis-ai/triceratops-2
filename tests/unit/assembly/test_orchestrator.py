"""Tests for DataAssemblyOrchestrator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from triceratops.assembly import AssembledInputs, AssemblyConfig, AssemblyMetadata
from triceratops.assembly.errors import (
    AssemblyConfigError,
    AssemblyLightCurveError,
    CatalogAcquisitionError,
)
from triceratops.assembly.orchestrator import DataAssemblyOrchestrator
from triceratops.domain.entities import StellarField, Star
from triceratops.domain.molusc import MoluscData
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.population.protocols import TRILEGALResult
from triceratops.scenarios.registry import ScenarioRegistry


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_star(tic_id: int = 11111) -> Star:
    return Star(
        tic_id=tic_id,
        ra_deg=50.0,
        dec_deg=15.0,
        tmag=10.5,
        jmag=9.8,
        hmag=9.5,
        kmag=9.4,
        bmag=11.2,
        vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0,
            radius_rsun=1.0,
            teff_k=5778.0,
            logg=4.44,
            metallicity_dex=0.0,
            parallax_mas=10.0,
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


def _make_resolved_target(
    tic_id: int = 11111, with_ephemeris: bool = False,
) -> ResolvedTarget:
    eph = Ephemeris(period_days=5.0, t0_btjd=1000.0) if with_ephemeris else None
    return ResolvedTarget(target_ref=f"TIC {tic_id}", tic_id=tic_id, ephemeris=eph)


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


def _make_contrast_curve() -> ContrastCurve:
    return ContrastCurve(
        separations_arcsec=np.array([0.5, 1.0, 2.0]),
        delta_mags=np.array([3.0, 5.0, 7.0]),
        band="TESS",
    )


def _make_molusc_data() -> MoluscData:
    return MoluscData(
        semi_major_axis_au=np.array([1.0, 2.0]),
        eccentricity=np.array([0.1, 0.2]),
        mass_ratio=np.array([0.5, 0.8]),
    )


class StubCatalogProvider:
    """Returns a fixed stellar field."""

    def __init__(self, stellar_field: StellarField | None = None) -> None:
        self._field = stellar_field or _make_stellar_field()
        self.call_count = 0
        self.last_args: dict | None = None

    def query_nearby_stars(
        self, tic_id: int, search_radius_px: int, mission: str,
    ) -> StellarField:
        self.call_count += 1
        self.last_args = dict(
            tic_id=tic_id, search_radius_px=search_radius_px, mission=mission,
        )
        return self._field


class BoomCatalogProvider:
    """Raises on call."""

    def query_nearby_stars(self, **kwargs: object) -> object:
        raise RuntimeError("catalog exploded")


class StubPopulationProvider:
    """Returns a fixed TRILEGALResult."""

    def __init__(self) -> None:
        self.call_count = 0

    def query(
        self, ra_deg: float = 0.0, dec_deg: float = 0.0,
        target_tmag: float = 10.0, cache_path: object = None,
    ) -> TRILEGALResult:
        self.call_count += 1
        return _make_trilegal()


class StubContrastSource:
    """Returns a fixed ContrastCurve."""

    def __init__(self) -> None:
        self.call_count = 0

    def load(self, band: str) -> ContrastCurve:
        self.call_count += 1
        return _make_contrast_curve()


class StubMoluscSource:
    """Returns fixed MoluscData."""

    def __init__(self) -> None:
        self.call_count = 0

    def load(self) -> MoluscData:
        self.call_count += 1
        return _make_molusc_data()


class StubExternalLcSource:
    """Returns an empty list of external light curves."""

    def __init__(self) -> None:
        self.call_count = 0

    def load(self) -> list:
        self.call_count += 1
        return []


def _make_trilegal_registry() -> ScenarioRegistry:
    """Registry with a single TRILEGAL-dependent scenario (BTP)."""

    @dataclass
    class _FakeScenario:
        _scenario_id: ScenarioID

        @property
        def scenario_id(self) -> ScenarioID:
            return self._scenario_id

        @property
        def is_eb(self) -> bool:
            return False

        @property
        def returns_twin(self) -> bool:
            return False

        def compute(self, **kwargs: object) -> object:
            return None

    registry = ScenarioRegistry()
    registry.register(_FakeScenario(_scenario_id=ScenarioID.BTP))
    return registry


def _make_non_trilegal_registry() -> ScenarioRegistry:
    """Registry with only a non-TRILEGAL scenario (TP)."""

    @dataclass
    class _FakeScenario:
        _scenario_id: ScenarioID

        @property
        def scenario_id(self) -> ScenarioID:
            return self._scenario_id

        @property
        def is_eb(self) -> bool:
            return False

        @property
        def returns_twin(self) -> bool:
            return False

        def compute(self, **kwargs: object) -> object:
            return None

    registry = ScenarioRegistry()
    registry.register(_FakeScenario(_scenario_id=ScenarioID.TP))
    return registry


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestOrchestratorConstruction:
    def test_constructs_with_catalog_provider_only(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        assert orch is not None

    def test_constructs_with_all_providers(self) -> None:
        orch = DataAssemblyOrchestrator(
            catalog_provider=StubCatalogProvider(),
            population_provider=StubPopulationProvider(),
            contrast_source=StubContrastSource(),
            molusc_source=StubMoluscSource(),
            external_lc_source=StubExternalLcSource(),
        )
        assert orch is not None


# ---------------------------------------------------------------------------
# assemble() return type and basic behavior
# ---------------------------------------------------------------------------


class TestAssembleBasic:
    def test_assemble_returns_assembled_inputs(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)
        result = orch.assemble(target, config)
        assert isinstance(result, AssembledInputs)

    def test_assemble_calls_catalog_provider_once(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)
        orch.assemble(target, config)
        assert catalog.call_count == 1
        assert catalog.last_args["tic_id"] == 11111
        assert catalog.last_args["search_radius_px"] == 10
        assert catalog.last_args["mission"] == "TESS"

    def test_assemble_config_defaults_used_when_none(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        # config=None uses AssemblyConfig defaults; default require_light_curve
        # is True so we must expect the error when no lc_source is provided.
        with pytest.raises(AssemblyLightCurveError, match="require_light_curve"):
            orch.assemble(target, config=None)


# ---------------------------------------------------------------------------
# Stellar field: provided vs catalog
# ---------------------------------------------------------------------------


class TestAssembleStellarField:
    def test_skips_catalog_when_stellar_field_provided(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)
        existing_field = _make_stellar_field(99999)

        result = orch.assemble(target, config, stellar_field=existing_field)

        assert catalog.call_count == 0
        assert result.stellar_field is existing_field

    def test_metadata_records_provided_when_stellar_field_given(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(
            target, config, stellar_field=_make_stellar_field(),
        )
        per_input = dict(result.metadata.per_input_source)
        assert per_input["stellar_field"] == "provided"

    def test_metadata_records_catalog_when_queried(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        per_input = dict(result.metadata.per_input_source)
        assert per_input["stellar_field"] == "catalog"
        assert "catalog" in result.metadata.source_labels

    def test_catalog_failure_raises_catalog_acquisition_error(self) -> None:
        orch = DataAssemblyOrchestrator(catalog_provider=BoomCatalogProvider())
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)
        with pytest.raises(CatalogAcquisitionError, match="catalog"):
            orch.assemble(target, config)


# ---------------------------------------------------------------------------
# TRILEGAL gating
# ---------------------------------------------------------------------------


class TestAssembleTrilegal:
    def test_trilegal_fetched_when_trilegal_scenario_requested(self) -> None:
        pop = StubPopulationProvider()
        registry = _make_trilegal_registry()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            population_provider=pop,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(
            target, config, scenario_ids=[ScenarioID.BTP],
        )
        assert pop.call_count == 1
        assert result.trilegal_population is not None
        assert isinstance(result.trilegal_population, TRILEGALResult)

    def test_trilegal_skipped_when_no_trilegal_scenarios(self) -> None:
        pop = StubPopulationProvider()
        registry = _make_non_trilegal_registry()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            population_provider=pop,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(
            target, config, scenario_ids=[ScenarioID.TP],
        )
        assert pop.call_count == 0
        assert result.trilegal_population is None

    def test_trilegal_fetched_when_scenario_ids_none(self) -> None:
        """scenario_ids=None uses all registered scenarios; if any are TRILEGAL, fetch."""
        pop = StubPopulationProvider()
        registry = _make_trilegal_registry()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            population_provider=pop,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config, scenario_ids=None)
        assert pop.call_count == 1
        assert result.trilegal_population is not None

    def test_trilegal_skipped_when_no_population_provider(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.trilegal_population is None


# ---------------------------------------------------------------------------
# Contrast curve
# ---------------------------------------------------------------------------


class TestAssembleContrastCurve:
    def test_contrast_curve_assembled_when_source_provided(self) -> None:
        cc_source = StubContrastSource()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            contrast_source=cc_source,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.contrast_curve is not None
        assert cc_source.call_count == 1

    def test_contrast_curve_none_when_no_source(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.contrast_curve is None

    def test_contrast_curve_skipped_when_disabled(self) -> None:
        cc_source = StubContrastSource()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            contrast_source=cc_source,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False, include_contrast_curve=False)

        result = orch.assemble(target, config)
        assert result.contrast_curve is None
        assert cc_source.call_count == 0


# ---------------------------------------------------------------------------
# MOLUSC
# ---------------------------------------------------------------------------


class TestAssembleMolusc:
    def test_molusc_assembled_when_source_provided(self) -> None:
        mol_source = StubMoluscSource()
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            molusc_source=mol_source,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.molusc_data is not None
        assert mol_source.call_count == 1

    def test_molusc_none_when_no_source(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.molusc_data is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestAssemblyMetadataOutput:
    def test_metadata_has_source_labels(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert isinstance(result.metadata, AssemblyMetadata)
        assert len(result.metadata.source_labels) > 0

    def test_metadata_has_created_at_utc(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.metadata.created_at_utc is not None
        assert "T" in result.metadata.created_at_utc  # ISO format

    def test_metadata_per_input_source_tracks_all_assembled(self) -> None:
        """All assembled inputs are tracked in per_input_source."""
        cc_source = StubContrastSource()
        mol_source = StubMoluscSource()
        pop = StubPopulationProvider()
        registry = _make_trilegal_registry()
        catalog = StubCatalogProvider()

        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            population_provider=pop,
            contrast_source=cc_source,
            molusc_source=mol_source,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        per_input = dict(result.metadata.per_input_source)
        assert "stellar_field" in per_input
        assert "contrast_curve" in per_input
        assert "molusc_data" in per_input
        assert "trilegal_population" in per_input


# ---------------------------------------------------------------------------
# Light curve edge cases
# ---------------------------------------------------------------------------


class TestAssembleLightCurve:
    def test_lc_assembly_raises_when_no_ephemeris(self) -> None:
        """Requesting LC assembly without ephemeris raises AssemblyLightCurveError."""

        class _FakeLcSource:
            def fetch_raw(self, config: object) -> object:
                raise AssertionError("should not be called")

        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            lc_source=_FakeLcSource(),  # type: ignore[arg-type]
        )
        target = _make_resolved_target(with_ephemeris=False)
        config = AssemblyConfig(include_light_curve=True)

        with pytest.raises(AssemblyLightCurveError, match="ephemeris"):
            orch.assemble(target, config)

    def test_lc_not_assembled_when_disabled(self) -> None:
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config)
        assert result.light_curve is None

    def test_lc_not_assembled_when_no_source(self) -> None:
        """No lc_source injected means no LC assembly even if enabled;
        with require_light_curve=True (default), this now raises."""
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target(with_ephemeris=True)
        config = AssemblyConfig(include_light_curve=True)

        with pytest.raises(AssemblyLightCurveError, match="require_light_curve"):
            orch.assemble(target, config)

    def test_lc_not_assembled_when_no_source_and_not_required(self) -> None:
        """No lc_source with require_light_curve=False should return None."""
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target(with_ephemeris=True)
        config = AssemblyConfig(include_light_curve=True, require_light_curve=False)

        result = orch.assemble(target, config)
        assert result.light_curve is None


# ---------------------------------------------------------------------------
# scenario_ids early validation
# ---------------------------------------------------------------------------


class TestAssembleScenarioIdValidation:
    def test_unknown_scenario_id_raises_before_io(self) -> None:
        """Unknown scenario IDs must raise AssemblyConfigError before catalog I/O."""
        catalog = StubCatalogProvider()
        registry = _make_non_trilegal_registry()  # only has TP
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        with pytest.raises(AssemblyConfigError, match="Unknown scenario IDs"):
            orch.assemble(target, config, scenario_ids=[ScenarioID.BTP])

        # Catalog must NOT have been called
        assert catalog.call_count == 0

    def test_valid_scenario_ids_pass_validation(self) -> None:
        """Known scenario IDs should not raise."""
        catalog = StubCatalogProvider()
        registry = _make_non_trilegal_registry()  # has TP
        orch = DataAssemblyOrchestrator(
            catalog_provider=catalog,
            registry=registry,
        )
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config, scenario_ids=[ScenarioID.TP])
        assert isinstance(result, AssembledInputs)

    def test_none_scenario_ids_skips_validation(self) -> None:
        """scenario_ids=None should not trigger validation."""
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(include_light_curve=False)

        result = orch.assemble(target, config, scenario_ids=None)
        assert isinstance(result, AssembledInputs)


# ---------------------------------------------------------------------------
# require_light_curve enforcement
# ---------------------------------------------------------------------------


class TestRequireLightCurve:
    def test_require_lc_raises_when_no_lc_assembled(self) -> None:
        """require_light_curve=True with no lc_source must raise at assembly time."""
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(
            include_light_curve=True,
            require_light_curve=True,
        )

        with pytest.raises(AssemblyLightCurveError, match="require_light_curve"):
            orch.assemble(target, config)

    def test_require_lc_false_allows_none(self) -> None:
        """require_light_curve=False should not raise when light_curve is None."""
        catalog = StubCatalogProvider()
        orch = DataAssemblyOrchestrator(catalog_provider=catalog)
        target = _make_resolved_target()
        config = AssemblyConfig(
            include_light_curve=False,
            require_light_curve=False,
        )

        result = orch.assemble(target, config)
        assert result.light_curve is None
