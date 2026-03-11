"""Tests for ValidationWorkspace."""
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
from triceratops.assembly.errors import CatalogAcquisitionError
from triceratops.validation.workspace import ValidationWorkspace

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubStarCatalogProvider:
    """Returns a pre-built StellarField without network calls."""

    def __init__(self, field: StellarField | None = None) -> None:
        self._field = field or _default_field()

    def query_nearby_stars(  # noqa: ARG002
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


def _neighbor_star(tic_id: int = 99999999) -> Star:
    return Star(
        tic_id=tic_id,
        ra_deg=83.83,
        dec_deg=-5.38,
        tmag=12.0,
        jmag=11.5,
        hmag=11.2,
        kmag=11.1,
        bmag=12.5,
        vmag=12.2,
    )


def _default_field() -> StellarField:
    return StellarField(
        target_id=12345678,
        mission="TESS",
        search_radius_pixels=10,
        stars=[_default_star(), _neighbor_star()],
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
    ) -> ScenarioResult | tuple[ScenarioResult, ScenarioResult]:
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace() -> ValidationWorkspace:
    return ValidationWorkspace(
        tic_id=12345678,
        sectors=np.array([1]),
        catalog_provider=StubStarCatalogProvider(),
    )


@pytest.fixture()
def transit_lc() -> LightCurve:
    time = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def small_config() -> Config:
    return Config(n_mc_samples=100, n_best_samples=10)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestWorkspaceConstruction:
    def test_constructs_with_stubs(self) -> None:
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=StubStarCatalogProvider(),
        )
        assert ws is not None
        assert ws.tic_id == 12345678

    def test_stars_property(self, workspace: ValidationWorkspace) -> None:
        assert isinstance(workspace.stars, list)
        assert len(workspace.stars) == 2

    def test_target_is_first_star(self, workspace: ValidationWorkspace) -> None:
        assert workspace.target is workspace.stars[0]
        assert workspace.target.tic_id == 12345678

    def test_construction_uses_assembly_pipeline(self) -> None:
        """Lazy catalog fetch goes through assemble_stellar_field,
        which wraps catalog errors in CatalogAcquisitionError."""

        class _BoomCatalog:
            def query_nearby_stars(self, **kwargs: object) -> object:
                raise RuntimeError("boom")

        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_BoomCatalog(),
        )
        # Construction succeeds (lazy), but first access triggers the error
        with pytest.raises(CatalogAcquisitionError, match="boom"):
            _ = ws.stars


# ---------------------------------------------------------------------------
# Star mutation tests
# ---------------------------------------------------------------------------


class TestStarMutation:
    def test_add_star_increases_count(self, workspace: ValidationWorkspace) -> None:
        original_count = len(workspace.stars)
        workspace.add_star(_neighbor_star(tic_id=77777777))
        assert len(workspace.stars) == original_count + 1

    def test_add_star_invalidates_result(self, workspace: ValidationWorkspace) -> None:
        workspace._last_result = ValidationResult(
            target_id=0, false_positive_probability=0.5,
            nearby_false_positive_probability=0.0, scenario_results=[],
        )
        workspace.add_star(_neighbor_star(tic_id=77777777))
        assert workspace._last_result is None

    def test_remove_star_decreases_count(self, workspace: ValidationWorkspace) -> None:
        original_count = len(workspace.stars)
        workspace.remove_star(99999999)
        assert len(workspace.stars) == original_count - 1

    def test_remove_star_nonexistent_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            workspace.remove_star(11111111)

    def test_update_star_changes_attribute(
        self, workspace: ValidationWorkspace,
    ) -> None:
        workspace.update_star(12345678, tmag=11.5)
        assert workspace.target.tmag == 11.5

    def test_update_star_nonexistent_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            workspace.update_star(11111111, tmag=11.5)

    def test_update_star_bad_attribute_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        with pytest.raises(AttributeError, match="no attribute"):
            workspace.update_star(12345678, nonexistent_field=42)

    def test_remove_target_raises(self, workspace: ValidationWorkspace) -> None:
        """Removing the target star must raise, not silently corrupt the field."""
        target_id = workspace.target.tic_id
        with pytest.raises(ValueError, match="Cannot remove the target"):
            workspace.remove_star(target_id)

    def test_add_duplicate_tic_raises(self, workspace: ValidationWorkspace) -> None:
        """Adding a star whose TIC ID already exists must raise."""
        existing_id = workspace.stars[1].tic_id
        with pytest.raises(ValueError, match="already exists"):
            workspace.add_star(_neighbor_star(tic_id=existing_id))

    def test_update_alias_without_stellar_params_raises_typeerror(
        self, workspace: ValidationWorkspace,
    ) -> None:
        """Alias update (Teff) on a star with stellar_params=None must raise TypeError."""
        # The neighbor star has no stellar_params
        neighbor_id = workspace.stars[1].tic_id
        with pytest.raises(TypeError, match="stellar_params is None"):
            workspace.update_star(neighbor_id, Teff=6000.0)

    def test_remove_star_invalidates_result(self, workspace: ValidationWorkspace) -> None:
        workspace._last_result = ValidationResult(
            target_id=0, false_positive_probability=0.5,
            nearby_false_positive_probability=0.0, scenario_results=[],
        )
        workspace.remove_star(99999999)
        assert workspace._last_result is None

    def test_update_star_invalidates_result(self, workspace: ValidationWorkspace) -> None:
        workspace._last_result = ValidationResult(
            target_id=0, false_positive_probability=0.5,
            nearby_false_positive_probability=0.0, scenario_results=[],
        )
        workspace.update_star(12345678, tmag=11.5)
        assert workspace._last_result is None


# ---------------------------------------------------------------------------
# Result access tests
# ---------------------------------------------------------------------------


class TestResultAccess:
    def test_fpp_before_compute_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        with pytest.raises(RuntimeError, match="compute_probs"):
            _ = workspace.fpp

    def test_nfpp_before_compute_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        with pytest.raises(RuntimeError, match="compute_probs"):
            _ = workspace.nfpp

    def test_results_none_before_compute(
        self, workspace: ValidationWorkspace,
    ) -> None:
        assert workspace.results is None


# ---------------------------------------------------------------------------
# compute_probs tests
# ---------------------------------------------------------------------------


class TestComputeProbsScenarioIdsPath:
    """compute_probs(scenario_ids=...) must route through compute_prepared(),
    not bypass the field validation gate via engine.compute() directly."""

    def _make_workspace_with_registry(self, registry: ScenarioRegistry) -> ValidationWorkspace:
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=StubStarCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        ws._engine._registry = registry
        return ws

    def test_scenario_ids_path_returns_validation_result(
        self, transit_lc: LightCurve,
    ) -> None:
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
        registry = ScenarioRegistry()
        registry.register(fake_tp)

        ws = self._make_workspace_with_registry(registry)
        vr = ws.compute_probs(transit_lc, period_days=5.0, scenario_ids=[ScenarioID.TP])
        assert isinstance(vr, ValidationResult)

    def test_scenario_ids_path_validates_field(
        self, transit_lc: LightCurve,
    ) -> None:
        """A corrupted field must raise ValueError even when scenario_ids is given."""
        registry = ScenarioRegistry()
        ws = self._make_workspace_with_registry(registry)

        # Trigger lazy fetch so _stellar_field is populated, then corrupt it
        _ = ws.stars
        wrong_star = _neighbor_star(tic_id=99999999)
        ws._stellar_field.stars[0] = wrong_star  # direct corruption — bypasses guards

        with pytest.raises(ValueError, match=r"stars\[0\]"):
            ws.compute_probs(transit_lc, period_days=5.0, scenario_ids=[ScenarioID.TP])


class TestComputeProbs:
    def test_compute_probs_returns_validation_result(
        self, transit_lc: LightCurve,
    ) -> None:
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
        registry = ScenarioRegistry()
        registry.register(fake_tp)

        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=StubStarCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        ws._engine._registry = registry

        vr = ws.compute_probs(transit_lc, period_days=5.0)
        assert isinstance(vr, ValidationResult)
        assert len(vr.scenario_results) == 1
        assert 0.0 <= ws.fpp <= 1.0

    def test_compute_probs_caches_result(
        self, workspace: ValidationWorkspace, transit_lc: LightCurve,
    ) -> None:
        # Use empty registry so no scenarios run (avoids pytransit import)
        workspace._engine._registry = ScenarioRegistry()
        vr = workspace.compute_probs(transit_lc, period_days=5.0)
        assert workspace.results is vr
        assert workspace.fpp == 1.0


# ---------------------------------------------------------------------------
# calc_depths tests (Gap 5)
# ---------------------------------------------------------------------------


class TestCalcDepths:
    """calc_depths() must update star.flux_ratio and star.transit_depth_required."""

    def test_calc_depths_sets_flux_ratio(self, workspace: ValidationWorkspace) -> None:
        """After calc_depths(), every star must have a non-None flux_ratio."""
        n_stars = len(workspace.stars)
        # One sector: star pixel coords (n_stars, 2), aperture pixels (4, 2)
        pixel_coords = [np.array([[5.0, 5.0], [6.0, 5.0]])]  # target + 1 neighbour
        aperture_pixels = [np.array([[5, 5], [5, 6], [6, 5], [6, 6]], dtype=float)]

        workspace.calc_depths(
            transit_depth=0.01,
            pixel_coords_per_sector=pixel_coords,
            aperture_pixels_per_sector=aperture_pixels,
        )

        for star in workspace.stars:
            assert star.flux_ratio is not None
            assert 0.0 <= star.flux_ratio <= 1.0

    def test_calc_depths_sets_transit_depth(self, workspace: ValidationWorkspace) -> None:
        """After calc_depths(), every star must have a non-None transit_depth_required."""
        pixel_coords = [np.array([[5.0, 5.0], [6.0, 5.0]])]
        aperture_pixels = [np.array([[5, 5], [5, 6], [6, 5], [6, 6]], dtype=float)]

        workspace.calc_depths(
            transit_depth=0.01,
            pixel_coords_per_sector=pixel_coords,
            aperture_pixels_per_sector=aperture_pixels,
        )

        for star in workspace.stars:
            assert star.transit_depth_required is not None

    def test_calc_depths_invalidates_result(self, workspace: ValidationWorkspace) -> None:
        """calc_depths() must invalidate any cached result."""
        workspace._last_result = ValidationResult(
            target_id=0, false_positive_probability=0.5,
            nearby_false_positive_probability=0.0, scenario_results=[],
        )
        pixel_coords = [np.array([[5.0, 5.0], [6.0, 5.0]])]
        aperture_pixels = [np.array([[5, 5], [5, 6], [6, 5], [6, 6]], dtype=float)]

        workspace.calc_depths(
            transit_depth=0.01,
            pixel_coords_per_sector=pixel_coords,
            aperture_pixels_per_sector=aperture_pixels,
        )

        assert workspace._last_result is None


# ---------------------------------------------------------------------------
# plot_fits RuntimeError guard (Gap 5)
# ---------------------------------------------------------------------------


class TestPlotFitsGuard:
    """plot_fits() must raise RuntimeError if compute_probs() was not called."""

    def test_plot_fits_before_compute_raises(
        self, workspace: ValidationWorkspace,
    ) -> None:
        """plot_fits() raises RuntimeError when _last_result is None."""
        with pytest.raises(RuntimeError, match="compute_probs"):
            workspace.plot_fits()
