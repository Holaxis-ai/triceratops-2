"""Tests for PreparedValidationInputs, PreparedValidationMetadata, and compute_prepared."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.scenarios.registry import ScenarioRegistry
from triceratops.validation.engine import ValidationEngine
from triceratops.validation.job import PreparedValidationInputs, PreparedValidationMetadata


# ---------------------------------------------------------------------------
# Helpers / fixtures shared with test_engine.py (minimal copies to avoid coupling)
# ---------------------------------------------------------------------------


def _make_scenario_result(sid: ScenarioID, lnZ: float, n: int = 10) -> ScenarioResult:
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


@pytest.fixture()
def star() -> Star:
    return Star(
        tic_id=99887766,
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


@pytest.fixture()
def stellar_field(star: Star) -> StellarField:
    return StellarField(
        target_id=99887766,
        mission="TESS",
        search_radius_pixels=10,
        stars=[star],
    )


@pytest.fixture()
def lc() -> LightCurve:
    t = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=t, flux=flux, flux_err=0.001)


@pytest.fixture()
def cfg() -> Config:
    return Config(n_mc_samples=100, n_best_samples=10)


# ---------------------------------------------------------------------------
# PreparedValidationInputs construction tests
# ---------------------------------------------------------------------------


class TestPreparedValidationInputsConstruction:
    def test_required_fields_only(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """Can construct with only required fields."""
        pvi = PreparedValidationInputs(
            target_id=99887766,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        assert pvi.target_id == 99887766
        assert pvi.period_days == 5.0

    def test_optional_fields_default_to_none(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """All optional fields default to None."""
        pvi = PreparedValidationInputs(
            target_id=1,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=1.0,
        )
        assert pvi.trilegal_population is None
        assert pvi.external_lcs is None
        assert pvi.contrast_curve is None
        assert pvi.molusc_file is None

    def test_with_contrast_curve(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """Can construct with a ContrastCurve."""
        cc = ContrastCurve(
            separations_arcsec=np.array([0.1, 1.0, 5.0]),
            delta_mags=np.array([0.0, 3.0, 6.0]),
            band="Vis",
        )
        pvi = PreparedValidationInputs(
            target_id=1,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=10.0,
            contrast_curve=cc,
        )
        assert pvi.contrast_curve is cc

    def test_with_molusc_file_path(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """molusc_file accepts a local path string (deferred to Phase 4 for materialisation)."""
        pvi = PreparedValidationInputs(
            target_id=1,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=10.0,
            molusc_file="/tmp/fake_molusc.csv",
        )
        assert pvi.molusc_file == "/tmp/fake_molusc.csv"

    def test_period_days_can_be_list(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """period_days can be a [min, max] range list."""
        pvi = PreparedValidationInputs(
            target_id=1,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=[3.0, 7.0],
        )
        assert pvi.period_days == [3.0, 7.0]


# ---------------------------------------------------------------------------
# PreparedValidationMetadata tests
# ---------------------------------------------------------------------------


class TestPreparedValidationMetadata:
    def test_default_construction(self) -> None:
        """All fields default to None / empty list."""
        meta = PreparedValidationMetadata()
        assert meta.prep_timestamp is None
        assert meta.source is None
        assert meta.trilegal_cache_origin is None
        assert meta.warnings == []

    def test_with_all_fields(self) -> None:
        """Can set all fields explicitly."""
        ts = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        meta = PreparedValidationMetadata(
            prep_timestamp=ts,
            source="MAST/local",
            trilegal_cache_origin="/cache/trilegal.csv",
            warnings=["missing g-band magnitude for TIC 12345"],
        )
        assert meta.prep_timestamp == ts
        assert meta.source == "MAST/local"
        assert meta.trilegal_cache_origin == "/cache/trilegal.csv"
        assert len(meta.warnings) == 1

    def test_warnings_is_independent_per_instance(self) -> None:
        """Each instance gets its own warnings list (field factory)."""
        m1 = PreparedValidationMetadata()
        m2 = PreparedValidationMetadata()
        m1.warnings.append("oops")
        assert m2.warnings == []


# ---------------------------------------------------------------------------
# compute_prepared() tests
# ---------------------------------------------------------------------------


class TestComputePrepared:
    def test_compute_prepared_returns_validation_result(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """compute_prepared() with a fake scenario returns a ValidationResult."""
        result = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=result)
        registry = ScenarioRegistry()
        registry.register(fake)

        engine = ValidationEngine(registry=registry)
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)

    def test_compute_prepared_fpp_in_unit_interval(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """FPP from compute_prepared is always in [0, 1]."""
        result = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=result)
        registry = ScenarioRegistry()
        registry.register(fake)

        engine = ValidationEngine(registry=registry)
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        vr = engine.compute_prepared(pvi)
        assert 0.0 <= vr.fpp <= 1.0

    def test_compute_prepared_matches_compute(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """compute_prepared and compute produce identical results for equivalent inputs."""
        result = _make_scenario_result(ScenarioID.TP, lnZ=-1.5)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=result)
        registry = ScenarioRegistry()
        registry.register(fake)

        engine = ValidationEngine(registry=registry)

        # via compute()
        vr_direct = engine.compute(
            light_curve=lc,
            stellar_field=stellar_field,
            period_days=5.0,
            config=cfg,
        )
        # via compute_prepared()
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        vr_prepared = engine.compute_prepared(pvi)

        assert vr_direct.fpp == pytest.approx(vr_prepared.fpp, abs=1e-12)
        assert vr_direct.nfpp == pytest.approx(vr_prepared.nfpp, abs=1e-12)

    def test_compute_prepared_with_empty_registry_gives_fpp_1(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """An empty registry produces fpp=1.0 (no planet scenario wins)."""
        engine = ValidationEngine(registry=ScenarioRegistry())
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        vr = engine.compute_prepared(pvi)
        assert vr.fpp == 1.0

    def test_compute_prepared_no_provider_called(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """compute_prepared must not call any provider (population or catalog)."""
        class _BoomProvider:
            def query(self, **kwargs: object) -> object:
                raise AssertionError("provider should not be called during compute_prepared")
            def query_nearby_stars(self, **kwargs: object) -> object:
                raise AssertionError("provider should not be called during compute_prepared")

        result = _make_scenario_result(ScenarioID.TP, lnZ=0.0)
        fake = _FakeScenario(_scenario_id=ScenarioID.TP, _result=result)
        registry = ScenarioRegistry()
        registry.register(fake)

        engine = ValidationEngine(
            registry=registry,
            catalog_provider=_BoomProvider(),
            population_provider=_BoomProvider(),
        )
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        # Should not raise
        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)


# ---------------------------------------------------------------------------
# Review-identified fixes: target_id consistency, zip length guard, eager TRILEGAL
# ---------------------------------------------------------------------------


class TestComputePreparedTargetIdGuard:
    """compute_prepared must reject payloads where target_id != stellar_field.target_id."""

    def test_matching_ids_accepted(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        engine = ValidationEngine(registry=ScenarioRegistry())
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,  # matches
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        # Should not raise
        vr = engine.compute_prepared(pvi)
        assert isinstance(vr, ValidationResult)

    def test_mismatched_ids_raises(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        engine = ValidationEngine(registry=ScenarioRegistry())
        pvi = PreparedValidationInputs(
            target_id=99999999,  # does NOT match stellar_field.target_id=99887766
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        with pytest.raises(ValueError, match="target_id.*does not match"):
            engine.compute_prepared(pvi)

    def test_error_message_includes_both_ids(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        engine = ValidationEngine(registry=ScenarioRegistry())
        pvi = PreparedValidationInputs(
            target_id=11111111,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
        )
        with pytest.raises(ValueError) as exc_info:
            engine.compute_prepared(pvi)
        msg = str(exc_info.value)
        assert "11111111" in msg
        assert str(stellar_field.target_id) in msg


class TestComputePreparedScenarioIdsGuard:
    """compute_prepared() must reject unregistered scenario_ids with a clear ValueError.

    PreparedValidationInputs supports direct construction (job.py:43), so
    compute_prepared() cannot rely on ValidationPreparer.prepare() having
    validated the IDs.  compute() resolves IDs via self._registry.get() which
    raises KeyError; compute_prepared() must raise ValueError with the IDs named.
    """

    def test_unregistered_id_raises_valueerror(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.validation.engine import ValidationEngine
        from triceratops.validation.job import PreparedValidationInputs

        engine = ValidationEngine()
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=[ScenarioID.EBX2P],
        )
        with pytest.raises(ValueError, match="not registered"):
            engine.compute_prepared(pvi)

    def test_error_message_names_unknown_ids(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.validation.engine import ValidationEngine
        from triceratops.validation.job import PreparedValidationInputs

        engine = ValidationEngine()
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=[ScenarioID.EBX2P, ScenarioID.DEBX2P],
        )
        with pytest.raises(ValueError) as exc_info:
            engine.compute_prepared(pvi)
        msg = str(exc_info.value)
        assert "EBX2P" in msg or "EBx2P" in msg

    def test_none_scenario_ids_passes_guard(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """scenario_ids=None (run all) must not be rejected by the guard."""
        from triceratops.validation.engine import ValidationEngine
        from triceratops.validation.job import PreparedValidationInputs

        engine = ValidationEngine()
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=None,
        )
        # Should not raise ValueError from the scenario_ids guard
        # (it may raise for other reasons like missing stellar_params, which is fine)
        try:
            engine.compute_prepared(pvi)
        except ValueError as e:
            assert "not registered" not in str(e), f"Guard fired unexpectedly: {e}"


class TestPreparerExternalLcLengthGuard:
    """ValidationPreparer.prepare() must reject mismatched external_lc_files / filt_lcs."""

    def _make_preparer(self) -> object:
        from unittest.mock import MagicMock
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.preparer import ValidationPreparer

        star = Star(
            tic_id=1234,
            ra_deg=10.0, dec_deg=5.0,
            tmag=12.0, jmag=11.5, hmag=11.3, kmag=11.2,
            bmag=12.5, vmag=12.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=1234, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf
        return ValidationPreparer(catalog_provider=mock_catalog)

    def test_matching_lengths_accepted(self, lc: LightCurve, cfg: Config, tmp_path) -> None:
        """Same-length lists should not raise (even if files don't exist for this unit test)."""
        preparer = self._make_preparer()
        # We expect an error about missing files, NOT about mismatched lengths
        with pytest.raises(Exception) as exc_info:
            preparer.prepare(
                target_id=1234,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                external_lc_files=[str(tmp_path / "a.txt"), str(tmp_path / "b.txt")],
                filt_lcs=["J", "i"],
            )
        assert "mismatched" not in str(exc_info.value).lower()

    def test_mismatched_lengths_raises_valueerror(self, lc: LightCurve, cfg: Config) -> None:
        preparer = self._make_preparer()
        with pytest.raises(ValueError, match="same length"):
            preparer.prepare(
                target_id=1234,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                external_lc_files=["a.txt", "b.txt", "c.txt"],  # 3 files
                filt_lcs=["J", "i"],                              # 2 filters
            )

    def test_error_message_includes_counts(self, lc: LightCurve, cfg: Config) -> None:
        preparer = self._make_preparer()
        with pytest.raises(ValueError) as exc_info:
            preparer.prepare(
                target_id=1234,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                external_lc_files=["a.txt"],
                filt_lcs=["J", "i", "r"],
            )
        msg = str(exc_info.value)
        assert "1" in msg and "3" in msg


class TestPreparerExternalLcAsymmetricGuard:
    """Preparer must raise when one of external_lc_files / filt_lcs is empty/None
    while the other is non-empty — the original `if x and y:` guard skipped these."""

    def _make_preparer(self) -> object:
        from unittest.mock import MagicMock
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.preparer import ValidationPreparer

        star = Star(
            tic_id=1234,
            ra_deg=10.0, dec_deg=5.0,
            tmag=12.0, jmag=11.5, hmag=11.3, kmag=11.2,
            bmag=12.5, vmag=12.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=1234, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf
        return ValidationPreparer(catalog_provider=mock_catalog)

    def _call_prepare(self, preparer, lc, cfg, **kwargs):
        import numpy as np
        return preparer.prepare(
            target_id=1234,
            sectors=np.array([1]),
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            **kwargs,
        )

    def test_files_provided_filters_none_raises(self, lc: LightCurve, cfg: Config) -> None:
        """Files non-empty, filt_lcs=None → should raise, not silently drop files."""
        preparer = self._make_preparer()
        with pytest.raises(ValueError, match="both be provided together"):
            self._call_prepare(preparer, lc, cfg,
                               external_lc_files=["a.txt"], filt_lcs=None)

    def test_files_provided_filters_empty_raises(self, lc: LightCurve, cfg: Config) -> None:
        """Files non-empty, filt_lcs=[] → should raise, not silently drop files."""
        preparer = self._make_preparer()
        with pytest.raises(ValueError, match="both be provided together"):
            self._call_prepare(preparer, lc, cfg,
                               external_lc_files=["a.txt"], filt_lcs=[])

    def test_filters_provided_files_none_raises(self, lc: LightCurve, cfg: Config) -> None:
        """filt_lcs non-empty, files=None → should raise, not silently drop filters."""
        preparer = self._make_preparer()
        with pytest.raises(ValueError, match="both be provided together"):
            self._call_prepare(preparer, lc, cfg,
                               external_lc_files=None, filt_lcs=["J"])


class TestPreparerTrilegalScenarioGate:
    """TRILEGAL fetch should be skipped when scenario_ids excludes all TRILEGAL scenarios."""

    def _make_preparer_with_population(self):
        from unittest.mock import MagicMock
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.preparer import ValidationPreparer

        star = Star(
            tic_id=9999,
            ra_deg=20.0, dec_deg=-10.0,
            tmag=11.0, jmag=10.5, hmag=10.3, kmag=10.2,
            bmag=11.5, vmag=11.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5800.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=8.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=9999, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf
        mock_pop = MagicMock()
        mock_pop.query.return_value = MagicMock()
        preparer = ValidationPreparer(
            catalog_provider=mock_catalog,
            population_provider=mock_pop,
        )
        return preparer, mock_pop

    def test_trilegal_not_fetched_when_all_scenarios_non_trilegal(
        self, lc: LightCurve, cfg: Config
    ) -> None:
        """When scenario_ids contains only non-TRILEGAL scenarios, provider.query() must not be called."""
        import numpy as np
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.scenarios.registry import DEFAULT_REGISTRY

        preparer, mock_pop = self._make_preparer_with_population()
        trilegal_ids = ScenarioID.trilegal_scenarios()
        non_trilegal_ids = [
            sid for sid in DEFAULT_REGISTRY
            if sid not in trilegal_ids
        ]
        preparer.prepare(
            target_id=9999,
            sectors=np.array([1]),
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=non_trilegal_ids,
        )
        mock_pop.query.assert_not_called()

    def test_trilegal_fetched_when_trilegal_scenario_included(
        self, lc: LightCurve, cfg: Config
    ) -> None:
        """When scenario_ids includes a registered TRILEGAL scenario, provider.query() must be called."""
        import numpy as np
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.scenarios.registry import DEFAULT_REGISTRY

        preparer, mock_pop = self._make_preparer_with_population()
        # Use only TRILEGAL IDs that are actually registered (BTP, BEB)
        registered_trilegal_ids = [
            sid for sid in ScenarioID.trilegal_scenarios()
            if sid in DEFAULT_REGISTRY
        ]
        assert registered_trilegal_ids, "Need at least one registered TRILEGAL scenario for this test"
        preparer.prepare(
            target_id=9999,
            sectors=np.array([1]),
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=registered_trilegal_ids,
        )
        mock_pop.query.assert_called_once()

    def test_trilegal_always_fetched_when_scenario_ids_none(
        self, lc: LightCurve, cfg: Config
    ) -> None:
        """When scenario_ids=None (default), TRILEGAL is fetched (default registry has BTP/BEB)."""
        import numpy as np

        preparer, mock_pop = self._make_preparer_with_population()
        preparer.prepare(
            target_id=9999,
            sectors=np.array([1]),
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=None,
        )
        mock_pop.query.assert_called_once()


class TestPreparerScenarioIdsValidation:
    """prepare() must reject unregistered ScenarioIDs early with a clear error.

    Previously, unregistered IDs (e.g. EBX2P, which exists in the ScenarioID
    enum but is not in DEFAULT_REGISTRY) were silently skipped during TRILEGAL
    gating via get_or_none(), but would crash with KeyError inside compute().
    The fix: validate all IDs before any IO and raise ValueError with the
    unknown IDs named.
    """

    def _make_preparer(self) -> object:
        from unittest.mock import MagicMock
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.preparer import ValidationPreparer

        star = Star(
            tic_id=5555,
            ra_deg=0.0, dec_deg=0.0,
            tmag=10.0, jmag=9.5, hmag=9.3, kmag=9.2,
            bmag=10.5, vmag=10.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=5555, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf
        return ValidationPreparer(catalog_provider=mock_catalog)

    def test_unregistered_id_raises_before_any_io(self, lc: LightCurve, cfg: Config) -> None:
        """EBX2P exists in ScenarioID enum but is not in DEFAULT_REGISTRY — must raise ValueError."""
        import numpy as np
        from triceratops.domain.scenario_id import ScenarioID

        preparer = self._make_preparer()
        with pytest.raises(ValueError, match="not registered in the registry"):
            preparer.prepare(
                target_id=5555,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                scenario_ids=[ScenarioID.EBX2P],
            )

    def test_error_message_names_unknown_ids(self, lc: LightCurve, cfg: Config) -> None:
        import numpy as np
        from triceratops.domain.scenario_id import ScenarioID

        preparer = self._make_preparer()
        with pytest.raises(ValueError) as exc_info:
            preparer.prepare(
                target_id=5555,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                scenario_ids=[ScenarioID.EBX2P, ScenarioID.DEBX2P],
            )
        msg = str(exc_info.value)
        assert "EBX2P" in msg or "EBx2P" in msg

    def test_registered_ids_are_accepted(self, lc: LightCurve, cfg: Config) -> None:
        """All IDs from DEFAULT_REGISTRY must pass validation without raising."""
        import numpy as np
        from triceratops.scenarios.registry import DEFAULT_REGISTRY

        preparer = self._make_preparer()
        registered_ids = list(DEFAULT_REGISTRY)
        # No ValueError — any exception here is from missing files/providers, not validation
        try:
            preparer.prepare(
                target_id=5555,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                scenario_ids=registered_ids,
            )
        except ValueError as e:
            assert "not registered" not in str(e), f"Unexpected validation error: {e}"

    def test_custom_registry_accepts_its_own_ids(self, lc: LightCurve, cfg: Config) -> None:
        """A preparer built with a custom registry should accept IDs in that registry,
        even if those IDs are not in DEFAULT_REGISTRY."""
        import numpy as np
        from unittest.mock import MagicMock
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry
        from triceratops.validation.preparer import ValidationPreparer

        # Build a registry with only TP — the rest are absent.
        custom_reg = ScenarioRegistry()
        custom_reg.register(DEFAULT_REGISTRY.get(ScenarioID.TP))

        star = Star(
            tic_id=6666,
            ra_deg=0.0, dec_deg=0.0,
            tmag=10.0, jmag=9.5, hmag=9.3, kmag=9.2,
            bmag=10.5, vmag=10.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=6666, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf

        preparer = ValidationPreparer(catalog_provider=mock_catalog, registry=custom_reg)

        # TP is in the custom registry — must not raise
        try:
            preparer.prepare(
                target_id=6666,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                scenario_ids=[ScenarioID.TP],
            )
        except ValueError as e:
            assert "not registered" not in str(e), f"Unexpected validation error: {e}"

        # EB is NOT in the custom registry — must raise
        with pytest.raises(ValueError, match="not registered in the registry"):
            preparer.prepare(
                target_id=6666,
                sectors=np.array([1]),
                light_curve=lc,
                config=cfg,
                period_days=5.0,
                scenario_ids=[ScenarioID.EB],
            )


class TestPrepareComputeScenarioContract:
    """scenario_ids must flow from prepare() through the payload into compute_prepared().

    Regression guard: the original fix gated TRILEGAL fetch in the preparer but
    hardcoded scenario_ids=None in compute_prepared(), breaking the contract —
    a caller could prepare for [TP] (no TRILEGAL), hand the payload to
    compute_prepared(), and get BTP/BEB run against a None trilegal_population.
    """

    def test_scenario_ids_stored_on_payload(self, lc: LightCurve, cfg: Config) -> None:
        """prepare(scenario_ids=X) must store X on the returned PreparedValidationInputs."""
        import numpy as np
        from unittest.mock import MagicMock
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.preparer import ValidationPreparer

        star = Star(
            tic_id=7777,
            ra_deg=0.0, dec_deg=0.0,
            tmag=10.0, jmag=9.5, hmag=9.3, kmag=9.2,
            bmag=10.5, vmag=10.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=7777, mission="TESS", search_radius_pixels=10, stars=[star])
        mock_catalog = MagicMock()
        mock_catalog.query_nearby_stars.return_value = sf
        preparer = ValidationPreparer(catalog_provider=mock_catalog)

        ids = [ScenarioID.TP, ScenarioID.EB]
        payload = preparer.prepare(
            target_id=7777,
            sectors=np.array([1]),
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=ids,
        )
        assert payload.scenario_ids == ids

    def test_compute_prepared_passes_scenario_ids_to_engine(
        self, lc: LightCurve, cfg: Config
    ) -> None:
        """compute_prepared() must call compute(scenario_ids=prepared.scenario_ids), not None."""
        from unittest.mock import MagicMock, patch
        from triceratops.domain.scenario_id import ScenarioID
        from triceratops.domain.value_objects import StellarParameters
        from triceratops.validation.engine import ValidationEngine
        from triceratops.validation.job import PreparedValidationInputs

        star = Star(
            tic_id=8888,
            ra_deg=0.0, dec_deg=0.0,
            tmag=10.0, jmag=9.5, hmag=9.3, kmag=9.2,
            bmag=10.5, vmag=10.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(target_id=8888, mission="TESS", search_radius_pixels=10, stars=[star])

        ids = [ScenarioID.TP]
        payload = PreparedValidationInputs(
            target_id=8888,
            stellar_field=sf,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=ids,
        )
        engine = ValidationEngine()
        with patch.object(engine, "compute", wraps=engine.compute) as mock_compute:
            engine.compute_prepared(payload)
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs["scenario_ids"] == ids
