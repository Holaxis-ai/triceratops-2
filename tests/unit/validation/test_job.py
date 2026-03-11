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
        assert pvi.molusc_data is None

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

    def test_with_molusc_data(
        self, stellar_field: StellarField, lc: LightCurve, cfg: Config
    ) -> None:
        """molusc_data accepts a MoluscData object."""
        from triceratops.domain.molusc import MoluscData
        md = MoluscData(
            semi_major_axis_au=np.array([20.0, 30.0]),
            eccentricity=np.array([0.1, 0.2]),
            mass_ratio=np.array([0.5, 0.6]),
        )
        pvi = PreparedValidationInputs(
            target_id=1,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=10.0,
            molusc_data=md,
        )
        assert pvi.molusc_data is md

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
        """scenario_ids=None (run all) must not be rejected by the scenario_ids guard.

        Uses a custom registry with no TRILEGAL scenarios so the TRILEGAL check
        does not fire — isolating the 'None is not rejected' guard behavior.
        """
        from triceratops.scenarios.registry import ScenarioRegistry
        from triceratops.validation.engine import ValidationEngine
        from triceratops.validation.errors import ValidationError
        from triceratops.validation.job import PreparedValidationInputs

        engine = ValidationEngine(registry=ScenarioRegistry())  # empty — no TRILEGAL
        pvi = PreparedValidationInputs(
            target_id=stellar_field.target_id,
            stellar_field=stellar_field,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            scenario_ids=None,
        )
        # Should not raise ValueError from the scenario_ids guard.
        # Other ValidationErrors (e.g. stellar_params) are fine — only the guard is tested.
        try:
            engine.compute_prepared(pvi)
        except ValueError as e:
            assert "not registered" not in str(e), f"Guard fired unexpectedly: {e}"


class TestPrepareComputeScenarioContract:
    """scenario_ids must flow from prepare() through the payload into compute_prepared().

    Regression guard: the original fix gated TRILEGAL fetch in the preparer but
    hardcoded scenario_ids=None in compute_prepared(), breaking the contract —
    a caller could prepare for [TP] (no TRILEGAL), hand the payload to
    compute_prepared(), and get BTP/BEB run against a None trilegal_population.
    """

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
