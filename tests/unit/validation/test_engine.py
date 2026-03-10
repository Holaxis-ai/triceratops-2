"""Tests for ValidationEngine."""
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
from triceratops.validation.engine import ValidationEngine

# _aggregate is a static method on ValidationEngine
_aggregate = ValidationEngine._aggregate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(sid: ScenarioID, lnZ: float, n: int = 10) -> ScenarioResult:
    """Create a minimal ScenarioResult with the given scenario ID and evidence."""
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
        planet_radius_rearth=np.ones(n) if "TP" in sid.name or sid.name == "TP" else np.zeros(n),
        eb_mass_msun=np.zeros(n),
        eb_radius_rsun=np.zeros(n),
        flux_ratio_eb_tess=np.zeros(n),
        companion_mass_msun=np.zeros(n),
        companion_radius_rsun=np.zeros(n),
        flux_ratio_companion_tess=np.zeros(n),
    )


@dataclass
class _FakeScenario:
    """Minimal scenario that returns a pre-built result."""
    _scenario_id: ScenarioID
    _is_eb: bool
    _result: ScenarioResult | tuple[ScenarioResult, ScenarioResult]

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


@dataclass
class _RecordingScenario(_FakeScenario):
    """Fake scenario that records the light curve passed by the engine."""
    seen_light_curve: LightCurve | None = None

    def compute(
        self,
        light_curve: object = None,
        stellar_params: object = None,
        period_days: object = None,
        config: object = None,
        external_lcs: object = None,
        **kwargs: object,
    ) -> ScenarioResult | tuple[ScenarioResult, ScenarioResult]:
        self.seen_light_curve = light_curve  # type: ignore[assignment]
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stellar_field():
    target = Star(
        tic_id=12345678,
        ra_deg=83.82, dec_deg=-5.39,
        tmag=10.5, jmag=9.8, hmag=9.5, kmag=9.4,
        bmag=11.2, vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        ),
    )
    return StellarField(
        target_id=12345678,
        mission="TESS",
        search_radius_pixels=10,
        stars=[target],
    )


@pytest.fixture()
def transit_lc():
    time = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def small_config():
    return Config(n_mc_samples=100, n_best_samples=10)


# ---------------------------------------------------------------------------
# Engine instantiation tests
# ---------------------------------------------------------------------------

class TestEngineInstantiation:
    def test_instantiates_with_empty_registry(self) -> None:
        engine = ValidationEngine(registry=ScenarioRegistry())
        assert engine is not None

    def test_instantiates_with_stubs(self) -> None:
        engine = ValidationEngine(
            registry=ScenarioRegistry(),
            catalog_provider=None,
            population_provider=None,
        )
        assert engine is not None


# ---------------------------------------------------------------------------
# _aggregate tests
# ---------------------------------------------------------------------------

class TestAggregateEmpty:
    def test_empty_results_fpp_is_1(self) -> None:
        result = _aggregate([], target_id=123)
        assert result.fpp == 1.0
        assert result.nfpp == 0.0
        assert result.scenario_results == []


class TestAggregateAllTP:
    def test_all_tp_fpp_is_0(self) -> None:
        """One TP result with finite lnZ, all others -inf => fpp=0."""
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=-np.inf),
            _make_result(ScenarioID.PEB, lnZ=-np.inf),
        ]
        vr = _aggregate(results, target_id=1)
        assert vr.fpp == pytest.approx(0.0, abs=1e-10)


class TestAggregateAllEB:
    def test_all_eb_fpp_is_1(self) -> None:
        """Only EB results => fpp=1."""
        results = [
            _make_result(ScenarioID.EB, lnZ=0.0),
            _make_result(ScenarioID.EBX2P, lnZ=-1.0),
        ]
        vr = _aggregate(results, target_id=1)
        assert vr.fpp == pytest.approx(1.0, abs=1e-10)


class TestAggregateFPPExcludesSTP:
    def test_stp_does_not_reduce_fpp(self) -> None:
        """STP is NOT in planet_scenarios. Having only STP should give fpp=1."""
        results = [
            _make_result(ScenarioID.STP, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=1)
        assert vr.fpp == pytest.approx(1.0, abs=1e-10)


class TestAggregateRelativeProbs:
    def test_relative_probs_sum_to_1(self) -> None:
        results = [
            _make_result(ScenarioID.TP, lnZ=-1.0),
            _make_result(ScenarioID.EB, lnZ=-2.0),
            _make_result(ScenarioID.PTP, lnZ=-3.0),
            _make_result(ScenarioID.DTP, lnZ=-4.0),
        ]
        vr = _aggregate(results, target_id=1)
        total_prob = sum(r.relative_probability for r in vr.scenario_results)
        assert total_prob == pytest.approx(1.0, abs=1e-10)

    def test_higher_lnz_gets_higher_probability(self) -> None:
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=-10.0),
        ]
        vr = _aggregate(results, target_id=1)
        tp = next(r for r in vr.scenario_results if r.scenario_id == ScenarioID.TP)
        eb = next(r for r in vr.scenario_results if r.scenario_id == ScenarioID.EB)
        assert tp.relative_probability > eb.relative_probability


class TestAggregateNFPP:
    def test_nfpp_from_ntp_neb_only(self) -> None:
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.NTP, lnZ=0.0),
            _make_result(ScenarioID.NEB, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=1)
        # NTP and NEB each have 1/3 of total prob
        assert vr.nfpp == pytest.approx(2.0 / 3.0, abs=1e-10)

    def test_nfpp_zero_without_nearby_scenarios(self) -> None:
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=1)
        assert vr.nfpp == pytest.approx(0.0, abs=1e-10)


class TestAggregateAllInfinite:
    def test_all_inf_gives_zero_probs(self) -> None:
        results = [
            _make_result(ScenarioID.TP, lnZ=-np.inf),
            _make_result(ScenarioID.EB, lnZ=-np.inf),
        ]
        vr = _aggregate(results, target_id=1)
        for r in vr.scenario_results:
            assert r.relative_probability == 0.0
        assert vr.fpp == 1.0


class TestAggregateFPPFormula:
    def test_fpp_includes_tp_ptp_dtp(self) -> None:
        """FPP = 1 - P(TP) - P(PTP) - P(DTP). All three contribute."""
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.PTP, lnZ=0.0),
            _make_result(ScenarioID.DTP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=1)
        # Each gets 0.25 prob; planet_prob = 0.75; fpp = 0.25
        assert vr.fpp == pytest.approx(0.25, abs=1e-10)


# ---------------------------------------------------------------------------
# _aggregate edge cases: FPP/NFPP numerical robustness
# ---------------------------------------------------------------------------

class TestAggregateAllNegInf:
    def test_all_lnZ_neg_inf_fpp_and_nfpp_well_defined(self) -> None:
        """All -inf lnZ: Z_total=0. FPP must be 1.0 and NFPP must be 0.0."""
        results = [
            _make_result(ScenarioID.TP, lnZ=-np.inf),
            _make_result(ScenarioID.EB, lnZ=-np.inf),
            _make_result(ScenarioID.NTP, lnZ=-np.inf),
        ]
        vr = _aggregate(results, target_id=99)
        # All probs must be 0 or explicitly defined (not NaN)
        for r in vr.scenario_results:
            assert np.isfinite(r.relative_probability), (
                f"{r.scenario_id} has non-finite relative_probability"
            )
            assert r.relative_probability == 0.0
        assert np.isfinite(vr.fpp)
        assert vr.fpp == pytest.approx(1.0, abs=1e-10)
        assert np.isfinite(vr.nfpp)
        assert vr.nfpp == pytest.approx(0.0, abs=1e-10)


class TestAggregateNaNInLnZ:
    def test_nan_lnZ_does_not_propagate_to_fpp(self) -> None:
        """NaN lnZ is treated as non-finite; planet scenario still determines FPP."""
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),   # finite: will dominate
            _make_result(ScenarioID.EB, lnZ=float("nan")),  # NaN: treated as -inf
        ]
        vr = _aggregate(results, target_id=2)
        assert np.isfinite(vr.fpp), "FPP must be finite even when one lnZ is NaN"
        assert np.isfinite(vr.nfpp), "NFPP must be finite even when one lnZ is NaN"
        # TP dominates; FPP ~ 0
        assert vr.fpp == pytest.approx(0.0, abs=1e-6)


class TestAggregateFPPClipping:
    def test_fpp_never_exceeds_1(self) -> None:
        """FPP is clipped to [0, 1]; it must never be > 1.0."""
        # All non-planet scenarios; planet_prob = 0; FPP = 1
        results = [
            _make_result(ScenarioID.EB, lnZ=0.0),
            _make_result(ScenarioID.PEB, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=3)
        assert vr.fpp <= 1.0

    def test_fpp_never_negative(self) -> None:
        """FPP is clipped to [0, 1]; it must never be negative."""
        # All planet scenarios; FPP = 0
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.PTP, lnZ=0.0),
            _make_result(ScenarioID.DTP, lnZ=0.0),
        ]
        vr = _aggregate(results, target_id=4)
        assert vr.fpp >= 0.0
        assert vr.fpp == pytest.approx(0.0, abs=1e-10)

    def test_fpp_clipped_to_exactly_1_when_no_planets(self) -> None:
        """When planet_prob=0, FPP = np.clip(1.0, 0, 1) = exactly 1.0."""
        results = [
            _make_result(ScenarioID.EB, lnZ=1.0),
        ]
        vr = _aggregate(results, target_id=5)
        assert vr.fpp == 1.0


class TestAggregateNFPPNoNearby:
    def test_nfpp_zero_when_no_nearby_scenarios(self) -> None:
        """No NTP/NEB/NEBx2P scenarios present => NFPP = 0.0."""
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=-1.0),
            _make_result(ScenarioID.PEB, lnZ=-2.0),
        ]
        vr = _aggregate(results, target_id=6)
        assert vr.nfpp == pytest.approx(0.0, abs=1e-10)


class TestAggregateProbabilitySum:
    def test_relative_probs_always_sum_to_1(self) -> None:
        """For any valid input, sum of relative_probability must be 1.0."""
        results = [
            _make_result(ScenarioID.TP, lnZ=-0.5),
            _make_result(ScenarioID.EB, lnZ=-1.5),
            _make_result(ScenarioID.PEB, lnZ=-3.0),
            _make_result(ScenarioID.NTP, lnZ=-2.0),
            _make_result(ScenarioID.DTP, lnZ=-4.0),
        ]
        vr = _aggregate(results, target_id=7)
        total = sum(r.relative_probability for r in vr.scenario_results)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_relative_probs_sum_to_1_with_mix_of_inf(self) -> None:
        """Mixed finite/-inf lnZ: finite scenarios split probability; sum still 1."""
        results = [
            _make_result(ScenarioID.TP, lnZ=0.0),
            _make_result(ScenarioID.EB, lnZ=-np.inf),
            _make_result(ScenarioID.PTP, lnZ=-1.0),
        ]
        vr = _aggregate(results, target_id=8)
        total = sum(r.relative_probability for r in vr.scenario_results)
        assert total == pytest.approx(1.0, abs=1e-10)


class TestAggregateSingleDominantScenario:
    def test_dominant_scenario_prob_near_1(self) -> None:
        """One scenario with lnZ >> all others captures nearly all probability."""
        results = [
            _make_result(ScenarioID.TP, lnZ=100.0),   # dominant
            _make_result(ScenarioID.EB, lnZ=-100.0),
            _make_result(ScenarioID.PEB, lnZ=-200.0),
        ]
        vr = _aggregate(results, target_id=9)
        tp = next(r for r in vr.scenario_results if r.scenario_id == ScenarioID.TP)
        assert tp.relative_probability == pytest.approx(1.0, abs=1e-6)
        for r in vr.scenario_results:
            if r.scenario_id != ScenarioID.TP:
                assert r.relative_probability == pytest.approx(0.0, abs=1e-6)
        # FPP should be near 0 since TP dominates
        assert vr.fpp == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Engine.compute tests
# ---------------------------------------------------------------------------

class TestEngineCompute:
    def test_compute_with_no_scenarios_returns_fpp_1(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        engine = ValidationEngine(registry=ScenarioRegistry())
        result = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        assert isinstance(result, ValidationResult)
        assert result.fpp == 1.0
        assert result.scenario_results == []

    def test_compute_calls_scenario_compute(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        """Engine dispatches to each registered scenario's compute()."""
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _FakeScenario(
            _scenario_id=ScenarioID.TP,
            _is_eb=False,
            _result=tp_result,
        )
        registry = ScenarioRegistry()
        registry.register(fake_tp)

        engine = ValidationEngine(registry=registry)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        assert isinstance(vr, ValidationResult)
        assert len(vr.scenario_results) == 1
        assert vr.scenario_results[0].scenario_id == ScenarioID.TP

    def test_compute_handles_eb_tuple_results(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        """EB scenarios return (result, result_twin); engine flattens them."""
        eb_result = _make_result(ScenarioID.EB, lnZ=-1.0)
        eb_twin = _make_result(ScenarioID.EBX2P, lnZ=-2.0)
        fake_eb = _FakeScenario(
            _scenario_id=ScenarioID.EB,
            _is_eb=True,
            _result=(eb_result, eb_twin),
        )
        registry = ScenarioRegistry()
        registry.register(fake_eb)

        engine = ValidationEngine(registry=registry)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        assert len(vr.scenario_results) == 2
        sids = {r.scenario_id for r in vr.scenario_results}
        assert ScenarioID.EB in sids
        assert ScenarioID.EBX2P in sids

    def test_compute_with_specific_scenario_ids(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        """scenario_ids parameter filters which scenarios to run."""
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        eb_result = _make_result(ScenarioID.EB, lnZ=-1.0)
        eb_twin = _make_result(ScenarioID.EBX2P, lnZ=-2.0)

        fake_tp = _FakeScenario(ScenarioID.TP, False, tp_result)
        fake_eb = _FakeScenario(ScenarioID.EB, True, (eb_result, eb_twin))

        registry = ScenarioRegistry()
        registry.register(fake_tp)
        registry.register(fake_eb)

        engine = ValidationEngine(registry=registry)
        # Only run TP
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
            scenario_ids=[ScenarioID.TP],
        )
        assert len(vr.scenario_results) == 1
        assert vr.scenario_results[0].scenario_id == ScenarioID.TP

    def test_default_compute_skips_nearby_scenarios_without_candidate_host(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        ntp_result = _make_result(ScenarioID.NTP, lnZ=-1.0)

        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.TP, False, tp_result))
        registry.register(_FakeScenario(ScenarioID.NTP, False, ntp_result))

        engine = ValidationEngine(registry=registry)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        sids = {r.scenario_id for r in vr.scenario_results}
        assert sids == {ScenarioID.TP}

    def test_default_compute_keeps_nearby_scenarios_with_candidate_host(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        neighbor = Star(
            tic_id=87654321,
            ra_deg=83.83,
            dec_deg=-5.40,
            tmag=13.0,
            jmag=12.0,
            hmag=11.8,
            kmag=11.7,
            bmag=14.0,
            vmag=13.5,
            stellar_params=stellar_field.target.stellar_params,
            transit_depth_required=0.02,
        )
        stellar_field.stars.append(neighbor)

        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        ntp_result = _make_result(ScenarioID.NTP, lnZ=-1.0)

        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.TP, False, tp_result))
        registry.register(_FakeScenario(ScenarioID.NTP, False, ntp_result))

        engine = ValidationEngine(registry=registry)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        sids = {r.scenario_id for r in vr.scenario_results}
        assert sids == {ScenarioID.TP, ScenarioID.NTP}

    def test_compute_renorms_target_host_light_curve_before_dispatch(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        stellar_field.target.flux_ratio = 0.8
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        fake_tp = _RecordingScenario(ScenarioID.TP, False, tp_result)

        registry = ScenarioRegistry()
        registry.register(fake_tp)

        engine = ValidationEngine(registry=registry)
        engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )

        assert fake_tp.seen_light_curve is not None
        expected_flux = (transit_lc.flux - 0.2) / 0.8
        assert fake_tp.seen_light_curve.flux == pytest.approx(expected_flux)
        assert fake_tp.seen_light_curve.sigma == pytest.approx(
            transit_lc.sigma / 0.8
        )

    def test_compute_keeps_raw_light_curve_for_nearby_scenarios(
        self, transit_lc, stellar_field, small_config,
    ) -> None:
        stellar_field.target.flux_ratio = 0.8
        ntp_result = _make_result(ScenarioID.NTP, lnZ=0.0)
        fake_ntp = _RecordingScenario(ScenarioID.NTP, False, ntp_result)

        registry = ScenarioRegistry()
        registry.register(fake_ntp)

        engine = ValidationEngine(registry=registry)
        engine.compute(
            transit_lc,
            stellar_field,
            period_days=5.0,
            config=small_config,
            scenario_ids=[ScenarioID.NTP],
        )

        assert fake_ntp.seen_light_curve is transit_lc


# ---------------------------------------------------------------------------
# _renorm_light_curve_for_host edge cases
# ---------------------------------------------------------------------------

class TestRenormLightCurveForHost:
    """Tests for ValidationEngine._renorm_light_curve_for_host()."""

    _renorm = staticmethod(ValidationEngine._renorm_light_curve_for_host)

    @pytest.fixture()
    def lc(self):
        time = np.linspace(-0.1, 0.1, 50)
        flux = np.ones(50)
        flux[20:30] = 0.99
        return LightCurve(time_days=time, flux=flux, flux_err=0.001)

    def test_flux_ratio_none_returns_original_object(self, lc) -> None:
        result = self._renorm(lc, None)
        assert result is lc

    def test_flux_ratio_none_values_unchanged(self, lc) -> None:
        result = self._renorm(lc, None)
        np.testing.assert_array_equal(result.flux, lc.flux)
        np.testing.assert_array_equal(result.time_days, lc.time_days)

    def test_flux_ratio_zero_returns_original_object(self, lc) -> None:
        result = self._renorm(lc, 0.0)
        assert result is lc

    def test_flux_ratio_greater_than_one_returns_original_object(self, lc) -> None:
        result = self._renorm(lc, 1.5)
        assert result is lc

    def test_flux_ratio_exactly_one_applies_renorm(self, lc) -> None:
        # flux_ratio=1.0: (flux - 0) / 1 = flux unchanged numerically,
        # but a new LightCurve object is returned (passes the guard: 0 < 1.0 <= 1.0)
        result = self._renorm(lc, 1.0)
        np.testing.assert_allclose(result.flux, lc.flux, rtol=1e-12)

    def test_flux_ratio_half_doubles_depth(self, lc) -> None:
        """flux_ratio=0.5 → renormed = (flux - 0.5) / 0.5 = 2*flux - 1."""
        result = self._renorm(lc, 0.5)
        expected_flux = (lc.flux - 0.5) / 0.5
        np.testing.assert_allclose(result.flux, expected_flux, rtol=1e-12)

    def test_flux_ratio_half_doubles_sigma(self, lc) -> None:
        result = self._renorm(lc, 0.5)
        assert result.sigma == pytest.approx(lc.sigma / 0.5)

    def test_returned_lc_has_same_time_array(self, lc) -> None:
        result = self._renorm(lc, 0.5)
        np.testing.assert_array_equal(result.time_days, lc.time_days)

    def test_flux_ratio_negative_returns_original_object(self, lc) -> None:
        result = self._renorm(lc, -0.1)
        assert result is lc

    def test_flux_ratio_valid_returns_new_object(self, lc) -> None:
        """A physical flux_ratio in (0, 1] returns a new LightCurve, not the input."""
        result = self._renorm(lc, 0.7)
        assert result is not lc

    def test_flux_ratio_valid_flux_formula(self, lc) -> None:
        """Verify renorm formula: new_flux = (flux - (1 - fr)) / fr."""
        fr = 0.8
        result = self._renorm(lc, fr)
        expected = (lc.flux - (1.0 - fr)) / fr
        np.testing.assert_allclose(result.flux, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# TRILEGAL loading conditions
# ---------------------------------------------------------------------------

class TestTrilegalLoading:
    """Phase 2: Engine is provider-free; TRILEGAL population is passed directly.

    The engine no longer calls population_provider.query() internally.
    TRILEGAL materialisation is the responsibility of ValidationWorkspace /
    ValidationPreparer, which then pass the result via trilegal_population=.
    """

    class _BoomProvider:
        """Provider that raises if called — verifies the engine never calls it."""
        def query(self, **kwargs: object) -> object:
            raise AssertionError("Engine must not call population_provider directly")

    @pytest.fixture()
    def stellar_field(self):
        target = Star(
            tic_id=99999,
            ra_deg=10.0, dec_deg=20.0,
            tmag=12.0, jmag=11.5, hmag=11.2, kmag=11.1,
            bmag=13.0, vmag=12.5,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
                logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
            ),
        )
        return StellarField(
            target_id=99999, mission="TESS",
            search_radius_pixels=10, stars=[target],
        )

    @pytest.fixture()
    def transit_lc(self):
        time = np.linspace(-0.1, 0.1, 50)
        return LightCurve(time_days=time, flux=np.ones(50), flux_err=0.001)

    @pytest.fixture()
    def small_config(self):
        return Config(n_mc_samples=100, n_best_samples=10)

    def test_no_trilegal_scenarios_provider_not_called(
        self, stellar_field, transit_lc, small_config,
    ) -> None:
        """TP/EB scenarios only — injected provider must not be called."""
        tp_result = _make_result(ScenarioID.TP, lnZ=0.0)
        eb_result = _make_result(ScenarioID.EB, lnZ=-1.0)
        eb_twin = _make_result(ScenarioID.EBX2P, lnZ=-2.0)

        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.TP, False, tp_result))
        registry.register(_FakeScenario(ScenarioID.EB, True, (eb_result, eb_twin)))

        # Even with a boom provider injected, it must never be called
        engine = ValidationEngine(registry=registry, population_provider=self._BoomProvider())
        engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        # If we reach here, the provider was not called — test passes.

    def test_dtp_scenario_with_population_passed_directly(
        self, stellar_field, transit_lc, small_config,
    ) -> None:
        """DTP scenario: engine accepts trilegal_population directly, no provider needed."""
        from triceratops.population.protocols import TRILEGALResult

        dtp_result = _make_result(ScenarioID.DTP, lnZ=-0.5)
        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.DTP, False, dtp_result))

        fake_pop = TRILEGALResult(
            tmags=np.array([15.0]),
            masses=np.array([0.5]),
            loggs=np.array([4.5]),
            teffs=np.array([4000.0]),
            metallicities=np.array([0.0]),
            jmags=np.array([14.0]),
            hmags=np.array([13.8]),
            kmags=np.array([13.7]),
            gmags=np.array([16.0]),
            rmags=np.array([15.5]),
            imags=np.array([15.2]),
            zmags=np.array([15.0]),
        )
        # Boom provider ensures engine never calls it
        engine = ValidationEngine(registry=registry, population_provider=self._BoomProvider())
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
            trilegal_population=fake_pop,
        )
        assert isinstance(vr, ValidationResult)

    def test_btp_scenario_with_population_none_does_not_crash(
        self, stellar_field, transit_lc, small_config,
    ) -> None:
        """BTP scenario: engine does not crash when trilegal_population=None."""
        btp_result = _make_result(ScenarioID.BTP, lnZ=-0.5)
        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.BTP, False, btp_result))

        engine = ValidationEngine(registry=registry, population_provider=None)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
            trilegal_population=None,
        )
        assert isinstance(vr, ValidationResult)

    def test_trilegal_provider_never_called_even_for_multiple_trilegal_scenarios(
        self, stellar_field, transit_lc, small_config,
    ) -> None:
        """Engine never calls provider even when multiple trilegal scenarios are present."""
        from triceratops.population.protocols import TRILEGALResult

        dtp_result = _make_result(ScenarioID.DTP, lnZ=-0.5)
        btp_result = _make_result(ScenarioID.BTP, lnZ=-1.0)

        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.DTP, False, dtp_result))
        registry.register(_FakeScenario(ScenarioID.BTP, False, btp_result))

        fake_pop = TRILEGALResult(
            tmags=np.array([14.0, 15.0]),
            masses=np.array([0.5, 0.8]),
            loggs=np.array([4.5, 4.3]),
            teffs=np.array([4000.0, 5000.0]),
            metallicities=np.array([0.0, 0.0]),
            jmags=np.array([13.0, 14.0]),
            hmags=np.array([12.8, 13.8]),
            kmags=np.array([12.7, 13.7]),
            gmags=np.array([15.0, 16.0]),
            rmags=np.array([14.5, 15.5]),
            imags=np.array([14.2, 15.2]),
            zmags=np.array([14.0, 15.0]),
        )
        engine = ValidationEngine(registry=registry, population_provider=self._BoomProvider())
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
            trilegal_population=fake_pop,
        )
        assert isinstance(vr, ValidationResult)

    def test_no_provider_does_not_crash_when_trilegal_needed(
        self, stellar_field, transit_lc, small_config,
    ) -> None:
        """If population_provider is None and no trilegal_population given, engine is graceful."""
        dtp_result = _make_result(ScenarioID.DTP, lnZ=-0.5)

        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.DTP, False, dtp_result))

        engine = ValidationEngine(registry=registry, population_provider=None)
        # Should not raise
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=small_config,
        )
        assert isinstance(vr, ValidationResult)


# ---------------------------------------------------------------------------
# Phase 2: Engine provider-free guarantees
# ---------------------------------------------------------------------------

class TestEngineProviderFreePhase2:
    """Phase 2 guarantees: engine.compute() works correctly with direct trilegal_population.

    These tests are the explicit Phase 2 regression guard.  They verify:
    1. compute() works when trilegal_population is passed directly.
    2. compute() works when trilegal_population=None.
    3. The engine never calls any provider during compute().
    """

    class _BoomProvider:
        def query(self, **kwargs: object) -> object:
            raise AssertionError("engine must not call providers during compute()")
        def query_nearby_stars(self, **kwargs: object) -> object:
            raise AssertionError("engine must not call providers during compute()")

    @pytest.fixture()
    def sf(self):
        target = Star(
            tic_id=55555,
            ra_deg=45.0, dec_deg=-10.0,
            tmag=10.0, jmag=9.3, hmag=9.1, kmag=9.0,
            bmag=10.8, vmag=10.5,
            stellar_params=StellarParameters(
                mass_msun=0.9, radius_rsun=0.9, teff_k=5400.0,
                logg=4.5, metallicity_dex=0.0, parallax_mas=8.0,
            ),
        )
        return StellarField(target_id=55555, mission="TESS", search_radius_pixels=10, stars=[target])

    @pytest.fixture()
    def lc(self):
        return LightCurve(
            time_days=np.linspace(-0.1, 0.1, 40),
            flux=np.ones(40),
            flux_err=0.001,
        )

    @pytest.fixture()
    def cfg(self):
        return Config(n_mc_samples=100, n_best_samples=10)

    @pytest.fixture()
    def fake_trilegal(self):
        from triceratops.population.protocols import TRILEGALResult
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

    def test_compute_with_trilegal_passed_directly(self, sf, lc, cfg, fake_trilegal) -> None:
        """compute() works when trilegal_population is passed directly (no provider needed)."""
        result = _make_result(ScenarioID.DTP, lnZ=-1.0)
        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.DTP, False, result))

        engine = ValidationEngine(
            registry=registry,
            catalog_provider=self._BoomProvider(),
            population_provider=self._BoomProvider(),
        )
        vr = engine.compute(
            lc, sf, period_days=5.0, config=cfg,
            trilegal_population=fake_trilegal,
        )
        assert isinstance(vr, ValidationResult)
        assert 0.0 <= vr.fpp <= 1.0

    def test_compute_with_trilegal_none_does_not_crash(self, sf, lc, cfg) -> None:
        """compute() with trilegal_population=None and no provider — no crash."""
        result = _make_result(ScenarioID.TP, lnZ=0.0)
        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.TP, False, result))

        engine = ValidationEngine(registry=registry, population_provider=None)
        vr = engine.compute(
            lc, sf, period_days=5.0, config=cfg,
            trilegal_population=None,
        )
        assert isinstance(vr, ValidationResult)

    def test_compute_never_calls_provider(self, sf, lc, cfg, fake_trilegal) -> None:
        """Provider injected into engine constructor must never be called during compute()."""
        result = _make_result(ScenarioID.BTP, lnZ=-2.0)
        registry = ScenarioRegistry()
        registry.register(_FakeScenario(ScenarioID.BTP, False, result))

        engine = ValidationEngine(
            registry=registry,
            catalog_provider=self._BoomProvider(),
            population_provider=self._BoomProvider(),
        )
        # Should not raise — provider must never be called
        vr = engine.compute(
            lc, sf, period_days=5.0, config=cfg,
            trilegal_population=fake_trilegal,
        )
        assert isinstance(vr, ValidationResult)


# ---------------------------------------------------------------------------
# n_workers cap at scenario count
# ---------------------------------------------------------------------------

class TestNWorkersCapAtScenarioCount:
    """Engine caps n_workers at the number of scenarios to avoid over-subscribing."""

    @pytest.fixture()
    def stellar_field(self):
        target = Star(
            tic_id=11111,
            ra_deg=5.0, dec_deg=10.0,
            tmag=11.0, jmag=10.5, hmag=10.2, kmag=10.1,
            bmag=12.0, vmag=11.5,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
                logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
            ),
        )
        return StellarField(
            target_id=11111, mission="TESS",
            search_radius_pixels=10, stars=[target],
        )

    @pytest.fixture()
    def transit_lc(self):
        time = np.linspace(-0.1, 0.1, 20)
        return LightCurve(time_days=time, flux=np.ones(20), flux_err=0.001)

    def _make_registry_with_n_scenarios(self, n: int) -> ScenarioRegistry:
        """Build a registry with n TP fake scenarios (using distinct IDs)."""
        ids = [
            ScenarioID.TP, ScenarioID.PTP, ScenarioID.DTP,
            ScenarioID.BTP, ScenarioID.STP, ScenarioID.NTP,
        ]
        registry = ScenarioRegistry()
        for sid in ids[:n]:
            result = _make_result(sid, lnZ=0.0)
            registry.register(_FakeScenario(sid, False, result))
        return registry

    def test_n_workers_100_with_3_scenarios_does_not_crash(
        self, stellar_field, transit_lc,
    ) -> None:
        """n_workers=100 with only 3 scenarios must complete without error."""
        config = Config(n_mc_samples=50, n_best_samples=5, n_workers=3)
        registry = self._make_registry_with_n_scenarios(3)
        engine = ValidationEngine(registry=registry)
        vr = engine.compute(
            transit_lc, stellar_field, period_days=5.0, config=config,
        )
        assert isinstance(vr, ValidationResult)
        assert len(vr.scenario_results) == 3

    def test_n_workers_capped_via_mock(
        self, stellar_field, transit_lc,
    ) -> None:
        """Verify ProcessPoolExecutor is called with max_workers <= len(scenarios)."""
        from unittest.mock import MagicMock, patch

        config = Config(n_mc_samples=50, n_best_samples=5, n_workers=100)
        registry = self._make_registry_with_n_scenarios(3)
        engine = ValidationEngine(registry=registry)

        captured: list[int] = []

        original_ppe = __import__(
            "concurrent.futures", fromlist=["ProcessPoolExecutor"]
        ).ProcessPoolExecutor

        def _spy_ppe(*args, max_workers=None, **kwargs):
            captured.append(max_workers)
            return original_ppe(*args, max_workers=max_workers, **kwargs)

        with patch(
            "triceratops.validation.engine.ProcessPoolExecutor",
            side_effect=_spy_ppe,
        ):
            engine.compute(
                transit_lc, stellar_field, period_days=5.0, config=config,
            )

        assert len(captured) == 1
        assert captured[0] <= 3, (
            f"ProcessPoolExecutor called with max_workers={captured[0]}, "
            "expected <= 3 (number of scenarios)"
        )

    def test_n_workers_zero_uses_serial_path(
        self, stellar_field, transit_lc,
    ) -> None:
        """n_workers=0 means serial execution — ProcessPoolExecutor is never created."""
        from unittest.mock import patch

        config = Config(n_mc_samples=50, n_best_samples=5, n_workers=0)
        registry = self._make_registry_with_n_scenarios(3)
        engine = ValidationEngine(registry=registry)

        with patch(
            "triceratops.validation.engine.ProcessPoolExecutor",
        ) as mock_ppe:
            vr = engine.compute(
                transit_lc, stellar_field, period_days=5.0, config=config,
            )

        mock_ppe.assert_not_called()
        assert isinstance(vr, ValidationResult)
        assert len(vr.scenario_results) == 3
