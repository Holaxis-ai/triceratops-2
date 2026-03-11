"""Tests for NTP and NEB scenarios (unknown + evolved)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.population.protocols import TRILEGALResult
from triceratops.scenarios.nearby_scenarios import (
    _EVOLVED_LOGG,
    NEBEvolvedScenario,
    NEBUnknownScenario,
    NTPEvolvedScenario,
    NTPUnknownScenario,
)

_LNL_MOD = "triceratops.scenarios.nearby_scenarios"


# ---------------------------------------------------------------------------
# Mock lnL functions
# ---------------------------------------------------------------------------

def _mock_lnL_planet_p(*, time, flux, sigma, rps, periods, incs, as_, rss,
                        u1s, u2s, eccs, argps, companion_flux_ratios, mask,
                        companion_is_host=False, exptime=0.00139, nsamples=20,
                        force_serial=False):
    n = len(rps)
    result = np.full(n, np.inf)
    result[mask] = 1.0
    return result


def _mock_lnL_eb_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
                    periods, incs, as_, u1s, u2s, eccs, argps,
                    companion_flux_ratios, mask, companion_is_host=False,
                    exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 1.5
    return result


def _mock_lnL_eb_twin_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
                         periods, incs, as_, u1s, u2s, eccs, argps,
                         companion_flux_ratios, mask, companion_is_host=False,
                         exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 2.0
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stellar_params():
    return StellarParameters(
        mass_msun=1.0, radius_rsun=2.0, teff_k=5778.0,
        logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
    )


@pytest.fixture()
def transit_lc():
    time = np.linspace(-0.1, 0.1, 100)
    flux = np.ones(100)
    flux[40:60] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def small_config():
    return Config(n_mc_samples=200, n_best_samples=10)


@pytest.fixture()
def mock_population():
    """TRILEGAL population with stars near Tmag=10."""
    n = 100
    rng = np.random.default_rng(42)
    return TRILEGALResult(
        tmags=rng.uniform(9.0, 11.0, n),  # many within +/- 1 of 10.0
        masses=rng.uniform(0.5, 1.5, n),
        loggs=rng.uniform(3.8, 4.8, n),  # most above 3.5
        teffs=rng.uniform(4000, 8000, n),  # most below 10000
        metallicities=rng.uniform(-0.5, 0.3, n),
        jmags=rng.uniform(9.0, 15.0, n),
        hmags=rng.uniform(8.5, 14.5, n),
        kmags=rng.uniform(8.0, 14.0, n),
        gmags=rng.uniform(10.5, 16.5, n),
        rmags=rng.uniform(10.0, 16.0, n),
        imags=rng.uniform(9.5, 15.5, n),
        zmags=rng.uniform(9.0, 15.0, n),
    )


# ---------------------------------------------------------------------------
# NTPUnknown Tests
# ---------------------------------------------------------------------------

class TestNTPUnknownIdentity:
    def test_scenario_id(self) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.NTP

    def test_is_not_eb(self) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        assert not s.is_eb
        assert not s.returns_twin

    def test_resolve_external_lcs_returns_empty(self) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        ext_lcs = [MagicMock(spec=ExternalLightCurve)]
        result = s._resolve_external_lc_ldcs(ext_lcs, MagicMock(spec=StellarParameters))  # type: ignore[arg-type]
        assert result == []


class TestNTPUnknownFiltering:
    def test_filters_by_tmag_range(self, stellar_params, mock_population) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            target_tmag=10.0,
        )
        # All masses_possible should be from stars with Tmag in [9, 11]
        assert len(samples["masses_possible"]) > 0
        assert samples["rps"].shape == (50,)

    def test_raises_without_trilegal(self, stellar_params, small_config) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="trilegal_population"):
            s._sample_priors(10, stellar_params, P_orb, small_config)

    def test_raises_no_matching_stars(self, stellar_params, small_config) -> None:
        """No stars within Tmag +/- 1 of target."""
        pop = TRILEGALResult(
            tmags=np.array([20.0, 21.0]),
            masses=np.array([0.5, 0.6]),
            loggs=np.array([4.0, 4.1]),
            teffs=np.array([5000.0, 5100.0]),
            metallicities=np.array([0.0, 0.0]),
            jmags=np.array([15.0, 16.0]),
            hmags=np.array([14.5, 15.5]),
            kmags=np.array([14.0, 15.0]),
            gmags=np.array([16.0, 17.0]),
            rmags=np.array([15.5, 16.5]),
            imags=np.array([15.0, 16.0]),
            zmags=np.array([14.5, 15.5]),
        )
        s = NTPUnknownScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="No TRILEGAL stars"):
            s._sample_priors(
                10, stellar_params, P_orb, small_config,
                trilegal_population=pop, target_tmag=10.0,
            )


class TestNTPUnknownCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config, mock_population,
    ) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population, target_tmag=10.0,
        )
        assert isinstance(result, ScenarioResult)
        assert not isinstance(result, tuple)
        assert result.scenario_id == ScenarioID.NTP

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_compute_result_has_finite_lnz(
        self, _mock, transit_lc, stellar_params, small_config, mock_population,
    ) -> None:
        s = NTPUnknownScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population, target_tmag=10.0,
        )
        assert isinstance(result, ScenarioResult)
        assert result.ln_evidence > -np.inf

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_per_star_host_mass_varies(
        self, _mock, transit_lc, stellar_params, small_config, mock_population,
    ) -> None:
        """N_unknown host properties vary per-sample (from TRILEGAL)."""
        s = NTPUnknownScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population, target_tmag=10.0,
        )
        assert isinstance(result, ScenarioResult)
        # Host masses should not all be the same (drawn from population)
        assert not np.all(result.host_mass_msun == result.host_mass_msun[0])


# ---------------------------------------------------------------------------
# NEBUnknown Tests
# ---------------------------------------------------------------------------

class TestNEBUnknownIdentity:
    def test_scenario_id(self) -> None:
        s = NEBUnknownScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.NEB

    def test_is_eb(self) -> None:
        s = NEBUnknownScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin

    def test_resolve_external_lcs_returns_empty(self) -> None:
        s = NEBUnknownScenario(FixedLDCCatalog())
        result = s._resolve_external_lc_ldcs([], MagicMock(spec=StellarParameters))
        assert result == []


class TestNEBUnknownCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config, mock_population,
    ) -> None:
        s = NEBUnknownScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population, target_tmag=10.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_primary_has_neb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config, mock_population,
    ) -> None:
        s = NEBUnknownScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population, target_tmag=10.0,
        )
        assert isinstance(result, tuple)
        primary, twin = result
        assert primary.scenario_id == ScenarioID.NEB
        assert twin.scenario_id == ScenarioID.NEBX2P


# ---------------------------------------------------------------------------
# NTPEvolved Tests
# ---------------------------------------------------------------------------

class TestNTPEvolvedIdentity:
    def test_scenario_id(self) -> None:
        s = NTPEvolvedScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.NTP

    def test_is_not_eb(self) -> None:
        s = NTPEvolvedScenario(FixedLDCCatalog())
        assert not s.is_eb

    def test_resolve_external_lcs_returns_empty(self) -> None:
        s = NTPEvolvedScenario(FixedLDCCatalog())
        result = s._resolve_external_lc_ldcs([], MagicMock(spec=StellarParameters))
        assert result == []


class TestNTPEvolvedLDC:
    def test_get_host_ldc_uses_logg_3(self) -> None:
        """Verify that LDC lookup uses logg=3.0, not stellar_params.logg."""
        mock_catalog = MagicMock()
        mock_catalog.get_coefficients.return_value = LimbDarkeningCoeffs(
            u1=0.4, u2=0.2, band="TESS",
        )
        s = NTPEvolvedScenario(mock_catalog)
        sp = StellarParameters(
            mass_msun=1.0, radius_rsun=2.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        )
        s._get_host_ldc(sp, "TESS", np.full(10, 5.0), {})
        call_args = mock_catalog.get_coefficients.call_args
        # logg argument (4th positional) should be 3.0
        assert call_args[0][3] == _EVOLVED_LOGG


class TestNTPEvolvedBug05:
    def test_rs_is_array_not_scalar(self, stellar_params) -> None:
        """BUG-05: R_s must be a numpy array, not a float."""
        s = NTPEvolvedScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        assert isinstance(samples["R_s"], np.ndarray)
        assert samples["R_s"].shape == (50,)
        assert np.all(samples["R_s"] == stellar_params.radius_rsun)


class TestNTPEvolvedCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = NTPEvolvedScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == ScenarioID.NTP

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_compute_has_finite_lnz(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = NTPEvolvedScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert result.ln_evidence > -np.inf

# ---------------------------------------------------------------------------
# NEBEvolved Tests
# ---------------------------------------------------------------------------

class TestNEBEvolvedIdentity:
    def test_scenario_id(self) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.NEB

    def test_is_eb(self) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin

    def test_resolve_external_lcs_returns_empty(self) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        result = s._resolve_external_lc_ldcs([], MagicMock(spec=StellarParameters))
        assert result == []


class TestNEBEvolvedLDC:
    def test_get_host_ldc_uses_logg_3(self) -> None:
        mock_catalog = MagicMock()
        mock_catalog.get_coefficients.return_value = LimbDarkeningCoeffs(
            u1=0.4, u2=0.2, band="TESS",
        )
        s = NEBEvolvedScenario(mock_catalog)
        sp = StellarParameters(
            mass_msun=1.0, radius_rsun=2.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        )
        s._get_host_ldc(sp, "TESS", np.full(10, 5.0), {})
        call_args = mock_catalog.get_coefficients.call_args
        assert call_args[0][3] == _EVOLVED_LOGG


class TestNEBEvolvedBug05:
    def test_rs_is_array_not_scalar(self, stellar_params) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        assert isinstance(samples["R_s"], np.ndarray)
        assert samples["R_s"].shape == (50,)


class TestNEBEvolvedCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_primary_has_neb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        primary, twin = result
        assert primary.scenario_id == ScenarioID.NEB
        assert twin.scenario_id == ScenarioID.NEBX2P

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_result_arrays_correct_length(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = NEBEvolvedScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        primary, twin = result
        n = small_config.n_best_samples
        assert len(primary.host_mass_msun) == n
        assert len(twin.host_mass_msun) == n
