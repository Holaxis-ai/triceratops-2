"""Tests for BTPScenario and BEBScenario."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_companion_mass_ratio,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
)
from triceratops.scenarios.background_scenarios import (
    BEBScenario,
    BTPScenario,
    _compute_bright_background_lnprior,
    _lookup_background_ldc_bulk,
    _sample_population_indices,
)

_LNL_MOD = "triceratops.scenarios.background_scenarios"


# ---------------------------------------------------------------------------
# Mock lnL functions (pytransit not usable under numpy 2.x)
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
        mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
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
    """Small TRILEGAL population for testing."""
    n = 50
    rng = np.random.default_rng(42)
    return TRILEGALResult(
        tmags=rng.uniform(10.0, 16.0, n),
        masses=rng.uniform(0.3, 1.5, n),
        loggs=rng.uniform(3.5, 5.0, n),
        teffs=rng.uniform(3500, 7000, n),
        metallicities=rng.uniform(-1.0, 0.5, n),
        jmags=rng.uniform(9.0, 15.0, n),
        hmags=rng.uniform(8.5, 14.5, n),
        kmags=rng.uniform(8.0, 14.0, n),
        gmags=rng.uniform(10.5, 16.5, n),
        rmags=rng.uniform(10.0, 16.0, n),
        imags=rng.uniform(9.5, 15.5, n),
        zmags=rng.uniform(9.0, 15.0, n),
    )


@pytest.fixture()
def host_mags():
    return {
        "tmag": 10.0, "jmag": 9.5, "hmag": 9.0, "kmag": 8.8,
        "gmag": 10.5, "rmag": 10.0, "imag": 9.5, "zmag": 9.0,
    }


# ---------------------------------------------------------------------------
# BTP Tests -- identity
# ---------------------------------------------------------------------------

class TestBTPIdentity:
    def test_btp_scenario_id(self) -> None:
        s = BTPScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.BTP

    def test_btp_is_not_eb(self) -> None:
        s = BTPScenario(FixedLDCCatalog())
        assert not s.is_eb
        assert not s.returns_twin

    def test_btp_satisfies_scenario_protocol(self) -> None:
        from triceratops.scenarios.base import Scenario
        s = BTPScenario(FixedLDCCatalog())
        assert isinstance(s, Scenario)


class TestBTPRaisesWithoutTrilegal:
    def test_btp_raises_without_trilegal(self, stellar_params, small_config) -> None:
        s = BTPScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="trilegal_population"):
            s._sample_priors(10, stellar_params, P_orb, small_config)


class TestBTPSamplePriors:
    def test_sample_priors_returns_expected_keys(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        s = BTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        for key in ("rps", "incs", "eccs", "argps", "idxs",
                     "fluxratios_comp", "lnprior_companion",
                     "masses_comp", "radii_comp", "loggs_comp",
                     "Teffs_comp", "Zs_comp"):
            assert key in samples, f"Missing key: {key}"
        assert samples["rps"].shape == (50,)

    def test_btp_uses_background_star_mass_for_rp(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        """BTP samples planet radius using background star mass, not target mass."""
        s = BTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        # rps should exist and be positive
        assert np.all(samples["rps"] > 0)


class TestBTPOrbitalGeometry:
    def test_orbital_geometry_uses_background_mass(
        self, stellar_params, mock_population, host_mags, small_config,
    ) -> None:
        """BTP orbital geometry uses background star mass, not target mass."""
        s = BTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        geometry = s._compute_orbital_geometry(
            samples, P_orb, stellar_params, cfg,
        )
        assert geometry["a"].shape == (50,)
        assert geometry["Ptra"].shape == (50,)
        assert geometry["b"].shape == (50,)
        assert geometry["coll"].shape == (50,)
        assert np.all(geometry["a"] > 0)


# ---------------------------------------------------------------------------
# BTP Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestBTPCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_btp_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert not isinstance(result, tuple)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_btp_scenario_id_in_result(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == ScenarioID.BTP

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_btp_host_is_background_star(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        """BTP host mass/radius should be background star, not target."""
        s = BTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        # Host mass should vary (background star masses), not all be 1.0 (target)
        # Companion mass should be target star mass (1.0)
        np.testing.assert_array_equal(
            result.companion_mass_msun,
            np.full(small_config.n_best_samples, stellar_params.mass_msun),
        )

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_btp_result_eb_fields_are_zero(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        np.testing.assert_array_equal(result.eb_mass_msun, 0)
        np.testing.assert_array_equal(result.eb_radius_rsun, 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_btp_result_arrays_have_correct_length(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        n = small_config.n_best_samples
        assert len(result.host_mass_msun) == n
        assert len(result.planet_radius_rearth) == n


class TestBrightBackgroundParity:
    def test_beb_prior_uses_combined_host_and_eb_brightness(self) -> None:
        idxs = np.array([0, 1])
        fluxratios_comp = np.array([0.20, 0.25])
        fluxratios_eb = np.array([0.10, 0.15])

        lnprior = _compute_bright_background_lnprior(
            50, idxs, fluxratios_comp, fluxratios_eb, None,
        )

        delta_mags = 2.5 * np.log10(
            (fluxratios_comp / (1 - fluxratios_comp))
            + (fluxratios_eb / (1 - fluxratios_eb))
        )
        expected = np.full(
            2, np.log((50 / 0.1) * (1 / 3600) ** 2 * 2.2**2),
        )
        expected[expected > 0.0] = 0.0
        expected[delta_mags > 0.0] = -np.inf

        np.testing.assert_array_equal(lnprior, expected)

    def test_lookup_background_ldc_bulk_uses_slice_then_metallicity(self) -> None:
        class SparseCatalog:
            def _load_filter(self, _band: str):
                return (
                    np.array([0.0, 0.5, 1.0]),
                    np.array([5000.0, 5000.0, 5500.0]),
                    np.array([4.5, 4.5, 4.0]),
                    np.array([0.1, 0.2, 0.9]),
                    np.array([0.01, 0.02, 0.09]),
                )

        u1, u2 = _lookup_background_ldc_bulk(
            SparseCatalog(),
            "TESS",
            np.array([5001.0]),
            np.array([4.49]),
            np.array([0.9]),
        )

        np.testing.assert_array_equal(u1, np.array([0.2]))
        np.testing.assert_array_equal(u2, np.array([0.02]))

    def test_beb_external_branch_keeps_background_star_as_host_radius(self) -> None:
        scenario = BEBScenario(FixedLDCCatalog())
        lc = LightCurve(
            time_days=np.array([-0.01, 0.01]),
            flux=np.array([1.0, 0.999]),
            flux_err=0.001,
        )
        ext_lc = ExternalLightCurve(
            light_curve=lc,
            band="i",
            ldc=LimbDarkeningCoeffs(u1=0.1, u2=0.2, band="i"),
        )
        samples = {
            "radii": np.array([0.4, 0.5]),
            "masses": np.array([0.3, 0.35]),
            "idxs": np.array([1, 0]),
            "radii_comp": np.array([1.1, 1.2]),
            "fluxratios": np.array([0.10, 0.12]),
            "fluxratios_comp": np.array([0.20, 0.25]),
            "delta_mags_map": {"delta_imags": np.array([1.0, 1.2])},
            "M_s": np.array([1.0, 1.0]),
            "masses_comp": np.array([1.3, 1.4]),
            "incs": np.array([89.0, 89.0]),
            "eccs": np.array([0.0, 0.0]),
            "argps": np.array([90.0, 90.0]),
            "P_orb": np.array([5.0, 5.0]),
        }
        geometry = {"a": np.array([10.0, 10.0])}
        comp_params = {
            "radii_comp": samples["radii_comp"],
            "u1s_comp": np.array([0.3, 0.4]),
            "u2s_comp": np.array([0.2, 0.3]),
            "loggs_comp": np.array([4.4, 4.5]),
            "Teffs_comp": np.array([5200.0, 5400.0]),
            "Zs_comp": np.array([0.0, 0.1]),
        }
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        def capture_lnL(**kwargs):
            calls.append((kwargs["rss"].copy(), kwargs["rcomps"].copy()))
            return np.zeros(len(kwargs["rss"]))

        scenario._beb_branch_lnL(
            lc,
            np.log(lc.sigma),
            samples,
            geometry,
            [ext_lc],
            idxs=samples["idxs"],
            comp_fr=samples["fluxratios_comp"][samples["idxs"]],
            comp_params=comp_params,
            mask=np.array([True, True]),
            lnL_fn=capture_lnL,
            a_key="a",
            period_mult=1,
            N=2,
        )

        assert len(calls) == 2
        expected_rss = samples["radii_comp"][samples["idxs"]]
        expected_rcomps = samples["radii"]
        np.testing.assert_array_equal(calls[0][0], expected_rss)
        np.testing.assert_array_equal(calls[0][1], expected_rcomps)
        np.testing.assert_array_equal(calls[1][0], expected_rss)
        np.testing.assert_array_equal(calls[1][1], expected_rcomps)


# ---------------------------------------------------------------------------
# BEB Tests -- identity
# ---------------------------------------------------------------------------

class TestBEBIdentity:
    def test_beb_scenario_id(self) -> None:
        s = BEBScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.BEB

    def test_beb_is_eb(self) -> None:
        s = BEBScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin

    def test_beb_satisfies_scenario_protocol(self) -> None:
        from triceratops.scenarios.base import Scenario
        s = BEBScenario(FixedLDCCatalog())
        assert isinstance(s, Scenario)


class TestBEBRaisesWithoutTrilegal:
    def test_beb_raises_without_trilegal(self, stellar_params, small_config) -> None:
        s = BEBScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="trilegal_population"):
            s._sample_priors(10, stellar_params, P_orb, small_config)


class TestBEBSamplePriors:
    def test_sample_priors_returns_expected_keys(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        for key in ("qs", "masses", "radii", "fluxratios",
                     "incs", "eccs", "argps", "idxs",
                     "fluxratios_comp", "lnprior_companion",
                     "masses_comp", "radii_comp", "distance_correction"):
            assert key in samples, f"Missing key: {key}"
        assert samples["qs"].shape == (50,)

    def test_beb_eb_masses_use_background_star_mass(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        """BEB: EB masses = qs * background_star_mass, not target mass."""
        s = BEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        idxs = samples["idxs"].astype(int)
        expected = samples["qs"] * mock_population.masses[idxs]
        np.testing.assert_allclose(samples["masses"], expected)

    def test_beb_preserves_original_rng_draw_order(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        scenario = BEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=24, n_best_samples=10)
        P_orb = np.full(24, 5.0)

        np.random.seed(321)
        expected_incs = sample_inclination(np.random.rand(24))
        expected_qs = sample_mass_ratio(np.random.rand(24), stellar_params.mass_msun)
        _ = sample_companion_mass_ratio(np.random.rand(24), stellar_params.mass_msun)
        expected_eccs = sample_eccentricity(
            np.random.rand(24), planet=False, period=float(np.mean(P_orb)),
        )
        expected_argps = sample_arg_periastron(np.random.rand(24))
        expected_idxs = _sample_population_indices(mock_population.n_stars, 24)

        np.random.seed(321)
        samples = scenario._sample_priors(
            24, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )

        np.testing.assert_allclose(samples["incs"], expected_incs)
        np.testing.assert_allclose(samples["qs"], expected_qs)
        np.testing.assert_allclose(samples["eccs"], expected_eccs)
        np.testing.assert_allclose(samples["argps"], expected_argps)
        np.testing.assert_array_equal(samples["idxs"], expected_idxs)


class TestBEBOrbitalGeometry:
    def test_orbital_geometry_has_twin_arrays(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        np.random.seed(42)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        geometry = s._compute_orbital_geometry(
            samples, P_orb, stellar_params, cfg,
        )
        for key in ("a", "a_twin", "Ptra", "Ptra_twin",
                     "b", "b_twin", "coll", "coll_twin"):
            assert key in geometry, f"Missing geometry key: {key}"
            assert geometry[key].shape == (50,)
        # Twin semi-major axis should be larger (longer period)
        assert np.all(geometry["a_twin"] > geometry["a"])


# ---------------------------------------------------------------------------
# BEB Tests -- legacy parallel-path parity
# ---------------------------------------------------------------------------

class TestBEBLegacyParallelMask:
    """Original BEB parallel path used coll_twin for the q<0.95 branch."""

    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    def test_q_lt_95_uses_coll_twin_in_parallel_path(
        self, _m_twin, _m_eb, transit_lc, stellar_params,
        mock_population, host_mags,
    ) -> None:
        """With coll=False, coll_twin=True: q<0.95 stays masked out."""
        s = BEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=100, n_best_samples=10)
        P_orb = np.full(100, 5.0)
        np.random.seed(123)

        samples = s._sample_priors(
            100, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )

        geometry = s._compute_orbital_geometry(
            samples, P_orb, stellar_params, cfg,
        )

        # Override geometry: coll=all_False, coll_twin=all_True
        geometry["coll"] = np.zeros(100, dtype=bool)
        geometry["coll_twin"] = np.ones(100, dtype=bool)

        ldc = s._get_host_ldc(stellar_params, "TESS", P_orb, {})
        lnL, lnL_twin = s._evaluate_lnL(
            transit_lc, np.log(transit_lc.sigma), samples, geometry,
            ldc, [], cfg,
        )

        # Original parallel path used coll_twin for q<0.95 too.
        q_lt_95 = samples["qs"] < 0.95
        if np.any(q_lt_95):
            assert np.all(lnL[q_lt_95] == -np.inf), \
                "Legacy BEB parity regression: q<0.95 should be masked by coll_twin"

        # q>=0.95 samples should be all -inf (coll_twin is True)
        q_ge_95 = samples["qs"] >= 0.95
        if np.any(q_ge_95):
            assert np.all(lnL_twin[q_ge_95] == -np.inf), \
                "q>=0.95 should be -inf because coll_twin=True"


# ---------------------------------------------------------------------------
# BEB Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestBEBCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_primary_has_beb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        assert primary.scenario_id == ScenarioID.BEB

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_twin_has_bebx2p_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        _primary, twin = result
        assert twin.scenario_id == ScenarioID.BEBX2P

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_primary_has_eb_mass(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        assert np.any(primary.eb_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_host_is_background_star(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        """BEB host mass should be background star mass, companion should be target."""
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        # Companion mass = target star mass
        np.testing.assert_array_equal(
            primary.companion_mass_msun,
            np.full(small_config.n_best_samples, stellar_params.mass_msun),
        )

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_result_arrays_have_correct_length(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, twin = result
        n = small_config.n_best_samples
        assert len(primary.host_mass_msun) == n
        assert len(twin.host_mass_msun) == n

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_beb_primary_planet_radius_is_zero(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        """BEB is an EB scenario, planet radius should be zero."""
        s = BEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        np.testing.assert_array_equal(primary.planet_radius_rearth, 0)
