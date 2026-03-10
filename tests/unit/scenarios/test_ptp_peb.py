"""Tests for PTP and PEB scenario implementations."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import CONST, Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.companion_scenarios import (
    PEBScenario,
    PTPScenario,
    _compute_companion_prior,
    _compute_companion_properties,
    _load_molusc_qs,
)

try:
    from pytransit import QuadraticModel  # noqa: F401
    _HAS_PYTRANSIT = True
except (ImportError, Exception):
    _HAS_PYTRANSIT = False

_skip_no_pytransit = pytest.mark.skipif(
    not _HAS_PYTRANSIT, reason="pytransit not available or incompatible"
)

_LNL_MOD = "triceratops.scenarios.companion_scenarios"


class _RecordingCatalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def get_coefficients(
        self, band: str, metallicity: float, teff: float, logg: float,
    ) -> LimbDarkeningCoeffs:
        self.calls.append((band, logg))
        return LimbDarkeningCoeffs(u1=0.4, u2=0.2, band=band)


def _derived_logg(stellar_params: StellarParameters) -> float:
    return float(np.log10(
        CONST.G * (stellar_params.mass_msun * CONST.Msun)
        / (stellar_params.radius_rsun * CONST.Rsun) ** 2
    ))


def _mock_lnL_planet_p(*args, **kwargs):  # noqa: ARG001
    rps = kwargs["rps"] if "rps" in kwargs else args[3]
    mask = kwargs["mask"] if "mask" in kwargs else args[13]
    n = len(rps)
    result = np.full(n, np.inf)
    result[mask] = 1.0
    return result


def _mock_lnL_eb_p(*args, **kwargs):  # noqa: ARG001
    rss = kwargs["rss"] if "rss" in kwargs else args[3]
    mask = kwargs["mask"] if "mask" in kwargs else args[14]
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 1.5
    return result


def _mock_lnL_eb_twin_p(*args, **kwargs):  # noqa: ARG001
    rss = kwargs["rss"] if "rss" in kwargs else args[3]
    mask = kwargs["mask"] if "mask" in kwargs else args[14]
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 2.0
    return result


@pytest.fixture()
def fixed_ldc():
    return FixedLDCCatalog(u1=0.4, u2=0.2)


@pytest.fixture()
def stellar_params():
    return StellarParameters(
        mass_msun=1.0,
        radius_rsun=1.0,
        teff_k=5778.0,
        logg=4.44,
        metallicity_dex=0.0,
        parallax_mas=10.0,
    )


@pytest.fixture()
def light_curve():
    np.random.seed(42)
    time = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50) + np.random.normal(0, 0.001, 50)
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def external_lc():
    time = np.linspace(-0.05, 0.05, 40)
    flux = np.ones(40)
    return ExternalLightCurve(
        light_curve=LightCurve(time_days=time, flux=flux, flux_err=0.002),
        band="J",
        ldc=None,
    )


@pytest.fixture()
def small_config():
    return Config(n_mc_samples=200, n_best_samples=50, parallel=True)


# --- PTP identity tests ---

def test_ptp_scenario_id(fixed_ldc):
    ptp = PTPScenario(fixed_ldc)
    assert ptp.scenario_id == ScenarioID.PTP


def test_ptp_is_not_eb(fixed_ldc):
    ptp = PTPScenario(fixed_ldc)
    assert ptp.is_eb is False
    assert ptp.returns_twin is False


# --- PEB identity tests ---

def test_peb_scenario_id(fixed_ldc):
    peb = PEBScenario(fixed_ldc)
    assert peb.scenario_id == ScenarioID.PEB


def test_peb_returns_twin_true(fixed_ldc):
    peb = PEBScenario(fixed_ldc)
    assert peb.is_eb is True
    assert peb.returns_twin is True


@pytest.mark.parametrize("scenario_cls", [PTPScenario, PEBScenario])
def test_target_host_ldc_uses_mass_radius_logg(scenario_cls, stellar_params):
    catalog = _RecordingCatalog()
    adjusted = StellarParameters(
        mass_msun=stellar_params.mass_msun,
        radius_rsun=stellar_params.radius_rsun,
        teff_k=stellar_params.teff_k,
        logg=3.05,
        metallicity_dex=stellar_params.metallicity_dex,
        parallax_mas=stellar_params.parallax_mas,
    )
    scenario = scenario_cls(catalog)

    scenario._get_host_ldc(adjusted, "TESS", np.full(5, 5.0), {})

    assert catalog.calls[0][1] == pytest.approx(_derived_logg(adjusted))


# --- PTP compute ---

@_skip_no_pytransit
def test_ptp_compute_no_contrast_no_molusc(
    fixed_ldc, stellar_params, light_curve, small_config,
):
    """PTP completes without error using no contrast curve and no MOLUSC."""
    np.random.seed(123)
    ptp = PTPScenario(fixed_ldc)
    result = ptp.compute(light_curve, stellar_params, 5.0, small_config)
    assert isinstance(result, ScenarioResult)
    assert result.scenario_id == ScenarioID.PTP
    assert np.isfinite(result.ln_evidence) or result.ln_evidence == -np.inf
    assert len(result.planet_radius_rearth) == small_config.n_best_samples
    assert np.all(result.eb_mass_msun == 0.0)
    # Should have companion data
    assert len(result.companion_mass_msun) == small_config.n_best_samples


# --- PEB compute ---

@_skip_no_pytransit
def test_peb_compute_returns_tuple(
    fixed_ldc, stellar_params, light_curve, small_config,
):
    """PEB returns a tuple of (result, result_twin)."""
    np.random.seed(456)
    peb = PEBScenario(fixed_ldc)
    result = peb.compute(light_curve, stellar_params, 5.0, small_config)
    assert isinstance(result, tuple)
    assert len(result) == 2
    res, res_twin = result
    assert isinstance(res, ScenarioResult)
    assert isinstance(res_twin, ScenarioResult)
    assert res.scenario_id == ScenarioID.PEB
    assert res_twin.scenario_id == ScenarioID.PEBX2P
    assert np.all(res.planet_radius_rearth == 0.0)
    assert len(res.eb_mass_msun) == small_config.n_best_samples


@patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
def test_ptp_external_lcs_are_accumulated(
    mock_lnl, fixed_ldc, stellar_params, light_curve, external_lc, small_config,
):
    ptp = PTPScenario(fixed_ldc)
    ptp.compute(
        light_curve,
        stellar_params,
        5.0,
        small_config,
        external_lcs=[external_lc],
    )
    assert mock_lnl.call_count == 2


@patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
@patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
def test_peb_external_lcs_are_accumulated(
    mock_lnl_eb, mock_lnl_eb_twin, fixed_ldc, stellar_params,
    light_curve, external_lc, small_config,
):
    peb = PEBScenario(fixed_ldc)
    peb.compute(
        light_curve,
        stellar_params,
        5.0,
        small_config,
        external_lcs=[external_lc],
    )
    assert mock_lnl_eb.call_count == 2
    assert mock_lnl_eb_twin.call_count == 2


# --- MOLUSC loading tests ---

def test_load_molusc_qs_pads_short_data():
    """MOLUSC data with fewer rows than N gets padded with zeros."""
    from triceratops.domain.molusc import MoluscData
    data = MoluscData(
        semi_major_axis_au=np.array([100.0, 200.0, 50.0]),
        eccentricity=np.array([0.1, 0.2, 0.3]),
        mass_ratio=np.array([0.5, 0.3, 0.4]),
    )
    qs = _load_molusc_qs(data, 10, 1.0)
    assert len(qs) == 10
    # First 3 rows all pass filter (a*(1-e) > 10)
    assert qs[0] > 0
    assert qs[1] > 0
    assert qs[2] > 0
    # Padded entries are zero
    assert qs[3] == 0.0
    assert qs[9] == 0.0


def test_load_molusc_qs_truncates_long_data():
    """MOLUSC data with more rows than N gets truncated."""
    from triceratops.domain.molusc import MoluscData
    data = MoluscData(
        semi_major_axis_au=np.array([100 + i for i in range(20)], dtype=float),
        eccentricity=np.full(20, 0.1),
        mass_ratio=np.array([0.3 + i * 0.01 for i in range(20)]),
    )
    qs = _load_molusc_qs(data, 5, 1.0)
    assert len(qs) == 5


def test_load_molusc_qs_wide_separation_filter():
    """Only rows with a*(1-e) > 10 are kept."""
    from triceratops.domain.molusc import MoluscData
    data = MoluscData(
        semi_major_axis_au=np.array([5.0, 100.0, 8.0]),
        eccentricity=np.array([0.1, 0.1, 0.5]),
        mass_ratio=np.array([0.5, 0.3, 0.4]),
    )
    qs = _load_molusc_qs(data, 5, 1.0)
    assert len(qs) == 5
    # Only 1 row passes filter
    assert qs[0] == pytest.approx(0.3)
    # Rest are padded zeros
    assert qs[1] == 0.0


def test_load_molusc_qs_minimum_mass_ratio():
    """Mass ratios below 0.1/M_s are clamped."""
    from triceratops.domain.molusc import MoluscData
    data = MoluscData(
        semi_major_axis_au=np.array([100.0]),
        eccentricity=np.array([0.1]),
        mass_ratio=np.array([0.01]),
    )
    qs = _load_molusc_qs(data, 3, 2.0)
    assert qs[0] == pytest.approx(0.05)


# --- Protocol compliance ---

def test_ptp_satisfies_scenario_protocol(fixed_ldc):
    """PTPScenario satisfies the Scenario structural protocol."""
    from triceratops.scenarios.base import Scenario

    ptp = PTPScenario(fixed_ldc)
    assert isinstance(ptp, Scenario)


def test_peb_satisfies_scenario_protocol(fixed_ldc):
    """PEBScenario satisfies the Scenario structural protocol."""
    from triceratops.scenarios.base import Scenario

    peb = PEBScenario(fixed_ldc)
    assert isinstance(peb, Scenario)


# --- Helper function tests ---

def test_compute_companion_properties_basic():
    """Companion properties computed from mass ratios."""
    qs = np.array([0.5, 0.3, 0.8])
    masses, radii, teffs, fluxratios = _compute_companion_properties(
        qs, M_s=1.0, R_s=1.0, Teff=5778.0, n=3,
    )
    assert masses.shape == (3,)
    assert np.allclose(masses, [0.5, 0.3, 0.8])
    assert radii.shape == (3,)
    assert teffs.shape == (3,)
    assert fluxratios.shape == (3,)
    # Flux ratios should be in (0, 1) for companions less massive than primary
    assert np.all(fluxratios > 0)
    assert np.all(fluxratios < 1)


def test_compute_companion_prior_with_molusc():
    """When molusc_data is not None, prior is all zeros."""
    from triceratops.domain.molusc import MoluscData
    prior = _compute_companion_prior(
        masses_comp=np.array([0.5, 0.3]),
        fluxratios_comp=np.array([0.3, 0.15]),
        M_s=1.0, plx=10.0, n=2,
        molusc_data=MoluscData(
            semi_major_axis_au=np.array([20.0]),
            eccentricity=np.array([0.0]),
            mass_ratio=np.array([0.5]),
        ),
        contrast_curve=None, filt="TESS", is_eb=False,
    )
    assert np.allclose(prior, 0.0)


def test_compute_companion_prior_no_contrast():
    """Without contrast curve, prior uses default separations."""
    n = 100
    np.random.seed(99)
    qs = np.random.uniform(0.1, 0.9, n)
    masses = qs * 1.0
    from triceratops.stellar.relations import StellarRelations
    sr = StellarRelations()
    flux_comp = sr.get_flux_ratio(masses, "TESS")
    flux_primary = sr.get_flux_ratio(np.array([1.0]), "TESS")
    fluxratios = flux_comp / (flux_comp + flux_primary)

    prior = _compute_companion_prior(
        masses_comp=masses,
        fluxratios_comp=fluxratios,
        M_s=1.0, plx=10.0, n=n,
        molusc_data=None,
        contrast_curve=None, filt="TESS", is_eb=False,
    )
    assert prior.shape == (n,)
    # Should have some finite and some -inf values
    assert np.any(np.isfinite(prior))


def test_compute_companion_prior_with_contrast_curve():
    """With contrast curve, prior uses the curve's separations and contrasts."""
    from triceratops.domain.value_objects import ContrastCurve

    cc = ContrastCurve(
        separations_arcsec=np.array([0.1, 0.5, 1.0, 2.0]),
        delta_mags=np.array([1.0, 3.0, 5.0, 7.0]),
        band="J",
    )
    n = 50
    np.random.seed(88)
    qs = np.random.uniform(0.1, 0.9, n)
    masses = qs * 1.0
    from triceratops.stellar.relations import StellarRelations
    sr = StellarRelations()
    flux_comp = sr.get_flux_ratio(masses, "TESS")
    flux_primary = sr.get_flux_ratio(np.array([1.0]), "TESS")
    fluxratios = flux_comp / (flux_comp + flux_primary)

    prior = _compute_companion_prior(
        masses_comp=masses,
        fluxratios_comp=fluxratios,
        M_s=1.0, plx=10.0, n=n,
        molusc_data=None,
        contrast_curve=cc, filt="J", is_eb=True,
    )
    assert prior.shape == (n,)


# --- Phase-level tests (no pytransit needed) ---

def test_ptp_sample_priors(fixed_ldc, stellar_params, small_config):
    """PTP _sample_priors returns correctly shaped arrays."""
    np.random.seed(77)
    ptp = PTPScenario(fixed_ldc)
    N = small_config.n_mc_samples
    P_orb = np.full(N, 5.0)
    samples = ptp._sample_priors(N, stellar_params, P_orb, small_config)
    assert samples["rps"].shape == (N,)
    assert samples["incs"].shape == (N,)
    assert samples["eccs"].shape == (N,)
    assert samples["argps"].shape == (N,)
    assert samples["qs_comp"].shape == (N,)
    assert samples["masses_comp"].shape == (N,)
    assert samples["radii_comp"].shape == (N,)
    assert samples["fluxratios_comp"].shape == (N,)


def test_ptp_compute_orbital_geometry(fixed_ldc, stellar_params, small_config):
    """PTP orbital geometry produces correctly shaped arrays."""
    np.random.seed(66)
    ptp = PTPScenario(fixed_ldc)
    N = small_config.n_mc_samples
    P_orb = np.full(N, 5.0)
    samples = ptp._sample_priors(N, stellar_params, P_orb, small_config)
    geometry = ptp._compute_orbital_geometry(
        samples, P_orb, stellar_params, small_config,
    )
    assert geometry["a"].shape == (N,)
    assert geometry["ptra"].shape == (N,)
    assert geometry["b"].shape == (N,)
    assert geometry["coll"].shape == (N,)
    assert np.all(geometry["a"] > 0)


def test_peb_sample_priors(fixed_ldc, stellar_params, small_config):
    """PEB _sample_priors returns correctly shaped arrays including EB properties."""
    np.random.seed(55)
    peb = PEBScenario(fixed_ldc)
    N = small_config.n_mc_samples
    P_orb = np.full(N, 5.0)
    samples = peb._sample_priors(N, stellar_params, P_orb, small_config)
    assert samples["qs"].shape == (N,)
    assert samples["masses_eb"].shape == (N,)
    assert samples["radii_eb"].shape == (N,)
    assert samples["fluxratios_eb"].shape == (N,)
    assert samples["masses_comp"].shape == (N,)
    assert samples["fluxratios_comp"].shape == (N,)


def test_peb_compute_orbital_geometry(fixed_ldc, stellar_params, small_config):
    """PEB orbital geometry includes twin arrays."""
    np.random.seed(44)
    peb = PEBScenario(fixed_ldc)
    N = small_config.n_mc_samples
    P_orb = np.full(N, 5.0)
    samples = peb._sample_priors(N, stellar_params, P_orb, small_config)
    geometry = peb._compute_orbital_geometry(
        samples, P_orb, stellar_params, small_config,
    )
    for key in ["a", "a_twin", "ptra", "ptra_twin", "b", "b_twin", "coll", "coll_twin"]:
        assert key in geometry, f"Missing geometry key: {key}"
        assert geometry[key].shape == (N,)
    # Twin semi-major axis should be larger (longer period)
    assert np.all(geometry["a_twin"] > geometry["a"])


def test_ptp_get_host_ldc(fixed_ldc, stellar_params):
    """PTP LDC lookup returns expected fixed values."""
    ptp = PTPScenario(fixed_ldc)
    ldc = ptp._get_host_ldc(stellar_params, "TESS", np.array([5.0]), {})
    assert ldc.u1 == 0.4
    assert ldc.u2 == 0.2
    assert ldc.band == "TESS"
