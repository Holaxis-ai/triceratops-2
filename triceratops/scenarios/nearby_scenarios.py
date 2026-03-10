"""NTP and NEB scenario implementations (nearby unknown + evolved).

NTP_unknown / NEB_unknown: target star properties unknown, drawn from
TRILEGAL peers with Tmag +/- 1. Per-star LDC via get_coefficients_bulk().

NTP_evolved / NEB_evolved: target is a known subgiant (logg=3.0).
Mass derived from logg and R_s.

BUG-03 fix: all four use config.lnz_const (default 650) via BaseScenario,
not the hardcoded 600 in the original.

BUG-05 fix: NEB_evolved passes radii[mask] (array) to lnL_EB_twin_p,
not scalar R_s.

Source: marginal_likelihoods.py:3672-3861 (lnZ_NTP_unknown),
        3864-4145 (lnZ_NEB_unknown), 4148-4285 (lnZ_NTP_evolved),
        4288-4503 (lnZ_NEB_evolved).
"""
# ruff: noqa: ARG002  -- ABC override signatures require unused params
from __future__ import annotations

import numpy as np

from triceratops.config.config import CONST, Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.likelihoods.geometry import (
    collision_check,
    impact_parameter,
    semi_major_axis,
    transit_probability,
)
from triceratops.likelihoods.lnl_functions import (
    lnL_eb_p,
    lnL_eb_twin_p,
    lnL_planet_p,
)
from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
    sample_planet_radius,
)
from triceratops.scenarios._eb_branching import build_eb_branch_masks
from triceratops.scenarios.base import BaseScenario
from triceratops.scenarios.constants import (
    LN2PI,
    MAIN_SEQUENCE_LOGG_MIN,
    MAIN_SEQUENCE_TEFF_MAX,
)
from triceratops.scenarios.kernels import build_transit_mask
from triceratops.stellar.relations import StellarRelations

_ln2pi = LN2PI
_relations = StellarRelations()

# Fixed logg for evolved star scenarios (subgiant assumption).
_EVOLVED_LOGG: float = 3.0


def _filter_trilegal_by_tmag(
    population: TRILEGALResult,
    target_tmag: float,
) -> dict[str, np.ndarray]:
    """Filter TRILEGAL population to stars with Tmag in [target-1, target+1].

    Source: marginal_likelihoods.py:3713-3721
    """
    mask = (target_tmag - 1 < population.tmags) & (population.tmags < target_tmag + 1)
    masses = population.masses[mask]
    loggs = population.loggs[mask]
    teffs = population.teffs[mask]
    zs = population.metallicities[mask]
    radii = np.sqrt(CONST.G * masses * CONST.Msun / 10**loggs) / CONST.Rsun

    return {
        "tmags": population.tmags[mask],
        "masses": masses,
        "loggs": loggs,
        "teffs": teffs,
        "zs": zs,
        "radii": radii,
    }


class NTPUnknownScenario(BaseScenario):
    """Transiting Planet on a Nearby star of Unknown identity.

    Target star properties drawn from TRILEGAL peers with Tmag +/- 1.
    Extra mask: logg >= 3.5 and Teff <= 10000 (main-sequence filter).
    No external light curve support.

    BUG-03 fix: uses config.lnz_const via BaseScenario (not hardcoded 600).

    Source: marginal_likelihoods.py:3672-3861
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.NTP

    @property
    def is_eb(self) -> bool:
        return False

    def _resolve_external_lc_ldcs(
        self, external_lcs: list[ExternalLightCurve],
        stellar_params: StellarParameters,
    ) -> list[ExternalLightCurve]:
        return []

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Placeholder LDC; actual per-star LDC computed in _evaluate_lnL."""
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, stellar_params.logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:3699-3790"""
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError("NTPUnknownScenario requires 'trilegal_population' in kwargs.")
        target_tmag: float = kwargs.get("target_tmag", 10.0)  # type: ignore[assignment]

        filtered = _filter_trilegal_by_tmag(population, target_tmag)
        n_possible = len(filtered["masses"])
        if n_possible == 0:
            raise ValueError(
                f"No TRILEGAL stars found with Tmag in "
                f"[{target_tmag - 1:.1f}, {target_tmag + 1:.1f}]"
            )

        idxs = np.random.randint(0, n_possible, size=n)

        # Per-star LDC via bulk lookup
        u1s_possible, u2s_possible = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            config.mission,
            filtered["teffs"],
            filtered["loggs"],
            filtered["zs"],
        )

        # Sample planet priors using per-sample host mass
        rps = sample_planet_radius(
            np.random.rand(n), filtered["masses"][idxs], config.flat_priors,
        )
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(
            np.random.rand(n), planet=True, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        return {
            "rps": rps, "incs": incs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "idxs": idxs,
            "masses_possible": filtered["masses"],
            "radii_possible": filtered["radii"],
            "teffs_possible": filtered["teffs"],
            "loggs_possible": filtered["loggs"],
            "u1s_possible": u1s_possible,
            "u2s_possible": u2s_possible,
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:3783-3795"""
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        rps = samples["rps"]
        idxs = samples["idxs"].astype(int)
        masses = samples["masses_possible"][idxs]
        radii = samples["radii_possible"][idxs]

        R_p_rsun = rps * (CONST.Rearth / CONST.Rsun)

        # Uses per-sample host mass for Kepler's 3rd law
        a = semi_major_axis(P_orb, masses)
        Ptra = transit_probability(a, radii, R_p_rsun, eccs, argps)
        b = impact_parameter(a, incs, radii, eccs, argps)
        coll = collision_check(a, radii, R_p_rsun, eccs)

        return {"a": a, "Ptra": Ptra, "b": b, "coll": coll}

    def _evaluate_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, None]:
        """Source: marginal_likelihoods.py:3797-3836"""
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        idxs = samples["idxs"].astype(int)
        loggs = samples["loggs_possible"][idxs]
        teffs = samples["teffs_possible"][idxs]

        # Extra mask: main-sequence filter (logg >= 3.5, Teff <= 10000)
        ms_mask = (loggs >= MAIN_SEQUENCE_LOGG_MIN) & (teffs <= MAIN_SEQUENCE_TEFF_MAX)
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
            extra_mask=ms_mask,
        )

        R_s_arr = samples["radii_possible"][idxs]
        u1_arr = samples["u1s_possible"][idxs]
        u2_arr = samples["u2s_possible"][idxs]
        companion_fr = np.zeros(N)

        chi2_half = lnL_planet_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rps=samples["rps"],
            periods=samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a"],
            rss=R_s_arr,
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        return lnL, None

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Source: marginal_likelihoods.py:3844-3861"""
        n = len(idx)
        idxs = samples["idxs"].astype(int)

        return ScenarioResult(
            scenario_id=ScenarioID.NTP,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["masses_possible"][idxs[idx]],
            host_radius_rsun=samples["radii_possible"][idxs[idx]],
            host_u1=samples["u1s_possible"][idxs[idx]],
            host_u2=samples["u2s_possible"][idxs[idx]],
            period_days=samples["P_orb"][idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry["b"][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=samples["rps"][idx],
            eb_mass_msun=np.zeros(n),
            eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n),
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )


class NEBUnknownScenario(BaseScenario):
    """Eclipsing Binary on a Nearby star of Unknown identity.

    Returns (result, result_twin).

    BUG-03 fix: uses config.lnz_const via BaseScenario.

    Source: marginal_likelihoods.py:3864-4145
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.NEB

    @property
    def is_eb(self) -> bool:
        return True

    def _resolve_external_lc_ldcs(
        self, external_lcs: list[ExternalLightCurve],
        stellar_params: StellarParameters,
    ) -> list[ExternalLightCurve]:
        return []

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, stellar_params.logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:3892-3986"""
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError("NEBUnknownScenario requires 'trilegal_population' in kwargs.")
        target_tmag: float = kwargs.get("target_tmag", 10.0)  # type: ignore[assignment]

        filtered = _filter_trilegal_by_tmag(population, target_tmag)
        n_possible = len(filtered["masses"])
        if n_possible == 0:
            raise ValueError(
                f"No TRILEGAL stars found with Tmag in "
                f"[{target_tmag - 1:.1f}, {target_tmag + 1:.1f}]"
            )

        idxs = np.random.randint(0, n_possible, size=n)

        # Per-star LDC
        u1s_possible, u2s_possible = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            config.mission,
            filtered["teffs"],
            filtered["loggs"],
            filtered["zs"],
        )

        # EB sampling -- qs drawn relative to M_s=1.0 per original (line 3903)
        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), 1.0)
        eccs = sample_eccentricity(
            np.random.rand(n), planet=False, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        # EB companion properties using per-sample host
        masses_host = filtered["masses"][idxs]
        radii_host = filtered["radii"][idxs]
        teffs_host = filtered["teffs"][idxs]

        masses = qs * masses_host
        radii, _teffs = _relations.get_radius_teff(
            masses,
            max_radii=radii_host,
            max_teffs=teffs_host,
        )
        fluxratios = (
            _relations.get_flux_ratio(masses, "TESS")
            / (_relations.get_flux_ratio(masses, "TESS")
               + _relations.get_flux_ratio(masses_host, "TESS"))
        )

        return {
            "incs": incs, "qs": qs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "idxs": idxs,
            "masses_possible": filtered["masses"],
            "radii_possible": filtered["radii"],
            "teffs_possible": filtered["teffs"],
            "loggs_possible": filtered["loggs"],
            "u1s_possible": u1s_possible,
            "u2s_possible": u2s_possible,
            "masses": masses,
            "radii": radii,
            "fluxratios": fluxratios,
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:3988-4009"""
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        masses = samples["masses"]
        radii = samples["radii"]
        idxs = samples["idxs"].astype(int)
        masses_host = samples["masses_possible"][idxs]
        radii_host = samples["radii_possible"][idxs]

        a = semi_major_axis(P_orb, masses_host + masses)
        Ptra = transit_probability(a, radii_host, radii, eccs, argps)
        b = impact_parameter(a, incs, radii_host, eccs, argps)
        coll = collision_check(a, radii_host, radii, eccs)

        a_twin = semi_major_axis(2 * P_orb, masses_host + masses)
        Ptra_twin = transit_probability(a_twin, radii_host, radii, eccs, argps)
        b_twin = impact_parameter(a_twin, incs, radii_host, eccs, argps)
        # Twin collision: 2*R_host (per original line 4009)
        coll_twin = (2 * radii_host * CONST.Rsun) > a_twin * (1 - eccs)

        return {
            "a": a, "Ptra": Ptra, "b": b, "coll": coll,
            "a_twin": a_twin, "Ptra_twin": Ptra_twin,
            "b_twin": b_twin, "coll_twin": coll_twin,
        }

    def _evaluate_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source: marginal_likelihoods.py:4011-4095"""
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        qs = samples["qs"]
        idxs = samples["idxs"].astype(int)
        loggs = samples["loggs_possible"][idxs]
        teffs = samples["teffs_possible"][idxs]

        R_s_arr = samples["radii_possible"][idxs]
        u1_arr = samples["u1s_possible"][idxs]
        u2_arr = samples["u2s_possible"][idxs]
        companion_fr = np.zeros(N)

        lnL = np.full(N, -np.inf)
        lnL_twin = np.full(N, -np.inf)

        # Extra mask: main-sequence filter
        ms_mask = (loggs >= MAIN_SEQUENCE_LOGG_MIN) & (teffs <= MAIN_SEQUENCE_TEFF_MAX)

        mask, mask_twin = build_eb_branch_masks(
            qs, samples["incs"],
            geometry["Ptra"], geometry["coll"],
            geometry["Ptra_twin"], geometry["coll_twin"],
            extra_mask=ms_mask,
        )

        chi2_half = lnL_eb_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=R_s_arr,
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a"],
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        chi2_half_twin = lnL_eb_twin_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=R_s_arr,
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=2 * samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a_twin"],
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask_twin,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL_twin = -0.5 * _ln2pi - lnsigma - chi2_half_twin

        return lnL, lnL_twin

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Source: marginal_likelihoods.py:4104-4145"""
        n = len(idx)
        sid = ScenarioID.NEBX2P if twin else ScenarioID.NEB
        b_key = "b_twin" if twin else "b"
        P_orb = (2 * samples["P_orb"] if twin else samples["P_orb"])
        idxs = samples["idxs"].astype(int)

        return ScenarioResult(
            scenario_id=sid,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["masses_possible"][idxs[idx]],
            host_radius_rsun=samples["radii_possible"][idxs[idx]],
            host_u1=samples["u1s_possible"][idxs[idx]],
            host_u2=samples["u2s_possible"][idxs[idx]],
            period_days=P_orb[idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry[b_key][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=samples["masses"][idx],
            eb_radius_rsun=samples["radii"][idx],
            flux_ratio_eb_tess=samples["fluxratios"][idx],
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )


class NTPEvolvedScenario(BaseScenario):
    """Transiting Planet on a Nearby Evolved (subgiant) star.

    logg fixed at 3.0. Mass derived from logg and R_s.
    No external light curve support.

    BUG-03 fix: uses config.lnz_const via BaseScenario.
    BUG-05 fix: R_s stored as np.full(n, R_s) array, not scalar.

    Source: marginal_likelihoods.py:4148-4285
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.NTP

    @property
    def is_eb(self) -> bool:
        return False

    def _resolve_external_lc_ldcs(
        self, external_lcs: list[ExternalLightCurve],
        stellar_params: StellarParameters,
    ) -> list[ExternalLightCurve]:
        return []

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """LDC lookup with logg forced to 3.0 (subgiant assumption)."""
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, _EVOLVED_LOGG,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:4209-4213

        BUG-05 fix: R_s is np.full(n, R_s), not a scalar.
        Mass derived: M_s = 10^logg * R_s^2 * Rsun^2 / (G * Msun)
        """
        R_s = stellar_params.radius_rsun
        M_s = (10**_EVOLVED_LOGG) * (R_s * CONST.Rsun) ** 2 / CONST.G / CONST.Msun

        rps = sample_planet_radius(
            np.random.rand(n), M_s, config.flat_priors,
        )
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(
            np.random.rand(n), planet=True, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        return {
            "rps": rps, "incs": incs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, M_s),
            "R_s": np.full(n, R_s),  # BUG-05 fix: array, not scalar
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:4215-4225"""
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        rps = samples["rps"]
        M_s = samples["M_s"][0]
        R_s = stellar_params.radius_rsun
        R_p_rsun = rps * (CONST.Rearth / CONST.Rsun)

        a = semi_major_axis(P_orb, M_s)
        Ptra = transit_probability(a, R_s, R_p_rsun, eccs, argps)
        b = impact_parameter(a, incs, R_s, eccs, argps)
        coll = collision_check(a, R_s, R_p_rsun, eccs)

        return {"a": a, "Ptra": Ptra, "b": b, "coll": coll}

    def _evaluate_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, None]:
        """Source: marginal_likelihoods.py:4227-4260"""
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
        )

        R_s_arr = samples["R_s"]  # BUG-05 fix: already an array
        u1_arr = np.full(N, float(ldc.u1))
        u2_arr = np.full(N, float(ldc.u2))
        companion_fr = np.zeros(N)

        chi2_half = lnL_planet_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rps=samples["rps"],
            periods=samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a"],
            rss=R_s_arr,
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        return lnL, None

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Source: marginal_likelihoods.py:4268-4285"""
        n = len(idx)

        return ScenarioResult(
            scenario_id=ScenarioID.NTP,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["M_s"][:n],
            host_radius_rsun=samples["R_s"][:n],
            host_u1=np.full(n, float(ldc.u1)),
            host_u2=np.full(n, float(ldc.u2)),
            period_days=samples["P_orb"][idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry["b"][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=samples["rps"][idx],
            eb_mass_msun=np.zeros(n),
            eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n),
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )


class NEBEvolvedScenario(BaseScenario):
    """Eclipsing Binary on a Nearby Evolved (subgiant) star.

    Returns (result, result_twin).

    BUG-03 fix: uses config.lnz_const via BaseScenario.
    BUG-05 fix: radii passed to lnL_EB_twin_p are arrays, not scalar R_s.

    Source: marginal_likelihoods.py:4288-4503
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.NEB

    @property
    def is_eb(self) -> bool:
        return True

    def _resolve_external_lc_ldcs(
        self, external_lcs: list[ExternalLightCurve],
        stellar_params: StellarParameters,
    ) -> list[ExternalLightCurve]:
        return []

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, _EVOLVED_LOGG,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:4350-4364

        BUG-05 fix: R_s stored as array.
        Note: original uses qs = sample_q(rand, 1.0) -- mass ratio drawn
        relative to M_s=1.0 (line 4352). We preserve this behavior.
        """
        R_s = stellar_params.radius_rsun
        Teff = stellar_params.teff_k
        M_s = (10**_EVOLVED_LOGG) * (R_s * CONST.Rsun) ** 2 / CONST.G / CONST.Msun

        incs = sample_inclination(np.random.rand(n))
        # Original: sample_q(rand, 1.0) -- not M_s
        qs = sample_mass_ratio(np.random.rand(n), 1.0)
        eccs = sample_eccentricity(
            np.random.rand(n), planet=False, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        masses = qs * M_s
        radii, _teffs = _relations.get_radius_teff(
            masses,
            max_radii=np.full(n, R_s),
            max_teffs=np.full(n, Teff),
        )
        fluxratios = (
            _relations.get_flux_ratio(masses, "TESS")
            / (_relations.get_flux_ratio(masses, "TESS")
               + _relations.get_flux_ratio(np.array([M_s]), "TESS"))
        )

        return {
            "incs": incs, "qs": qs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, M_s),
            "R_s": np.full(n, R_s),  # BUG-05 fix: array
            "masses": masses,
            "radii": radii,
            "fluxratios": fluxratios,
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Source: marginal_likelihoods.py:4366-4381"""
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        masses = samples["masses"]
        radii = samples["radii"]

        M_s = samples["M_s"][0]
        R_s = stellar_params.radius_rsun

        a = semi_major_axis(P_orb, M_s + masses)
        Ptra = transit_probability(a, R_s, radii, eccs, argps)
        b = impact_parameter(a, incs, R_s, eccs, argps)
        coll = collision_check(a, R_s, radii, eccs)

        a_twin = semi_major_axis(2 * P_orb, M_s + masses)
        # Original line 4371: Ptra_twin uses (R_s + R_s), not (radii + R_s)
        Ptra_twin = transit_probability(a_twin, R_s, R_s, eccs, argps)
        b_twin = impact_parameter(a_twin, incs, R_s, eccs, argps)
        coll_twin = (2 * R_s * CONST.Rsun) > a_twin * (1 - eccs)

        return {
            "a": a, "Ptra": Ptra, "b": b, "coll": coll,
            "a_twin": a_twin, "Ptra_twin": Ptra_twin,
            "b_twin": b_twin, "coll_twin": coll_twin,
        }

    def _evaluate_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source: marginal_likelihoods.py:4383-4453

        BUG-05 fix: lnL_EB_twin_p receives radii[mask] (array), not scalar R_s.
        Original line 4419 passes scalar R_s where radii[mask] is needed.
        """
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        qs = samples["qs"]
        R_s_arr = samples["R_s"]  # BUG-05 fix: array
        u1_arr = np.full(N, float(ldc.u1))
        u2_arr = np.full(N, float(ldc.u2))
        companion_fr = np.zeros(N)

        lnL = np.full(N, -np.inf)
        lnL_twin = np.full(N, -np.inf)

        mask, mask_twin = build_eb_branch_masks(
            qs, samples["incs"],
            geometry["Ptra"], geometry["coll"],
            geometry["Ptra_twin"], geometry["coll_twin"],
        )

        chi2_half = lnL_eb_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=R_s_arr,
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a"],
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        # BUG-05 fix: pass radii (array) not scalar R_s
        chi2_half_twin = lnL_eb_twin_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=R_s_arr,
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=2 * samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a_twin"],
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=companion_fr,
            mask=mask_twin,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL_twin = -0.5 * _ln2pi - lnsigma - chi2_half_twin

        return lnL, lnL_twin

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Source: marginal_likelihoods.py:4462-4503"""
        n = len(idx)
        sid = ScenarioID.NEBX2P if twin else ScenarioID.NEB
        b_key = "b_twin" if twin else "b"
        P_orb = (2 * samples["P_orb"] if twin else samples["P_orb"])

        # Note: original twin result (line 4498) stores R_EB = R_s, not radii
        R_EB = (np.full(n, stellar_params.radius_rsun) if twin
                else samples["radii"][idx])

        return ScenarioResult(
            scenario_id=sid,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["M_s"][:n],
            host_radius_rsun=samples["R_s"][:n],
            host_u1=np.full(n, float(ldc.u1)),
            host_u2=np.full(n, float(ldc.u2)),
            period_days=P_orb[idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry[b_key][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=samples["masses"][idx],
            eb_radius_rsun=R_EB,
            flux_ratio_eb_tess=samples["fluxratios"][idx],
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )
