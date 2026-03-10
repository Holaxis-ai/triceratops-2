"""TTP and TEB scenario implementations (target star).

Source: marginal_likelihoods.py:106-333 (lnZ_TTP) and 336-671 (lnZ_TEB).
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
)
from triceratops.scenarios.kernels import build_transit_mask
from triceratops.stellar.relations import StellarRelations

_ln2pi = LN2PI
_relations = StellarRelations()


class TTPScenario(BaseScenario):
    """True Transiting Planet on the target star.

    Source: marginal_likelihoods.py:106-333
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.TP

    @property
    def is_eb(self) -> bool:
        return False

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        logg = self._stellar_logg_from_mass_radius(stellar_params)
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        rps = sample_planet_radius(
            np.random.rand(n), stellar_params.mass_msun, config.flat_priors,
        )
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(
            np.random.rand(n), planet=True, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))
        return {
            "rps": rps, "incs": incs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, stellar_params.mass_msun),
            "R_s": np.full(n, stellar_params.radius_rsun),
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        rps = samples["rps"]

        M_s = stellar_params.mass_msun
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
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
        )

        R_s_arr = samples["R_s"]
        u1_arr = np.full(N, float(ldc.u1))
        u2_arr = np.full(N, float(ldc.u2))
        companion_fr = np.zeros(N)

        # Main LC
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

        # External LCs
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            ext_u1 = np.full(N, float(ext_lc.ldc.u1))
            ext_u2 = np.full(N, float(ext_lc.ldc.u2))
            elc = ext_lc.light_curve
            ext_chi2 = lnL_planet_p(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rps=samples["rps"],
                periods=samples["P_orb"],
                incs=samples["incs"],
                as_=geometry["a"],
                rss=R_s_arr,
                u1s=ext_u1,
                u2s=ext_u2,
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=companion_fr,
                mask=mask,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        return lnL, None

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        n = len(idx)
        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        return ScenarioResult(
            scenario_id=ScenarioID.TP,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.full(n, stellar_params.mass_msun),
            host_radius_rsun=np.full(n, stellar_params.radius_rsun),
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
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
        )


class TEBScenario(BaseScenario):
    """True Eclipsing Binary on the target star.

    Returns (result, result_twin) -- twin is the q>=0.95 half-period alias.

    Source: marginal_likelihoods.py:336-671
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.EB

    @property
    def is_eb(self) -> bool:
        return True

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        logg = self._stellar_logg_from_mass_radius(stellar_params)
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), stellar_params.mass_msun)
        eccs = sample_eccentricity(
            np.random.rand(n), planet=False, period=np.mean(P_orb),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Teff = stellar_params.teff_k

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
            "R_s": np.full(n, R_s),
            "masses": masses,
            "radii": radii,
            "fluxratios": fluxratios,
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        masses = samples["masses"]
        radii = samples["radii"]

        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun

        # Standard period geometry
        a = semi_major_axis(P_orb, M_s + masses)
        Ptra = transit_probability(a, R_s, radii, eccs, argps)
        b = impact_parameter(a, incs, R_s, eccs, argps)
        coll = collision_check(a, R_s, radii, eccs)

        # Twin (2x period) geometry
        a_twin = semi_major_axis(2 * P_orb, M_s + masses)
        Ptra_twin = transit_probability(a_twin, R_s, radii, eccs, argps)
        b_twin = impact_parameter(a_twin, incs, R_s, eccs, argps)
        # Original uses (2*R_s) for twin collision check
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
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        qs = samples["qs"]
        R_s_arr = samples["R_s"]
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

        # External LCs for q < 0.95
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            elc = ext_lc.light_curve
            ext_u1 = np.full(N, float(ext_lc.ldc.u1))
            ext_u2 = np.full(N, float(ext_lc.ldc.u2))
            # Flux ratios in the external band
            ext_fr = (
                _relations.get_flux_ratio(samples["masses"], ext_lc.band)
                / (_relations.get_flux_ratio(samples["masses"], ext_lc.band)
                   + _relations.get_flux_ratio(
                       np.array([float(samples["M_s"][0])]), ext_lc.band))
            )
            ext_chi2 = lnL_eb_p(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rss=R_s_arr,
                rcomps=samples["radii"],
                eb_flux_ratios=ext_fr,
                periods=samples["P_orb"],
                incs=samples["incs"],
                as_=geometry["a"],
                u1s=ext_u1,
                u2s=ext_u2,
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=companion_fr,
                mask=mask,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

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

        # External LCs for q >= 0.95
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            elc = ext_lc.light_curve
            ext_u1 = np.full(N, float(ext_lc.ldc.u1))
            ext_u2 = np.full(N, float(ext_lc.ldc.u2))
            ext_fr = (
                _relations.get_flux_ratio(samples["masses"], ext_lc.band)
                / (_relations.get_flux_ratio(samples["masses"], ext_lc.band)
                   + _relations.get_flux_ratio(
                       np.array([float(samples["M_s"][0])]), ext_lc.band))
            )
            ext_chi2_twin = lnL_eb_twin_p(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rss=R_s_arr,
                rcomps=samples["radii"],
                eb_flux_ratios=ext_fr,
                periods=2 * samples["P_orb"],
                incs=samples["incs"],
                as_=geometry["a_twin"],
                u1s=ext_u1,
                u2s=ext_u2,
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=companion_fr,
                mask=mask_twin,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL_twin = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2_twin
            lnL_twin = lnL_twin + ext_lnL_twin

        return lnL, lnL_twin

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        n = len(idx)
        sid = ScenarioID.EBX2P if twin else ScenarioID.EB
        b_key = "b_twin" if twin else "b"
        P_orb = (2 * samples["P_orb"] if twin else samples["P_orb"])

        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        ext_fr_eb = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            fr = (
                _relations.get_flux_ratio(samples["masses"][idx], ext_lc.band)
                / (_relations.get_flux_ratio(samples["masses"][idx], ext_lc.band)
                   + _relations.get_flux_ratio(
                       np.array([stellar_params.mass_msun]), ext_lc.band))
            )
            ext_fr_eb.append(fr)

        return ScenarioResult(
            scenario_id=sid,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.full(n, stellar_params.mass_msun),
            host_radius_rsun=np.full(n, stellar_params.radius_rsun),
            host_u1=np.full(n, float(ldc.u1)),
            host_u2=np.full(n, float(ldc.u2)),
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
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_eb=ext_fr_eb,
        )
