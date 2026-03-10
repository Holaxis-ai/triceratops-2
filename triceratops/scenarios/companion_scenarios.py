"""PTP, PEB, STP, and SEB scenario implementations.

PTP: Planet on a Physically-bound companion star.
PEB: Eclipsing Binary involving a Physically-bound companion.
STP: Planet transiting a Sibling (companion-hosted) star.
SEB: Eclipsing Binary on a Sibling (companion) star.

All scenarios add companion star sampling via sample_companion_mass_ratio
(or pre-loaded MoluscData), optional contrast curve prior, and flux dilution from
the unresolved companion.

Source: marginal_likelihoods.py:674-913 (PTP), 916-1294 (PEB),
        1296-1614 (STP), 1617-2056 (SEB).

BUG-07 fix (commit ac244189): companion scenario compute() overrides now rank
best-fit samples by lnL + lnprior_comp (posterior score) rather than raw lnL.
The original TRICERATOPS-PLUS code used raw lnL for sample selection while
correctly using lnL + lnprior_companion for evidence. See
working_docs/Original_bugs/BUG-07_companion_best_sample_ranking.md.
"""
# ruff: noqa: ARG002  -- ABC override signatures require unused params
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from triceratops.config.config import CONST, Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.likelihoods.geometry import (
    impact_parameter,
    semi_major_axis,
)
from triceratops.likelihoods.lnl_functions import (
    lnL_eb_p,
    lnL_eb_twin_p,
    lnL_planet_p,
)
from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_companion_mass_ratio,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
    sample_planet_radius,
)
from triceratops.scenarios._companion_helpers import (
    _bulk_companion_ldc,
    _compute_companion_prior,
    _compute_companion_properties,
    _compute_seb_companion_prior,
    _flux_ratio_in_band,
    _ln2pi,
    _load_molusc_qs,
    _relations,
)
from triceratops.scenarios._eb_branching import build_eb_branch_masks
from triceratops.scenarios.base import BaseScenario
from triceratops.scenarios.kernels import (
    build_transit_mask,
    compute_lnZ,
    pack_best_indices,
    resolve_period,
)


class PTPScenario(BaseScenario):
    """Planet on a Physically-Bound Companion star.

    The planet orbits the target star, but an unresolved companion dilutes
    the observed transit. The companion's presence affects the transit depth.

    Source: marginal_likelihoods.py:674-913
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.PTP

    @property
    def is_eb(self) -> bool:
        return False

    @property
    def returns_twin(self) -> bool:
        return False

    def _get_host_ldc(
        self,
        stellar_params: StellarParameters,
        mission: str,
        P_orb: np.ndarray,
        kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """LDC for target star. Source: marginal_likelihoods.py:724-733."""
        logg = self._stellar_logg_from_mass_radius(stellar_params)
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission,
            stellar_params.metallicity_dex,
            stellar_params.teff_k,
            logg,
        )

    def _sample_priors(
        self,
        n: int,
        stellar_params: StellarParameters,
        P_orb: np.ndarray,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample PTP priors including companion mass ratio.

        Source: marginal_likelihoods.py:788-822.
        """
        molusc_data = kwargs.get("molusc_data")
        M_s = stellar_params.mass_msun

        if molusc_data is not None:
            qs_comp = _load_molusc_qs(molusc_data, n, M_s)
        else:
            qs_comp = sample_companion_mass_ratio(np.random.rand(n), M_s)

        masses_comp, radii_comp, _Teffs_comp, fluxratios_comp = (
            _compute_companion_properties(
                qs_comp, M_s, stellar_params.radius_rsun,
                stellar_params.teff_k, n,
            )
        )

        rps = sample_planet_radius(np.random.rand(n), M_s, config.flat_priors)
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(np.random.rand(n), planet=True, period=P_orb)
        argps = sample_arg_periastron(np.random.rand(n))

        return {
            "rps": rps,
            "incs": incs,
            "eccs": eccs,
            "argps": argps,
            "qs_comp": qs_comp,
            "masses_comp": masses_comp,
            "radii_comp": radii_comp,
            "fluxratios_comp": fluxratios_comp,
        }

    def _compute_orbital_geometry(
        self,
        samples: dict[str, np.ndarray],
        P_orb: np.ndarray,
        stellar_params: StellarParameters,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Compute PTP orbital geometry.

        Source: marginal_likelihoods.py:721, 824-830.
        Semi-major axis uses M_s alone (line 721).
        Transit probability uses (Rp + R_s) (line 828).
        """
        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Rearth = CONST.Rearth
        Rsun = CONST.Rsun

        rps = samples["rps"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        incs = samples["incs"]

        a = semi_major_axis(P_orb, M_s)

        e_corr = (1 + eccs * np.sin(argps * np.pi / 180)) / (1 - eccs**2)
        ptra = (rps * Rearth + R_s * Rsun) / a * e_corr
        b = impact_parameter(a, incs, R_s, eccs, argps)
        coll = (rps * Rearth + R_s * Rsun) > a * (1 - eccs)

        return {"a": a, "ptra": ptra, "b": b, "coll": coll}

    def _evaluate_lnL(
        self,
        light_curve: LightCurve,
        lnsigma: float,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, None]:
        """Evaluate PTP log-likelihoods.

        Source: marginal_likelihoods.py:834-864.
        Uses companion_is_host=False (same as original line 848).
        """
        N = config.n_mc_samples
        rps = samples["rps"]
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        qs_comp = samples["qs_comp"]
        fluxratios_comp = samples["fluxratios_comp"]
        R_s_val = samples["R_s"][0]

        a = geometry["a"]
        ptra = geometry["ptra"]
        coll = geometry["coll"]
        P_orb = samples["P_orb"]

        time = light_curve.time_days
        flux = light_curve.flux
        sigma = light_curve.sigma

        extra_mask = qs_comp != 0.0
        mask = build_transit_mask(incs, ptra, coll, extra_mask=extra_mask)

        lnL = np.full(N, -np.inf)
        force_serial = (not config.parallel) and not bool(external_lcs)

        if np.any(mask):
            R_s_arr = np.full(N, R_s_val)
            u1_arr = np.full(N, ldc.u1)
            u2_arr = np.full(N, ldc.u2)

            # lnL_planet_p returns full-length array (inf for non-masked)
            chi2_half = lnL_planet_p(
                time, flux, sigma,
                rps, P_orb, incs, a, R_s_arr,
                u1_arr, u2_arr,
                eccs, argps,
                fluxratios_comp,
                mask,
                companion_is_host=False,
                exptime=light_curve.cadence_days,
                nsamples=light_curve.supersampling_rate,
                force_serial=force_serial,
            )
            lnL = -0.5 * _ln2pi - lnsigma - chi2_half

            for ext_lc in external_lcs:
                if ext_lc.ldc is None:
                    continue
                elc = ext_lc.light_curve
                ext_u1 = np.full(N, float(ext_lc.ldc.u1))
                ext_u2 = np.full(N, float(ext_lc.ldc.u2))
                ext_fluxratios_comp = _flux_ratio_in_band(
                    samples["masses_comp"],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
                ext_chi2 = lnL_planet_p(
                    time=elc.time_days,
                    flux=elc.flux,
                    sigma=elc.sigma,
                    rps=rps,
                    periods=P_orb,
                    incs=incs,
                    as_=a,
                    rss=R_s_arr,
                    u1s=ext_u1,
                    u2s=ext_u2,
                    eccs=eccs,
                    argps=argps,
                    companion_flux_ratios=ext_fluxratios_comp,
                    mask=mask,
                    companion_is_host=False,
                    exptime=elc.cadence_days,
                    nsamples=elc.supersampling_rate,
                    force_serial=force_serial,
                )
                ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
                lnL = lnL + ext_lnL

        return lnL, None

    def _pack_result(
        self,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        lnZ: float,
        idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve],
        twin: bool = False,
    ) -> ScenarioResult:
        """Pack PTP result. Source: marginal_likelihoods.py:886-913."""
        n = len(idx)
        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        ext_fr_comp = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            ext_fr_comp.append(
                _flux_ratio_in_band(
                    samples["masses_comp"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )
        return ScenarioResult(
            scenario_id=self.scenario_id,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.full(n, stellar_params.mass_msun),
            host_radius_rsun=np.full(n, stellar_params.radius_rsun),
            host_u1=np.full(n, ldc.u1),
            host_u2=np.full(n, ldc.u2),
            period_days=samples["P_orb"][idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry["b"][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=samples["rps"][idx],
            eb_mass_msun=np.zeros(n),
            eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n),
            companion_mass_msun=samples["masses_comp"][idx],
            companion_radius_rsun=samples["radii_comp"][idx],
            flux_ratio_companion_tess=samples["fluxratios_comp"][idx],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> ScenarioResult:
        """Override compute to add lnprior_companion to lnZ computation.

        In PTP, lnprior_companion is added to lnL before computing Z
        (source: line 883).
        """
        N = config.n_mc_samples
        P_orb = resolve_period(period_days, N)

        lnsigma = np.log(light_curve.sigma)
        ldc = self._get_host_ldc(stellar_params, config.mission, P_orb, dict(kwargs))

        resolved_ext_lcs = (
            self._resolve_external_lc_ldcs(external_lcs, stellar_params)
            if external_lcs else []
        )

        samples = self._sample_priors(N, stellar_params, P_orb, config, **kwargs)
        samples["M_s"] = np.full(N, stellar_params.mass_msun)
        samples["R_s"] = np.full(N, stellar_params.radius_rsun)
        samples["P_orb"] = P_orb

        # Compute companion prior
        molusc_data = kwargs.get("molusc_data")
        contrast_curve = kwargs.get("contrast_curve")
        filt = str(kwargs.get("filt", "TESS"))
        lnprior_comp = _compute_companion_prior(
            samples["masses_comp"], samples["fluxratios_comp"],
            stellar_params.mass_msun, stellar_params.parallax_mas, N,
            molusc_data, contrast_curve, filt, is_eb=False,
        )

        geometry = self._compute_orbital_geometry(
            samples, P_orb, stellar_params, config, **kwargs
        )

        lnL, _ = self._evaluate_lnL(
            light_curve, lnsigma, samples, geometry, ldc,
            resolved_ext_lcs, config,
        )

        # Add companion prior before computing Z (source: line 883)
        lnZ = compute_lnZ(lnL + lnprior_comp, config.lnz_const)

        n_best = config.n_best_samples
        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx = pack_best_indices(lnL + lnprior_comp, n_best)
        return self._pack_result(
            samples, geometry, ldc, lnZ, idx, stellar_params,
            resolved_ext_lcs,
        )


class PEBScenario(BaseScenario):
    """Eclipsing Binary involving a Physically-Bound Companion.

    Returns (result, result_twin) where result_twin is the q>=0.95 case.

    Source: marginal_likelihoods.py:916-1294
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.PEB

    @property
    def is_eb(self) -> bool:
        return True

    def _get_host_ldc(
        self,
        stellar_params: StellarParameters,
        mission: str,
        P_orb: np.ndarray,
        kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """LDC for target star. Source: marginal_likelihoods.py:968-978."""
        logg = self._stellar_logg_from_mass_radius(stellar_params)
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission,
            stellar_params.metallicity_dex,
            stellar_params.teff_k,
            logg,
        )

    def _sample_priors(
        self,
        n: int,
        stellar_params: StellarParameters,
        P_orb: np.ndarray,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample PEB priors.

        Source: marginal_likelihoods.py:1034-1071.
        """
        molusc_data = kwargs.get("molusc_data")
        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun

        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), M_s)
        eccs = sample_eccentricity(np.random.rand(n), planet=False, period=P_orb)
        argps = sample_arg_periastron(np.random.rand(n))

        if molusc_data is not None:
            qs_comp = _load_molusc_qs(molusc_data, n, M_s)
        else:
            qs_comp = sample_companion_mass_ratio(np.random.rand(n), M_s)

        # EB properties (source: lines 1051-1060)
        masses_eb = qs * M_s
        radii_eb, _Teffs_eb = _relations.get_radius_teff(
            masses_eb, np.full(n, R_s), np.full(n, stellar_params.teff_k),
        )
        flux_eb = _relations.get_flux_ratio(masses_eb, "TESS")
        flux_primary = _relations.get_flux_ratio(np.array([M_s]), "TESS")
        fluxratios_eb = flux_eb / (flux_eb + flux_primary)

        # Companion properties (source: lines 1062-1071)
        masses_comp, radii_comp, _Teffs_comp, fluxratios_comp = (
            _compute_companion_properties(
                qs_comp, M_s, R_s, stellar_params.teff_k, n,
            )
        )

        return {
            "incs": incs,
            "qs": qs,
            "eccs": eccs,
            "argps": argps,
            "qs_comp": qs_comp,
            "masses_eb": masses_eb,
            "radii_eb": radii_eb,
            "fluxratios_eb": fluxratios_eb,
            "masses_comp": masses_comp,
            "radii_comp": radii_comp,
            "fluxratios_comp": fluxratios_comp,
        }

    def _compute_orbital_geometry(
        self,
        samples: dict[str, np.ndarray],
        P_orb: np.ndarray,
        stellar_params: StellarParameters,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Compute PEB orbital geometry.

        Source: marginal_likelihoods.py:1106-1121.
        """
        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Rsun = CONST.Rsun

        masses_eb = samples["masses_eb"]
        radii_eb = samples["radii_eb"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        incs = samples["incs"]

        e_corr = (1 + eccs * np.sin(argps * np.pi / 180)) / (1 - eccs**2)

        # q < 0.95 path (source: line 1108)
        a = semi_major_axis(P_orb, M_s + masses_eb)
        ptra = (radii_eb * Rsun + R_s * Rsun) / a * e_corr

        # q >= 0.95 twin: double period (source: line 1110)
        a_twin = semi_major_axis(2 * P_orb, M_s + masses_eb)
        ptra_twin = (radii_eb * Rsun + R_s * Rsun) / a_twin * e_corr

        b = impact_parameter(a, incs, R_s, eccs, argps)
        b_twin = impact_parameter(a_twin, incs, R_s, eccs, argps)

        # Collisions (source: lines 1120-1121)
        coll = (radii_eb * Rsun + R_s * Rsun) > a * (1 - eccs)
        coll_twin = (2 * R_s * Rsun) > a_twin * (1 - eccs)

        return {
            "a": a,
            "a_twin": a_twin,
            "ptra": ptra,
            "ptra_twin": ptra_twin,
            "b": b,
            "b_twin": b_twin,
            "coll": coll,
            "coll_twin": coll_twin,
        }

    def _evaluate_lnL(
        self,
        light_curve: LightCurve,
        lnsigma: float,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate PEB log-likelihoods for q<0.95 and q>=0.95.

        Source: marginal_likelihoods.py:1126-1218.
        """
        N = config.n_mc_samples

        qs = samples["qs"]
        qs_comp = samples["qs_comp"]
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        radii_eb = samples["radii_eb"]
        fluxratios_eb = samples["fluxratios_eb"]
        fluxratios_comp = samples["fluxratios_comp"]
        R_s_val = samples["R_s"][0]
        P_orb = samples["P_orb"]

        a = geometry["a"]
        a_twin = geometry["a_twin"]
        ptra = geometry["ptra"]
        ptra_twin = geometry["ptra_twin"]
        coll = geometry["coll"]
        coll_twin = geometry["coll_twin"]

        time = light_curve.time_days
        flux = light_curve.flux
        sigma = light_curve.sigma

        lnL = np.full(N, -np.inf)
        lnL_twin = np.full(N, -np.inf)
        force_serial = (not config.parallel) and not bool(external_lcs)

        R_s_arr = np.full(N, R_s_val)
        u1_arr = np.full(N, ldc.u1)
        u2_arr = np.full(N, ldc.u2)

        # q < 0.95 / q >= 0.95 branching (source: lines 1127-1186)
        mask, mask_twin = build_eb_branch_masks(
            qs, incs, ptra, coll, ptra_twin, coll_twin,
            extra_mask=(qs_comp != 0.0),
        )

        if np.any(mask):
            # lnL_eb_p returns full-length array (inf for non-masked)
            chi2_half = lnL_eb_p(
                time, flux, sigma,
                R_s_arr, radii_eb, fluxratios_eb,
                P_orb, incs, a,
                u1_arr, u2_arr,
                eccs, argps,
                fluxratios_comp,
                mask,
                companion_is_host=False,
                exptime=light_curve.cadence_days,
                nsamples=light_curve.supersampling_rate,
                force_serial=force_serial,
            )
            lnL = -0.5 * _ln2pi - lnsigma - chi2_half

            for ext_lc in external_lcs:
                if ext_lc.ldc is None:
                    continue
                elc = ext_lc.light_curve
                ext_u1 = np.full(N, float(ext_lc.ldc.u1))
                ext_u2 = np.full(N, float(ext_lc.ldc.u2))
                ext_fluxratios_eb = _flux_ratio_in_band(
                    samples["masses_eb"],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
                ext_fluxratios_comp = _flux_ratio_in_band(
                    samples["masses_comp"],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
                ext_chi2 = lnL_eb_p(
                    time=elc.time_days,
                    flux=elc.flux,
                    sigma=elc.sigma,
                    rss=R_s_arr,
                    rcomps=radii_eb,
                    eb_flux_ratios=ext_fluxratios_eb,
                    periods=P_orb,
                    incs=incs,
                    as_=a,
                    u1s=ext_u1,
                    u2s=ext_u2,
                    eccs=eccs,
                    argps=argps,
                    companion_flux_ratios=ext_fluxratios_comp,
                    mask=mask,
                    companion_is_host=False,
                    exptime=elc.cadence_days,
                    nsamples=elc.supersampling_rate,
                    force_serial=force_serial,
                )
                ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
                lnL = lnL + ext_lnL

        if np.any(mask_twin):
            # lnL_eb_twin_p returns full-length array (inf for non-masked)
            chi2_half_twin = lnL_eb_twin_p(
                time, flux, sigma,
                R_s_arr, radii_eb, fluxratios_eb,
                2 * P_orb, incs, a_twin,
                u1_arr, u2_arr,
                eccs, argps,
                fluxratios_comp,
                mask_twin,
                companion_is_host=False,
                exptime=light_curve.cadence_days,
                nsamples=light_curve.supersampling_rate,
                force_serial=force_serial,
            )
            lnL_twin = -0.5 * _ln2pi - lnsigma - chi2_half_twin

            for ext_lc in external_lcs:
                if ext_lc.ldc is None:
                    continue
                elc = ext_lc.light_curve
                ext_u1 = np.full(N, float(ext_lc.ldc.u1))
                ext_u2 = np.full(N, float(ext_lc.ldc.u2))
                ext_fluxratios_eb = _flux_ratio_in_band(
                    samples["masses_eb"],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
                ext_fluxratios_comp = _flux_ratio_in_band(
                    samples["masses_comp"],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
                ext_chi2_twin = lnL_eb_twin_p(
                    time=elc.time_days,
                    flux=elc.flux,
                    sigma=elc.sigma,
                    rss=R_s_arr,
                    rcomps=radii_eb,
                    eb_flux_ratios=ext_fluxratios_eb,
                    periods=2 * P_orb,
                    incs=incs,
                    as_=a_twin,
                    u1s=ext_u1,
                    u2s=ext_u2,
                    eccs=eccs,
                    argps=argps,
                    companion_flux_ratios=ext_fluxratios_comp,
                    mask=mask_twin,
                    companion_is_host=False,
                    exptime=elc.cadence_days,
                    nsamples=elc.supersampling_rate,
                    force_serial=force_serial,
                )
                ext_lnL_twin = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2_twin
                lnL_twin = lnL_twin + ext_lnL_twin

        return lnL, lnL_twin

    def _pack_result(
        self,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        lnZ: float,
        idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve],
        twin: bool = False,
    ) -> ScenarioResult:
        """Pack PEB result. Source: marginal_likelihoods.py:1230-1278."""
        n = len(idx)
        b_key = "b_twin" if twin else "b"
        P_orb = samples["P_orb"]
        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        ext_fr_eb = []
        ext_fr_comp = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            ext_fr_eb.append(
                _flux_ratio_in_band(
                    samples["masses_eb"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )
            ext_fr_comp.append(
                _flux_ratio_in_band(
                    samples["masses_comp"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )

        return ScenarioResult(
            scenario_id=ScenarioID.PEBX2P if twin else self.scenario_id,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.full(n, stellar_params.mass_msun),
            host_radius_rsun=np.full(n, stellar_params.radius_rsun),
            host_u1=np.full(n, ldc.u1),
            host_u2=np.full(n, ldc.u2),
            period_days=(2 * P_orb[idx]) if twin else P_orb[idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry[b_key][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=samples["masses_eb"][idx],
            eb_radius_rsun=samples["radii_eb"][idx],
            flux_ratio_eb_tess=samples["fluxratios_eb"][idx],
            companion_mass_msun=samples["masses_comp"][idx],
            companion_radius_rsun=samples["radii_comp"][idx],
            flux_ratio_companion_tess=samples["fluxratios_comp"][idx],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_eb=ext_fr_eb,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> tuple[ScenarioResult, ScenarioResult]:
        """Override compute to add lnprior_companion to lnZ computation.

        In PEB, lnprior_companion is added to both lnL and lnL_twin before
        computing Z (source: lines 1223-1228, 1253-1258).
        """
        N = config.n_mc_samples
        P_orb = resolve_period(period_days, N)

        lnsigma = np.log(light_curve.sigma)
        ldc = self._get_host_ldc(stellar_params, config.mission, P_orb, dict(kwargs))

        resolved_ext_lcs = (
            self._resolve_external_lc_ldcs(external_lcs, stellar_params)
            if external_lcs else []
        )

        samples = self._sample_priors(N, stellar_params, P_orb, config, **kwargs)
        samples["M_s"] = np.full(N, stellar_params.mass_msun)
        samples["R_s"] = np.full(N, stellar_params.radius_rsun)
        samples["P_orb"] = P_orb

        # Compute companion prior
        molusc_data = kwargs.get("molusc_data")
        contrast_curve = kwargs.get("contrast_curve")
        filt = str(kwargs.get("filt", "TESS"))
        lnprior_comp = _compute_companion_prior(
            samples["masses_comp"], samples["fluxratios_comp"],
            stellar_params.mass_msun, stellar_params.parallax_mas, N,
            molusc_data, contrast_curve, filt, is_eb=True,
        )

        geometry = self._compute_orbital_geometry(
            samples, P_orb, stellar_params, config, **kwargs
        )

        lnL, lnL_twin = self._evaluate_lnL(
            light_curve, lnsigma, samples, geometry, ldc,
            resolved_ext_lcs, config,
        )

        # Add companion prior before computing Z (source: lines 1223-1228)
        lnZ = compute_lnZ(lnL + lnprior_comp, config.lnz_const)
        lnZ_twin = compute_lnZ(lnL_twin + lnprior_comp, config.lnz_const)

        n_best = config.n_best_samples
        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx = pack_best_indices(lnL + lnprior_comp, n_best)
        result = self._pack_result(
            samples, geometry, ldc, lnZ, idx, stellar_params,
            resolved_ext_lcs, twin=False,
        )

        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx_twin = pack_best_indices(lnL_twin + lnprior_comp, n_best)
        result_twin = self._pack_result(
            samples, geometry, ldc, lnZ_twin, idx_twin, stellar_params,
            resolved_ext_lcs, twin=True,
        )

        return result, result_twin


# ---------------------------------------------------------------------------
# STP + SEB: Sibling (companion-hosted) scenarios
# ---------------------------------------------------------------------------


class STPScenario(BaseScenario):
    """Planet transiting a Sibling (companion-hosted) star.

    The transit occurs around the companion star (companion_is_host=True).
    LDC lookup uses per-companion-sample Teff/logg arrays (bulk lookup).

    Source: marginal_likelihoods.py:1296-1614
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.STP

    @property
    def is_eb(self) -> bool:
        return False

    @property
    def returns_twin(self) -> bool:
        return False

    def _get_host_ldc(
        self,
        stellar_params: StellarParameters,
        mission: str,
        P_orb: np.ndarray,
        kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Return placeholder LDC; actual per-companion LDC computed in _evaluate_lnL.

        Source: marginal_likelihoods.py:1372-1410.
        """
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission,
            stellar_params.metallicity_dex,
            stellar_params.teff_k,
            stellar_params.logg,
        )

    def _sample_priors(
        self,
        n: int,
        stellar_params: StellarParameters,
        P_orb: np.ndarray,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample STP priors.

        Source: marginal_likelihoods.py:1349-1512.
        Planet radius sampled with companion mass as host mass (line 1509).
        """
        molusc_data = kwargs.get("molusc_data")
        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Teff = stellar_params.teff_k

        if molusc_data is not None:
            qs_comp = _load_molusc_qs(molusc_data, n, M_s)
        else:
            qs_comp = sample_companion_mass_ratio(np.random.rand(n), M_s)

        masses_comp, radii_comp, teffs_comp, fluxratios_comp = (
            _compute_companion_properties(qs_comp, M_s, R_s, Teff, n)
        )
        loggs_comp = np.log10(
            CONST.G * masses_comp * CONST.Msun / (radii_comp * CONST.Rsun) ** 2
        )

        # Planet radius sampled with companion mass as host (line 1509)
        rps = sample_planet_radius(np.random.rand(n), masses_comp, config.flat_priors)
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(np.random.rand(n), planet=True, period=P_orb)
        argps = sample_arg_periastron(np.random.rand(n))

        return {
            "rps": rps,
            "incs": incs,
            "eccs": eccs,
            "argps": argps,
            "qs_comp": qs_comp,
            "masses_comp": masses_comp,
            "radii_comp": radii_comp,
            "teffs_comp": teffs_comp,
            "loggs_comp": loggs_comp,
            "fluxratios_comp": fluxratios_comp,
        }

    def _compute_orbital_geometry(
        self,
        samples: dict[str, np.ndarray],
        P_orb: np.ndarray,
        stellar_params: StellarParameters,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """STP orbital geometry using companion mass as host.

        Source: marginal_likelihoods.py:1514-1524.
        Semi-major axis uses companion mass only (line 1516).
        Impact parameter denominates with R_comp (line 1521).
        """
        Rearth = CONST.Rearth
        Rsun = CONST.Rsun

        masses_comp = samples["masses_comp"]
        radii_comp = samples["radii_comp"]
        rps = samples["rps"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        incs = samples["incs"]

        a = semi_major_axis(P_orb, masses_comp)

        e_corr = (1 + eccs * np.sin(argps * np.pi / 180)) / (1 - eccs**2)
        ptra = (rps * Rearth + radii_comp * Rsun) / a * e_corr

        r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
        b = r * np.cos(incs * np.pi / 180) / (radii_comp * Rsun)

        coll = (rps * Rearth + radii_comp * Rsun) > a * (1 - eccs)

        return {"a": a, "ptra": ptra, "b": b, "coll": coll}

    def _evaluate_lnL(
        self,
        light_curve: LightCurve,
        lnsigma: float,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, None]:
        """STP likelihoods with per-companion LDC and companion_is_host=True.

        Source: marginal_likelihoods.py:1527-1554.
        """
        N = config.n_mc_samples

        rps = samples["rps"]
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        qs_comp = samples["qs_comp"]
        radii_comp = samples["radii_comp"]
        fluxratios_comp = samples["fluxratios_comp"]
        P_orb = samples["P_orb"]

        a = geometry["a"]
        ptra = geometry["ptra"]
        coll = geometry["coll"]

        time = light_curve.time_days
        flux = light_curve.flux
        sigma = light_curve.sigma

        extra_mask = qs_comp != 0.0
        mask = build_transit_mask(incs, ptra, coll, extra_mask=extra_mask)

        # Per-companion LDC via bulk lookup (lines 1396-1410)
        u1s_comp, u2s_comp = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            config.mission,
            samples["teffs_comp"],
            samples["loggs_comp"],
            np.full(N, 0.0),
        )

        force_serial = (not config.parallel) and not bool(external_lcs)
        chi2_half = lnL_planet_p(
            time=time,
            flux=flux,
            sigma=sigma,
            rps=rps,
            periods=P_orb,
            incs=incs,
            as_=a,
            rss=radii_comp,
            u1s=u1s_comp,
            u2s=u2s_comp,
            eccs=eccs,
            argps=argps,
            companion_flux_ratios=fluxratios_comp,
            mask=mask,
            companion_is_host=True,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        for ext_lc in external_lcs:
            elc = ext_lc.light_curve
            ext_u1s_comp, ext_u2s_comp = _bulk_companion_ldc(
                self._ldc,
                ext_lc.band,
                samples["teffs_comp"],
                samples["loggs_comp"],
            )
            ext_fluxratios_comp = _flux_ratio_in_band(
                samples["masses_comp"],
                float(samples["M_s"][0]),
                ext_lc.band,
            )
            ext_chi2 = lnL_planet_p(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rps=rps,
                periods=P_orb,
                incs=incs,
                as_=a,
                rss=radii_comp,
                u1s=ext_u1s_comp,
                u2s=ext_u2s_comp,
                eccs=eccs,
                argps=argps,
                companion_flux_ratios=ext_fluxratios_comp,
                mask=mask,
                companion_is_host=True,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        return lnL, None

    def _pack_result(
        self,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        lnZ: float,
        idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve],
        twin: bool = False,
    ) -> ScenarioResult:
        """Pack STP result. Host = companion, Comp = target star.

        Source: marginal_likelihoods.py:1582-1614.
        """
        n = len(idx)
        u1s_comp, u2s_comp = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            "TESS",
            samples["teffs_comp"][idx],
            samples["loggs_comp"][idx],
            np.full(n, 0.0),
        )
        ext_u1 = []
        ext_u2 = []
        ext_fr_comp = []
        for ext_lc in external_lcs:
            ext_u1s_comp, ext_u2s_comp = _bulk_companion_ldc(
                self._ldc,
                ext_lc.band,
                samples["teffs_comp"][idx],
                samples["loggs_comp"][idx],
            )
            ext_u1.append(ext_u1s_comp)
            ext_u2.append(ext_u2s_comp)
            ext_fr_comp.append(
                _flux_ratio_in_band(
                    samples["masses_comp"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )
        return ScenarioResult(
            scenario_id=self.scenario_id,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["masses_comp"][idx],
            host_radius_rsun=samples["radii_comp"][idx],
            host_u1=u1s_comp,
            host_u2=u2s_comp,
            period_days=samples["P_orb"][idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry["b"][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=samples["rps"][idx],
            eb_mass_msun=np.zeros(n),
            eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n),
            companion_mass_msun=np.full(n, stellar_params.mass_msun),
            companion_radius_rsun=np.full(n, stellar_params.radius_rsun),
            flux_ratio_companion_tess=samples["fluxratios_comp"][idx],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> ScenarioResult:
        """Override compute to add lnprior_companion before lnZ.

        Source: marginal_likelihoods.py:1573-1580.
        """
        N = config.n_mc_samples
        P_orb = resolve_period(period_days, N)

        lnsigma = np.log(light_curve.sigma)
        ldc = self._get_host_ldc(stellar_params, config.mission, P_orb, dict(kwargs))

        resolved_ext_lcs = (
            self._resolve_external_lc_ldcs(external_lcs, stellar_params)
            if external_lcs else []
        )

        samples = self._sample_priors(N, stellar_params, P_orb, config, **kwargs)
        samples["M_s"] = np.full(N, stellar_params.mass_msun)
        samples["P_orb"] = P_orb

        molusc_data = kwargs.get("molusc_data")
        contrast_curve = kwargs.get("contrast_curve")
        filt = str(kwargs.get("filt", "TESS"))
        lnprior_comp = _compute_companion_prior(
            samples["masses_comp"], samples["fluxratios_comp"],
            stellar_params.mass_msun, stellar_params.parallax_mas, N,
            molusc_data, contrast_curve, filt, is_eb=False,
        )

        geometry = self._compute_orbital_geometry(
            samples, P_orb, stellar_params, config, **kwargs,
        )

        lnL, _ = self._evaluate_lnL(
            light_curve, lnsigma, samples, geometry, ldc,
            resolved_ext_lcs, config,
        )

        lnZ = compute_lnZ(lnL + lnprior_comp, config.lnz_const)

        n_best = config.n_best_samples
        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx = pack_best_indices(lnL + lnprior_comp, n_best)
        return self._pack_result(
            samples, geometry, ldc, lnZ, idx, stellar_params,
            resolved_ext_lcs,
        )


class SEBScenario(BaseScenario):
    """Eclipsing Binary on a Sibling (companion) star.

    Returns (result, result_twin). companion_is_host=True.

    Source: marginal_likelihoods.py:1617-2056
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.SEB

    @property
    def is_eb(self) -> bool:
        return True

    def _get_host_ldc(
        self,
        stellar_params: StellarParameters,
        mission: str,
        P_orb: np.ndarray,
        kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Placeholder LDC; per-companion LDC computed in _evaluate_lnL."""
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission,
            stellar_params.metallicity_dex,
            stellar_params.teff_k,
            stellar_params.logg,
        )

    def _sample_priors(
        self,
        n: int,
        stellar_params: StellarParameters,
        P_orb: np.ndarray,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample SEB priors.

        Source: marginal_likelihoods.py:1658-1742.
        EB masses = qs * masses_comp (line 1736).
        EB radii/Teffs use companion as reference (line 1737).
        """
        molusc_data = kwargs.get("molusc_data")
        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Teff = stellar_params.teff_k

        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), M_s)
        eccs = sample_eccentricity(np.random.rand(n), planet=False, period=P_orb)
        argps = sample_arg_periastron(np.random.rand(n))

        if molusc_data is not None:
            qs_comp = _load_molusc_qs(molusc_data, n, M_s)
        else:
            qs_comp = sample_companion_mass_ratio(np.random.rand(n), M_s)

        # Companion properties (lines 1684-1694)
        masses_comp, radii_comp, teffs_comp, fluxratios_comp = (
            _compute_companion_properties(qs_comp, M_s, R_s, Teff, n)
        )
        loggs_comp = np.log10(
            CONST.G * masses_comp * CONST.Msun / (radii_comp * CONST.Rsun) ** 2
        )

        # EB properties: EB orbits the companion (lines 1735-1742)
        masses_eb = qs * masses_comp
        radii_eb, _teffs_eb = _relations.get_radius_teff(
            masses_eb, radii_comp, teffs_comp,
        )
        flux_eb = _relations.get_flux_ratio(masses_eb, "TESS")
        flux_primary = _relations.get_flux_ratio(np.array([M_s]), "TESS")
        fluxratios_eb = flux_eb / (flux_eb + flux_primary)

        return {
            "incs": incs,
            "qs": qs,
            "eccs": eccs,
            "argps": argps,
            "qs_comp": qs_comp,
            "masses_comp": masses_comp,
            "radii_comp": radii_comp,
            "teffs_comp": teffs_comp,
            "loggs_comp": loggs_comp,
            "fluxratios_comp": fluxratios_comp,
            "masses_eb": masses_eb,
            "radii_eb": radii_eb,
            "fluxratios_eb": fluxratios_eb,
        }

    def _compute_orbital_geometry(
        self,
        samples: dict[str, np.ndarray],
        P_orb: np.ndarray,
        stellar_params: StellarParameters,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """SEB orbital geometry.

        Source: marginal_likelihoods.py:1860-1879.
        Semi-major axis uses (masses_comp + masses_eb) (line 1863).
        Transit probability uses (radii_eb + radii_comp) (line 1865).
        Impact parameter denominates with radii_comp (line 1873).
        Collision check uses (radii_eb + radii_comp) (line 1878).
        Twin collision uses (2*radii_comp) (line 1879).
        """
        Rsun = CONST.Rsun

        masses_comp = samples["masses_comp"]
        masses_eb = samples["masses_eb"]
        radii_comp = samples["radii_comp"]
        radii_eb = samples["radii_eb"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        incs = samples["incs"]

        e_corr = (1 + eccs * np.sin(argps * np.pi / 180)) / (1 - eccs**2)

        # q < 0.95 (line 1863)
        a = semi_major_axis(P_orb, masses_comp + masses_eb)
        ptra = (radii_eb * Rsun + radii_comp * Rsun) / a * e_corr

        # q >= 0.95 twin at 2*P (line 1867)
        a_twin = semi_major_axis(2 * P_orb, masses_comp + masses_eb)
        ptra_twin = (radii_eb * Rsun + radii_comp * Rsun) / a_twin * e_corr

        # Impact parameter with companion radius as denominator (line 1873)
        r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
        b = r * np.cos(incs * np.pi / 180) / (radii_comp * Rsun)
        r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
        b_twin = r_twin * np.cos(incs * np.pi / 180) / (radii_comp * Rsun)

        # Collisions (lines 1878-1879)
        coll = (radii_eb * Rsun + radii_comp * Rsun) > a * (1 - eccs)
        coll_twin = (2 * radii_comp * Rsun) > a_twin * (1 - eccs)

        return {
            "a": a,
            "a_twin": a_twin,
            "ptra": ptra,
            "ptra_twin": ptra_twin,
            "b": b,
            "b_twin": b_twin,
            "coll": coll,
            "coll_twin": coll_twin,
        }

    def _evaluate_lnL(
        self,
        light_curve: LightCurve,
        lnsigma: float,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SEB likelihoods with per-companion LDC and companion_is_host=True.

        Source: marginal_likelihoods.py:1884-1943.
        """
        N = config.n_mc_samples
        qs = samples["qs"]
        qs_comp = samples["qs_comp"]

        # Per-companion LDC via bulk lookup (lines 1726-1733)
        u1s_comp, u2s_comp = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            config.mission,
            samples["teffs_comp"],
            samples["loggs_comp"],
            np.full(N, 0.0),
        )

        force_serial = (not config.parallel) and not bool(external_lcs)

        # q < 0.95 / q >= 0.95 branching (lines 1886-1943)
        mask, mask_twin = build_eb_branch_masks(
            qs, samples["incs"],
            geometry["ptra"], geometry["coll"],
            geometry["ptra_twin"], geometry["coll_twin"],
            extra_mask=(qs_comp != 0.0),
        )
        lnL = self._seb_branch_lnL(
            light_curve, lnsigma, samples, u1s_comp, u2s_comp, mask,
            lnL_fn=lnL_eb_p, a=geometry["a"], period_mult=1,
            force_serial=force_serial,
        )
        lnL_twin = self._seb_branch_lnL(
            light_curve, lnsigma, samples, u1s_comp, u2s_comp, mask_twin,
            lnL_fn=lnL_eb_twin_p, a=geometry["a_twin"], period_mult=2,
            force_serial=force_serial,
        )

        for ext_lc in external_lcs:
            elc = ext_lc.light_curve
            ext_u1s_comp, ext_u2s_comp = _bulk_companion_ldc(
                self._ldc,
                ext_lc.band,
                samples["teffs_comp"],
                samples["loggs_comp"],
            )
            ext_fluxratios_eb = _flux_ratio_in_band(
                samples["masses_eb"],
                float(samples["M_s"][0]),
                ext_lc.band,
            )
            ext_fluxratios_comp = _flux_ratio_in_band(
                samples["masses_comp"],
                float(samples["M_s"][0]),
                ext_lc.band,
            )
            ext_lnl_sigma = np.log(elc.sigma)
            lnL = lnL + self._seb_branch_lnL(
                elc,
                ext_lnl_sigma,
                samples,
                ext_u1s_comp,
                ext_u2s_comp,
                mask,
                lnL_fn=lnL_eb_p,
                a=geometry["a"],
                period_mult=1,
                eb_flux_ratios=ext_fluxratios_eb,
                companion_flux_ratios=ext_fluxratios_comp,
                force_serial=force_serial,
            )
            lnL_twin = lnL_twin + self._seb_branch_lnL(
                elc,
                ext_lnl_sigma,
                samples,
                ext_u1s_comp,
                ext_u2s_comp,
                mask_twin,
                lnL_fn=lnL_eb_twin_p,
                a=geometry["a_twin"],
                period_mult=2,
                eb_flux_ratios=ext_fluxratios_eb,
                companion_flux_ratios=ext_fluxratios_comp,
                force_serial=force_serial,
            )

        return lnL, lnL_twin

    @staticmethod
    def _seb_branch_lnL(
        light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray],
        u1s_comp: np.ndarray, u2s_comp: np.ndarray,
        mask: np.ndarray, lnL_fn: Callable[..., np.ndarray],
        a: np.ndarray, period_mult: int,
        eb_flux_ratios: np.ndarray | None = None,
        companion_flux_ratios: np.ndarray | None = None,
        force_serial: bool = False,
    ) -> np.ndarray:
        """Compute SEB lnL for one branch (standard or twin)."""
        chi2_half = lnL_fn(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=samples["radii_comp"],
            rcomps=samples["radii_eb"],
            eb_flux_ratios=(
                samples["fluxratios_eb"]
                if eb_flux_ratios is None else eb_flux_ratios
            ),
            periods=period_mult * samples["P_orb"],
            incs=samples["incs"],
            as_=a,
            u1s=u1s_comp,
            u2s=u2s_comp,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=(
                samples["fluxratios_comp"]
                if companion_flux_ratios is None else companion_flux_ratios
            ),
            mask=mask,
            companion_is_host=True,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        return -0.5 * _ln2pi - lnsigma - chi2_half

    def _pack_result(
        self,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        lnZ: float,
        idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve],
        twin: bool = False,
    ) -> ScenarioResult:
        """Pack SEB result. Host = companion, Comp = target star.

        Source: marginal_likelihoods.py:1990-2054.
        """
        n = len(idx)
        b_key = "b_twin" if twin else "b"
        P_orb = samples["P_orb"]

        u1s_comp, u2s_comp = self._ldc.get_coefficients_bulk(  # type: ignore[union-attr]
            "TESS",
            samples["teffs_comp"][idx],
            samples["loggs_comp"][idx],
            np.full(n, 0.0),
        )
        ext_u1 = []
        ext_u2 = []
        ext_fr_eb = []
        ext_fr_comp = []
        for ext_lc in external_lcs:
            ext_u1s_comp, ext_u2s_comp = _bulk_companion_ldc(
                self._ldc,
                ext_lc.band,
                samples["teffs_comp"][idx],
                samples["loggs_comp"][idx],
            )
            ext_u1.append(ext_u1s_comp)
            ext_u2.append(ext_u2s_comp)
            ext_fr_eb.append(
                _flux_ratio_in_band(
                    samples["masses_eb"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )
            ext_fr_comp.append(
                _flux_ratio_in_band(
                    samples["masses_comp"][idx],
                    float(samples["M_s"][0]),
                    ext_lc.band,
                )
            )

        return ScenarioResult(
            scenario_id=ScenarioID.SEBX2P if twin else self.scenario_id,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=samples["masses_comp"][idx],
            host_radius_rsun=samples["radii_comp"][idx],
            host_u1=u1s_comp,
            host_u2=u2s_comp,
            period_days=(2 * P_orb[idx]) if twin else P_orb[idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry[b_key][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=samples["masses_eb"][idx],
            eb_radius_rsun=samples["radii_eb"][idx],
            flux_ratio_eb_tess=samples["fluxratios_eb"][idx],
            companion_mass_msun=np.full(n, stellar_params.mass_msun),
            companion_radius_rsun=np.full(n, stellar_params.radius_rsun),
            flux_ratio_companion_tess=samples["fluxratios_comp"][idx],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_eb=ext_fr_eb,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> tuple[ScenarioResult, ScenarioResult]:
        """Override compute to add lnprior_companion before lnZ.

        Source: marginal_likelihoods.py:1979-2054.
        SEB companion prior includes both EB and companion flux ratios.
        """
        N = config.n_mc_samples
        P_orb = resolve_period(period_days, N)

        lnsigma = np.log(light_curve.sigma)
        ldc = self._get_host_ldc(stellar_params, config.mission, P_orb, dict(kwargs))

        resolved_ext_lcs = (
            self._resolve_external_lc_ldcs(external_lcs, stellar_params)
            if external_lcs else []
        )

        samples = self._sample_priors(N, stellar_params, P_orb, config, **kwargs)
        samples["M_s"] = np.full(N, stellar_params.mass_msun)
        samples["P_orb"] = P_orb

        # SEB companion prior includes both EB and companion flux (lines 1819-1858)
        molusc_data = kwargs.get("molusc_data")
        contrast_curve = kwargs.get("contrast_curve")
        filt = str(kwargs.get("filt", "TESS"))
        lnprior_comp = _compute_seb_companion_prior(
            samples["masses_comp"], samples["fluxratios_comp"],
            samples["masses_eb"], samples["fluxratios_eb"],
            stellar_params.mass_msun, stellar_params.parallax_mas, N,
            molusc_data, contrast_curve, filt,
        )

        geometry = self._compute_orbital_geometry(
            samples, P_orb, stellar_params, config, **kwargs,
        )

        lnL, lnL_twin = self._evaluate_lnL(
            light_curve, lnsigma, samples, geometry, ldc,
            resolved_ext_lcs, config,
        )

        lnZ = compute_lnZ(lnL + lnprior_comp, config.lnz_const)
        lnZ_twin = compute_lnZ(lnL_twin + lnprior_comp, config.lnz_const)

        n_best = config.n_best_samples
        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx = pack_best_indices(lnL + lnprior_comp, n_best)
        result = self._pack_result(
            samples, geometry, ldc, lnZ, idx, stellar_params,
            resolved_ext_lcs, twin=False,
        )

        # BUG-07 fix: rank by posterior (lnL + lnprior_comp), not raw likelihood
        idx_twin = pack_best_indices(lnL_twin + lnprior_comp, n_best)
        result_twin = self._pack_result(
            samples, geometry, ldc, lnZ_twin, idx_twin, stellar_params,
            resolved_ext_lcs, twin=True,
        )

        return result, result_twin
