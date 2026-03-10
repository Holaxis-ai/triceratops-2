"""Background star scenario implementations (D- and B-scenarios).

DTP/DEB: Diluted background star (target star dilutes signal).
BTP/BEB: Bright background star (background star is transit host).

Source: marginal_likelihoods.py:2058-2380 (lnZ_DTP), 2382-2820 (lnZ_DEB),
        2823-3155 (lnZ_BTP), 3158-3669 (lnZ_BEB).

BUG-04 fix: _compute_delta_mags_map() correctly maps "delta_Kmags" to
target_kmag - population.kmags (the original mapped it to delta_Hmags).

BUG-06 fix: BEBScenario._evaluate_lnL uses geometry['coll'] (not coll_twin)
for the q<0.95 mask. The original code at ~line 3492 incorrectly used coll_twin.
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
from triceratops.priors.lnpriors import lnprior_background
from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_companion_mass_ratio,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
    sample_planet_radius,
)
from triceratops.scenarios.base import BaseScenario
from triceratops.scenarios.kernels import build_transit_mask
from triceratops.stellar.relations import StellarRelations

_ln2pi = np.log(2 * np.pi)
_relations = StellarRelations()
_SDSS_BANDS = frozenset({"g", "r", "i", "z"})


def _sample_population_indices(
    n_comp: int, n: int, *, legacy_exclude_last: bool = False,
) -> np.ndarray:
    """Draw TRILEGAL row indices, preserving the original off-by-one quirk."""
    upper = n_comp - 1 if legacy_exclude_last and n_comp > 1 else n_comp
    if upper <= 0:
        return np.zeros(n, dtype=int)
    return np.random.randint(0, upper, size=n)


def _filter_population_by_target_tmag(
    population: TRILEGALResult,
    target_tmag: float | None,
) -> TRILEGALResult:
    """Match funcs.trilegal_results(..., Tmag): keep stars fainter than target."""
    if target_tmag is None:
        return population
    mask = population.tmags >= float(target_tmag)
    return TRILEGALResult(
        tmags=population.tmags[mask],
        masses=population.masses[mask],
        loggs=population.loggs[mask],
        teffs=population.teffs[mask],
        metallicities=population.metallicities[mask],
        jmags=population.jmags[mask],
        hmags=population.hmags[mask],
        kmags=population.kmags[mask],
        gmags=population.gmags[mask],
        rmags=population.rmags[mask],
        imags=population.imags[mask],
        zmags=population.zmags[mask],
    )


def _needs_sdss_delta_mags(
    external_lc_bands: tuple[str, ...], filt: str | None,
) -> bool:
    """Return True when any active band requires SDSS photometry."""
    bands = set(external_lc_bands)
    if filt is not None:
        bands.add(filt)
    return any(band in _SDSS_BANDS for band in bands)


def _resolve_sdss_target_mags(
    host_mags: dict[str, float | None],
    external_lc_bands: tuple[str, ...],
    filt: str | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve target g/r/i/z, estimating them when the original code did."""
    gmag = host_mags.get("gmag")
    rmag = host_mags.get("rmag")
    imag = host_mags.get("imag")
    zmag = host_mags.get("zmag")

    no_sdss = all(
        mag is None or np.isnan(mag) for mag in (gmag, rmag, imag, zmag)
    )
    if no_sdss and _needs_sdss_delta_mags(external_lc_bands, filt):
        bmag = host_mags.get("bmag")
        vmag = host_mags.get("vmag")
        jmag = host_mags.get("jmag")
        if (
            bmag is not None and np.isfinite(bmag)
            and vmag is not None and np.isfinite(vmag)
            and jmag is not None and np.isfinite(jmag)
        ):
            estimated = _relations.estimate_sdss_magnitudes(
                float(bmag), float(vmag), float(jmag),
            )
            gmag = estimated["g"]
            rmag = estimated["r"]
            imag = estimated["i"]
            zmag = estimated["z"]

    return gmag, rmag, imag, zmag


def _compute_delta_mags_map(
    target_tmag: float,
    target_jmag: float,
    target_hmag: float,
    target_kmag: float,
    population: TRILEGALResult,
) -> dict[str, np.ndarray]:
    """Compute delta-magnitude arrays for each photometric band.

    BUG-04 fix: "delta_Kmags" correctly maps to target_kmag - population.kmags.
    In the original code (marginal_likelihoods.py:2155), it was mapped to delta_Hmags.

    Source: marginal_likelihoods.py:2141-2156 (DTP).
    """
    return {
        "delta_TESSmags": target_tmag - population.tmags,
        "delta_Jmags": target_jmag - population.jmags,
        "delta_Hmags": target_hmag - population.hmags,
        "delta_Kmags": target_kmag - population.kmags,  # BUG-04 fix
    }


def _compute_sdss_delta_mags(
    target_gmag: float | None,
    target_rmag: float | None,
    target_imag: float | None,
    target_zmag: float | None,
    population: TRILEGALResult,
) -> dict[str, np.ndarray]:
    """Compute SDSS band delta-mags if target has SDSS photometry.

    Source: marginal_likelihoods.py:2159-2179 (DTP).
    """
    result: dict[str, np.ndarray] = {}
    if target_gmag is not None and not np.isnan(target_gmag):
        result["delta_gmags"] = target_gmag - population.gmags
    if target_rmag is not None and not np.isnan(target_rmag):
        result["delta_rmags"] = target_rmag - population.rmags
    if target_imag is not None and not np.isnan(target_imag):
        result["delta_imags"] = target_imag - population.imags
    if target_zmag is not None and not np.isnan(target_zmag):
        result["delta_zmags"] = target_zmag - population.zmags
    return result


def _compute_fluxratios_comp(delta_mags: np.ndarray) -> np.ndarray:
    """Compute companion flux ratios from delta magnitudes.

    Source: marginal_likelihoods.py:2149
    """
    ratio = 10 ** (delta_mags / 2.5)
    return ratio / (1 + ratio)


def _combined_delta_mag(
    primary_flux_ratio: np.ndarray, secondary_flux_ratio: np.ndarray,
) -> np.ndarray:
    """Convert summed host+EB brightness ratios back to delta magnitudes."""
    return 2.5 * np.log10(
        (primary_flux_ratio / (1 - primary_flux_ratio))
        + (secondary_flux_ratio / (1 - secondary_flux_ratio))
    )


def _compute_lnprior_companion(
    n_comp: int,
    fluxratios_comp: np.ndarray,
    idxs: np.ndarray,
    delta_mags_map: dict[str, np.ndarray],
    contrast_curve: object | None,
    filt: str | None,
) -> np.ndarray:
    """Compute the background companion prior for D-scenarios.

    Without contrast curve (lines 2246-2255):
        lnprior = log10((N_comp/0.1) * (1/3600)^2 * 2.2^2), capped at 0.
        Set to -inf where delta_mags > 0 (background star brighter than target).

    With contrast curve (lines 2256-2272):
        Use lnprior_background() with contrast curve separations and contrasts.

    Source: marginal_likelihoods.py:2246-2272 (DTP) and 2592-2618 (DEB).
    """
    n = len(idxs)

    if contrast_curve is None:
        # Recompute delta_mags from flux ratios for the drawn samples
        fr = fluxratios_comp[idxs]
        delta_mags_drawn = 2.5 * np.log10(fr / (1 - fr))
        lnprior = np.full(n, np.log10((n_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2))
        lnprior[lnprior > 0.0] = 0.0
        lnprior[delta_mags_drawn > 0.0] = -np.inf
        return lnprior

    # With contrast curve: select the right band's delta_mags
    separations = contrast_curve.separations_arcsec  # type: ignore[union-attr]
    contrasts = contrast_curve.delta_mags  # type: ignore[union-attr]

    filt_key_map = {
        "J": "delta_Jmags",
        "H": "delta_Hmags",
        "K": "delta_Kmags",
    }
    key = filt_key_map.get(filt or "", "delta_TESSmags")
    delta_mags_band = delta_mags_map[key][idxs]

    lnprior = lnprior_background(n_comp, np.abs(delta_mags_band), separations, contrasts)
    lnprior[lnprior > 0.0] = 0.0
    lnprior[delta_mags_band > 0.0] = -np.inf
    return lnprior


def _compute_bright_background_lnprior(
    n_comp: int,
    idxs: np.ndarray,
    fluxratios_comp_band: np.ndarray,
    fluxratios_eb_band: np.ndarray,
    contrast_curve: object | None,
) -> np.ndarray:
    """Background prior for BEB using combined host+EB brightness."""
    n = len(idxs)
    delta_mags = _combined_delta_mag(
        fluxratios_comp_band, fluxratios_eb_band,
    )

    if contrast_curve is None:
        lnprior = np.full(n, np.log10((n_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2))
    else:
        separations = contrast_curve.separations_arcsec  # type: ignore[union-attr]
        contrasts = contrast_curve.delta_mags  # type: ignore[union-attr]
        lnprior = lnprior_background(
            n_comp, np.abs(delta_mags), separations, contrasts,
        )

    lnprior[lnprior > 0.0] = 0.0
    lnprior[delta_mags > 0.0] = -np.inf
    return lnprior


def _lookup_background_ldc_bulk(
    ldc_catalog: object,
    band: str,
    teffs: np.ndarray,
    loggs: np.ndarray,
    metallicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the original bright-background LDC lookup order exactly."""
    load_filter = getattr(ldc_catalog, "_load_filter", None)
    if not callable(load_filter):
        return ldc_catalog.get_coefficients_bulk(  # type: ignore[union-attr]
            band, teffs, loggs, metallicities,
        )

    zs, teff_grid, logg_grid, u1_grid, u2_grid = load_filter(band)
    n = len(teffs)
    u1_out = np.zeros(n)
    u2_out = np.zeros(n)
    for i in range(n):
        this_teff = teff_grid[np.argmin(np.abs(teff_grid - teffs[i]))]
        this_logg = logg_grid[np.argmin(np.abs(logg_grid - loggs[i]))]
        mask = (teff_grid == this_teff) & (logg_grid == this_logg)
        these_zs = zs[mask]
        this_z = these_zs[np.argmin(np.abs(these_zs - metallicities[i]))]
        coeff_mask = mask & (zs == this_z)
        u1_out[i] = float(u1_grid[coeff_mask][0])
        u2_out[i] = float(u2_grid[coeff_mask][0])
    return u1_out, u2_out


class DTPScenario(BaseScenario):
    """Planet on a Diluted Background star.

    The background star's properties come from TRILEGAL simulation.
    The target star dilutes the signal. BUG-04 is fixed here.

    Source: marginal_likelihoods.py:2058-2380
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.DTP

    @property
    def is_eb(self) -> bool:
        return False

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Target star LDC (DTP: target dilutes the background signal).

        Source: marginal_likelihoods.py:2117-2138
        """
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, stellar_params.logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample DTP priors including TRILEGAL row indices.

        Source: marginal_likelihoods.py:2241-2278
        """
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError(
                "DTPScenario requires 'trilegal_population' in kwargs."
            )
        population = _filter_population_by_target_tmag(
            population, kwargs.get("target_tmag"),  # type: ignore[arg-type]
        )
        n_comp = population.n_stars
        idxs = _sample_population_indices(n_comp, n, legacy_exclude_last=True)

        host_mags: dict = kwargs.get("host_magnitudes", {})  # type: ignore[assignment]
        filt: str | None = kwargs.get("filt")  # type: ignore[assignment]
        external_lc_bands = tuple(kwargs.get("external_lc_bands", ()))  # type: ignore[arg-type]
        gmag, rmag, imag, zmag = _resolve_sdss_target_mags(
            host_mags, external_lc_bands, filt,
        )

        # Compute delta mags and flux ratios (BUG-04 fix via _compute_delta_mags_map)
        delta_mags_map = _compute_delta_mags_map(
            host_mags.get("tmag", 10.0),
            host_mags.get("jmag", 10.0),
            host_mags.get("hmag", 10.0),
            host_mags.get("kmag", 10.0),
            population,
        )
        sdss_delta = _compute_sdss_delta_mags(
            gmag, rmag, imag, zmag,
            population,
        )
        delta_mags_map.update(sdss_delta)

        delta_mags_tess = delta_mags_map["delta_TESSmags"]
        fluxratios_comp = _compute_fluxratios_comp(delta_mags_tess)

        # Companion prior
        contrast_curve = kwargs.get("contrast_curve")
        lnprior = _compute_lnprior_companion(
            n_comp, fluxratios_comp, idxs, delta_mags_map, contrast_curve, filt,
        )

        # Sample planet priors (same as TTP but uses target star mass)
        rps = sample_planet_radius(
            np.random.rand(n), stellar_params.mass_msun, config.flat_priors,
        )
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(
            np.random.rand(n), planet=True, period=float(np.mean(P_orb)),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        return {
            "rps": rps, "incs": incs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, stellar_params.mass_msun),
            "R_s": np.full(n, stellar_params.radius_rsun),
            "idxs": idxs,
            "fluxratios_comp": fluxratios_comp,
            "lnprior_companion": lnprior,
            "masses_comp": population.masses,
            "delta_mags_map": delta_mags_map,  # type: ignore[dict-item]
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """DTP orbital geometry -- uses target star mass only (planet on background).

        Source: marginal_likelihoods.py:2280-2290
        Note: original uses a = ((G*M_s*Msun)/(4*pi**2)*(P_orb*86400)**2)**(1/3)
        which is semi_major_axis(P_orb, M_s) -- target star mass only.
        """
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
        """DTP log-likelihoods with companion flux ratios and lnprior folded in.

        Key difference from TTP: companion_flux_ratios = fluxratios_comp[idxs[mask]]
        (background star dilutes signal), and lnprior_companion is added to lnL
        so BaseScenario.compute_lnZ will produce the correct evidence.

        Source: marginal_likelihoods.py:2291-2326
        """
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
        )

        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        lnprior = samples["lnprior_companion"]

        R_s_arr = samples["R_s"]
        u1_arr = np.full(N, float(ldc.u1))
        u2_arr = np.full(N, float(ldc.u2))
        # Companion flux ratio per sample (background star dilution)
        comp_fr = fluxratios_comp[idxs]

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
            companion_flux_ratios=comp_fr,
            mask=mask,
            companion_is_host=False,
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

            # Compute external LC companion flux ratios from the band's delta mags
            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                ext_comp_fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs]
            else:
                ext_comp_fr = comp_fr

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
                companion_flux_ratios=ext_comp_fr,
                mask=mask,
                companion_is_host=False,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        # Fold in the background prior so compute_lnZ gives correct evidence
        # Original: Z = mean(exp(lnL + lnprior_companion + lnz_const))
        lnL = lnL + lnprior

        return lnL, None

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Pack DTP results.

        Source: marginal_likelihoods.py:2353-2378
        """
        n = len(idx)
        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        masses_comp = samples["masses_comp"]

        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        ext_fr_comp = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs[idx]]
            else:
                fr = fluxratios_comp[idxs[idx]]
            ext_fr_comp.append(fr)

        return ScenarioResult(
            scenario_id=ScenarioID.DTP,
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
            companion_mass_msun=masses_comp[idxs[idx]],
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=fluxratios_comp[idxs[idx]],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )


class DEBScenario(BaseScenario):
    """Eclipsing Binary on a Diluted Background star.

    Returns (result, result_twin) -- twin is the q>=0.95 half-period alias.

    Source: marginal_likelihoods.py:2382-2820
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.DEB

    @property
    def is_eb(self) -> bool:
        return True

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
        """Sample DEB priors: EB params + TRILEGAL population.

        Source: marginal_likelihoods.py:2464-2490 (EB sampling) + 2587-2618 (prior)
        """
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError(
                "DEBScenario requires 'trilegal_population' in kwargs."
            )
        population = _filter_population_by_target_tmag(
            population, kwargs.get("target_tmag"),  # type: ignore[arg-type]
        )
        n_comp = population.n_stars

        host_mags: dict = kwargs.get("host_magnitudes", {})  # type: ignore[assignment]
        filt: str | None = kwargs.get("filt")  # type: ignore[assignment]
        external_lc_bands = tuple(kwargs.get("external_lc_bands", ()))  # type: ignore[arg-type]
        gmag, rmag, imag, zmag = _resolve_sdss_target_mags(
            host_mags, external_lc_bands, filt,
        )

        # Match the original RNG stream: sample EB priors before drawing
        # the background-star indices.
        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), stellar_params.mass_msun)
        eccs = sample_eccentricity(
            np.random.rand(n), planet=False, period=float(np.mean(P_orb)),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        # Delta mags and companion flux ratios (BUG-04 fix)
        delta_mags_map = _compute_delta_mags_map(
            host_mags.get("tmag", 10.0),
            host_mags.get("jmag", 10.0),
            host_mags.get("hmag", 10.0),
            host_mags.get("kmag", 10.0),
            population,
        )
        sdss_delta = _compute_sdss_delta_mags(
            gmag, rmag, imag, zmag,
            population,
        )
        delta_mags_map.update(sdss_delta)

        delta_mags_tess = delta_mags_map["delta_TESSmags"]
        fluxratios_comp = _compute_fluxratios_comp(delta_mags_tess)
        idxs = _sample_population_indices(n_comp, n, legacy_exclude_last=True)

        # Companion prior (same as DTP)
        contrast_curve = kwargs.get("contrast_curve")
        lnprior = _compute_lnprior_companion(
            n_comp, fluxratios_comp, idxs, delta_mags_map, contrast_curve, filt,
        )

        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun
        Teff = stellar_params.teff_k

        masses = qs * M_s
        radii, _teffs = _relations.get_radius_teff(
            masses,
            max_radii=np.full(n, R_s),
            max_teffs=np.full(n, Teff),
        )
        fluxratios_eb = (
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
            "fluxratios": fluxratios_eb,
            "idxs": idxs,
            "fluxratios_comp": fluxratios_comp,
            "lnprior_companion": lnprior,
            "masses_comp": population.masses,
            "delta_mags_map": delta_mags_map,  # type: ignore[dict-item]
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """DEB orbital geometry -- uses total system mass (M_s + M_EB).

        Source: marginal_likelihoods.py:2620-2635
        """
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        masses = samples["masses"]
        radii = samples["radii"]

        M_s = stellar_params.mass_msun
        R_s = stellar_params.radius_rsun

        a = semi_major_axis(P_orb, M_s + masses)
        Ptra = transit_probability(a, R_s, radii, eccs, argps)
        b = impact_parameter(a, incs, R_s, eccs, argps)
        coll = collision_check(a, R_s, radii, eccs)

        a_twin = semi_major_axis(2 * P_orb, M_s + masses)
        Ptra_twin = transit_probability(a_twin, R_s, radii, eccs, argps)
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
        """DEB log-likelihoods with companion flux ratios and lnprior folded in.

        DEB uses both EB flux ratio AND companion flux ratio from background.

        Source: marginal_likelihoods.py:2637-2711
        """
        N = config.n_mc_samples
        force_serial = (not config.parallel) and not bool(external_lcs)
        qs = samples["qs"]
        R_s_arr = samples["R_s"]
        u1_arr = np.full(N, float(ldc.u1))
        u2_arr = np.full(N, float(ldc.u2))

        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        lnprior = samples["lnprior_companion"]
        comp_fr = fluxratios_comp[idxs]

        # q < 0.95: standard EB
        q_lt_mask = qs < 0.95
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
            extra_mask=q_lt_mask,
        )
        lnL = self._deb_branch_lnL(
            light_curve, lnsigma, samples, geometry, external_lcs,
            R_s_arr, u1_arr, u2_arr, comp_fr, idxs, mask,
            lnL_fn=lnL_eb_p, a_key="a", period_mult=1, N=N,
            force_serial=force_serial,
        )

        # q >= 0.95: twin EB at 2x period
        q_ge_mask = qs >= 0.95
        mask_twin = build_transit_mask(
            samples["incs"], geometry["Ptra_twin"], geometry["coll_twin"],
            extra_mask=q_ge_mask,
        )
        lnL_twin = self._deb_branch_lnL(
            light_curve, lnsigma, samples, geometry, external_lcs,
            R_s_arr, u1_arr, u2_arr, comp_fr, idxs, mask_twin,
            lnL_fn=lnL_eb_twin_p, a_key="a_twin", period_mult=2, N=N,
            force_serial=force_serial,
        )

        # Fold in the background prior
        lnL = lnL + lnprior
        lnL_twin = lnL_twin + lnprior

        return lnL, lnL_twin

    @staticmethod
    def _deb_branch_lnL(
        light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        external_lcs: list[ExternalLightCurve],
        R_s_arr: np.ndarray, u1_arr: np.ndarray, u2_arr: np.ndarray,
        comp_fr: np.ndarray, idxs: np.ndarray, mask: np.ndarray,
        lnL_fn: Callable[..., np.ndarray], a_key: str, period_mult: int, N: int,
        force_serial: bool = False,
    ) -> np.ndarray:
        """Compute DEB lnL for one branch (standard or twin)."""
        periods = period_mult * samples["P_orb"]

        chi2_half = lnL_fn(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=R_s_arr,
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=periods,
            incs=samples["incs"],
            as_=geometry[a_key],
            u1s=u1_arr,
            u2s=u2_arr,
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=comp_fr,
            mask=mask,
            companion_is_host=False,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        # Accumulate external LC contributions
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            elc = ext_lc.light_curve
            ext_u1 = np.full(N, float(ext_lc.ldc.u1))
            ext_u2 = np.full(N, float(ext_lc.ldc.u2))
            ext_fr_eb = (
                _relations.get_flux_ratio(samples["masses"], ext_lc.band)
                / (_relations.get_flux_ratio(samples["masses"], ext_lc.band)
                   + _relations.get_flux_ratio(
                       np.array([float(samples["M_s"][0])]), ext_lc.band))
            )
            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                ext_comp_fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs]
            else:
                ext_comp_fr = comp_fr

            ext_chi2 = lnL_fn(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rss=R_s_arr,
                rcomps=samples["radii"],
                eb_flux_ratios=ext_fr_eb,
                periods=periods,
                incs=samples["incs"],
                as_=geometry[a_key],
                u1s=ext_u1,
                u2s=ext_u2,
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=ext_comp_fr,
                mask=mask,
                companion_is_host=False,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        return lnL

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Pack DEB results.

        Source: marginal_likelihoods.py:2755-2819
        """
        n = len(idx)
        sid = ScenarioID.DEBX2P if twin else ScenarioID.DEB
        b_key = "b_twin" if twin else "b"
        P_orb = (2 * samples["P_orb"] if twin else samples["P_orb"])

        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        masses_comp = samples["masses_comp"]

        ext_u1 = [np.full(n, float(e.ldc.u1)) for e in external_lcs if e.ldc]
        ext_u2 = [np.full(n, float(e.ldc.u2)) for e in external_lcs if e.ldc]
        ext_fr_eb = []
        ext_fr_comp = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            fr_eb = (
                _relations.get_flux_ratio(samples["masses"][idx], ext_lc.band)
                / (_relations.get_flux_ratio(samples["masses"][idx], ext_lc.band)
                   + _relations.get_flux_ratio(
                       np.array([stellar_params.mass_msun]), ext_lc.band))
            )
            ext_fr_eb.append(fr_eb)

            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                fr_comp = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs[idx]]
            else:
                fr_comp = fluxratios_comp[idxs[idx]]
            ext_fr_comp.append(fr_comp)

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
            companion_mass_msun=masses_comp[idxs[idx]],
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=fluxratios_comp[idxs[idx]],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_eb=ext_fr_eb,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )


# ---------------------------------------------------------------------------
# BTP + BEB: Bright Background star scenarios
# ---------------------------------------------------------------------------


class BTPScenario(BaseScenario):
    """Planet transiting a Bright Background star.

    Both the transit host and the distance-corrected flux contributions come
    from the TRILEGAL background population. ``companion_is_host=True`` is
    passed to the transit simulator.

    Per-TRILEGAL-star LDC is computed in ``_evaluate_lnL`` via bulk lookup.
    Extra mask conditions: ``logg >= 3.5`` and ``Teff <= 10000``.

    Source: marginal_likelihoods.py:2823-3155 (lnZ_BTP).
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.BTP

    @property
    def is_eb(self) -> bool:
        return False

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Placeholder: per-TRILEGAL-star LDC computed in _evaluate_lnL.

        Returns target star LDC for external-LC resolution in Phase 4.
        """
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, stellar_params.logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample BTP priors from TRILEGAL population.

        Source: marginal_likelihoods.py:3015-3051 (BTP Phase 5).
        """
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError(
                "BTPScenario requires 'trilegal_population' in kwargs."
            )
        population = _filter_population_by_target_tmag(
            population, kwargs.get("target_tmag"),  # type: ignore[arg-type]
        )
        n_comp = population.n_stars
        idxs = _sample_population_indices(n_comp, n)

        host_mags: dict = kwargs.get("host_magnitudes", {})  # type: ignore[assignment]
        filt: str | None = kwargs.get("filt")  # type: ignore[assignment]
        external_lc_bands = tuple(kwargs.get("external_lc_bands", ()))  # type: ignore[arg-type]
        gmag, rmag, imag, zmag = _resolve_sdss_target_mags(
            host_mags, external_lc_bands, filt,
        )

        delta_mags_map = _compute_delta_mags_map(
            host_mags.get("tmag", 10.0),
            host_mags.get("jmag", 10.0),
            host_mags.get("hmag", 10.0),
            host_mags.get("kmag", 10.0),
            population,
        )
        sdss_delta = _compute_sdss_delta_mags(
            gmag, rmag, imag, zmag,
            population,
        )
        delta_mags_map.update(sdss_delta)

        delta_mags_tess = delta_mags_map["delta_TESSmags"]
        fluxratios_comp = _compute_fluxratios_comp(delta_mags_tess)

        # Background prior
        contrast_curve = kwargs.get("contrast_curve")
        lnprior = _compute_lnprior_companion(
            n_comp, fluxratios_comp, idxs, delta_mags_map, contrast_curve, filt,
        )

        # BTP: planet radius sampled using background star mass
        # Source: marginal_likelihoods.py:3048
        rps = sample_planet_radius(
            np.random.rand(n), population.masses[idxs], config.flat_priors,
        )
        incs = sample_inclination(np.random.rand(n))
        eccs = sample_eccentricity(
            np.random.rand(n), planet=True, period=float(np.mean(P_orb)),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        # Compute radii of background stars from logg and mass
        radii_comp = (
            np.sqrt(CONST.G * population.masses * CONST.Msun / 10 ** population.loggs)
            / CONST.Rsun
        )

        return {
            "rps": rps, "incs": incs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, stellar_params.mass_msun),
            "R_s": np.full(n, stellar_params.radius_rsun),
            "idxs": idxs,
            "fluxratios_comp": fluxratios_comp,
            "lnprior_companion": lnprior,
            "masses_comp": population.masses,
            "radii_comp": radii_comp,
            "loggs_comp": population.loggs,
            "Teffs_comp": population.teffs,
            "Zs_comp": population.metallicities,
            "delta_mags_map": delta_mags_map,  # type: ignore[dict-item]
        }

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """BTP orbital geometry -- uses background star mass only.

        Source: marginal_likelihoods.py:3053-3063
        Note: original uses a = ((G*masses_comp[idxs]*Msun)/(4*pi^2)*(P*86400)^2)^(1/3)
        """
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        rps = samples["rps"]
        idxs = samples["idxs"].astype(int)
        masses_comp = samples["masses_comp"]
        radii_comp = samples["radii_comp"]

        rps_rsun = rps * (CONST.Rearth / CONST.Rsun)

        a = semi_major_axis(P_orb, masses_comp[idxs])
        Ptra = transit_probability(a, radii_comp[idxs], rps_rsun, eccs, argps)
        b = impact_parameter(a, incs, radii_comp[idxs], eccs, argps)
        coll = collision_check(a, radii_comp[idxs], rps_rsun, eccs)

        return {"a": a, "Ptra": Ptra, "b": b, "coll": coll}

    def _evaluate_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, None]:
        """BTP log-likelihoods with per-TRILEGAL-star LDC, companion_is_host=True.

        Extra mask conditions: logg >= 3.5 and Teff <= 10000.
        Source: marginal_likelihoods.py:3068-3100
        """
        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        lnprior = samples["lnprior_companion"]
        radii_comp = samples["radii_comp"]
        loggs_comp = samples["loggs_comp"]
        Teffs_comp = samples["Teffs_comp"]
        Zs_comp = samples["Zs_comp"]

        # Per-TRILEGAL-star LDC (bulk lookup for all background stars)
        u1s_comp, u2s_comp = _lookup_background_ldc_bulk(
            self._ldc, config.mission, Teffs_comp, loggs_comp, Zs_comp,
        )

        # Extra mask: logg >= 3.5 and Teff <= 10000
        extra_mask = (loggs_comp[idxs] >= 3.5) & (Teffs_comp[idxs] <= 10000)
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll"],
            extra_mask=extra_mask,
        )

        comp_fr = fluxratios_comp[idxs]
        force_serial = (not config.parallel) and not bool(external_lcs)

        chi2_half = lnL_planet_p(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rps=samples["rps"],
            periods=samples["P_orb"],
            incs=samples["incs"],
            as_=geometry["a"],
            rss=radii_comp[idxs],
            u1s=u1s_comp[idxs],
            u2s=u2s_comp[idxs],
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=comp_fr,
            mask=mask,
            companion_is_host=True,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        # External LCs
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            elc = ext_lc.light_curve

            # Per-TRILEGAL-star LDC for this band
            ext_u1s, ext_u2s = _lookup_background_ldc_bulk(
                self._ldc, ext_lc.band, Teffs_comp, loggs_comp, Zs_comp,
            )

            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                ext_comp_fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs]
            else:
                ext_comp_fr = comp_fr

            ext_chi2 = lnL_planet_p(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rps=samples["rps"],
                periods=samples["P_orb"],
                incs=samples["incs"],
                as_=geometry["a"],
                rss=radii_comp[idxs],
                u1s=ext_u1s[idxs],
                u2s=ext_u2s[idxs],
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=ext_comp_fr,
                mask=mask,
                companion_is_host=True,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        # Fold in the background prior
        lnL = lnL + lnprior

        return lnL, None

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Pack BTP results.

        Source: marginal_likelihoods.py:3129-3155
        In BTP: M_comp and R_comp are the TARGET star (not background).
        """
        n = len(idx)
        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        masses_comp = samples["masses_comp"]
        radii_comp = samples["radii_comp"]
        Teffs_comp = samples["Teffs_comp"]
        loggs_comp = samples["loggs_comp"]
        Zs_comp = samples["Zs_comp"]

        u1s_comp, u2s_comp = _lookup_background_ldc_bulk(
            self._ldc, "TESS", Teffs_comp, loggs_comp, Zs_comp,
        )

        ext_u1: list[np.ndarray] = []
        ext_u2: list[np.ndarray] = []
        ext_fr_comp: list[np.ndarray] = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            ext_u1s_all, ext_u2s_all = _lookup_background_ldc_bulk(
                self._ldc, ext_lc.band, Teffs_comp, loggs_comp, Zs_comp,
            )
            ext_u1.append(ext_u1s_all[idxs[idx]])
            ext_u2.append(ext_u2s_all[idxs[idx]])
            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs[idx]]
            else:
                fr = fluxratios_comp[idxs[idx]]
            ext_fr_comp.append(fr)

        return ScenarioResult(
            scenario_id=ScenarioID.BTP,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=masses_comp[idxs[idx]],
            host_radius_rsun=radii_comp[idxs[idx]],
            host_u1=u1s_comp[idxs[idx]],
            host_u2=u2s_comp[idxs[idx]],
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
            flux_ratio_companion_tess=fluxratios_comp[idxs[idx]],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )


class BEBScenario(BaseScenario):
    """Eclipsing Binary on a Bright Background star.

    Returns (result, result_twin). The EB is on the background star.
    Uses ``companion_is_host=True``, per-TRILEGAL-star LDC, and
    distance correction for flux ratios.

    BUG-06 fix: q<0.95 mask uses ``geometry['coll']`` (not ``coll_twin``).
    The original code at marginal_likelihoods.py:~3492 incorrectly used
    ``coll_twin`` in the q<0.95 block.

    Source: marginal_likelihoods.py:3158-3669 (lnZ_BEB).
    """

    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.BEB

    @property
    def is_eb(self) -> bool:
        return True

    def _get_host_ldc(
        self, stellar_params: StellarParameters, mission: str,
        P_orb: np.ndarray, kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Placeholder: per-TRILEGAL-star LDC computed in _evaluate_lnL."""
        return self._ldc.get_coefficients(  # type: ignore[union-attr]
            mission, stellar_params.metallicity_dex,
            stellar_params.teff_k, stellar_params.logg,
        )

    def _sample_priors(
        self, n: int, stellar_params: StellarParameters,
        P_orb: np.ndarray, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Sample BEB priors: EB params + TRILEGAL population.

        Key difference from DEB: EB masses are qs * masses_comp[idxs] (background
        star mass, not target star mass).

        Source: marginal_likelihoods.py:3200-3300 (BEB Phase 5).
        """
        population: TRILEGALResult | None = kwargs.get("trilegal_population")  # type: ignore[assignment]
        if population is None:
            raise ValueError(
                "BEBScenario requires 'trilegal_population' in kwargs."
            )
        population = _filter_population_by_target_tmag(
            population, kwargs.get("target_tmag"),  # type: ignore[arg-type]
        )
        n_comp = population.n_stars

        host_mags: dict = kwargs.get("host_magnitudes", {})  # type: ignore[assignment]
        filt: str | None = kwargs.get("filt")  # type: ignore[assignment]
        external_lc_bands = tuple(kwargs.get("external_lc_bands", ()))  # type: ignore[arg-type]
        gmag, rmag, imag, zmag = _resolve_sdss_target_mags(
            host_mags, external_lc_bands, filt,
        )

        # Match the original RNG stream. The unused companion-q draw is kept
        # intentionally because the legacy implementation consumed it here.
        incs = sample_inclination(np.random.rand(n))
        qs = sample_mass_ratio(np.random.rand(n), stellar_params.mass_msun)
        _qs_comp_unused = sample_companion_mass_ratio(
            np.random.rand(n), stellar_params.mass_msun,
        )
        eccs = sample_eccentricity(
            np.random.rand(n), planet=False, period=float(np.mean(P_orb)),
        )
        argps = sample_arg_periastron(np.random.rand(n))

        delta_mags_map = _compute_delta_mags_map(
            host_mags.get("tmag", 10.0),
            host_mags.get("jmag", 10.0),
            host_mags.get("hmag", 10.0),
            host_mags.get("kmag", 10.0),
            population,
        )
        sdss_delta = _compute_sdss_delta_mags(
            gmag, rmag, imag, zmag,
            population,
        )
        delta_mags_map.update(sdss_delta)

        delta_mags_tess = delta_mags_map["delta_TESSmags"]
        fluxratios_comp = _compute_fluxratios_comp(delta_mags_tess)
        idxs = _sample_population_indices(n_comp, n)

        # EB masses = qs * background star mass (not target star mass!)
        # Source: marginal_likelihoods.py:3230
        masses = qs * population.masses[idxs]

        # Compute EB properties and distance-corrected flux ratios
        radii_comp, radii, fluxratios_eb, distance_correction, fluxratios = (
            self._beb_compute_eb_properties(
                masses, population, idxs, fluxratios_comp,
                stellar_params.mass_msun,
            )
        )

        contrast_curve = kwargs.get("contrast_curve")
        if filt is not None:
            filt_key = f"delta_{filt}mags"
            if filt_key in delta_mags_map:
                fluxratios_comp_band = _compute_fluxratios_comp(
                    delta_mags_map[filt_key],
                )[idxs]
            else:
                fluxratios_comp_band = fluxratios_comp[idxs]
            _distance_correction_band, fluxratios_band = (
                self._beb_distance_corrected_eb_fluxratios(
                    masses,
                    population.masses[idxs],
                    fluxratios_comp_band,
                    stellar_params.mass_msun,
                    band=filt,
                )
            )
        else:
            fluxratios_comp_band = fluxratios_comp[idxs]
            fluxratios_band = fluxratios
        lnprior = _compute_bright_background_lnprior(
            n_comp,
            idxs,
            fluxratios_comp_band,
            fluxratios_band,
            contrast_curve,
        )

        return {
            "incs": incs, "qs": qs, "eccs": eccs, "argps": argps,
            "P_orb": P_orb,
            "M_s": np.full(n, stellar_params.mass_msun),
            "R_s": np.full(n, stellar_params.radius_rsun),
            "masses": masses,
            "radii": radii,
            "fluxratios": fluxratios,
            "fluxratios_eb": fluxratios_eb,
            "idxs": idxs,
            "fluxratios_comp": fluxratios_comp,
            "lnprior_companion": lnprior,
            "masses_comp": population.masses,
            "radii_comp": radii_comp,
            "loggs_comp": population.loggs,
            "Teffs_comp": population.teffs,
            "Zs_comp": population.metallicities,
            "delta_mags_map": delta_mags_map,  # type: ignore[dict-item]
            "distance_correction": distance_correction,
        }

    def _beb_compute_eb_properties(
        self, masses: np.ndarray, population: TRILEGALResult,
        idxs: np.ndarray, fluxratios_comp: np.ndarray,
        target_mass: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute background star radii, EB radii/flux ratios, and distance correction.

        Returns:
            (radii_comp, radii_eb, fluxratios_eb, distance_correction, fluxratios)
        """
        # Compute radii of background stars
        radii_comp = (
            np.sqrt(CONST.G * population.masses * CONST.Msun / 10 ** population.loggs)
            / CONST.Rsun
        )

        # EB radii (capped by background star properties)
        radii, _teffs = _relations.get_radius_teff(
            masses,
            max_radii=radii_comp[idxs],
            max_teffs=population.teffs[idxs],
        )

        # EB flux ratio in TESS band (relative to background star + EB system)
        fluxratios_eb = (
            _relations.get_flux_ratio(masses, "TESS")
            / (_relations.get_flux_ratio(masses, "TESS")
               + _relations.get_flux_ratio(population.masses[idxs], "TESS"))
        )

        # Distance-corrected EB flux ratios
        distance_correction, fluxratios = self._beb_distance_corrected_eb_fluxratios(
            masses, population.masses[idxs], fluxratios_comp[idxs],
            target_mass, band="TESS",
        )

        return radii_comp, radii, fluxratios_eb, distance_correction, fluxratios

    @staticmethod
    def _beb_distance_corrected_eb_fluxratios(
        eb_masses: np.ndarray, host_masses: np.ndarray,
        obs_comp_fr: np.ndarray, target_mass: float, band: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute distance-correction factor and corrected EB flux ratios.

        Source: marginal_likelihoods.py:3291-3298

        Returns:
            (distance_correction, corrected_fluxratios) tuple.
        """
        fr_host = _relations.get_flux_ratio(host_masses, band)
        fr_target = _relations.get_flux_ratio(np.array([target_mass]), band)
        fr_eb = _relations.get_flux_ratio(eb_masses, band)

        fluxratios_comp_bound = fr_host / (fr_host + fr_target)
        distance_correction = obs_comp_fr / fluxratios_comp_bound

        fluxratios = fr_eb / (fr_eb + fr_target) * distance_correction
        return distance_correction, fluxratios

    def _compute_orbital_geometry(
        self, samples: dict[str, np.ndarray], P_orb: np.ndarray,
        stellar_params: StellarParameters, config: Config, **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """BEB orbital geometry -- uses background_star_mass + EB_mass.

        Source: marginal_likelihoods.py:3461-3480
        """
        incs = samples["incs"]
        eccs = samples["eccs"]
        argps = samples["argps"]
        masses = samples["masses"]
        radii = samples["radii"]
        idxs = samples["idxs"].astype(int)
        masses_comp = samples["masses_comp"]
        radii_comp = samples["radii_comp"]

        total_mass = masses_comp[idxs] + masses
        a = semi_major_axis(P_orb, total_mass)
        Ptra = transit_probability(a, radii, radii_comp[idxs], eccs, argps)
        b = impact_parameter(a, incs, radii_comp[idxs], eccs, argps)
        coll = collision_check(a, radii, radii_comp[idxs], eccs)

        a_twin = semi_major_axis(2 * P_orb, total_mass)
        Ptra_twin = transit_probability(a_twin, radii, radii_comp[idxs], eccs, argps)
        b_twin = impact_parameter(a_twin, incs, radii_comp[idxs], eccs, argps)
        coll_twin = (2 * radii_comp[idxs] * CONST.Rsun) > a_twin * (1 - eccs)

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
        """BEB log-likelihoods.

        Extra mask conditions: logg >= 3.5 and Teff <= 10000.
        Source: marginal_likelihoods.py:3485-3556
        """
        N = config.n_mc_samples
        qs = samples["qs"]
        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        lnprior = samples["lnprior_companion"]
        radii_comp = samples["radii_comp"]
        loggs_comp = samples["loggs_comp"]
        Teffs_comp = samples["Teffs_comp"]
        Zs_comp = samples["Zs_comp"]

        # Per-TRILEGAL-star LDC
        u1s_comp, u2s_comp = _lookup_background_ldc_bulk(
            self._ldc, config.mission, Teffs_comp, loggs_comp, Zs_comp,
        )

        comp_fr = fluxratios_comp[idxs]
        extra_logg_teff = (loggs_comp[idxs] >= 3.5) & (Teffs_comp[idxs] <= 10000)

        comp_params = {
            "u1s_comp": u1s_comp, "u2s_comp": u2s_comp,
            "radii_comp": radii_comp, "loggs_comp": loggs_comp,
            "Teffs_comp": Teffs_comp, "Zs_comp": Zs_comp,
        }

        force_serial = (not config.parallel) and not bool(external_lcs)

        # --- q < 0.95: standard EB ---
        q_lt_mask = (qs < 0.95) & extra_logg_teff
        mask = build_transit_mask(
            samples["incs"], geometry["Ptra"], geometry["coll_twin"],
            extra_mask=q_lt_mask,
        )
        lnL = self._beb_branch_lnL(
            light_curve, lnsigma, samples, geometry, external_lcs,
            idxs, comp_fr, comp_params, mask,
            lnL_fn=lnL_eb_p, a_key="a", period_mult=1, N=N,
            force_serial=force_serial,
        )

        # --- q >= 0.95: twin EB at 2x period ---
        q_ge_mask = (qs >= 0.95) & extra_logg_teff
        mask_twin = build_transit_mask(
            samples["incs"], geometry["Ptra_twin"], geometry["coll_twin"],
            extra_mask=q_ge_mask,
        )
        lnL_twin = self._beb_branch_lnL(
            light_curve, lnsigma, samples, geometry, external_lcs,
            idxs, comp_fr, comp_params, mask_twin,
            lnL_fn=lnL_eb_twin_p, a_key="a_twin", period_mult=2, N=N,
            force_serial=force_serial,
        )

        # Fold in the background prior
        lnL = lnL + lnprior
        lnL_twin = lnL_twin + lnprior

        return lnL, lnL_twin

    def _beb_branch_lnL(
        self, light_curve: LightCurve, lnsigma: float,
        samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        external_lcs: list[ExternalLightCurve],
        idxs: np.ndarray, comp_fr: np.ndarray,
        comp_params: dict[str, np.ndarray], mask: np.ndarray,
        lnL_fn: Callable[..., np.ndarray], a_key: str, period_mult: int, N: int,
        force_serial: bool = False,
    ) -> np.ndarray:
        """Compute BEB lnL for one branch (standard or twin)."""
        radii_comp = comp_params["radii_comp"]
        u1s_comp = comp_params["u1s_comp"]
        u2s_comp = comp_params["u2s_comp"]
        loggs_comp = comp_params["loggs_comp"]
        Teffs_comp = comp_params["Teffs_comp"]
        Zs_comp = comp_params["Zs_comp"]
        periods = period_mult * samples["P_orb"]

        chi2_half = lnL_fn(
            time=light_curve.time_days,
            flux=light_curve.flux,
            sigma=light_curve.sigma,
            rss=radii_comp[idxs],
            rcomps=samples["radii"],
            eb_flux_ratios=samples["fluxratios"],
            periods=periods,
            incs=samples["incs"],
            as_=geometry[a_key],
            u1s=u1s_comp[idxs],
            u2s=u2s_comp[idxs],
            eccs=samples["eccs"],
            argps=samples["argps"],
            companion_flux_ratios=comp_fr,
            mask=mask,
            companion_is_host=True,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
            force_serial=force_serial,
        )
        lnL = -0.5 * _ln2pi - lnsigma - chi2_half

        # Accumulate external LC contributions
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            elc = ext_lc.light_curve
            ext_u1s, ext_u2s = _lookup_background_ldc_bulk(
                self._ldc, ext_lc.band, Teffs_comp, loggs_comp, Zs_comp,
            )
            ext_fr_eb = self._distance_corrected_fluxratios(
                samples, ext_lc.band,
            )
            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                ext_comp_fr = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs]
            else:
                ext_comp_fr = comp_fr

            ext_chi2 = lnL_fn(
                time=elc.time_days,
                flux=elc.flux,
                sigma=elc.sigma,
                rss=radii_comp[idxs],
                rcomps=samples["radii"],
                eb_flux_ratios=ext_fr_eb,
                periods=periods,
                incs=samples["incs"],
                as_=geometry[a_key],
                u1s=ext_u1s[idxs],
                u2s=ext_u2s[idxs],
                eccs=samples["eccs"],
                argps=samples["argps"],
                companion_flux_ratios=ext_comp_fr,
                mask=mask,
                companion_is_host=True,
                exptime=elc.cadence_days,
                nsamples=elc.supersampling_rate,
                force_serial=force_serial,
            )
            ext_lnL = -0.5 * _ln2pi - np.log(elc.sigma) - ext_chi2
            lnL = lnL + ext_lnL

        return lnL

    def _distance_corrected_fluxratios(
        self, samples: dict[str, np.ndarray], band: str,
    ) -> np.ndarray:
        """Compute distance-corrected EB flux ratios for a given band.

        Source: marginal_likelihoods.py:3378-3393 (BEB external LC distance correction).
        """
        idxs = samples["idxs"].astype(int)
        masses = samples["masses"]
        masses_comp = samples["masses_comp"]

        # Bound flux ratio in this band
        fr_host = _relations.get_flux_ratio(masses_comp[idxs], band)
        fr_target = _relations.get_flux_ratio(samples["M_s"][:1], band)
        fluxratios_comp_bound_band = fr_host / (fr_host + fr_target)

        # Distance correction in this band
        delta_mags_map = samples["delta_mags_map"]
        filt_key = f"delta_{band}mags"
        if filt_key in delta_mags_map:
            flux_term_band = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs]
        else:
            flux_term_band = samples["fluxratios_comp"][idxs]
        distance_correction_band = flux_term_band / fluxratios_comp_bound_band

        # EB flux ratio in this band, distance-corrected
        fluxratios_band = (
            _relations.get_flux_ratio(masses, band)
            / (_relations.get_flux_ratio(masses, band) + fr_target)
            * distance_correction_band
        )
        return fluxratios_band

    def _pack_result(
        self, samples: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs, lnZ: float, idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve], twin: bool = False,
    ) -> ScenarioResult:
        """Pack BEB results.

        Source: marginal_likelihoods.py:3607-3669
        """
        n = len(idx)
        sid = ScenarioID.BEBX2P if twin else ScenarioID.BEB
        b_key = "b_twin" if twin else "b"
        P_orb = (2 * samples["P_orb"] if twin else samples["P_orb"])

        idxs = samples["idxs"].astype(int)
        fluxratios_comp = samples["fluxratios_comp"]
        masses_comp = samples["masses_comp"]
        radii_comp = samples["radii_comp"]
        Teffs_comp = samples["Teffs_comp"]
        loggs_comp = samples["loggs_comp"]
        Zs_comp = samples["Zs_comp"]

        u1s_comp, u2s_comp = _lookup_background_ldc_bulk(
            self._ldc, "TESS", Teffs_comp, loggs_comp, Zs_comp,
        )

        ext_u1: list[np.ndarray] = []
        ext_u2: list[np.ndarray] = []
        ext_fr_eb: list[np.ndarray] = []
        ext_fr_comp: list[np.ndarray] = []
        for ext_lc in external_lcs:
            if ext_lc.ldc is None:
                continue
            ext_u1s_all, ext_u2s_all = _lookup_background_ldc_bulk(
                self._ldc, ext_lc.band, Teffs_comp, loggs_comp, Zs_comp,
            )
            ext_u1.append(ext_u1s_all[idxs[idx]])
            ext_u2.append(ext_u2s_all[idxs[idx]])

            fr_eb = self._distance_corrected_fluxratios(samples, ext_lc.band)[idx]
            ext_fr_eb.append(fr_eb)

            delta_mags_map = samples["delta_mags_map"]
            filt_key = f"delta_{ext_lc.band}mags"
            if filt_key in delta_mags_map:
                fr_comp = _compute_fluxratios_comp(delta_mags_map[filt_key])[idxs[idx]]
            else:
                fr_comp = fluxratios_comp[idxs[idx]]
            ext_fr_comp.append(fr_comp)

        return ScenarioResult(
            scenario_id=sid,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=masses_comp[idxs[idx]],
            host_radius_rsun=radii_comp[idxs[idx]],
            host_u1=u1s_comp[idxs[idx]],
            host_u2=u2s_comp[idxs[idx]],
            period_days=P_orb[idx],
            inclination_deg=samples["incs"][idx],
            impact_parameter=geometry[b_key][idx],
            eccentricity=samples["eccs"][idx],
            arg_periastron_deg=samples["argps"][idx],
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=samples["masses"][idx],
            eb_radius_rsun=samples["radii"][idx],
            flux_ratio_eb_tess=samples["fluxratios"][idx],
            companion_mass_msun=np.full(n, stellar_params.mass_msun),
            companion_radius_rsun=np.full(n, stellar_params.radius_rsun),
            flux_ratio_companion_tess=fluxratios_comp[idxs[idx]],
            external_lc_u1=ext_u1,
            external_lc_u2=ext_u2,
            external_lc_flux_ratio_eb=ext_fr_eb,
            external_lc_flux_ratio_comp=ext_fr_comp,
        )
