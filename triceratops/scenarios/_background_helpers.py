"""Module-level helper functions for background star scenarios.

Extracted from background_scenarios.py to improve modularity and testability.
These helpers are shared across DTPScenario, DEBScenario, BTPScenario, and
BEBScenario but are not tightly bound to any one class.
"""
from __future__ import annotations

import numpy as np

from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.lnpriors import lnprior_background
from triceratops.stellar.relations import StellarRelations

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
        lnprior = np.full(n, np.log((n_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2))
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
        lnprior = np.full(n, np.log((n_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2))
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
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the original bright-background LDC lookup order exactly.

    The optional ``cache`` dict is keyed by ``band``.  Pass
    ``samples.get("_ldc_cache")`` at each call site so that repeated calls
    within one scenario.compute() (e.g. _evaluate_lnL then _pack_result) skip
    the Python loop entirely after the first call.
    """
    if cache is not None and band in cache:
        return cache[band]

    load_filter = getattr(ldc_catalog, "_load_filter", None)
    if not callable(load_filter):
        result = ldc_catalog.get_coefficients_bulk(  # type: ignore[union-attr]
            band, teffs, loggs, metallicities,
        )
        if cache is not None:
            cache[band] = result
        return result

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
    result = u1_out, u2_out
    if cache is not None:
        cache[band] = result
    return result
