"""Module-level helper functions for background star scenarios.

Extracted from background_scenarios.py to improve modularity and testability.
These helpers are shared across DTPScenario, DEBScenario, BTPScenario, and
BEBScenario but are not tightly bound to any one class.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from triceratops.domain.value_objects import normalize_contrast_curves
from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.lnpriors import (
    combine_allowed_separations,
    lnprior_background,
    lnprior_background_from_separations,
    separation_at_contrast_or_inf_if_blind,
)
from triceratops.scenarios.constants import (
    ARCSEC_TO_DEG,
    COMPANION_DEFAULT_SEP_ARCSEC,
)
from triceratops.stellar.relations import StellarRelations, canonicalize_filter_name

_relations = StellarRelations()
_SDSS_BANDS = frozenset({"g", "r", "i", "z"})


def _max_curve_owa_arcsec(curves: tuple) -> float:
    return max(float(np.max(curve.separations_arcsec)) for curve in curves)


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
    external_lc_bands: tuple[str, ...],
    filt: str | None,
    contrast_curve: object | None = None,
) -> bool:
    """Return True when any active band requires SDSS photometry."""
    bands = {canonicalize_filter_name(band) for band in external_lc_bands}
    if filt is not None:
        bands.add(canonicalize_filter_name(filt))
    for curve in normalize_contrast_curves(contrast_curve):
        bands.add(canonicalize_filter_name(curve.band))
    return any(band in _SDSS_BANDS for band in bands)


def _resolve_sdss_target_mags(
    host_mags: dict[str, float | None],
    external_lc_bands: tuple[str, ...],
    filt: str | None,
    contrast_curve: object | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve target g/r/i/z, estimating them when the original code did."""
    gmag = host_mags.get("gmag")
    rmag = host_mags.get("rmag")
    imag = host_mags.get("imag")
    zmag = host_mags.get("zmag")

    no_sdss = all(
        mag is None or np.isnan(mag) for mag in (gmag, rmag, imag, zmag)
    )
    if no_sdss and _needs_sdss_delta_mags(
        external_lc_bands, filt, contrast_curve,
    ):
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


def _delta_mag_key_for_band(band: str | None) -> str:
    """Map a curve/filter band to the available TRILEGAL delta-mag array."""
    canonical = canonicalize_filter_name(band or "TESS")
    filt_key_map = {
        "TESS": "delta_TESSmags",
        "Vis": "delta_TESSmags",
        "Kepler": "delta_TESSmags",
        "J": "delta_Jmags",
        "H": "delta_Hmags",
        "K": "delta_Kmags",
        "g": "delta_gmags",
        "r": "delta_rmags",
        "i": "delta_imags",
        "z": "delta_zmags",
    }
    if canonical not in filt_key_map:
        raise ValueError(
            f"Unsupported contrast curve band {band!r}; expected one of "
            "TESS, Vis, Kepler, J, H, K, g, r, i, z or a supported alias."
        )
    return filt_key_map[canonical]


def _delta_mags_for_band(
    delta_mags_map: dict[str, np.ndarray],
    band: str | None,
) -> np.ndarray:
    """Return delta magnitudes for a curve band, failing clearly if absent."""
    key = _delta_mag_key_for_band(band)
    if key not in delta_mags_map:
        raise ValueError(
            f"Contrast curve band {band!r} requires {key}, but those "
            "target/population magnitudes were not available."
        )
    return delta_mags_map[key]


def _compute_lnprior_companion(
    n_comp: int,
    fluxratios_comp: np.ndarray,
    idxs: np.ndarray,
    delta_mags_map: dict[str, np.ndarray],
    contrast_curve: object | None,
    filt: str | None,
    numerical_mode: str = "corrected",
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

    curves = normalize_contrast_curves(contrast_curve)

    if not curves:
        # Recompute delta_mags from flux ratios for the drawn samples
        fr = fluxratios_comp[idxs]
        delta_mags_drawn = 2.5 * np.log10(fr / (1 - fr))
        if numerical_mode == "legacy":
            lnprior = np.full(
                n,
                np.log10(
                    (n_comp / 0.1)
                    * ARCSEC_TO_DEG**2
                    * COMPANION_DEFAULT_SEP_ARCSEC**2
                ),
            )
        elif numerical_mode == "corrected":
            lnprior = np.full(
                n,
                np.log(
                    (n_comp / 0.1)
                    * ARCSEC_TO_DEG**2
                    * COMPANION_DEFAULT_SEP_ARCSEC**2
                ),
            )
        else:
            raise ValueError(
                "numerical_mode must be 'corrected' or 'legacy', "
                f"got {numerical_mode!r}"
            )
        lnprior[lnprior > 0.0] = 0.0
        lnprior[delta_mags_drawn > 0.0] = -np.inf
        return lnprior

    allowed_separations = []
    bright_mask = np.zeros(n, dtype=bool)
    for curve in curves:
        delta_mags_band = _delta_mags_for_band(delta_mags_map, curve.band)[idxs]
        allowed_separations.append(
            separation_at_contrast_or_inf_if_blind(
                np.abs(delta_mags_band),
                curve.separations_arcsec,
                curve.delta_mags,
            )
        )
        bright_mask |= delta_mags_band > 0.0

    if len(curves) == 1:
        curve = curves[0]
        delta_mags_band = _delta_mags_for_band(delta_mags_map, curve.band)[idxs]
        lnprior = lnprior_background(
            n_comp,
            np.abs(delta_mags_band),
            curve.separations_arcsec,
            curve.delta_mags,
            numerical_mode=numerical_mode,
        )
    else:
        combined_separations = combine_allowed_separations(
            allowed_separations,
            all_blind_fallback_arcsec=_max_curve_owa_arcsec(curves),
        )
        lnprior = lnprior_background_from_separations(
            n_comp,
            combined_separations,
            numerical_mode=numerical_mode,
        )
    lnprior[lnprior > 0.0] = 0.0
    lnprior[bright_mask] = -np.inf
    return lnprior


def _compute_bright_background_lnprior(
    n_comp: int,
    idxs: np.ndarray,
    fluxratios_comp_band: np.ndarray,
    fluxratios_eb_band: np.ndarray,
    contrast_curve: object | None,
    numerical_mode: str = "corrected",
    band_fluxratio_resolver: Callable[[str], tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Background prior for BEB using combined host+EB brightness."""
    n = len(idxs)
    delta_mags = _combined_delta_mag(
        fluxratios_comp_band, fluxratios_eb_band,
    )
    curves = normalize_contrast_curves(contrast_curve)

    if not curves:
        if numerical_mode == "legacy":
            lnprior = np.full(
                n,
                np.log10(
                    (n_comp / 0.1)
                    * ARCSEC_TO_DEG**2
                    * COMPANION_DEFAULT_SEP_ARCSEC**2
                ),
            )
        elif numerical_mode == "corrected":
            lnprior = np.full(
                n,
                np.log(
                    (n_comp / 0.1)
                    * ARCSEC_TO_DEG**2
                    * COMPANION_DEFAULT_SEP_ARCSEC**2
                ),
            )
        else:
            raise ValueError(
                "numerical_mode must be 'corrected' or 'legacy', "
                f"got {numerical_mode!r}"
            )
        bright_mask = delta_mags > 0.0
    else:
        allowed_separations = []
        bright_mask = np.zeros(n, dtype=bool)
        for curve in curves:
            if band_fluxratio_resolver is None:
                raise ValueError(
                    "Contrast curves for bright-background scenarios require "
                    "a band_fluxratio_resolver."
                )
            comp_fr, eb_fr = band_fluxratio_resolver(curve.band)
            delta_mags_curve = _combined_delta_mag(comp_fr, eb_fr)
            allowed_separations.append(
                separation_at_contrast_or_inf_if_blind(
                    np.abs(delta_mags_curve),
                    curve.separations_arcsec,
                    curve.delta_mags,
                )
            )
            bright_mask |= delta_mags_curve > 0.0

        combined_separations = combine_allowed_separations(
            allowed_separations,
            all_blind_fallback_arcsec=_max_curve_owa_arcsec(curves),
        )
        lnprior = lnprior_background_from_separations(
            n_comp,
            combined_separations,
            numerical_mode=numerical_mode,
        )

    lnprior[lnprior > 0.0] = 0.0
    lnprior[bright_mask] = -np.inf
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
