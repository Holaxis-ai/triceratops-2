"""Module-level helper functions for companion-star scenarios (PTP, PEB, STP, SEB).

These helpers handle companion star photometry, SDSS magnitude handling,
MOLUSC file loading, and flux ratio utilities that are shared across
the four companion-star scenario classes.

This module is intentionally separate from background scenario helpers
(which deal with TRILEGAL stellar populations).
"""
from __future__ import annotations

import numpy as np
from pandas import read_csv

from triceratops.priors.lnpriors import lnprior_bound_companion
from triceratops.stellar.relations import StellarRelations

_relations = StellarRelations()

_ln2pi = np.log(2 * np.pi)


def _load_molusc_qs(
    molusc_file: str,
    n: int,
    primary_mass: float,
) -> np.ndarray:
    """Load companion mass ratios from a MOLUSC output file.

    Source: marginal_likelihoods.py:788-797 (PTP), 1040-1050 (PEB).

    Filters: keeps only rows where a*(1-e) > 10 AU (wide-separation companions).
    Pads with zeros if shorter than n, or truncates if longer.

    Args:
        molusc_file: Path to the MOLUSC CSV.
        n: Desired output array length.
        primary_mass: Primary star mass in Msun.

    Returns:
        Array of mass ratios, shape (n,).
    """
    df = read_csv(molusc_file)
    molusc_a = np.asarray(df["semi-major axis(AU)"])
    molusc_e = np.asarray(df["eccentricity"])
    df2 = df[molusc_a * (1 - molusc_e) > 10]
    qs = np.asarray(df2["mass ratio"]).copy()
    qs[qs < 0.1 / primary_mass] = 0.1 / primary_mass
    if len(qs) < n:
        qs = np.pad(qs, (0, n - len(qs)))
    else:
        qs = qs[:n]
    return qs


def _compute_companion_properties(
    qs_comp: np.ndarray,
    M_s: float,
    R_s: float,
    Teff: float,
    n: int,
    filt: str = "TESS",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute companion star masses, radii, Teffs, and TESS flux ratios.

    Source: marginal_likelihoods.py:799-801 (PTP), 1062-1071 (PEB).

    Returns:
        (masses_comp, radii_comp, Teffs_comp, fluxratios_comp)
    """
    masses_comp = qs_comp * M_s
    radii_comp, Teffs_comp = _relations.get_radius_teff(
        masses_comp,
        max_radii=np.full(n, R_s),
        max_teffs=np.full(n, Teff),
    )
    flux_comp = _relations.get_flux_ratio(masses_comp, filt)
    flux_primary = _relations.get_flux_ratio(np.array([M_s]), filt)
    fluxratios_comp = flux_comp / (flux_comp + flux_primary)
    return masses_comp, radii_comp, Teffs_comp, fluxratios_comp


def _compute_companion_prior(
    masses_comp: np.ndarray,
    fluxratios_comp: np.ndarray,
    M_s: float,
    plx: float,
    n: int,
    molusc_file: object | None,
    contrast_curve: object | None,
    filt: str,
    is_eb: bool,
) -> np.ndarray:
    """Compute the companion prior (lnprior_companion).

    Source: marginal_likelihoods.py:803-817 (PTP), 1074-1104 (PEB).

    Returns:
        lnprior_companion array, shape (N,).
    """
    if molusc_file is not None:
        return np.zeros(n)

    if contrast_curve is None:
        # No contrast curve: use default wide separations
        # Source: line 805-808 (PTP), 1076-1084 (PEB)
        delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
        lnprior_comp = lnprior_bound_companion(
            delta_mags=np.abs(delta_mags),
            separations_arcsec=np.array([2.2]),
            contrasts=np.array([1.0]),
            primary_mass_msun=M_s,
            parallax_mas=plx,
            is_eb=is_eb,
        )
        lnprior_comp[lnprior_comp > 0.0] = 0.0
        lnprior_comp[delta_mags > 0.0] = -np.inf
        return lnprior_comp
    else:
        # With contrast curve: compute flux ratio in contrast curve filter
        # Source: line 810-815 (PTP), 1087-1102 (PEB)
        flux_comp_cc = _relations.get_flux_ratio(masses_comp, filt)
        flux_primary_cc = _relations.get_flux_ratio(np.array([M_s]), filt)
        fluxratios_comp_cc = flux_comp_cc / (flux_comp_cc + flux_primary_cc)
        delta_mags = 2.5 * np.log10(fluxratios_comp_cc / (1 - fluxratios_comp_cc))
        lnprior_comp = lnprior_bound_companion(
            delta_mags=np.abs(delta_mags),
            separations_arcsec=contrast_curve.separations_arcsec,  # type: ignore[union-attr]
            contrasts=contrast_curve.delta_mags,  # type: ignore[union-attr]
            primary_mass_msun=M_s,
            parallax_mas=plx,
            is_eb=is_eb,
        )
        lnprior_comp[lnprior_comp > 0.0] = 0.0
        lnprior_comp[delta_mags > 0.0] = -np.inf
        return lnprior_comp


def _flux_ratio_in_band(
    masses: np.ndarray,
    primary_mass_msun: float,
    band: str,
) -> np.ndarray:
    """Return flux ratio F_draw / (F_draw + F_primary) in the requested band."""
    flux_draw = _relations.get_flux_ratio(masses, band)
    flux_primary = _relations.get_flux_ratio(np.array([primary_mass_msun]), band)
    return flux_draw / (flux_draw + flux_primary)


def _bulk_companion_ldc(
    ldc_catalog: object,
    band: str,
    teffs: np.ndarray,
    loggs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bulk LDC lookup for companion-hosted scenarios in an external band."""
    return ldc_catalog.get_coefficients_bulk(  # type: ignore[union-attr]
        band,
        teffs,
        loggs,
        np.zeros(len(teffs)),
    )


def _compute_seb_companion_prior(
    masses_comp: np.ndarray,
    fluxratios_comp: np.ndarray,
    masses_eb: np.ndarray,
    fluxratios_eb: np.ndarray,
    M_s: float,
    plx: float,
    n: int,
    molusc_file: object | None,
    contrast_curve: object | None,
    filt: str,
) -> np.ndarray:
    """Companion prior for SEB including both EB and companion flux ratios.

    Source: marginal_likelihoods.py:1819-1858.
    delta_mags includes BOTH companion and EB contributions.
    """
    if molusc_file is not None:
        return np.zeros(n)

    if contrast_curve is None:
        delta_mags = 2.5 * np.log10(
            fluxratios_comp / (1 - fluxratios_comp)
            + fluxratios_eb / (1 - fluxratios_eb)
        )
        lnprior_comp = lnprior_bound_companion(
            delta_mags=np.abs(delta_mags),
            separations_arcsec=np.array([2.2]),
            contrasts=np.array([1.0]),
            primary_mass_msun=M_s,
            parallax_mas=plx,
            is_eb=True,
        )
        lnprior_comp[lnprior_comp > 0.0] = 0.0
        lnprior_comp[delta_mags > 0.0] = -np.inf
        return lnprior_comp
    else:
        # With contrast curve: recompute flux ratios in CC filter
        flux_comp_cc = _relations.get_flux_ratio(masses_comp, filt)
        flux_primary_cc = _relations.get_flux_ratio(np.array([M_s]), filt)
        fluxratios_comp_cc = flux_comp_cc / (flux_comp_cc + flux_primary_cc)

        flux_eb_cc = _relations.get_flux_ratio(masses_eb, filt)
        fluxratios_eb_cc = flux_eb_cc / (flux_eb_cc + flux_primary_cc)

        delta_mags = 2.5 * np.log10(
            fluxratios_comp_cc / (1 - fluxratios_comp_cc)
            + fluxratios_eb_cc / (1 - fluxratios_eb_cc)
        )
        lnprior_comp = lnprior_bound_companion(
            delta_mags=np.abs(delta_mags),
            separations_arcsec=contrast_curve.separations_arcsec,  # type: ignore[union-attr]
            contrasts=contrast_curve.delta_mags,  # type: ignore[union-attr]
            primary_mass_msun=M_s,
            parallax_mas=plx,
            is_eb=True,
        )
        lnprior_comp[lnprior_comp > 0.0] = 0.0
        lnprior_comp[delta_mags > 0.0] = -np.inf
        return lnprior_comp
