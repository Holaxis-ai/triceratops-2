"""PSF flux ratio and transit depth computation.

Replaces the dblquad(Gauss2D) logic in triceratops.calc_depths() with the
exact analytic integral of a separable 2D Gaussian over a pixel box.

The 2D Gaussian integral over a pixel box [x-0.5, x+0.5] × [y-0.5, y+0.5]
centred at (mu_x, mu_y) with amplitude A and std sigma is:

    A * [Φ((x+0.5-mu_x)/sigma) - Φ((x-0.5-mu_x)/sigma)]
      * [Φ((y+0.5-mu_y)/sigma) - Φ((y-0.5-mu_y)/sigma)]

where Φ = scipy.special.ndtr (the standard normal CDF).  This is
mathematically exact — not an approximation — and replaces K×S×P calls to
adaptive quadrature with a single vectorised ndtr computation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from triceratops.domain.entities import StellarField


def compute_flux_ratios(
    field: StellarField,
    pixel_coords_per_sector: list[np.ndarray],
    aperture_pixels_per_sector: list[np.ndarray],
    sigma_psf_px: float = 0.75,
) -> list[float]:
    """Compute the fraction of aperture flux contributed by each star.

    Ports the PSF integration loop from triceratops.calc_depths() (lines 542-581)
    using the analytic separable Gaussian CDF form instead of dblquad.

    Args:
        field: StellarField with all stars.
        pixel_coords_per_sector: List of arrays, each shape (N_stars, 2),
            giving (col, row) pixel positions for each star in each sector.
        aperture_pixels_per_sector: List of arrays, each shape (N_pixels, 2),
            giving (col, row) of each aperture pixel per sector.
        sigma_psf_px: PSF sigma in pixels (original default: 0.75).

    Returns:
        List of flux ratios (one per star). Sum ~= 1.0.
    """
    n_sectors = len(pixel_coords_per_sector)
    n_stars = len(field.stars)

    # Relative brightness of each star normalised to brightest
    tmags = np.array([s.tmag for s in field.stars])
    min_tmag = np.min(tmags)
    brightness = 10.0 ** ((min_tmag - tmags) / 2.5)  # shape (S,)

    flux_ratio_per_sector = np.zeros((n_sectors, n_stars))

    for k in range(n_sectors):
        pix = aperture_pixels_per_sector[k]    # (P, 2): col, row
        coords = pixel_coords_per_sector[k]    # (S, 2): col, row

        px = pix[:, 0][:, None]                # (P, 1) col
        py = pix[:, 1][:, None]                # (P, 1) row
        mu_x = coords[:, 0][None, :]           # (1, S) col
        mu_y = coords[:, 1][None, :]           # (1, S) row

        # Analytic integral over each pixel box for each star: shape (P, S)
        pixel_flux = (
            brightness[None, :]
            * (ndtr((px + 0.5 - mu_x) / sigma_psf_px) - ndtr((px - 0.5 - mu_x) / sigma_psf_px))
            * (ndtr((py + 0.5 - mu_y) / sigma_psf_px) - ndtr((py - 0.5 - mu_y) / sigma_psf_px))
        )

        rel_flux = pixel_flux.sum(axis=0)      # (S,)
        total_flux = float(rel_flux.sum())
        if total_flux > 0:
            flux_ratio_per_sector[k, :] = rel_flux / total_flux

    avg_ratios = np.mean(flux_ratio_per_sector, axis=0)
    return avg_ratios.tolist()


def compute_transit_depths(
    flux_ratios: list[float],
    observed_transit_depth: float,
) -> list[float]:
    """Compute the intrinsic transit depth required if each star is the host.

    Ports the depth scaling from triceratops.calc_depths() (lines 584-588).
    The formula: tdepth = 1 - (flux_ratio - observed_depth) / flux_ratio
                        = observed_depth / flux_ratio

    Args:
        flux_ratios: Output of compute_flux_ratios().
        observed_transit_depth: The measured transit depth (fractional).

    Returns:
        List of intrinsic depths, one per star. Zero-flux stars get inf.
        Depths > 1.0 are set to 0.0 (unphysical).
    """
    depths: list[float] = []
    for fr in flux_ratios:
        if fr > 0:
            d = observed_transit_depth / fr
            depths.append(0.0 if d > 1.0 else d)
        else:
            depths.append(float("inf"))
    return depths
