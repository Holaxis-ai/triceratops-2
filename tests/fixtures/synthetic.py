"""Synthetic light curve generators for testing.

All generators are deterministic when ``rng_seed`` is specified, so tests
remain reproducible without global RNG state.
"""
from __future__ import annotations

import numpy as np

from triceratops_new.domain.entities import LightCurve


def make_transit_lightcurve(
    R_p_rearth: float = 2.0,
    P_orb_days: float = 5.0,
    impact_param: float = 0.1,
    noise_ppm: float = 500.0,
    n_points: int = 200,
    cadence_days: float = 0.00139,
    rng_seed: int = 42,
) -> LightCurve:
    """Generate a noisy synthetic transit light curve for testing.

    Uses a simple trapezoidal approximation (NOT pytransit) to avoid
    circular dependency. Suitable for testing that likelihoods produce
    finite values, not for precision fits.

    Args:
        R_p_rearth: Planet radius in Earth radii.
        P_orb_days: Orbital period in days.
        impact_param: Impact parameter (0 = central transit).
        noise_ppm: White noise level in parts-per-million.
        n_points: Number of time samples.
        cadence_days: Exposure time per sample.
        rng_seed: Random seed for reproducibility.

    Returns:
        LightCurve centred on transit midpoint.
    """
    rng = np.random.default_rng(rng_seed)
    time = np.linspace(-0.2, 0.2, n_points)
    # Approximate transit depth from radius ratio squared (R_s=1 Rsun assumed)
    depth = (R_p_rearth * 6.371e8 / 6.957e10) ** 2
    # Trapezoidal transit approximation centred at t=0
    width = 0.04  # transit half-width in days (approximate)
    flux = np.where(np.abs(time) < width, 1.0 - depth * (1 - impact_param**2), 1.0)
    noise = rng.normal(0.0, noise_ppm * 1e-6, size=n_points)
    flux_err = noise_ppm * 1e-6
    return LightCurve(
        time_days=time,
        flux=flux + noise,
        flux_err=flux_err,
        cadence_days=cadence_days,
        supersampling_rate=20,
    )


def make_flat_lightcurve(
    noise_ppm: float = 500.0,
    n_points: int = 200,
    cadence_days: float = 0.00139,
    rng_seed: int = 99,
) -> LightCurve:
    """Generate a flat (no-transit) light curve for testing null-signal scenarios."""
    rng = np.random.default_rng(rng_seed)
    time = np.linspace(-0.2, 0.2, n_points)
    noise = rng.normal(0.0, noise_ppm * 1e-6, size=n_points)
    return LightCurve(
        time_days=time,
        flux=1.0 + noise,
        flux_err=noise_ppm * 1e-6,
        cadence_days=cadence_days,
        supersampling_rate=20,
    )
