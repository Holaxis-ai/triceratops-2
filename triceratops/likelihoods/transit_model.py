"""Transit simulation using pytransit QuadraticModel with thread-local storage.

Replaces the module-level mutable singletons ``tm`` and ``tm_sec`` from
likelihoods.py:12-13 with thread-local instances.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from triceratops.config.config import CONST

_local = threading.local()


def _ensure_pytransit_numpy_compat(np_module: Any = np) -> None:
    """Restore NumPy aliases removed in 1.24/2.x that older pytransit imports."""
    if not hasattr(np_module, "int"):
        np_module.int = int
    if not hasattr(np_module, "trapz"):
        np_module.trapz = np_module.trapezoid


def _get_transit_model() -> Any:
    """Get (or create) the thread-local QuadraticModel for the primary transit."""
    if not hasattr(_local, "tm"):
        _ensure_pytransit_numpy_compat()
        from pytransit import QuadraticModel
        _local.tm = QuadraticModel(interpolate=False)
    return _local.tm


def _get_secondary_transit_model() -> Any:
    """Get (or create) the thread-local QuadraticModel for secondary eclipses."""
    if not hasattr(_local, "tm_sec"):
        _ensure_pytransit_numpy_compat()
        from pytransit import QuadraticModel
        _local.tm_sec = QuadraticModel(interpolate=False)
    return _local.tm_sec


def simulate_planet_transit(
    time: np.ndarray,
    rp: float,
    period: float,
    inc: float,
    a: float,
    rs: float,
    u1: float,
    u2: float,
    ecc: float,
    argp: float,
    companion_flux_ratio: float = 0.0,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
) -> np.ndarray:
    """Simulate a planet transit light curve using pytransit QuadraticModel.

    Port of likelihoods.py:simulate_TP_transit().

    Args:
        time: Array of times in days, centred at transit midpoint.
        rp: Planet radius in Earth radii.
        period: Orbital period in days.
        inc: Inclination in degrees.
        a: Semi-major axis in cm.
        rs: Host star radius in Solar radii.
        u1, u2: Quadratic limb-darkening coefficients.
        ecc: Eccentricity.
        argp: Argument of periastron in degrees.
        companion_flux_ratio: Fraction of total aperture flux from unresolved companion.
        companion_is_host: If True, the planet orbits the companion (S/B scenarios).
        exptime: Exposure time in days.
        nsamples: Supersampling rate for integration.

    Returns:
        Normalised flux array, shape (len(time),).
    """
    Rearth = CONST.Rearth
    Rsun = CONST.Rsun
    pi = np.pi

    F_target = 1.0
    F_comp = companion_flux_ratio / (1 - companion_flux_ratio) if companion_flux_ratio > 0 else 0.0

    tm = _get_transit_model()
    tm.set_data(time, exptimes=exptime, nsamples=nsamples)
    flux = tm.evaluate_ps(
        k=rp * Rearth / (rs * Rsun),
        ldc=np.array([float(u1), float(u2)]),
        t0=0.0,
        p=period,
        a=a / (rs * Rsun),
        i=inc * (pi / 180.0),
        e=ecc,
        w=(90 - argp) * (pi / 180.0),
    )

    if companion_flux_ratio > 0:
        if companion_is_host:
            F_dilute = F_target / F_comp
        else:
            F_dilute = F_comp / F_target
        flux = (flux + F_dilute) / (1 + F_dilute)

    return flux


def simulate_eb_transit(
    time: np.ndarray,
    rs: float,
    rcomp: float,
    eb_flux_ratio: float,
    period: float,
    inc: float,
    a: float,
    u1: float,
    u2: float,
    ecc: float,
    argp: float,
    companion_flux_ratio: float = 0.0,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
) -> tuple[np.ndarray, float]:
    """Simulate an eclipsing binary light curve.

    Port of likelihoods.py:simulate_EB_transit().

    Returns:
        (flux, secondary_depth): normalised flux array and the secondary eclipse depth.
    """
    Rsun = CONST.Rsun
    pi = np.pi

    F_target = 1.0
    F_comp = companion_flux_ratio / (1 - companion_flux_ratio) if companion_flux_ratio > 0 else 0.0
    F_EB = eb_flux_ratio / (1 - eb_flux_ratio) if eb_flux_ratio > 0 else 0.0

    # Primary eclipse
    tm = _get_transit_model()
    tm.set_data(time, exptimes=exptime, nsamples=nsamples)
    k = rcomp / rs
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    flux = tm.evaluate_ps(
        k=k,
        ldc=np.array([float(u1), float(u2)]),
        t0=0.0,
        p=period,
        a=a / (rs * Rsun),
        i=inc * (pi / 180.0),
        e=ecc,
        w=(90 - argp) * (pi / 180.0),
    )

    # Secondary eclipse depth
    tm_sec = _get_secondary_transit_model()
    tm_sec.set_data(np.linspace(-0.05, 0.05, 25))
    sec_flux = tm_sec.evaluate_ps(
        k=1 / k,
        ldc=np.array([float(u1), float(u2)]),
        t0=0.0,
        p=period,
        a=a / (rs * Rsun),
        i=inc * (pi / 180.0),
        e=ecc,
        w=(90 - argp + 180) * (pi / 180.0),
    )
    sec_flux_val = float(np.min(sec_flux))

    # Dilution adjustments
    if companion_is_host:
        flux = (flux + F_EB / F_comp) / (1 + F_EB / F_comp)
        sec_flux_val = (sec_flux_val + F_comp / F_EB) / (1 + F_comp / F_EB) if F_EB > 0 else 1.0
        F_dilute = F_target / (F_comp + F_EB)
        flux = (flux + F_dilute) / (1 + F_dilute)
        secdepth = 1 - (sec_flux_val + F_dilute) / (1 + F_dilute)
    else:
        flux = (flux + F_EB / F_target) / (1 + F_EB / F_target) if F_EB > 0 else flux
        sec_flux_val = (sec_flux_val + F_target / F_EB) / (1 + F_target / F_EB) if F_EB > 0 else 1.0
        F_dilute = F_comp / (F_target + F_EB) if (F_target + F_EB) > 0 else 0.0
        flux = (flux + F_dilute) / (1 + F_dilute) if F_dilute > 0 else flux
        secdepth = 1 - (sec_flux_val + F_dilute) / (1 + F_dilute)

    return flux, secdepth


def simulate_planet_transit_p(
    time: np.ndarray,
    rps: np.ndarray,
    period: float | np.ndarray,
    incs: np.ndarray,
    as_: np.ndarray,
    rss: np.ndarray,
    u1s: np.ndarray,
    u2s: np.ndarray,
    eccs: np.ndarray,
    argps: np.ndarray,
    companion_flux_ratios: np.ndarray,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
) -> np.ndarray:
    """Vectorised planet transit simulation for N samples.

    Port of likelihoods.py:simulate_TP_transit_p().

    Returns:
        flux array of shape (N, len(time)).
    """
    Rearth = CONST.Rearth
    Rsun = CONST.Rsun
    pi = np.pi

    F_target = 1.0
    F_comp = companion_flux_ratios / (1 - companion_flux_ratios)
    F_comp = F_comp.reshape(F_comp.shape[0], 1)

    k = rps * Rearth / (rss * Rsun)
    t0 = np.full_like(k, 0.0)
    if np.isscalar(period):
        P_arr = np.full_like(k, float(period))
    else:
        P_arr = np.asarray(period, dtype=float)
    a_norm = as_ / (rss * Rsun)
    inc_rad = incs * (pi / 180.0)
    w = (90 - argps) * (pi / 180.0)
    pvp = np.array([k, t0, P_arr, a_norm, inc_rad, eccs, w]).T
    ldc = np.array([u1s, u2s]).T

    tm = _get_transit_model()
    tm.set_data(time, exptimes=exptime, nsamples=nsamples)
    flux = tm.evaluate_pv(pvp=pvp, ldc=ldc)

    if companion_is_host:
        F_dilute = F_target / F_comp
    else:
        F_dilute = F_comp / F_target
    flux = (flux + F_dilute) / (1 + F_dilute)

    return flux


def simulate_eb_transit_p(
    time: np.ndarray,
    r_ebs: np.ndarray,
    eb_flux_ratios: np.ndarray,
    period: float | np.ndarray,
    incs: np.ndarray,
    as_: np.ndarray,
    rss: np.ndarray,
    u1s: np.ndarray,
    u2s: np.ndarray,
    eccs: np.ndarray,
    argps: np.ndarray,
    companion_flux_ratios: np.ndarray,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised EB transit simulation for N samples.

    Port of likelihoods.py:simulate_EB_transit_p().

    Returns:
        (flux, secdepth): flux shape (N, len(time)), secdepth shape (N, 1).
    """
    Rsun = CONST.Rsun
    pi = np.pi

    F_target = 1.0
    F_comp = companion_flux_ratios / (1 - companion_flux_ratios)
    F_comp = F_comp.reshape(F_comp.shape[0], 1)
    F_EB = eb_flux_ratios / (1 - eb_flux_ratios)
    F_EB = F_EB.reshape(F_EB.shape[0], 1)

    # Primary eclipse
    k = r_ebs / rss
    k[(k - 1.0) < 1e-6] *= 0.999
    t0 = np.full_like(k, 0.0)
    if np.isscalar(period):
        P_arr = np.full_like(k, float(period))
    else:
        P_arr = np.asarray(period, dtype=float)
    a_norm = as_ / (rss * Rsun)
    inc_rad = incs * (pi / 180.0)
    w = (90 - argps) * (pi / 180.0)
    pvp = np.array([k, t0, P_arr, a_norm, inc_rad, eccs, w]).T
    ldc = np.array([u1s, u2s]).T

    tm = _get_transit_model()
    tm.set_data(time, exptimes=exptime, nsamples=nsamples)
    flux = tm.evaluate_pv(pvp=pvp, ldc=ldc)

    # Secondary eclipse depth
    k_sec = rss / r_ebs
    k_sec[(k_sec - 1.0) < 1e-6] *= 0.999
    w_sec = (90 - argps + 180) * (pi / 180.0)
    pvp_sec = np.array([k_sec, t0, P_arr, a_norm, inc_rad, eccs, w_sec]).T

    tm_sec = _get_secondary_transit_model()
    tm_sec.set_data(np.linspace(-0.05, 0.05, 25))
    sec_flux = tm_sec.evaluate_pv(pvp=pvp_sec, ldc=ldc)
    sec_flux = np.atleast_2d(sec_flux)  # (N, n_pts); pytransit squeezes N=1 to 1D
    sec_flux = np.min(sec_flux, axis=1)
    sec_flux = sec_flux.reshape(sec_flux.shape[0], 1)

    # Dilution
    if companion_is_host:
        flux = (flux + F_EB / F_comp) / (1 + F_EB / F_comp)
        sec_flux = (sec_flux + F_comp / F_EB) / (1 + F_comp / F_EB)
        F_dilute = F_target / (F_comp + F_EB)
        flux = (flux + F_dilute) / (1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute) / (1 + F_dilute)
    else:
        flux = (flux + F_EB / F_target) / (1 + F_EB / F_target)
        sec_flux = (sec_flux + F_target / F_EB) / (1 + F_target / F_EB)
        F_dilute = F_comp / (F_target + F_EB)
        flux = (flux + F_dilute) / (1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute) / (1 + F_dilute)

    return flux, secdepth
