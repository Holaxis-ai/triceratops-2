"""Pure phase-fold utility and preparation helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from triceratops.lightcurve.ephemeris import Ephemeris

# ---------------------------------------------------------------------------
# Cadence string → cadence_days mapping (spec §6)
# ---------------------------------------------------------------------------

_CADENCE_DAYS: dict[str, float] = {
    "20sec": 20 / 86400,
    "2min": 120 / 86400,
    "10min": 600 / 86400,
    "30min": 1800 / 86400,
}


# ---------------------------------------------------------------------------
# Public: fold_and_clip
# ---------------------------------------------------------------------------


def fold_and_clip(
    time_btjd: np.ndarray,
    period_days: float,
    t0_btjd: float,
) -> np.ndarray:
    """Phase-fold time series, centred at 0 (transit midpoint).

    Returns phase in days with range (-period/2, +period/2).
    """
    return ((time_btjd - t0_btjd) % period_days) - period_days / 2


# ---------------------------------------------------------------------------
# Private helpers used by prep.py
# ---------------------------------------------------------------------------


def _upper_sigma_mask(
    flux: np.ndarray,
    sigma: float,
    iters: int,
) -> np.ndarray:
    """Return boolean mask for upper-only sigma clipping.

    sigma_lower is always inf — transit dips are never clipped.
    """
    mask = np.ones(len(flux), dtype=bool)
    for _ in range(iters):
        subset = flux[mask]
        if len(subset) == 0:
            break
        med = np.median(subset)
        std = np.std(subset)
        if std == 0:
            break
        # Upper-only: clip points above median + sigma * std
        # sigma_lower=inf means never clip below
        mask &= flux <= med + sigma * std
    return mask


def _bin_timeseries(
    time: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a time series into uniform-width time bins."""
    bin_days = bin_minutes / (24 * 60)
    t_start = time[0]
    t_end = time[-1]
    n_bins = max(1, int(np.ceil((t_end - t_start) / bin_days)))

    bin_edges = np.linspace(t_start, t_start + n_bins * bin_days, n_bins + 1)
    bin_indices = np.digitize(time, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned_t = []
    binned_f = []
    binned_e = []
    for i in range(n_bins):
        in_bin = bin_indices == i
        if not np.any(in_bin):
            continue
        weights = 1.0 / ferr[in_bin] ** 2
        w_sum = np.sum(weights)
        binned_t.append(np.sum(time[in_bin] * weights) / w_sum)
        binned_f.append(np.sum(flux[in_bin] * weights) / w_sum)
        binned_e.append(1.0 / np.sqrt(w_sum))

    return (
        np.array(binned_t, dtype=np.float64),
        np.array(binned_f, dtype=np.float64),
        np.array(binned_e, dtype=np.float64),
    )


def _bin_phase(
    phase_days: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin phase-folded data into n_bins uniform phase bins."""
    p_min = phase_days.min()
    p_max = phase_days.max()
    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_indices = np.digitize(phase_days, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned_p = []
    binned_f = []
    binned_e = []
    for i in range(n_bins):
        in_bin = bin_indices == i
        if not np.any(in_bin):
            continue
        weights = 1.0 / ferr[in_bin] ** 2
        w_sum = np.sum(weights)
        binned_p.append(np.sum(phase_days[in_bin] * weights) / w_sum)
        binned_f.append(np.sum(flux[in_bin] * weights) / w_sum)
        binned_e.append(1.0 / np.sqrt(w_sum))

    return (
        np.array(binned_p, dtype=np.float64),
        np.array(binned_f, dtype=np.float64),
        np.array(binned_e, dtype=np.float64),
    )


def _savitzky_golay_flatten(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: int,
    polyorder: int,
    ephemeris: Ephemeris,
) -> np.ndarray:
    """Flatten flux using Savitzky-Golay filter with a transit mask.

    A transit mask is constructed so the filter does not fit the transit
    dip as a trend (spec §9 gotcha #8).
    """
    # Build transit mask: mask points within 1 transit duration of any transit
    mask = np.ones(len(time), dtype=bool)  # True = use in fit
    if ephemeris.duration_hours is not None:
        half_dur = ephemeris.duration_hours / 24.0 / 2.0
        phase = fold_and_clip(time, ephemeris.period_days, ephemeris.t0_btjd)
        mask = np.abs(phase) > half_dur  # True = out-of-transit

    # Interpolate in-transit points for the filter input
    flux_for_filter = flux.copy()
    if not np.all(mask):
        # Replace in-transit points with linear interpolation of OOT
        oot_time = time[mask]
        oot_flux = flux[mask]
        if len(oot_time) >= 2:
            flux_for_filter[~mask] = np.interp(time[~mask], oot_time, oot_flux)

    # Clamp window_length to data length
    wl = min(window_length, len(flux_for_filter))
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        # Not enough points to filter — return as-is
        return flux

    trend = savgol_filter(flux_for_filter, wl, polyorder)
    return flux / trend


def _cadence_days(cadence_str: str, exptime_seconds: float) -> float:
    """Convert cadence string or exposure time to cadence in days."""
    if cadence_str in _CADENCE_DAYS:
        return _CADENCE_DAYS[cadence_str]
    return exptime_seconds / 86400
