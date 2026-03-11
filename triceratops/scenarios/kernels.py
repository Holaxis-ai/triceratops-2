"""Kernel functions shared by all 14 scenario implementations.

Each function replaces a block that was copy-pasted 10-14 times across
marginal_likelihoods.py. Every function is pure (no I/O, no global state).
"""
from __future__ import annotations

import numpy as np

from triceratops.domain.entities import ExternalLightCurve, LightCurve


def resolve_period(
    period_spec: float | int | np.floating | np.integer | list[float] | tuple[float, float],
    n: int,
) -> np.ndarray:
    """Expand a period specification to an array of N samples.

    Args:
        period_spec: Either a scalar period in days, or a 2-element sequence
                     [min_period, max_period] for uniform sampling.
        n: Number of samples to generate.

    Returns:
        Array of period values in days, shape (n,).
    """
    if isinstance(period_spec, (int, float, np.floating, np.integer)):
        return np.full(n, float(period_spec))
    seq = list(period_spec)
    if len(seq) != 2:
        raise ValueError(
            f"period_spec as sequence must have exactly 2 elements [lo, hi], "
            f"got {len(seq)}"
        )
    lo, hi = float(seq[0]), float(seq[-1])
    return np.random.uniform(lo, hi, size=n)


def compute_lnZ(
    lnL: np.ndarray,
) -> float:
    """Compute the log marginal likelihood (evidence) from a lnL array.

    Uses the log-sum-exp trick for numerical stability, which is correct
    regardless of the number of data points.

    lnZ = log(mean(exp(lnL)))
        = lnL_max + log(sum(exp(lnL_finite - lnL_max)) / N)

    Args:
        lnL: Array of per-sample log-likelihoods, shape (N,). May contain -inf.

    Returns:
        lnZ: float. Will be -inf if all lnL values are -inf.
    """
    finite_mask = np.isfinite(lnL)
    if not np.any(finite_mask):
        return float(-np.inf)
    lnL_finite = lnL[finite_mask]
    lnL_max = float(np.max(lnL_finite))
    sum_exp = float(np.sum(np.exp(lnL_finite - lnL_max)))
    N = len(lnL)
    return lnL_max + float(np.log(sum_exp / N))


def pack_best_indices(
    lnL: np.ndarray,
    n_best: int,
) -> np.ndarray:
    """Return indices of the top n_best samples by log-likelihood.

    Source: marginal_likelihoods.py:304 -- ``idx = (-lnL).argsort()[:N_samples]``

    Args:
        lnL: Array of per-sample log-likelihoods. May contain -inf.
        n_best: Number of top samples to retain.

    Returns:
        Integer index array of length min(n_best, N), sorted descending by lnL.
    """
    n_actual = min(n_best, len(lnL))
    if n_actual >= len(lnL):
        return (-lnL).argsort()
    # argpartition is O(N); argsort of only the top-k subset is O(k log k)
    part = np.argpartition(lnL, -n_actual)[-n_actual:]
    return part[(-lnL[part]).argsort()]


def load_external_lcs(
    lc_files: list[str],
    filter_names: list[str],
    ldc_catalog: object,
    stellar_metallicity: float,
    stellar_teff: float,
    stellar_logg: float,
    renorm: bool = False,
    star_flux_ratios: list[float] | None = None,
) -> list[ExternalLightCurve]:
    """Load and pre-process external (ground-based) light curve files.

    Args:
        lc_files: List of file paths.
        filter_names: List of filter names corresponding to each file.
        ldc_catalog: Object with .get_coefficients(filter, Z, Teff, logg).
        stellar_metallicity: [M/H] of the host star.
        stellar_teff: Teff of the host star in K.
        stellar_logg: logg of the host star.
        renorm: If True, renormalise each LC by its flux ratio.
        star_flux_ratios: List of flux ratios per external LC. Required if renorm=True.

    Returns:
        List of ExternalLightCurve objects.

    Raises:
        ValueError: If mismatched lengths or > 7 files.
    """
    if len(lc_files) != len(filter_names):
        raise ValueError(
            f"Number of LC files ({len(lc_files)}) must match "
            f"number of filter names ({len(filter_names)})"
        )
    if len(lc_files) > 7:
        raise ValueError(
            f"Maximum 7 external light curves supported, got {len(lc_files)}"
        )
    result = []
    for i, (path, filt) in enumerate(zip(lc_files, filter_names)):
        data = np.loadtxt(path)
        time = data[:, 0]
        flux = data[:, 1]
        flux_err = data[:, 2] if data.shape[1] > 2 else np.full(len(flux), np.std(flux))

        if renorm and star_flux_ratios is not None:
            fr = star_flux_ratios[i]
            flux = (flux - (1.0 - fr)) / fr
            flux_err = flux_err / fr

        ldc = ldc_catalog.get_coefficients(  # type: ignore[union-attr]
            filt, stellar_metallicity, stellar_teff, stellar_logg
        )

        lc = LightCurve(
            time_days=time,
            flux=flux,
            flux_err=float(np.mean(flux_err)),
            cadence_days=float(time[1] - time[0]) if len(time) > 1 else 0.00139,
        )
        result.append(ExternalLightCurve(light_curve=lc, band=filt, ldc=ldc))
    return result


def build_transit_mask(
    inc_deg: np.ndarray,
    ptra: np.ndarray,
    coll: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the boolean mask selecting transiting, non-colliding samples.

    Source: marginal_likelihoods.py Phase 7 (parallel path).

    Args:
        inc_deg: Inclinations in degrees, shape (N,).
        ptra: Geometric transit probabilities, shape (N,).
        coll: Boolean collision array, shape (N,). True = collision (reject).
        extra_mask: Optional additional boolean mask.

    Returns:
        Boolean mask, shape (N,). True = evaluate likelihood for this sample.
    """
    valid_ptra = ptra <= 1.0
    # For valid entries, inc_min = arccos(Ptra). Invalid entries get 90 deg.
    safe_ptra = np.where(valid_ptra, ptra, 1.0)
    inc_min = np.where(valid_ptra, np.degrees(np.arccos(safe_ptra)), 90.0)
    mask = valid_ptra & (inc_deg >= inc_min) & (~coll)
    if extra_mask is not None:
        mask = mask & extra_mask
    return mask
