"""Prior sampling functions.

All functions receive an array ``u`` of uniform [0,1) samples and return
samples from the corresponding prior distribution. The caller is responsible
for generating ``u`` (typically ``np.random.rand(N)``). This makes all functions
pure, deterministic given ``u``, and independently testable.

All distributions are ported from triceratops/priors.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, powerlaw


def sample_planet_radius(
    u: np.ndarray,
    host_mass: float,
    flat: bool = False,
) -> np.ndarray:
    """Sample planet radii from the broken power-law prior.

    When flat=True, returns samples uniform in [0.5, 20.0] Rearth.
    When flat=False, uses the broken power-law from Fressin et al. 2013
    as implemented in priors.py:16-115.

    Args:
        u: Uniform [0,1) samples, shape (N,).
        host_mass: Host star mass in Solar masses (scalar).
        flat: If True, use a flat (uniform) prior.

    Returns:
        Planet radii in Earth radii, shape (N,).

    Source: priors.py:16-115
    """
    x = u.copy()

    if flat:
        return x / (1.0 / 19.5) + 0.5

    R_break1 = 3.0
    R_break2 = 6.0
    R_min = 0.5
    R_max = 20.0

    # Power coefficients for M > 0.45
    p1 = 0.0
    p2 = -4.0
    p3 = -0.5
    # Power coefficients for M <= 0.45
    p4 = 0.0
    p5 = -7.0
    p6 = -0.5

    # Normalizing constants for M > 0.45
    A1 = R_break1**p1 / R_break1**p2
    A2 = R_break2**p2 / R_break2**p3
    I1 = (R_break1 ** (p1 + 1) - R_min ** (p1 + 1)) / (p1 + 1)
    I2 = A1 * (R_break2 ** (p2 + 1) - R_break1 ** (p2 + 1)) / (p2 + 1)
    I3 = A2 * A1 * (R_max ** (p3 + 1) - R_break2 ** (p3 + 1)) / (p3 + 1)
    Norm1 = 1 / (I1 + I2 + I3)

    # Normalizing constants for M <= 0.45
    A3 = R_break1**p4 / R_break1**p5
    A4 = R_break2**p5 / R_break2**p6
    I4 = (R_break1 ** (p4 + 1) - R_min ** (p4 + 1)) / (p4 + 1)
    I5 = A3 * (R_break2 ** (p5 + 1) - R_break1 ** (p5 + 1)) / (p5 + 1)
    I6 = A4 * A3 * (R_max ** (p6 + 1) - R_break2 ** (p6 + 1)) / (p6 + 1)
    Norm2 = 1 / (I4 + I5 + I6)

    # Masks for M > 0.45 regime
    mask1 = (x <= Norm1 * I1) & (host_mass > 0.45)
    mask2 = (x > Norm1 * I1) & (x <= Norm1 * (I1 + I2)) & (host_mass > 0.45)
    mask3 = (x > Norm1 * (I1 + I2)) & (x <= Norm1 * (I1 + I2 + I3)) & (host_mass > 0.45)

    # Masks for M <= 0.45 regime
    mask4 = (x <= Norm2 * I4) & (host_mass <= 0.45)
    mask5 = (x > Norm2 * I4) & (x <= Norm2 * (I4 + I5)) & (host_mass <= 0.45)
    mask6 = (x > Norm2 * (I4 + I5)) & (x <= Norm2 * (I4 + I5 + I6)) & (host_mass <= 0.45)

    # Inverse CDF for M > 0.45
    x[mask1] = (x[mask1] / Norm1 * (p1 + 1) + R_min ** (p1 + 1)) ** (1 / (p1 + 1))
    x[mask2] = (
        (x[mask2] / Norm1 - I1) * (p2 + 1) / A1 + R_break1 ** (p2 + 1)
    ) ** (1 / (p2 + 1))
    x[mask3] = (
        (x[mask3] / Norm1 - I1 - I2) * (p3 + 1) / (A1 * A2) + R_break2 ** (p3 + 1)
    ) ** (1 / (p3 + 1))

    # Inverse CDF for M <= 0.45
    x[mask4] = (x[mask4] / Norm2 * (p4 + 1) + R_min ** (p4 + 1)) ** (1 / (p4 + 1))
    x[mask5] = (
        (x[mask5] / Norm2 - I4) * (p5 + 1) / A3 + R_break1 ** (p5 + 1)
    ) ** (1 / (p5 + 1))
    x[mask6] = (
        (x[mask6] / Norm2 - I4 - I5) * (p6 + 1) / (A3 * A4) + R_break2 ** (p6 + 1)
    ) ** (1 / (p6 + 1))

    return x


def sample_inclination(
    u: np.ndarray,
    lower: float = 0.0,
    upper: float = 90.0,
) -> np.ndarray:
    """Sample orbital inclinations uniform in cos(inc).

    Source: priors.py:119-132

    Args:
        u: Uniform [0,1) samples, shape (N,).
        lower: Lower bound in degrees (default 0).
        upper: Upper bound in degrees (default 90).

    Returns:
        Inclinations in degrees, shape (N,).
    """
    lower_rad = lower * np.pi / 180
    upper_rad = upper * np.pi / 180
    Norm = 1 / (np.cos(lower_rad) - np.cos(upper_rad))
    return np.arccos(np.cos(lower_rad) - u / Norm) * 180 / np.pi


def sample_eccentricity(
    u: np.ndarray,
    planet: bool = True,
    period: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Sample orbital eccentricities.

    For planets: Beta(0.867, 3.030) distribution (Kipping 2013).
    For binaries: power-law distribution dependent on period (Moe & Di Stefano 2017).

    Note: the original uses beta.rvs/powerlaw.rvs ignoring ``u``. This port uses
    the inverse CDF (ppf) so results are deterministic given ``u``.

    Source: priors.py:134-155

    Args:
        u: Uniform [0,1) samples, shape (N,).
        planet: If True, sample from planet eccentricity prior; else EB prior.
        period: Orbital period in days. Used for EB prior only.

    Returns:
        Eccentricities in [0, 1), shape (N,).
    """
    if planet:
        return beta.ppf(u, 0.867, 3.030)
    else:
        if np.isscalar(period):
            if period <= 10:
                # nu+1 = 0.2 => nu = -0.8; powerlaw(a) parameterises x^(a-1) on [0,1]
                return powerlaw.ppf(u, 0.2)
            else:
                return powerlaw.ppf(u, 0.6)
        else:
            # period is an array: element-wise
            result = np.empty_like(u)
            short = np.asarray(period) <= 10
            result[short] = powerlaw.ppf(u[short], 0.2)
            result[~short] = powerlaw.ppf(u[~short], 0.6)
            return result


def sample_arg_periastron(u: np.ndarray) -> np.ndarray:
    """Sample argument of periastron uniform in [0, 360) degrees.

    Source: priors.py:157-166

    Args:
        u: Uniform [0,1) samples, shape (N,).

    Returns:
        Arguments of periastron in degrees, shape (N,).
    """
    return u * 360.0


def _inverse_cdf_broken_power_q(
    x: np.ndarray,
    q_min: float,
    p1: float | None,
    p2: float,
    F_twin: float,
) -> np.ndarray:
    """Shared inverse-CDF logic for mass ratio sampling.

    Handles the three-segment (p1 is not None) and two-segment (p1 is None)
    broken power-law cases from sample_q and sample_q_companion.
    """
    result = x.copy()

    if p1 is not None and q_min < 0.3:
        # Three-segment case
        A1 = 0.3**p1 / 0.3**p2
        A2 = (
            1
            + F_twin / (1 - F_twin)
            * ((1.0 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1))
            / ((1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1))
        )
        I1 = (0.3 ** (p1 + 1) - q_min ** (p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1)
        Norm = 1 / (I1 + I2 + I3)

        mask1 = result <= Norm * I1
        mask2 = (result > Norm * I1) & (result <= Norm * (I1 + I2))
        mask3 = (result > Norm * (I1 + I2)) & (result <= Norm * (I1 + I2 + I3))

        result[mask1] = (
            result[mask1] / Norm * (p1 + 1) + q_min ** (p1 + 1)
        ) ** (1 / (p1 + 1))
        result[mask2] = (
            (result[mask2] / Norm - I1) * (p2 + 1) / A1 + 0.3 ** (p2 + 1)
        ) ** (1 / (p2 + 1))
        result[mask3] = (
            (result[mask3] / Norm - I1 - I2) * (p2 + 1) / (A1 * A2) + 0.95 ** (p2 + 1)
        ) ** (1 / (p2 + 1))
    else:
        # Two-segment case (q_min >= 0.3)
        A2 = (
            1
            + F_twin / (1 - F_twin)
            * ((1.0 ** (p2 + 1) - q_min ** (p2 + 1)) / (p2 + 1))
            / ((1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1))
        )
        I2 = (0.95 ** (p2 + 1) - q_min ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * (1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1)
        Norm = 1 / (I2 + I3)

        mask2 = result <= Norm * I2
        mask3 = (result > Norm * I2) & (result <= Norm * (I2 + I3))

        result[mask2] = (
            result[mask2] / Norm * (p2 + 1) + q_min ** (p2 + 1)
        ) ** (1 / (p2 + 1))
        result[mask3] = (
            (result[mask3] / Norm - I2) * (p2 + 1) / A2 + 0.95 ** (p2 + 1)
        ) ** (1 / (p2 + 1))

    return result


def sample_mass_ratio(
    u: np.ndarray,
    primary_mass: float,
) -> np.ndarray:
    """Sample EB companion mass ratios q = M_secondary / M_primary.

    Source: priors.py:168-274

    Args:
        u: Uniform [0,1) samples, shape (N,).
        primary_mass: Mass of the primary star in Solar masses.

    Returns:
        Mass ratios, shape (N,). Values in (0, 1].
    """
    if primary_mass <= 0.1:
        return np.full(len(u), 1.0)

    if primary_mass >= 1.0:
        q_min = 0.1
    else:
        q_min = 0.1 / primary_mass

    p1: float | None = 0.3 if q_min < 0.3 else None
    p2 = -0.5
    F_twin = 0.30

    return _inverse_cdf_broken_power_q(u, q_min, p1, p2, F_twin)


def sample_companion_mass_ratio(
    u: np.ndarray,
    primary_mass: float,
) -> np.ndarray:
    """Sample wide-companion (P/S scenarios) mass ratios.

    Source: priors.py:277-383

    Args:
        u: Uniform [0,1) samples, shape (N,).
        primary_mass: Mass of the primary star in Solar masses.

    Returns:
        Companion mass ratios, shape (N,).
    """
    if primary_mass <= 0.1:
        return np.full(len(u), 1.0)

    if primary_mass >= 1.0:
        q_min = 0.1
    else:
        q_min = 0.1 / primary_mass

    p1: float | None = 0.3 if q_min < 0.3 else None
    p2 = -0.95
    F_twin = 0.05

    return _inverse_cdf_broken_power_q(u, q_min, p1, p2, F_twin)
