"""Log-prior functions for computing importance weights.

These are added to the log-likelihood in Phase 7 of each scenario's
compute() method. All functions return log-prior values (float or np.ndarray).

Source references point to priors.py in the original codebase.
"""
from __future__ import annotations

import numpy as np

from triceratops.config.config import CONST


def _separation_at_contrast(
    delta_mags: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Interpolate the separation at which a given delta-mag can be detected.

    Port of funcs.py:277-293 (separation_at_contrast).
    """
    return np.interp(delta_mags, contrasts, separations)


def lnprior_host_mass_planet(mass_msun: np.ndarray) -> float:
    """Log prior on host star mass for planet scenarios.

    Source: priors.py:386-405

    Note: the original computes the prior but returns 0.0 unconditionally
    ("omitted due to bias"). We preserve that behavior.
    """
    return 0.0


def lnprior_host_mass_binary(mass_msun: np.ndarray) -> float:
    """Log prior on host star mass for EB scenarios.

    Source: priors.py:408-479

    Note: the original computes the prior but returns 0.0 unconditionally
    ("omitted due to bias"). We preserve that behavior.
    """
    return 0.0


def lnprior_period_planet(period_days: float, flat_priors: bool = False) -> float:
    """Log prior on orbital period for planet scenarios.

    Source: priors.py:482-536

    Args:
        period_days: Orbital period in days (scalar).
        flat_priors: If True, use a flat prior.

    Returns:
        Log probability of planet having an orbital period P_orb +/- 0.1 days.
    """
    if not flat_priors:
        P_break = 10.0
        P_min = 0.1
        P_max = 50.0
        p1 = 1.5
        p2 = 0.0
        A = P_break**p1 / P_break**p2
        I1_norm = (P_break ** (p1 + 1) - P_min ** (p1 + 1)) / (p1 + 1)
        I2_norm = A * (P_max ** (p2 + 1) - P_break ** (p2 + 1)) / (p2 + 1)
        Norm = 1 / (I1_norm + I2_norm)

        P_orb = period_days
        if P_orb < P_min + 0.1:
            P_orb = P_min + 0.1
        elif P_orb > P_max - 0.1:
            P_orb = P_max - 0.1

        if P_orb <= P_break - 0.1:
            I1 = ((P_orb + 0.1) ** (p1 + 1) - (P_orb - 0.1) ** (p1 + 1)) / (p1 + 1)
            prob = Norm * I1
        elif P_orb >= P_break + 0.1:
            I1 = A * ((P_orb + 0.1) ** (p2 + 1) - (P_orb - 0.1) ** (p2 + 1)) / (p2 + 1)
            prob = Norm * I1
        else:
            I1 = (P_break ** (p1 + 1) - (P_orb - 0.1) ** (p1 + 1)) / (p1 + 1)
            I2 = A * ((P_orb + 0.1) ** (p2 + 1) - P_break ** (p2 + 1)) / (p2 + 1)
            prob = Norm * (I1 + I2)
    else:
        P_min = 0.1
        P_max = 50.0
        p1 = 0.0
        I1_norm = (P_max ** (p1 + 1) - P_min ** (p1 + 1)) / (p1 + 1)
        Norm = 1 / I1_norm

        P_orb = period_days
        if P_orb < P_min + 0.1:
            P_orb = P_min + 0.1
        elif P_orb > P_max - 0.1:
            P_orb = P_max - 0.1

        I1 = ((P_orb + 0.1) ** (p1 + 1) - (P_orb - 0.1) ** (p1 + 1)) / (p1 + 1)
        prob = Norm * I1

    return float(np.log(prob))


def lnprior_period_binary(period_days: float) -> float:
    """Log prior on orbital period for EB scenarios.

    Source: priors.py:539-577

    Args:
        period_days: Orbital period in days (scalar).

    Returns:
        Log probability of binary having an orbital period P_orb +/- 0.1 days.
    """
    P_break = 0.3
    P_min = 0.1
    P_max = 50.0
    p1 = 5.0
    p2 = 0.5

    A = P_break**p1 / P_break**p2
    I1_norm = (P_break ** (p1 + 1) - P_min ** (p1 + 1)) / (p1 + 1)
    I2_norm = A * (P_max ** (p2 + 1) - P_break ** (p2 + 1)) / (p2 + 1)
    Norm = 1 / (I1_norm + I2_norm)

    P_orb = period_days
    if P_orb < P_min + 0.1:
        P_orb = P_min + 0.1
    elif P_orb > P_max - 0.1:
        P_orb = P_max - 0.1

    if P_orb <= P_break - 0.1:
        I1 = ((P_orb + 0.1) ** (p1 + 1) - (P_orb - 0.1) ** (p1 + 1)) / (p1 + 1)
        prob = Norm * I1
    elif P_orb >= P_break + 0.1:
        I1 = A * ((P_orb + 0.1) ** (p2 + 1) - (P_orb - 0.1) ** (p2 + 1)) / (p2 + 1)
        prob = Norm * I1
    else:
        I1 = (P_break ** (p1 + 1) - (P_orb - 0.1) ** (p1 + 1)) / (p1 + 1)
        I2 = A * ((P_orb + 0.1) ** (p2 + 1) - P_break ** (p2 + 1)) / (p2 + 1)
        prob = Norm * (I1 + I2)

    return float(np.log(prob))


def _compute_companion_rate(
    primary_mass: float,
    parallax_mas: float,
    delta_mags: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
    include_short_period: bool,
) -> np.ndarray:
    """Shared computation for bound companion priors (TP and EB).

    Args:
        primary_mass: Target star mass in solar masses.
        parallax_mas: Parallax in milliarcseconds.
        delta_mags: Contrasts of simulated companions.
        separations: Contrast curve separations (arcsec).
        contrasts: Contrast curve delta-mag limits.
        include_short_period: If True, include contributions from short-period
            companions (EB scenarios). If False, set them to zero (TP scenarios).

    Returns:
        Array of log-prior values.

    Source: priors.py:580-984 (lnprior_bound_TP and lnprior_bound_EB)
    """
    pi = np.pi

    plx = parallax_mas
    if np.isnan(plx):
        plx = 0.1
    d = 1000 / plx
    seps = d * _separation_at_contrast(delta_mags, separations, contrasts)

    if primary_mass >= 1.0:
        M_s = primary_mass
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s)) ** 2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s)) ** 2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s)) ** 2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = (
            (4 * pi**2) / (CONST.G * M_s * CONST.Msun) * (seps * CONST.au) ** 3
        ) ** (1 / 2) / 86400

        t2_partial = (
            0.5
            * (np.log10(max_Porbs) - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (np.log10(max_Porbs) - 1.0))
        )
        t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
        t3_partial = (
            0.5 * alpha * (np.log10(max_Porbs) ** 2 - 5.4 * np.log10(max_Porbs) + 6.8)
            + f2 * (np.log10(max_Porbs) - 2.0)
        )
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (np.log10(max_Porbs) - 3.4)
            + f2 * (np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (
                0.238095 * np.log10(max_Porbs) ** 2
                - 0.952381 * np.log10(max_Porbs)
                + 0.485714
            )
        )
        t4 = (
            alpha * dlogP * (5.5 - 3.4)
            + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP) * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714)
        )
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * np.log10(max_Porbs)))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        logP = np.log10(max_Porbs)

        if include_short_period:
            # EB: include short-period contributions
            f_comp[logP < 1.0] = 0.0
            mask = (logP >= 1.0) & (logP < 2.0)
            f_comp[mask] = t2_partial[mask]
            mask = (logP >= 2.0) & (logP < 3.4)
            f_comp[mask] = t2 + t3_partial[mask]
            mask = (logP >= 3.4) & (logP < 5.5)
            f_comp[mask] = t2 + t3 + t4_partial[mask]
            mask = (logP >= 5.5) & (logP < 8.0)
            f_comp[mask] = t2 + t3 + t4 + t5_partial[mask]
            mask = logP >= 8.0
            f_comp[mask] = t2 + t3 + t4 + t5
        else:
            # TP: only long-period contributions
            f_comp[logP < 3.4] = 0.0
            mask = (logP >= 3.4) & (logP < 5.5)
            f_comp[mask] = t4_partial[mask]
            mask = (logP >= 5.5) & (logP < 8.0)
            f_comp[mask] = t4 + t5_partial[mask]
            mask = logP >= 8.0
            f_comp[mask] = t4 + t5

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(f_comp)
    else:
        # M_s < 1.0: compute using M_s=1.0 then scale
        M_act = primary_mass
        M_s = 1.0
        f1 = 0.020 + 0.04 * np.log10(M_s) + 0.07 * (np.log10(M_s)) ** 2
        f2 = 0.039 + 0.07 * np.log10(M_s) + 0.01 * (np.log10(M_s)) ** 2
        f3 = 0.078 - 0.05 * np.log10(M_s) + 0.04 * (np.log10(M_s)) ** 2
        alpha = 0.018
        dlogP = 0.7
        max_Porbs = (
            (4 * pi**2) / (CONST.G * M_s * CONST.Msun) * (seps * CONST.au) ** 3
        ) ** (1 / 2) / 86400

        t2_partial = (
            0.5
            * (np.log10(max_Porbs) - 1.0)
            * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (np.log10(max_Porbs) - 1.0))
        )
        t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
        t3_partial = (
            0.5 * alpha * (np.log10(max_Porbs) ** 2 - 5.4 * np.log10(max_Porbs) + 6.8)
            + f2 * (np.log10(max_Porbs) - 2.0)
        )
        t3 = 0.5 * alpha * (3.4**2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
        t4_partial = (
            alpha * dlogP * (np.log10(max_Porbs) - 3.4)
            + f2 * (np.log10(max_Porbs) - 3.4)
            + (f3 - f2 - alpha * dlogP)
            * (
                0.238095 * np.log10(max_Porbs) ** 2
                - 0.952381 * np.log10(max_Porbs)
                + 0.485714
            )
        )
        t4 = (
            alpha * dlogP * (5.5 - 3.4)
            + f2 * (5.5 - 3.4)
            + (f3 - f2 - alpha * dlogP) * (0.238095 * 5.5**2 - 0.952381 * 5.5 + 0.485714)
        )
        t5_partial = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * np.log10(max_Porbs)))
        t5 = f3 * (3.33333 - 17.3566 * np.exp(-0.3 * 8.0))

        f_comp = np.zeros(len(seps))
        logP = np.log10(max_Porbs)

        if include_short_period:
            f_comp[logP < 1.0] = 0.0
            mask = (logP >= 1.0) & (logP < 2.0)
            f_comp[mask] = t2_partial[mask]
            mask = (logP >= 2.0) & (logP < 3.4)
            f_comp[mask] = t2 + t3_partial[mask]
            mask = (logP >= 3.4) & (logP < 5.5)
            f_comp[mask] = t2 + t3 + t4_partial[mask]
            mask = (logP >= 5.5) & (logP < 8.0)
            f_comp[mask] = t2 + t3 + t4 + t5_partial[mask]
            mask = logP >= 8.0
            f_comp[mask] = t2 + t3 + t4 + t5
        else:
            f_comp[logP < 3.4] = 0.0
            mask = (logP >= 3.4) & (logP < 5.5)
            f_comp[mask] = t4_partial[mask]
            mask = (logP >= 5.5) & (logP < 8.0)
            f_comp[mask] = t4 + t5_partial[mask]
            mask = logP >= 8.0
            f_comp[mask] = t4 + t5

        f_act = 0.65 * f_comp + 0.35 * f_comp * M_act
        f_act[f_act < 0.0] = 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(f_act)


def lnprior_bound_companion(
    delta_mags: np.ndarray,
    separations_arcsec: np.ndarray | None,
    contrasts: np.ndarray | None,
    primary_mass_msun: float,
    parallax_mas: float,
    is_eb: bool = False,
) -> np.ndarray:
    """Log prior for a physically-bound companion (P and S scenarios).

    Source: priors.py:580-984 (lnprior_bound_TP and lnprior_bound_EB)

    Args:
        delta_mags: Contrast values of simulated companions, shape (N,).
        separations_arcsec: Contrast curve separations. If None, returns zeros.
        contrasts: Contrast curve delta-mag limits. If None, returns zeros.
        primary_mass_msun: Primary star mass in solar masses.
        parallax_mas: Parallax in milliarcseconds.
        is_eb: If True, compute EB variant (include short-period). If False, TP variant.

    Returns:
        Array of log-prior values, shape (N,). -inf for rejected samples.
    """
    if separations_arcsec is None or contrasts is None:
        return np.zeros(len(delta_mags))

    return _compute_companion_rate(
        primary_mass=primary_mass_msun,
        parallax_mas=parallax_mas,
        delta_mags=delta_mags,
        separations=separations_arcsec,
        contrasts=contrasts,
        include_short_period=is_eb,
    )


def lnprior_background(
    n_comp: int,
    delta_mags: np.ndarray,
    separations_arcsec: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Log prior for a background star scenario (D and B scenarios).

    Source: priors.py:986-1005

    Args:
        n_comp: Total number of background stars in TRILEGAL sample.
        delta_mags: |delta_mag| values for each MC draw, shape (N,).
        separations_arcsec: Contrast curve separations array.
        contrasts: Contrast curve delta-mag limits array.

    Returns:
        Array of log-prior values, shape (N,).
    """
    seps = _separation_at_contrast(delta_mags, separations_arcsec, contrasts)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((n_comp / 0.1) * (1 / 3600) ** 2 * seps**2)
