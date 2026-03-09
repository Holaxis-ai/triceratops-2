"""Pure value objects: immutable data types used across the system."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _is_nan(v: object) -> bool:
    """Return True if v is a float NaN (handles None, str, int safely)."""
    try:
        import math
        return math.isnan(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class StellarParameters:
    """Physical properties of a single star as used by scenario computations."""

    mass_msun: float           # Solar masses
    radius_rsun: float         # Solar radii
    teff_k: float              # Effective temperature in Kelvin
    logg: float                # log10(g / cm s^-2)
    metallicity_dex: float     # [M/H] in dex; 0.0 = solar
    parallax_mas: float        # mas; use 10.0 as fallback when unknown

    @classmethod
    def from_tic_row(cls, row: dict) -> StellarParameters:
        """Construct from a TIC catalog row dict. Fills defaults for NaN/None fields.

        Args:
            row: dict with keys matching TIC column names (mass, rad, Teff, plx).
                 Any field may be NaN or None; defaults will be substituted.
        """
        import math

        from triceratops.config.config import CONST

        plx = float(row.get("plx") or 10.0) if not _is_nan(row.get("plx")) else 10.0
        mass = float(row.get("mass") or 1.0) if not _is_nan(row.get("mass")) else 1.0
        rad = float(row.get("rad") or 1.0) if not _is_nan(row.get("rad")) else 1.0
        teff = float(row.get("Teff") or 5778.0) if not _is_nan(row.get("Teff")) else 5778.0
        logg = math.log10(CONST.G * mass * CONST.Msun / (rad * CONST.Rsun) ** 2)
        return cls(
            mass_msun=mass,
            radius_rsun=rad,
            teff_k=teff,
            logg=logg,
            metallicity_dex=0.0,
            parallax_mas=plx,
        )


@dataclass(frozen=True)
class OrbitalParameters:
    """Orbital configuration for a Monte Carlo draw or best-fit solution."""

    period_days: float | np.ndarray
    inclination_deg: float | np.ndarray
    eccentricity: float | np.ndarray
    arg_periastron_deg: float | np.ndarray
    impact_parameter: float | np.ndarray


@dataclass(frozen=True)
class LimbDarkeningCoeffs:
    """Quadratic limb-darkening coefficients (u1, u2) for a single star+band combination."""

    u1: float | np.ndarray
    u2: float | np.ndarray
    band: str   # "TESS", "Kepler", "J", "H", "K", "g", "r", "i", "z"

    @property
    def as_ldc_array(self) -> np.ndarray:
        """Return [[u1, u2]] as required by pytransit QuadraticModel.evaluate_ps()."""
        return np.array([[float(self.u1), float(self.u2)]])


@dataclass(frozen=True)
class ContrastCurve:
    """AO or speckle contrast curve: angular separation vs achievable delta-magnitude.

    Separations must be in ascending order.
    """

    separations_arcsec: np.ndarray   # ascending, shape (M,)
    delta_mags: np.ndarray           # corresponding contrast limits, shape (M,)
    band: str                        # filter label, e.g. "J", "K"

    def max_detectable_delta_mag(self, separation_arcsec: float) -> float:
        """Interpolate the contrast limit at a given angular separation.

        Returns 0.0 if separation is below the inner working angle,
        and the outermost value if beyond the outer boundary.
        """
        if separation_arcsec < self.separations_arcsec[0]:
            return 0.0
        if separation_arcsec > self.separations_arcsec[-1]:
            return float(self.delta_mags[-1])
        return float(
            np.interp(separation_arcsec, self.separations_arcsec, self.delta_mags)
        )
