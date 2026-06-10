"""Mass-radius-Teff spline relations and flux-ratio lookups.

Ports the Torres (main sequence) and CDwarf (M-dwarf) splines from
funcs.py:15-172, the flux_relation() function, and
estimate_sdss_magnitudes() from funcs.py:466-490.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# ---------------------------------------------------------------------------
# Hardcoded spline node arrays — ground truth values from funcs.py:15-47.
# Do not change these.
# ---------------------------------------------------------------------------

_MASS_NODES_TORRES = np.array([
    0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085,
    1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0,
])
_TEFF_NODES_TORRES = np.array([
    3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650,
    7300, 8180, 9790, 11400, 15200, 30000, 42000,
])
_RAD_NODES_TORRES = np.array([
    0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55,
    1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0,
])

_MASS_NODES_CDWRF = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63])
_TEFF_NODES_CDWRF = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000])
_RAD_NODES_CDWRF = np.array([0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6])

# Breakpoint: below this mass use CDwarf splines; at or above use Torres.
# From funcs.py:50 — must be preserved exactly.
_MASS_BREAKPOINT_MSUN: float = 0.63

# ---------------------------------------------------------------------------
# Flux-ratio spline nodes — from funcs.py:77-172.
# ---------------------------------------------------------------------------

# TESS / Vis band
_MASS_NODES_FLUX = np.array([
    0.1, 0.15, 0.23, 0.4, 0.58, 0.7, 0.9, 1.15, 1.45, 2.2, 2.8,
])
_FLUX_NODES = np.array([
    -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2,
])

# SDSS griz bands — nodes from funcs.py:88-126
_MASS_NODES_GRIZ = np.array([
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25,
    1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85,
    1.9, 1.95,
])

_FLUX_NODES_G = np.array([
    -4.25679829, -3.63466117, -3.02202606, -2.69030897, -2.49659157,
    -2.2505271, -2.021323, -1.85840076, -1.67780772, -1.43277653,
    -1.140759, -0.84440326, -0.59320315, -0.37137966, -0.15264831,
    0.01301527, 0.09435761, 0.15101542, 0.23020827, 0.32669619,
    0.42882, 0.52743055, 0.6211622, 0.71015347, 0.79454293,
    0.8744691, 0.95007055, 1.02148581, 1.08885342, 1.15231194,
    1.2119999, 1.26805586, 1.32061835, 1.36982593, 1.41581714,
    1.45873052, 1.49870461, 1.53587798,
])

_FLUX_NODES_R = np.array([
    -3.7645869, -3.18831114, -2.62976165, -2.3250714, -2.14875222,
    -1.93215166, -1.72755743, -1.57706801, -1.42152281, -1.22624787,
    -1.00041153, -0.76654975, -0.55048523, -0.34401933, -0.13795185,
    0.01884458, 0.09806609, 0.15556025, 0.2352632, 0.33169079,
    0.43331161, 0.53107891, 0.62365058, 0.7111735, 0.79379456,
    0.87166065, 0.94491863, 1.0137154, 1.07819783, 1.13851281,
    1.19480723, 1.24722796, 1.29592188, 1.34103588, 1.38271685,
    1.42111165, 1.45636718, 1.48863032,
])

_FLUX_NODES_I = np.array([
    -3.13478939, -2.74389379, -2.32962701, -2.0762982, -1.91415895,
    -1.73602243, -1.56722207, -1.4299837, -1.29128114, -1.12766897,
    -0.93574407, -0.72873667, -0.52782333, -0.32896605, -0.12933175,
    0.02266234, 0.09986847, 0.1567847, 0.23612551, 0.33219174,
    0.43336024, 0.53051308, 0.62230102, 0.70887614, 0.79039055,
    0.86699633, 0.93884558, 1.00609039, 1.06888284, 1.12737504,
    1.18171907, 1.23206702, 1.278571, 1.32138308, 1.36065537,
    1.39653995, 1.42918892, 1.45875437,
])

_FLUX_NODES_Z = np.array([
    -2.75986184, -2.47632399, -2.13642875, -1.91102978, -1.75970715,
    -1.60431339, -1.45586353, -1.32688125, -1.19931114, -1.05625109,
    -0.88746278, -0.6997811, -0.50990205, -0.31663808, -0.12196758,
    0.02613279, 0.10149454, 0.15766348, 0.23647588, 0.33208016,
    0.43278959, 0.52943638, 0.6206639, 0.70662507, 0.78747281,
    0.86336005, 0.93443971, 1.00086471, 1.06278797, 1.12036242,
    1.17374098, 1.22307657, 1.26852211, 1.31023053, 1.34835475,
    1.38304769, 1.41446227, 1.44275142,
])

# J, H, K bands — from funcs.py:144-172. Note the /2.5 division.
_MASS_NODES_J = np.array([0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3])
_FLUX_NODES_J = np.array([-5.7, -3.8, -1.6, 0, 1.2, 2.9, 3.3, 4, 6]) / 2.5

_MASS_NODES_H = np.array([0.1, 0.23, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3])
_FLUX_NODES_H = np.array([-4.9, -2.8, -0.9, 0.6, 1.5, 3, 3.3, 4, 6]) / 2.5

_MASS_NODES_K = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3])
_FLUX_NODES_K = np.array([-4.7, -2.9, -1.7, -0.7, 0.6, 1.6, 3, 3.3, 4, 6]) / 2.5

# Supported filter names for get_flux_ratio
_SUPPORTED_FILTERS = frozenset({"TESS", "Vis", "Kepler", "J", "H", "K", "g", "r", "i", "z"})
_FILTER_ALIASES = {
    "Jcont": "J",
    "Hcont": "H",
    "Ks": "K",
    "Kcont": "K",
    "Brgamma": "K",
    "Kp": "Kepler",
    "LP600": "Vis",
    "562nm": "Vis",
    "832nm": "Vis",
}


def canonicalize_filter_name(filter_name: str) -> str:
    """Map common mission/imaging aliases onto the canonical flux-relation filters."""
    return _FILTER_ALIASES.get(filter_name, filter_name)


class StellarRelations:
    """Mass -> (radius, Teff) and mass -> flux-ratio splines.

    Wraps the Torres (main sequence) and CDwarf (M-dwarf) spline relations
    from funcs.py. The splines are built on first use (lazy initialisation).

    Breakpoint at M = 0.63 Msun: below this value the CDwarf splines are used;
    at or above this value the Torres splines are used.
    """

    def __init__(self) -> None:
        self._teff_spline_torres: InterpolatedUnivariateSpline | None = None
        self._rad_spline_torres: InterpolatedUnivariateSpline | None = None
        self._teff_spline_cdwrf: InterpolatedUnivariateSpline | None = None
        self._rad_spline_cdwrf: InterpolatedUnivariateSpline | None = None
        self._flux_splines: dict[str, InterpolatedUnivariateSpline] | None = None

    def _ensure_splines(self) -> None:
        """Build splines on first use."""
        if self._teff_spline_torres is not None:
            return
        self._teff_spline_torres = InterpolatedUnivariateSpline(
            _MASS_NODES_TORRES, _TEFF_NODES_TORRES,
        )
        self._rad_spline_torres = InterpolatedUnivariateSpline(
            _MASS_NODES_TORRES, _RAD_NODES_TORRES,
        )
        self._teff_spline_cdwrf = InterpolatedUnivariateSpline(
            _MASS_NODES_CDWRF, _TEFF_NODES_CDWRF,
        )
        self._rad_spline_cdwrf = InterpolatedUnivariateSpline(
            _MASS_NODES_CDWRF, _RAD_NODES_CDWRF,
        )

    def _ensure_flux_splines(self) -> None:
        """Build flux-ratio splines on first use."""
        if self._flux_splines is not None:
            return
        self._flux_splines = {
            "TESS": InterpolatedUnivariateSpline(_MASS_NODES_FLUX, _FLUX_NODES),
            "Vis": InterpolatedUnivariateSpline(_MASS_NODES_FLUX, _FLUX_NODES),
            "Kepler": InterpolatedUnivariateSpline(_MASS_NODES_FLUX, _FLUX_NODES),
            "g": InterpolatedUnivariateSpline(_MASS_NODES_GRIZ, _FLUX_NODES_G),
            "r": InterpolatedUnivariateSpline(_MASS_NODES_GRIZ, _FLUX_NODES_R),
            "i": InterpolatedUnivariateSpline(_MASS_NODES_GRIZ, _FLUX_NODES_I),
            "z": InterpolatedUnivariateSpline(_MASS_NODES_GRIZ, _FLUX_NODES_Z),
            "J": InterpolatedUnivariateSpline(_MASS_NODES_J, _FLUX_NODES_J),
            "H": InterpolatedUnivariateSpline(_MASS_NODES_H, _FLUX_NODES_H),
            "K": InterpolatedUnivariateSpline(_MASS_NODES_K, _FLUX_NODES_K),
        }

    def get_radius_teff(
        self,
        masses: np.ndarray,
        max_radii: np.ndarray | None = None,
        max_teffs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute stellar radii and effective temperatures from masses.

        Ports funcs.stellar_relations() (funcs.py:50-75).
        The breakpoint at M = 0.63 Msun selects between CDwarf and Torres splines.

        Args:
            masses: Array of stellar masses in Solar masses, shape (N,).
            max_radii: Optional upper-bound array for radii clamping, shape (N,).
            max_teffs: Optional upper-bound array for Teff clamping, shape (N,).

        Returns:
            (radii, teffs): Two arrays of shape (N,).
                radii in Solar radii, teffs in Kelvin.
        """
        self._ensure_splines()
        masses = np.asarray(masses, dtype=float)

        radii = np.where(
            masses < _MASS_BREAKPOINT_MSUN,
            self._rad_spline_cdwrf(masses),  # type: ignore[misc]
            self._rad_spline_torres(masses),  # type: ignore[misc]
        )
        teffs = np.where(
            masses < _MASS_BREAKPOINT_MSUN,
            self._teff_spline_cdwrf(masses),  # type: ignore[misc]
            self._teff_spline_torres(masses),  # type: ignore[misc]
        )

        # Clamp to bounds (matching original funcs.py:71-74)
        if max_radii is not None:
            max_radii = np.asarray(max_radii, dtype=float)
            radii = np.minimum(radii, max_radii)
        radii = np.maximum(radii, 0.1)

        if max_teffs is not None:
            max_teffs = np.asarray(max_teffs, dtype=float)
            teffs = np.minimum(teffs, max_teffs)
        teffs = np.maximum(teffs, 2800.0)

        return radii, teffs

    def get_flux_ratio(
        self, masses: np.ndarray, filter_name: str,
    ) -> np.ndarray:
        """Compute the flux ratio of stars relative to a solar-type star.

        Ports funcs.flux_relation() (funcs.py:174-201).

        Args:
            masses: Array of stellar masses in Solar masses, shape (N,).
            filter_name: One of "TESS", "Vis", "Kepler", "J", "H", "K",
                "g", "r", "i", "z".

        Returns:
            Array of flux ratios, shape (N,). Values are dimensionless.

        Raises:
            ValueError: If filter_name is not recognized.
        """
        filter_name = canonicalize_filter_name(filter_name)
        if filter_name not in _SUPPORTED_FILTERS:
            raise ValueError(
                f"Unknown filter {filter_name!r}. "
                f"Must be one of: {sorted(_SUPPORTED_FILTERS)}"
            )
        self._ensure_flux_splines()
        masses = np.asarray(masses, dtype=float)
        spline = self._flux_splines[filter_name]  # type: ignore[index]
        return np.asarray(10 ** spline(masses))

    def estimate_sdss_magnitudes(
        self, bmag: float, vmag: float, jmag: float,
    ) -> dict[str, float]:
        """Estimate SDSS g, r, i, z magnitudes from B, V, and J magnitudes.

        Ports funcs.estimate_sdss_magnitudes() (funcs.py:466-490).

        Args:
            bmag: B-band magnitude.
            vmag: V-band magnitude.
            jmag: J-band magnitude.

        Returns:
            Dict with keys "g", "r", "i", "z" -- estimated SDSS magnitudes.
        """
        b_v = bmag - vmag

        # g estimates from multiple sources, averaged
        g_from_v1 = vmag + 0.60 * b_v - 0.12  # Jester et al. 2005
        g_from_v2 = vmag + 0.634 * b_v - 0.108  # Bilir, Karaali, Tuncel 2005
        g_from_v3 = vmag + 0.63 * b_v - 0.124  # Jordi, Grebel, Ammon 2006
        g_from_b = bmag + (-0.370) * b_v - 0.124  # Jordi, Grebel, Ammon 2006
        g = (g_from_v1 + g_from_v2 + g_from_b + g_from_v3) / 4

        # r from Jester et al. 2005
        r = vmag - 0.42 * b_v + 0.11

        # i from Eq. 13, Bilir et al. 2008 (MNRAS 384, 1178)
        i = r - ((g - jmag) - 1.379 * (g - r) - 0.518) / 1.702

        # z from Jordi, Grebel, Ammon 2006
        r_i = ((r - i) + 0.236) / 1.007
        z = -1.584 * r_i + 0.386 + r

        return {"g": g, "r": r, "i": i, "z": z}
