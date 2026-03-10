"""Domain entities: mutable or composite data types representing domain concepts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .value_objects import LimbDarkeningCoeffs, StellarParameters


@dataclass
class Star:
    """One catalogued star in the photometric field."""

    tic_id: int
    ra_deg: float
    dec_deg: float
    tmag: float
    jmag: float
    hmag: float
    kmag: float
    bmag: float
    vmag: float
    gmag: float | None = None
    rmag: float | None = None
    imag: float | None = None
    zmag: float | None = None
    stellar_params: StellarParameters | None = None
    separation_arcsec: float = 0.0
    position_angle_deg: float = 0.0
    # Computed by flux_contributions module (P1-022), not set at construction:
    flux_ratio: float | None = None
    transit_depth_required: float | None = None

    def mag_for_band(self, band: str) -> float | None:
        """Return the magnitude for the given filter band, or None if unavailable.

        Args:
            band: One of "TESS", "J", "H", "K", "B", "V", "g", "r", "i", "z".
        """
        _map = {
            "TESS": self.tmag, "J": self.jmag, "H": self.hmag, "K": self.kmag,
            "B": self.bmag, "V": self.vmag, "g": self.gmag,
            "r": self.rmag, "i": self.imag, "z": self.zmag,
        }
        return _map.get(band)


@dataclass
class StellarField:
    """All stars in the photometric search aperture, target at index 0.

    Invariants
    ----------
    - ``stars`` is non-empty.
    - ``stars[0].tic_id == target_id`` (target is always index 0).
    - No two stars share a TIC ID.

    Use the mutation methods (``add_neighbor``, ``remove_neighbor``,
    ``update_star``) rather than mutating ``stars`` directly so that
    these invariants are maintained.  Call ``validate()`` to check them
    explicitly before a compute job.
    """

    target_id: int
    mission: str
    search_radius_pixels: int
    stars: list[Star]          # stars[0] is always the target

    @property
    def target(self) -> Star:
        return self.stars[0]

    @property
    def neighbors(self) -> list[Star]:
        return self.stars[1:]

    def stars_with_flux_data(self) -> list[Star]:
        """Return stars that have a non-None, positive transit_depth_required."""
        return [
            s for s in self.stars
            if s.transit_depth_required is not None and s.transit_depth_required > 0
        ]

    # ------------------------------------------------------------------
    # Guarded mutation API
    # ------------------------------------------------------------------

    def add_neighbor(self, star: Star) -> None:
        """Append a neighbor star to the field.

        Raises:
            ValueError: If a star with the same TIC ID already exists.
        """
        existing_ids = {s.tic_id for s in self.stars}
        if star.tic_id in existing_ids:
            raise ValueError(
                f"Star with TIC ID {star.tic_id} already exists in the field. "
                "Use update_star() to modify an existing star."
            )
        self.stars.append(star)

    def remove_neighbor(self, tic_id: int) -> None:
        """Remove a neighbor star by TIC ID.

        Raises:
            ValueError: If ``tic_id`` is the target star's ID.
            ValueError: If no star with ``tic_id`` is found.
        """
        if tic_id == self.target_id:
            raise ValueError(
                f"Cannot remove the target star (TIC ID {tic_id}). "
                "The target must always be at stars[0]."
            )
        original_len = len(self.stars)
        self.stars[:] = [s for s in self.stars if s.tic_id != tic_id]
        if len(self.stars) == original_len:
            raise ValueError(f"Star with TIC ID {tic_id} not found in the field.")

    def update_star(self, tic_id: int, **kwargs: object) -> None:
        """Update fields on a star by TIC ID.

        Accepts both direct ``Star`` attribute names and TIC-style aliases:

        ============  ==================================
        Alias         Maps to
        ============  ==================================
        ``Teff``      ``stellar_params.teff_k``
        ``mass``      ``stellar_params.mass_msun``
        ``logg``      ``stellar_params.logg``
        ``metallicity`` ``stellar_params.metallicity_dex``
        ============  ==================================

        Raises:
            ValueError: If no star with ``tic_id`` is found.
            TypeError: If an alias update is requested but
                ``stellar_params`` is ``None``.
            AttributeError: If an unknown attribute name is given.
        """
        from dataclasses import replace

        _STELLAR_PARAMS_ALIASES: dict[str, str] = {
            "Teff": "teff_k",
            "mass": "mass_msun",
            "logg": "logg",
            "metallicity": "metallicity_dex",
        }

        for star in self.stars:
            if star.tic_id == tic_id:
                stellar_updates: dict[str, object] = {}
                for attr, value in kwargs.items():
                    if attr in _STELLAR_PARAMS_ALIASES:
                        stellar_updates[_STELLAR_PARAMS_ALIASES[attr]] = value
                    elif hasattr(star, attr):
                        object.__setattr__(star, attr, value)
                    else:
                        raise AttributeError(f"Star has no attribute {attr!r}")
                if stellar_updates:
                    if star.stellar_params is None:
                        raise TypeError(
                            f"Cannot update stellar parameter aliases {list(stellar_updates)} "
                            f"on star TIC {tic_id}: stellar_params is None. "
                            "Set stellar_params explicitly before using alias updates."
                        )
                    object.__setattr__(
                        star, "stellar_params",
                        replace(star.stellar_params, **stellar_updates),
                    )
                return
        raise ValueError(f"Star with TIC ID {tic_id} not found in the field.")

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert all structural invariants.

        Called by ``ValidationPreparer.prepare()`` after catalog assembly
        and by ``ValidationEngine.compute_prepared()`` before compute.
        May also be called explicitly in debug/test flows.

        Raises:
            ValueError: If any invariant is violated.
        """
        if not self.stars:
            raise ValueError(
                "StellarField.stars is empty; the field must contain at least the target star."
            )
        if self.stars[0].tic_id != self.target_id:
            raise ValueError(
                f"StellarField invariant violated: stars[0].tic_id "
                f"({self.stars[0].tic_id}) != target_id ({self.target_id}). "
                "The target star must always be at index 0."
            )
        seen: set[int] = set()
        for star in self.stars:
            if star.tic_id in seen:
                raise ValueError(
                    f"StellarField invariant violated: duplicate TIC ID {star.tic_id}."
                )
            seen.add(star.tic_id)


@dataclass
class LightCurve:
    """A phase-folded, normalised photometric time series ready for model fitting."""

    time_days: np.ndarray        # days from transit midpoint; t=0 at centre
    flux: np.ndarray             # normalised flux; 1.0 = out of transit
    flux_err: float              # scalar per-point uncertainty (sigma)
    cadence_days: float = 0.00139   # exposure time; 0.00139 ~ 2-min TESS cadence
    supersampling_rate: int = 20    # pytransit integration supersampling

    @property
    def sigma(self) -> float:
        return self.flux_err

    def with_renorm(self, flux_ratio: float) -> LightCurve:
        """Return a new LightCurve renormalised to a single star's contribution.

        This is the vectorised equivalent of renorm_flux() from funcs.py:225-238.
        flux_ratio is the fraction of aperture flux from the host star (0 < fr <= 1).
        """
        renormed = (self.flux - (1.0 - flux_ratio)) / flux_ratio
        renormed_err = self.flux_err / flux_ratio
        return LightCurve(
            time_days=self.time_days,
            flux=renormed,
            flux_err=renormed_err,
            cadence_days=self.cadence_days,
            supersampling_rate=self.supersampling_rate,
        )


@dataclass
class ExternalLightCurve:
    """A ground-based follow-up observation in a specific photometric band."""

    light_curve: LightCurve
    band: str                                   # "J", "H", "K", "g", "r", "i", "z"
    ldc: LimbDarkeningCoeffs | None = None  # resolved by BaseScenario._resolve_external_lc_ldcs
