"""Provider protocols for star catalog and aperture queries.

These define the dependency-inversion boundary: all network I/O is behind
these protocols, allowing tests to inject stubs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from triceratops.domain.entities import StellarField


@runtime_checkable
class StarCatalogProvider(Protocol):
    """Protocol for querying stars near a target from a catalog."""

    def query_nearby_stars(
        self,
        tic_id: int,
        search_radius_px: int,
        mission: str,
    ) -> StellarField:
        """Return all catalogued stars within search_radius_px of the target.

        Args:
            tic_id: TIC ID (or KIC/EPIC ID for Kepler/K2).
            search_radius_px: Search radius in pixels.
            mission: Survey mission name.  Only ``"TESS"`` is supported for
                prepared compute.  Kepler/K2 support is experimental.

        Returns:
            StellarField with target at index 0.
        """
        ...


@runtime_checkable
class ApertureProvider(Protocol):
    """Protocol for retrieving pixel-level cutout images."""

    def get_cutouts(
        self,
        ra_deg: float,
        dec_deg: float,
        size_px: int,
        sectors: np.ndarray,
        mission: str,
    ) -> list[np.ndarray]:
        """Return a list of 2D pixel arrays, one per sector/quarter/campaign.

        Args:
            ra_deg, dec_deg: Coordinates of the target.
            size_px: Size of the cutout in pixels (square).
            sectors: Array of sector/quarter/campaign numbers.
            mission: Survey mission name.  Only ``"TESS"`` is fully supported;
                non-TESS missions raise ``NotImplementedError`` in the default
                provider.

        Returns:
            List of 2D arrays of shape (size_px, size_px). One per sector.
        """
        ...
