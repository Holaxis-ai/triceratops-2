"""Protocol and data types for background star population providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TRILEGALResult:
    """Background star population data from a TRILEGAL simulation.

    All arrays have the same length N (number of background stars returned).
    """

    tmags: np.ndarray
    masses: np.ndarray
    loggs: np.ndarray
    teffs: np.ndarray
    metallicities: np.ndarray
    jmags: np.ndarray
    hmags: np.ndarray
    kmags: np.ndarray
    gmags: np.ndarray
    rmags: np.ndarray
    imags: np.ndarray
    zmags: np.ndarray

    def __len__(self) -> int:
        return len(self.tmags)

    @property
    def n_stars(self) -> int:
        return len(self.tmags)


@runtime_checkable
class PopulationSynthesisProvider(Protocol):
    """Protocol for querying background star populations."""

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        target_tmag: float,
        cache_path: Path | None = None,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> TRILEGALResult:
        """Return a background star population for the given sky position.

        Raises:
            TRILEGALQueryError: If the query fails.
        """
        ...
