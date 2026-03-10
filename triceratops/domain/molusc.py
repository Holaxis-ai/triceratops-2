"""MoluscData: pre-loaded MOLUSC companion population data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MoluscData:
    """Pre-loaded MOLUSC companion population data.

    Contains the three columns from a MOLUSC CSV needed by
    _load_molusc_qs() to filter and extract mass ratios.

    All arrays must have the same length (one entry per MOLUSC row).
    """

    semi_major_axis_au: np.ndarray  # "semi-major axis(AU)" column
    eccentricity: np.ndarray  # "eccentricity" column
    mass_ratio: np.ndarray  # "mass ratio" column

    def __post_init__(self) -> None:
        fields = {
            "semi_major_axis_au": self.semi_major_axis_au,
            "eccentricity": self.eccentricity,
            "mass_ratio": self.mass_ratio,
        }
        for name, arr in fields.items():
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(
                    f"MoluscData.{name} must be a floating-point array, "
                    f"got dtype={arr.dtype!r}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"MoluscData.{name} contains non-finite values (NaN or inf)."
                )
        la, le, lm = len(self.semi_major_axis_au), len(self.eccentricity), len(self.mass_ratio)
        if not (la == le == lm):
            raise ValueError(
                f"All MoluscData arrays must have the same length, "
                f"got semi_major_axis_au={la}, eccentricity={le}, mass_ratio={lm}"
            )
