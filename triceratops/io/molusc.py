"""Load MOLUSC CSV files into MoluscData objects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import read_csv

from triceratops.domain.molusc import MoluscData


def load_molusc_file(path: Path) -> MoluscData:
    """Load a MOLUSC CSV file into a MoluscData object.

    Reads the CSV once at prep time. The returned MoluscData is
    picklable and carries no filesystem references.
    """
    df = read_csv(path)
    if len(df) == 0:
        raise ValueError(f"MOLUSC file {path} has zero data rows")
    required = {"semi-major axis(AU)", "eccentricity", "mass ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MOLUSC file {path} is missing columns: {missing}")
    return MoluscData(
        semi_major_axis_au=np.asarray(df["semi-major axis(AU)"]),
        eccentricity=np.asarray(df["eccentricity"]),
        mass_ratio=np.asarray(df["mass ratio"]),
    )
