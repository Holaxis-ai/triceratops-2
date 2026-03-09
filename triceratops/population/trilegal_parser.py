"""TRILEGAL CSV parsing -- pure function, no I/O beyond reading the file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from triceratops.population.protocols import TRILEGALResult


def _jk_to_tmag(j: float, k: float) -> float:
    """Convert 2MASS J, K to TESS magnitude using Stassun et al. 2018 (Sec 2.2.1.1)."""
    jk = j - k
    if -0.1 <= jk <= 0.70:
        return j + 1.22163 * jk ** 3 - 1.74299 * jk ** 2 + 1.89115 * jk + 0.0563
    elif 0.7 < jk <= 1.0:
        return j - 269.372 * jk ** 3 + 668.453 * jk ** 2 - 545.64 * jk + 147.811
    elif jk < -0.1:
        return j + 0.5
    else:  # jk > 1.0
        return j + 1.75


def parse_trilegal_csv(
    path: Path,
    target_tmag: float | None = None,
) -> TRILEGALResult:
    """Parse a TRILEGAL output CSV into a TRILEGALResult.

    Ports funcs.trilegal_results() (funcs.py:382-463).

    Handles two CSV formats:
    1. Real TRILEGAL output: columns like "TESS", "Mact", "logTe", "[M/H]",
       "J", "H", "Ks", "g", "r", "i", "z"
    2. Stub/simple format: columns like "Tmag", "mass", "logg", "Teff",
       "metallicity", "Jmag", "Hmag", "Kmag", "gmag", "rmag", "imag", "zmag"

    Args:
        path: Path to the TRILEGAL output CSV.
        target_tmag: If given, return only stars with Tmag >= target_tmag
            (fainter than the target, matching original filter).

    Returns:
        TRILEGALResult with arrays populated from the CSV.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TRILEGAL CSV not found: {path}")

    df = pd.read_csv(path)

    # Drop the last 2 rows for real TRILEGAL output (termination markers)
    if len(df) > 2 and "Mact" in df.columns:
        df = df.iloc[:-2]

    # Determine format and extract arrays
    if "Mact" in df.columns:
        # Real TRILEGAL format
        masses = df["Mact"].values.astype(float)
        loggs = df["logg"].values.astype(float)
        teffs = (10.0 ** df["logTe"].values).astype(float)
        metallicities = df["[M/H]"].values.astype(float)
        jmags = df["J"].values.astype(float)
        hmags = df["H"].values.astype(float)
        kmags = df["Ks"].values.astype(float)
        gmags = df["g"].values.astype(float)
        rmags = df["r"].values.astype(float)
        imags = df["i"].values.astype(float)
        zmags = df["z"].values.astype(float)

        if "TESS" in df.columns:
            tmags = df["TESS"].values.astype(float)
        else:
            # Compute Tmag from J-K color (Stassun et al. 2018)
            tmags = np.array([
                _jk_to_tmag(j, k)
                for j, k in zip(jmags, kmags)
            ])
    else:
        # Stub/simple format
        tmags = df["Tmag"].values.astype(float)
        masses = df["mass"].values.astype(float)
        loggs = df["logg"].values.astype(float)
        teffs = df["Teff"].values.astype(float)
        metallicities = df["metallicity"].values.astype(float)
        jmags = df["Jmag"].values.astype(float)
        hmags = df["Hmag"].values.astype(float)
        kmags = df["Kmag"].values.astype(float)
        gmags = df["gmag"].values.astype(float)
        rmags = df["rmag"].values.astype(float)
        imags = df["imag"].values.astype(float)
        zmags = df["zmag"].values.astype(float)

    # Filter: keep only stars fainter than the target (Tmag >= target_tmag)
    if target_tmag is not None:
        mask = tmags >= target_tmag
        tmags = tmags[mask]
        masses = masses[mask]
        loggs = loggs[mask]
        teffs = teffs[mask]
        metallicities = metallicities[mask]
        jmags = jmags[mask]
        hmags = hmags[mask]
        kmags = kmags[mask]
        gmags = gmags[mask]
        rmags = rmags[mask]
        imags = imags[mask]
        zmags = zmags[mask]

    return TRILEGALResult(
        tmags=tmags,
        masses=masses,
        loggs=loggs,
        teffs=teffs,
        metallicities=metallicities,
        jmags=jmags,
        hmags=hmags,
        kmags=kmags,
        gmags=gmags,
        rmags=rmags,
        imags=imags,
        zmags=zmags,
    )
