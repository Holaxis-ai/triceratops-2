"""Stub implementations of provider protocols for testing.

These return fixed, pre-defined data without making any network or disk I/O calls.
Suitable for use in @pytest.mark.unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from triceratops_new.domain.entities import Star, StellarField
from triceratops_new.domain.value_objects import StellarParameters


class StubStarCatalogProvider:
    """Returns a fixed two-star StellarField without any network calls.

    The target star (tic_id=12345678) has solar-like parameters.
    One neighbor star (tic_id=12345679) is 2 arcsec away with M=0.5 Msun.

    Use this anywhere a StarCatalogProvider is required in tests.
    """

    def __init__(
        self,
        target_tic_id: int = 12345678,
        mission: str = "TESS",
        search_radius_px: int = 10,
    ) -> None:
        self._target_tic_id = target_tic_id
        self._mission = mission
        self._search_radius_px = search_radius_px

    def query_nearby_stars(
        self,
        tic_id: int,
        search_radius_px: int,
        mission: str,
    ) -> StellarField:
        target = Star(
            tic_id=self._target_tic_id,
            ra_deg=83.82,
            dec_deg=-5.39,
            tmag=10.5,
            jmag=9.8,
            hmag=9.5,
            kmag=9.4,
            bmag=11.2,
            vmag=10.8,
            stellar_params=StellarParameters(
                mass_msun=1.0,
                radius_rsun=1.0,
                teff_k=5778.0,
                logg=4.44,
                metallicity_dex=0.0,
                parallax_mas=10.0,
            ),
            separation_arcsec=0.0,
            position_angle_deg=0.0,
        )
        neighbor = Star(
            tic_id=self._target_tic_id + 1,
            ra_deg=83.82 + 2 / 3600,
            dec_deg=-5.39,
            tmag=13.0,
            jmag=12.0,
            hmag=11.8,
            kmag=11.7,
            bmag=14.0,
            vmag=13.5,
            stellar_params=StellarParameters(
                mass_msun=0.5,
                radius_rsun=0.5,
                teff_k=3800.0,
                logg=4.7,
                metallicity_dex=0.0,
                parallax_mas=10.0,
            ),
            separation_arcsec=2.0,
            position_angle_deg=90.0,
        )
        return StellarField(
            target_id=tic_id,
            mission=mission,
            search_radius_pixels=search_radius_px,
            stars=[target, neighbor],
        )


@dataclass
class StubTRILEGALResult:
    """Minimal stand-in for the real TRILEGALResult (P1-011).

    Used by StubPopulationSynthesisProvider until the population module is built.
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


class StubPopulationSynthesisProvider:
    """Returns a fixed TRILEGAL-like background population without web requests.

    The fixture CSV at tests/fixtures/trilegal/stub_trilegal.csv is loaded
    once and its columns returned as arrays.
    """

    _FIXTURE_PATH = Path(__file__).parent / "trilegal" / "stub_trilegal.csv"

    def __init__(self, fixture_path: Path | None = None) -> None:
        self._path = fixture_path or self._FIXTURE_PATH
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self._path)
        return self._df

    def query(self, ra: float, dec: float, target_tmag: float) -> StubTRILEGALResult:
        """Return fixture data regardless of input coordinates."""
        df = self._load()
        return StubTRILEGALResult(
            tmags=df["Tmag"].values,
            masses=df["mass"].values,
            loggs=df["logg"].values,
            teffs=df["Teff"].values,
            metallicities=df["metallicity"].values,
            jmags=df["Jmag"].values,
            hmags=df["Hmag"].values,
            kmags=df["Kmag"].values,
            gmags=df["gmag"].values,
            rmags=df["rmag"].values,
            imags=df["imag"].values,
            zmags=df["zmag"].values,
        )
