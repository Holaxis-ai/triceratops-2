"""Concrete catalog/aperture providers using MAST and Tesscut.

All network I/O in the package is isolated to this module.
"""

from __future__ import annotations

from typing import overload

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs, Tesscut  # type: ignore[import-untyped]
from astroquery.vizier import Vizier  # type: ignore[import-untyped]

from triceratops.config.config import MissionConfig
from triceratops.domain.entities import Star, StellarField
from triceratops.domain.value_objects import StellarParameters


@overload
def _safe_float(val: object, default: float) -> float: ...
@overload
def _safe_float(val: object, default: None) -> float | None: ...
def _safe_float(val: object, default: float | None = 0.0) -> float | None:
    """Convert to float, returning *default* for NaN / None."""
    if val is None:
        return default
    try:
        f = float(val)  # type: ignore[arg-type]
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


class MASTCatalogProvider:
    """Concrete StarCatalogProvider that queries the MAST TIC catalog.

    Ports the inline astroquery calls from triceratops.py:66-245.
    Fixes BUG-01 (DataFrame.append -> pd.concat) and
    BUG-02 (reshape wrong arg type).
    """

    def query_nearby_stars(
        self,
        tic_id: int,
        search_radius_px: int,
        mission: str,
    ) -> StellarField:
        """Query TIC for all stars within search_radius_px of the target."""
        mission_cfg = MissionConfig.for_mission(mission)
        pixel_size = mission_cfg.pixel_size_arcsec * u.arcsec
        radius = search_radius_px * pixel_size

        # Resolve coordinates for Kepler/K2 via Vizier
        ra: float | None = None
        dec: float | None = None
        if mission == "Kepler":
            result = (
                Vizier(columns=["_RA", "_DE"])
                .query_constraints(KIC=str(tic_id), catalog="J/ApJS/229/30/catalog")  # type: ignore[attr-defined]
                [0].as_array()
            )
            ra = float(result[0]["_RA"])
            dec = float(result[0]["_DE"])
        elif mission == "K2":
            result = (
                Vizier(columns=["RAJ2000", "DEJ2000"])
                .query_constraints(ID=str(tic_id), catalog="IV/34/epic")  # type: ignore[attr-defined]
                [0].as_array()
            )
            ra = float(result[0]["RAJ2000"])
            dec = float(result[0]["DEJ2000"])

        # For Kepler/K2, resolve to nearest TIC ID
        resolved_id = tic_id
        if ra is not None and dec is not None:
            resolved_id = Catalogs.query_region(  # type: ignore[attr-defined]
                SkyCoord(ra, dec, unit="deg"),
                radius=radius,
                catalog="TIC",
            )[0]["ID"]

        # Query TIC
        df = Catalogs.query_object(  # type: ignore[attr-defined]
            "TIC" + str(resolved_id),
            radius=radius,
            catalog="TIC",
        )
        cols = [
            "ID", "Bmag", "Vmag", "Tmag", "Jmag", "Hmag", "Kmag",
            "gmag", "rmag", "imag", "zmag",
            "ra", "dec", "mass", "rad", "Teff", "d", "plx",
        ]
        stars_df = df[cols].to_pandas()

        # MAST can return duplicate-coordinate rows where the requested TIC is not
        # first in the table. Reorder explicitly so the exact resolved TIC becomes
        # the target row used for separations and StellarField.target.
        target_mask = stars_df["ID"].astype(int) == int(resolved_id)
        if target_mask.any():
            target_rows = stars_df[target_mask]
            other_rows = stars_df[~target_mask]
            stars_df = target_rows.reset_index(drop=True)
            if not other_rows.empty:
                import pandas as pd

                stars_df = pd.concat(
                    [stars_df, other_rows.reset_index(drop=True)],
                    ignore_index=True,
                )

        # Compute separations and position angles
        target_coord = SkyCoord(
            stars_df["ra"].values[0],
            stars_df["dec"].values[0],
            unit="deg",
        )
        separations = [0.0]
        position_angles = [0.0]
        for i in range(1, len(stars_df)):
            star_coord = SkyCoord(
                stars_df["ra"].values[i],
                stars_df["dec"].values[i],
                unit="deg",
            )
            separations.append(
                round(target_coord.separation(star_coord).to(u.arcsec).value, 3)
            )
            position_angles.append(
                round(target_coord.position_angle(star_coord).to(u.deg).value, 3)
            )

        # Build Star objects
        star_list: list[Star] = []
        for i in range(len(stars_df)):
            row = stars_df.iloc[i]
            sp = StellarParameters.from_tic_row(row.to_dict())
            star_list.append(Star(
                tic_id=int(row["ID"]),
                ra_deg=float(row["ra"]),
                dec_deg=float(row["dec"]),
                tmag=_safe_float(row.get("Tmag"), 99.0),
                jmag=_safe_float(row.get("Jmag"), 99.0),
                hmag=_safe_float(row.get("Hmag"), 99.0),
                kmag=_safe_float(row.get("Kmag"), 99.0),
                bmag=_safe_float(row.get("Bmag"), 99.0),
                vmag=_safe_float(row.get("Vmag"), 99.0),
                gmag=_safe_float(row.get("gmag"), None),
                rmag=_safe_float(row.get("rmag"), None),
                imag=_safe_float(row.get("imag"), None),
                zmag=_safe_float(row.get("zmag"), None),
                stellar_params=sp,
                separation_arcsec=separations[i],
                position_angle_deg=position_angles[i],
            ))

        return StellarField(
            target_id=int(stars_df["ID"].values[0]),
            mission=mission,
            search_radius_pixels=search_radius_px,
            stars=star_list,
        )


class TesscutApertureProvider:
    """Concrete ApertureProvider using astroquery Tesscut.

    Ports the cutout logic from triceratops.py:131-207.
    """

    def get_cutouts(
        self,
        ra_deg: float,
        dec_deg: float,
        size_px: int,
        sectors: np.ndarray,
        mission: str,
    ) -> list[np.ndarray]:
        """Return pixel cutout images, one per sector."""
        images: list[np.ndarray] = []
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")

        for sector in sectors:
            if mission == "TESS":
                cutout_hdu = Tesscut.get_cutouts(
                    coordinates=coord, size=size_px, sector=int(sector),
                )[0]
                cutout_table = cutout_hdu[1].data
                images.append(np.nanmean(cutout_table["FLUX"], axis=0))
            else:
                raise NotImplementedError(
                    "Kepler/K2 cutouts require lightkurve; not yet ported"
                )

        return images
