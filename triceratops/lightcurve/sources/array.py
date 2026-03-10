"""ArrayRawSource — construct RawLightCurveData from numpy arrays."""
from __future__ import annotations

import warnings

import numpy as np

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.raw import RawLightCurveData


class ArrayRawSource:
    """Raw source from user-supplied numpy arrays.

    The caller is responsible for per-sector-median normalisation.
    A soft check warns if median(flux) deviates > 5% from 1.0.
    """

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        sectors: tuple[int, ...],
        cadence: str,
        exptime_seconds: float,
        target_id: int | None = None,
    ) -> None:
        self._time = np.asarray(time, dtype=np.float64)
        self._flux = np.asarray(flux, dtype=np.float64)
        self._flux_err = np.asarray(flux_err, dtype=np.float64)
        self._sectors = sectors
        self._cadence = cadence
        self._exptime_seconds = exptime_seconds
        self._target_id = target_id

    def fetch_raw(self, config: LightCurveConfig) -> RawLightCurveData:
        warn_list: list[str] = []

        # Soft normalisation check
        median_flux = float(np.median(self._flux))
        if abs(median_flux - 1.0) > 0.05:
            msg = (
                f"flux median is {median_flux:.4f}, expected ~1.0. "
                "ArrayRawSource requires caller-normalised flux."
            )
            warnings.warn(msg, stacklevel=2)
            warn_list.append(msg)

        return RawLightCurveData(
            time_btjd=self._time,
            flux=self._flux,
            flux_err=self._flux_err,
            sectors=self._sectors,
            cadence=self._cadence,
            exptime_seconds=self._exptime_seconds,
            target_id=self._target_id,
            warnings=tuple(warn_list),
        )
