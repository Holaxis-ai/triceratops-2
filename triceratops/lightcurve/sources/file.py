"""FileRawSource — load raw photometry from FITS or plain-text files on disk."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.errors import LightCurveNotFoundError
from triceratops.lightcurve.raw import RawLightCurveData

# Column name aliases for plain-text files
_TIME_COLS = ("time", "time_btjd", "btjd", "bjd", "t")
_FLUX_COLS = ("flux", "pdcsap_flux", "sap_flux", "f")
_FERR_COLS = ("flux_err", "flux_error", "pdcsap_flux_err", "sap_flux_err", "e", "ferr")


class FileRawSource:
    """Raw source from a FITS or plain-text file on disk.

    Normalises each sector by its median before returning.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch_raw(self, config: LightCurveConfig) -> RawLightCurveData:
        if not self.path.exists():
            raise LightCurveNotFoundError(f"File not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix in (".fits", ".fit"):
            return self._load_fits(config)
        return self._load_text(config)

    def _load_fits(self, config: LightCurveConfig) -> RawLightCurveData:
        from astropy.io import fits

        warn_list: list[str] = []

        with fits.open(self.path) as hdul:
            # Find the binary table extension
            data = None
            header = hdul[0].header
            for hdu in hdul[1:]:
                if hasattr(hdu, "data") and hdu.data is not None and hasattr(hdu, "columns"):
                    data = hdu.data
                    header = hdu.header
                    break
            if data is None:
                raise LightCurveNotFoundError(
                    f"No binary table extension found in {self.path}"
                )

            # Extract columns
            time = self._find_column(data, _TIME_COLS, "time")
            flux = self._find_column(data, _FLUX_COLS, "flux")
            flux_err = self._find_column(data, _FERR_COLS, "flux_err")

            # Remove NaN rows
            valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
            time = time[valid]
            flux = flux[valid]
            flux_err = flux_err[valid]

            if len(time) == 0:
                raise LightCurveNotFoundError(
                    f"No valid data rows in {self.path}"
                )

            # Sort by time
            order = np.argsort(time)
            time = time[order].astype(np.float64)
            flux = flux[order].astype(np.float64)
            flux_err = flux_err[order].astype(np.float64)

            # Normalise by median
            flux, flux_err, norm_warn = _normalise_by_median(flux, flux_err)
            warn_list.extend(norm_warn)

            # Extract sector from header
            sector = header.get("SECTOR")
            sectors = (int(sector),) if sector is not None else (0,)

            # Cadence from header
            exptime = float(header.get("TIMEDEL", header.get("EXPTIME", 120.0)))
            if exptime < 1.0:
                exptime *= 86400.0

            cadence = _cadence_from_exptime(exptime)

        return RawLightCurveData(
            time_btjd=time,
            flux=flux,
            flux_err=flux_err,
            sectors=sectors,
            cadence=cadence,
            exptime_seconds=exptime,
            target_id=None,
            warnings=tuple(warn_list),
        )

    def _load_text(self, config: LightCurveConfig) -> RawLightCurveData:
        warn_list: list[str] = []

        # Try comma-delimited first, then whitespace
        text = self.path.read_text()
        if "," in text.split("\n")[0]:
            delimiter = ","
        else:
            delimiter = None  # whitespace

        # Find header line (first non-comment line) or assume column order
        lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.startswith("#")]
        if len(lines) < 2:
            raise LightCurveNotFoundError(f"Not enough data in {self.path}")

        # Try to parse header
        first_line = lines[0]
        try:
            float(first_line.split(delimiter)[0].strip())
            has_header = False
        except ValueError:
            has_header = True

        if has_header:
            headers = [h.strip().lower() for h in first_line.split(delimiter or None)]
            data_lines = lines[1:]
        else:
            headers = ["time", "flux", "flux_err"]
            data_lines = lines

        # Parse data
        rows = []
        for line in data_lines:
            parts = line.split(delimiter or None)
            try:
                rows.append([float(p.strip()) for p in parts[:len(headers)]])
            except ValueError:
                continue

        if len(rows) == 0:
            raise LightCurveNotFoundError(f"No valid data rows in {self.path}")

        arr = np.array(rows, dtype=np.float64)

        # Find columns by header name
        time_idx = _find_col_index(headers, _TIME_COLS, default=0)
        flux_idx = _find_col_index(headers, _FLUX_COLS, default=1)
        ferr_idx = _find_col_index(headers, _FERR_COLS, default=2)

        time = arr[:, time_idx]
        flux = arr[:, flux_idx]
        flux_err = arr[:, ferr_idx] if ferr_idx < arr.shape[1] else np.full(len(time), 1e-4)

        # Remove NaN/non-finite
        valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
        time = time[valid]
        flux = flux[valid]
        flux_err = flux_err[valid]

        if len(time) == 0:
            raise LightCurveNotFoundError(f"No valid data rows in {self.path}")

        # Sort by time
        order = np.argsort(time)
        time = time[order]
        flux = flux[order]
        flux_err = flux_err[order]

        # Normalise by median
        flux, flux_err, norm_warn = _normalise_by_median(flux, flux_err)
        warn_list.extend(norm_warn)

        return RawLightCurveData(
            time_btjd=time,
            flux=flux,
            flux_err=flux_err,
            sectors=(0,),  # unknown from text file
            cadence="2min",  # default assumption for text files
            exptime_seconds=120.0,
            target_id=None,
            warnings=tuple(warn_list),
        )

    @staticmethod
    def _find_column(data: object, aliases: tuple[str, ...], label: str) -> np.ndarray:
        col_names = [c.lower() for c in data.names]  # type: ignore[union-attr]
        for alias in aliases:
            if alias.lower() in col_names:
                idx = col_names.index(alias.lower())
                return np.array(data[data.names[idx]], dtype=np.float64)  # type: ignore[index]
        raise LightCurveNotFoundError(
            f"Could not find {label} column (tried: {aliases})"
        )


def _normalise_by_median(
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalise flux by its median, propagating to flux_err."""
    warn_list: list[str] = []
    median = float(np.median(flux))

    if abs(median - 1.0) < 0.05:
        # Already normalised — check and warn if slightly off
        if abs(median - 1.0) > 0.01:
            msg = f"flux median is {median:.4f}, close to but not exactly 1.0"
            warnings.warn(msg, stacklevel=3)
            warn_list.append(msg)
        return flux, flux_err, warn_list

    if median == 0:
        msg = "flux median is 0.0, cannot normalise"
        warnings.warn(msg, stacklevel=3)
        warn_list.append(msg)
        return flux, flux_err, warn_list

    flux_normed = flux / median
    flux_err_normed = flux_err / median
    return flux_normed, flux_err_normed, warn_list


def _find_col_index(
    headers: list[str], aliases: tuple[str, ...], default: int
) -> int:
    for alias in aliases:
        if alias.lower() in headers:
            return headers.index(alias.lower())
    return default


def _cadence_from_exptime(exptime_seconds: float) -> str:
    if exptime_seconds < 60:
        return "20sec"
    if exptime_seconds < 300:
        return "2min"
    if exptime_seconds < 900:
        return "10min"
    return "30min"
