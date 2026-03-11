"""LightkurveSource — acquire and prepare a light curve from MAST via lightkurve."""
from __future__ import annotations

import logging
import os
import time as time_mod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.convert import convert_folded_to_domain
from triceratops.lightcurve.errors import (
    DownloadTimeoutError,
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurveNotFoundError,
)
from triceratops.lightcurve.result import LightCurvePreparationResult

if TYPE_CHECKING:
    from triceratops.lightcurve.ephemeris import Ephemeris

log = logging.getLogger(__name__)

_CADENCE_MAP: dict[str, tuple[str, float | None]] = {
    "20sec": ("SPOC", 20.0),
    "2min": ("SPOC", 120.0),
    "10min": ("QLP", 600.0),
    "30min": ("SPOC", 1800.0),
}

_MAX_RETRIES = 3


def _cache_dir() -> Path:
    base = os.environ.get("TRICERATOPS_CACHE_DIR", str(Path.home() / ".triceratops" / "cache"))
    cache = Path(base) / "lightkurve"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _quality_bitmask(quality_mask: str) -> str:
    return {"none": "none", "default": "default", "hard": "hard"}[quality_mask]


def process_lightcurve_collection(
    lc_coll: Any,
    config: LightCurveConfig,
    *,
    tic_id: int,
) -> Any:
    """Apply the standard lightkurve preprocessing pipeline to a collection."""
    lc = lc_coll.stitch()
    lc = lc.remove_nans()

    if len(lc.time) == 0:
        raise LightCurveNotFoundError(
            f"TIC {tic_id}: all cadences removed after stitch/NaN removal"
        )

    if config.sigma_clip is not None:
        lc = lc.remove_outliers(
            sigma_upper=config.sigma_clip,
            sigma_lower=float("inf"),
        )

    if config.detrend_method == "flatten":
        lc = lc.flatten(
            window_length=config.flatten_window_length,
            polyorder=config.flatten_polyorder,
        )

    return lc


def fold_lightcurve(
    lc: Any,
    ephemeris: Ephemeris,
) -> Any:
    """Fold a light curve on the supplied ephemeris."""
    import astropy.time

    epoch = astropy.time.Time(ephemeris.t0_btjd, format="btjd", scale="tdb")
    return lc.fold(period=ephemeris.period_days, epoch_time=epoch)


def trim_folded_lightcurve(
    lc_folded: Any,
    ephemeris: Ephemeris,
    config: LightCurveConfig,
    *,
    tic_id: int,
) -> Any:
    """Trim a folded light curve to the configured transit window."""
    if ephemeris.duration_hours is not None:
        half_window = config.phase_window_factor * ephemeris.duration_hours / 24.0
    else:
        half_window = ephemeris.period_days * 0.25
    lc_trimmed = lc_folded[np.abs(lc_folded.phase.value) < half_window]

    if len(lc_trimmed) == 0:
        raise LightCurveEmptyError(
            f"TIC {tic_id}: no cadences in transit window after fold and trim"
        )
    return lc_trimmed


def fold_and_trim_lightcurve(
    lc: Any,
    ephemeris: Ephemeris,
    config: LightCurveConfig,
    *,
    tic_id: int,
) -> Any:
    """Fold a light curve on the supplied ephemeris and trim to the transit window."""
    lc_folded = fold_lightcurve(lc, ephemeris)
    return trim_folded_lightcurve(lc_folded, ephemeris, config, tic_id=tic_id)


def resolve_cadence_label(stitched: Any, config_cadence: str) -> str:
    """Resolve the cadence label used for downstream exposure-time selection."""
    if config_cadence != "auto":
        return config_cadence
    exptime = stitched.meta.get("TIMEDEL", stitched.meta.get("EXPTIME"))
    if exptime is not None:
        exptime_sec = float(exptime)
        if exptime_sec < 1.0:
            exptime_sec *= 86400.0
        if exptime_sec < 60:
            return "20sec"
        if exptime_sec < 300:
            return "2min"
        if exptime_sec < 900:
            return "10min"
        return "30min"
    return "2min"


class LightkurveSource:
    """Acquire and prepare a TESS light curve from MAST using lightkurve.

    Uses lightkurve's own fold, sigma-clip, and flatten methods.
    Returns a compute-ready LightCurvePreparationResult.
    """

    def __init__(self, tic_id: int, _override_collection: Any = None) -> None:
        self.tic_id = tic_id
        self._override_collection = _override_collection

    def prepare(
        self,
        ephemeris: Ephemeris,
        config: LightCurveConfig | None = None,
    ) -> LightCurvePreparationResult:
        """Download from MAST, process with lightkurve, return a domain LightCurve."""
        config = config or LightCurveConfig()
        lc_folded, sectors, cadence_used = self.prepare_folded(ephemeris, config)
        lc_domain = convert_folded_to_domain(lc_folded, cadence=cadence_used, config=config)

        return LightCurvePreparationResult(
            light_curve=lc_domain,
            ephemeris=ephemeris,
            sectors_used=sectors,
            cadence_used=cadence_used,
            warnings=[],
        )

    def prepare_folded(
        self,
        ephemeris: Ephemeris,
        config: LightCurveConfig | None = None,
    ) -> tuple[Any, tuple[int, ...], str]:
        """Download from MAST and return a folded, trimmed lightkurve object.

        Uses lightkurve for: stitch, remove_outliers (upper-only), flatten, fold.
        Does NOT reimplement any photometry processing.

        Returns the folded, trimmed lightkurve object plus source metadata.
        This lower-level API exists so orchestration code can apply
        lightkurve-native transforms, such as optional binning, before
        converting into the domain LightCurve type.
        """
        import lightkurve as lk

        config = config or LightCurveConfig()

        if not (np.isfinite(ephemeris.period_days) and ephemeris.period_days > 0):
            raise EphemerisRequiredError("period_days must be finite and positive")
        if not (np.isfinite(ephemeris.t0_btjd) and ephemeris.t0_btjd < 10_000):
            raise EphemerisRequiredError(
                "t0_btjd must be in BTJD (BJD – 2,457,000); "
                f"got {ephemeris.t0_btjd} which looks like full JD"
            )

        # Step 1: Acquire
        if self._override_collection is not None:
            lc_coll = self._override_collection
        else:
            lc_coll = self._search_and_download(config, lk)

        lc = process_lightcurve_collection(lc_coll, config, tic_id=self.tic_id)
        lc_trimmed = fold_and_trim_lightcurve(lc, ephemeris, config, tic_id=self.tic_id)

        # Step 7: Convert to domain type
        sectors = self._extract_sectors(lc_coll, lc)
        cadence_used = resolve_cadence_label(lc, config.cadence)
        return lc_trimmed, sectors, cadence_used

    def _search_and_download(self, config: LightCurveConfig, lk: Any) -> Any:
        search_kwargs: dict[str, Any] = {
            "target": f"TIC {self.tic_id}",
            "mission": "TESS",
        }
        if config.cadence != "auto" and config.cadence in _CADENCE_MAP:
            author, exptime = _CADENCE_MAP[config.cadence]
            search_kwargs["author"] = author
            search_kwargs["exptime"] = exptime
        if isinstance(config.sectors, tuple):
            search_kwargs["sector"] = list(config.sectors)

        search = lk.search_lightcurve(**search_kwargs)
        if len(search) == 0:
            raise LightCurveNotFoundError(
                f"No TESS light curves found for TIC {self.tic_id}"
            )

        search_filtered = self._select_sectors(search, config.sectors)
        return self._download_with_retry(
            search_filtered,
            quality_bitmask=_quality_bitmask(config.quality_mask),
            flux_column=config.flux_type,
        )

    @staticmethod
    def _select_sectors(search: Any, sectors: Any) -> Any:
        if isinstance(sectors, tuple):
            return search
        if sectors == "auto":
            return search[-1:]
        return search

    @staticmethod
    def _download_with_retry(search: Any, quality_bitmask: str, flux_column: str) -> Any:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return search.download_all(
                    quality_bitmask=quality_bitmask,
                    flux_column=flux_column,
                    download_dir=str(_cache_dir()),
                )
            except Exception as exc:
                last_err = exc
                exc_str = str(exc).lower()
                retryable = any(
                    kw in exc_str
                    for kw in ("timeout", "429", "500", "502", "503", "connection")
                )
                if not retryable:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning(
                    "MAST download attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time_mod.sleep(wait)
        raise DownloadTimeoutError(
            f"MAST download failed after {_MAX_RETRIES} attempts: {last_err}",
            retryable=True,
        )

    @staticmethod
    def _extract_sectors(lc_coll: Any, stitched: Any) -> tuple[int, ...]:
        raw = lc_coll.sector  # numpy array, np.nan for missing
        valid = raw[~np.isnan(raw)]
        if len(valid) > 0:
            return tuple(sorted(int(s) for s in valid))
        meta_sectors = stitched.meta.get("SECTOR")
        if meta_sectors is not None:
            if isinstance(meta_sectors, (list, tuple)):
                return tuple(sorted(int(s) for s in meta_sectors))
            return (int(meta_sectors),)
        return (0,)
