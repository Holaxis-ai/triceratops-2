"""Light-curve error hierarchy."""
from __future__ import annotations


class LightCurveError(Exception):
    """Base exception for all light-curve preparation errors."""


class LightCurveNotFoundError(LightCurveError):
    """No light curve found for the given target/sector/cadence."""


class SectorNotAvailableError(LightCurveError):
    """Requested sector is not available for this target."""


class EphemerisRequiredError(LightCurveError):
    """Ephemeris is missing or invalid."""


class DownloadTimeoutError(LightCurveError):
    """Download failed after exhausting retries."""

    def __init__(self, message: str = "download timed out", *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LightCurveEmptyError(LightCurveError):
    """No cadences survived the processing pipeline."""


class LightCurvePreparationError(LightCurveError):
    """Generic preparation failure (e.g. flux_err collapsed to non-positive)."""
