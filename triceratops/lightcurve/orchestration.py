"""Convenience orchestration wrappers for light-curve preparation.

Thin wrappers that chain acquisition → prepare_from_raw().
Not the architecture — just ergonomics.
"""
from __future__ import annotations

from pathlib import Path

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import ResolvedTarget
from triceratops.lightcurve.errors import EphemerisRequiredError
from triceratops.lightcurve.prep import prepare_from_raw
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.lightcurve.sources.file import FileRawSource
from triceratops.lightcurve.sources.lightkurve import LightkurveRawSource


def prepare_lightcurve_from_tic(
    target: ResolvedTarget,
    config: LightCurveConfig | None = None,
) -> LightCurvePreparationResult:
    """Convenience: LightkurveRawSource(target.tic_id).fetch_raw() -> prepare_from_raw().

    Requires target.ephemeris to be present; raises EphemerisRequiredError if None.
    """
    if target.ephemeris is None:
        raise EphemerisRequiredError(
            f"ResolvedTarget for TIC {target.tic_id} has no ephemeris. "
            "Resolve ephemeris first (e.g. via ExoFopEphemerisResolver) or "
            "provide one manually."
        )
    config = config or LightCurveConfig()
    source = LightkurveRawSource(target.tic_id)
    raw = source.fetch_raw(config)
    return prepare_from_raw(raw, target.ephemeris, config)


def prepare_lightcurve_from_file(
    path: str | Path,
    target: ResolvedTarget,
    config: LightCurveConfig | None = None,
) -> LightCurvePreparationResult:
    """Convenience: FileRawSource(path).fetch_raw() -> prepare_from_raw().

    Requires target.ephemeris to be present; raises EphemerisRequiredError if None.
    """
    if target.ephemeris is None:
        raise EphemerisRequiredError(
            f"ResolvedTarget for TIC {target.tic_id} has no ephemeris. "
            "Resolve ephemeris first or provide one manually."
        )
    config = config or LightCurveConfig()
    source = FileRawSource(path)
    raw = source.fetch_raw(config)
    return prepare_from_raw(raw, target.ephemeris, config)
