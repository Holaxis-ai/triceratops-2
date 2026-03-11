"""Light-curve assembly: adapter for the LC sub-pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from triceratops.assembly.errors import AssemblyLightCurveError
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.result import LightCurvePreparationResult

if TYPE_CHECKING:
    from triceratops.assembly.protocols import ArtifactStore, LightCurveSource
    from triceratops.domain.entities import LightCurve
    from triceratops.lightcurve.ephemeris import Ephemeris


def assemble_light_curve(
    lc_source: LightCurveSource,
    artifact_store: ArtifactStore | None,
    ephemeris: Ephemeris,
    lc_config: LightCurveConfig | None,
    require: bool,
) -> tuple[LightCurve | None, str, list[str], list[str]]:
    """Prepare a light curve via the LC source.

    Returns:
        (light_curve, source_label, warnings, artifact_ids) tuple.
        light_curve is None iff require=False and prep failed.

    Raises:
        AssemblyLightCurveError: If require=True and the LC pipeline fails.
    """
    artifact_ids: list[str] = []
    config = lc_config if lc_config is not None else LightCurveConfig()
    try:
        result = cast("LightCurvePreparationResult", lc_source.prepare(ephemeris, config))
    except Exception as exc:
        if require:
            raise AssemblyLightCurveError(
                f"Light-curve preparation failed: {exc}"
            ) from exc
        return (None, "lc_source", [str(exc)], [])

    lc = result.light_curve
    warnings = list(result.warnings)

    if artifact_store is not None:
        aid = artifact_store.put_prepared_lc(lc)
        artifact_ids.append(aid)

    return (lc, "lc_source", warnings, artifact_ids)
