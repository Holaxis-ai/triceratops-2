"""Convert a pre-processed, pre-folded lightkurve LightCurve to the domain type."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.errors import LightCurveEmptyError, LightCurvePreparationError
_CADENCE_DAYS: dict[str, float] = {
    "20sec": 20 / 86400,
    "2min": 120 / 86400,
    "10min": 600 / 86400,
    "30min": 1800 / 86400,
}

if TYPE_CHECKING:
    import lightkurve


def convert_folded_to_domain(
    lk_lc: lightkurve.LightCurve,
    *,
    cadence: str = "auto",
    config: LightCurveConfig | None = None,
) -> LightCurve:
    """Convert a pre-processed, pre-folded lightkurve LightCurve to the domain type.

    Use this when you have run your own lightkurve pipeline and just need
    the type conversion step. Bypasses all acquisition and transformation.
    The input must already be phase-folded with transit at phase=0.

    IMPORTANT: uses .phase.value (TimeDelta in days). On FoldedLightCurve,
    .phase is an alias for .time — both hold phase, not the original BTJD.
    """
    config = config or LightCurveConfig()

    # Extract arrays — .phase.value is TimeDelta in days, NOT .time.value
    time_days = lk_lc.phase.value.astype(np.float64)
    flux = lk_lc.flux.value.astype(np.float64)
    flux_err_arr = lk_lc.flux_err.value.astype(np.float64)

    # NaN sweep
    finite = np.isfinite(time_days) & np.isfinite(flux) & np.isfinite(flux_err_arr)
    time_days = time_days[finite]
    flux = flux[finite]
    flux_err_arr = flux_err_arr[finite]
    if len(time_days) == 0:
        raise LightCurveEmptyError(
            "No finite cadences in pre-folded lightkurve object"
        )

    # Scalar error collapse
    flux_err_scalar = float(np.mean(flux_err_arr))
    if not (np.isfinite(flux_err_scalar) and flux_err_scalar > 0):
        raise LightCurvePreparationError(
            "flux_err collapsed to non-positive scalar"
        )

    # cadence_days: config override → cadence string → EXPTIME metadata
    cadence_days = (
        config.cadence_days_override
        or _CADENCE_DAYS.get(cadence)
        or lk_lc.meta.get("EXPTIME", 120) / 86400
    )

    return LightCurve(
        time_days=time_days,
        flux=flux,
        flux_err=flux_err_scalar,
        cadence_days=cadence_days,
        supersampling_rate=config.supersampling_rate,
    )
