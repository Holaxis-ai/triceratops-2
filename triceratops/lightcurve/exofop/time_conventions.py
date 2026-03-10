from __future__ import annotations

import math

BJD_TO_BTJD_OFFSET = 2_457_000.0
_ABSOLUTE_BJD_THRESHOLD = 2_400_000.0
_MJD_TO_JD_OFFSET = 2_400_000.5
_BJD_MINUS_2450000_OFFSET = 2_450_000.0


def normalize_epoch_to_btjd(
    epoch: float | None,
    *,
    bjd_reference: float | None = None,
    mjd_reference: float | None = None,
) -> float | None:
    """Normalize a transit epoch to BTJD.

    Expects either:
    - absolute BJD values (~2.4M+), or
    - already-offset BTJD values (~0-10k).
    """
    if epoch is None:
        return None
    value = float(epoch)
    if not math.isfinite(value):
        raise ValueError(f"Epoch must be finite, got {value!r}")
    if bjd_reference is not None and math.isfinite(bjd_reference):
        return value + (float(bjd_reference) - BJD_TO_BTJD_OFFSET)
    if mjd_reference is not None and math.isfinite(mjd_reference):
        return value + (float(mjd_reference) + _MJD_TO_JD_OFFSET - BJD_TO_BTJD_OFFSET)
    if value >= _ABSOLUTE_BJD_THRESHOLD:
        return value - BJD_TO_BTJD_OFFSET
    # Common offsets seen in catalogs:
    # - MJD (JD-2400000.5): ~50k-70k
    # - BJD-2450000: ~7k-20k for modern surveys
    if 50_000.0 <= value < 90_000.0:
        return value + (_MJD_TO_JD_OFFSET - BJD_TO_BTJD_OFFSET)
    if 7_000.0 <= value < 20_000.0:
        return value + (_BJD_MINUS_2450000_OFFSET - BJD_TO_BTJD_OFFSET)
    return value


def looks_like_absolute_bjd(epoch: float) -> bool:
    """Best-effort guard for mislabeled absolute-BJD values."""
    return math.isfinite(epoch) and float(epoch) >= _ABSOLUTE_BJD_THRESHOLD
