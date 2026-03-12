"""Light-curve configuration types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LightCurveConfig:
    """Configuration for light-curve fetching and preparation.

    All fields are primitives. Validated in __post_init__.
    No ephemeris fields — those belong on Ephemeris.
    """

    # --- basic parameters ---
    sectors: tuple[int, ...] | Literal["all", "auto"] = "auto"
    source: Literal["pdcsap", "sap", "tpf"] = "pdcsap"
    cadence: Literal["20sec", "2min", "10min", "30min", "auto"] = "auto"
    quality_mask: Literal["none", "default", "hard"] = "default"
    detrend_method: Literal["flatten", "none"] = "none"
    sigma_clip: float | None = 5.0

    # --- advanced parameters ---
    flatten_window_length: int = 401
    flatten_polyorder: int = 3
    phase_window_factor: float = 2.0
    cadence_days_override: float | None = None
    supersampling_rate: int = 20

    def __post_init__(self) -> None:
        if self.flatten_window_length < 3 or self.flatten_window_length % 2 == 0:
            raise ValueError(
                f"flatten_window_length must be an odd integer >= 3, "
                f"got {self.flatten_window_length}"
            )
        if self.flatten_polyorder < 1 or self.flatten_polyorder > 5:
            raise ValueError(
                f"flatten_polyorder must be between 1 and 5, got {self.flatten_polyorder}"
            )
        if self.phase_window_factor < 1.0:
            raise ValueError(
                f"phase_window_factor must be >= 1.0, got {self.phase_window_factor}"
            )
        if self.supersampling_rate < 1:
            raise ValueError(
                f"supersampling_rate must be >= 1, got {self.supersampling_rate}"
            )
        if self.sigma_clip is not None and self.sigma_clip <= 0:
            raise ValueError(
                f"sigma_clip must be positive or None, got {self.sigma_clip}"
            )
        if self.cadence_days_override is not None and self.cadence_days_override <= 0:
            raise ValueError(
                f"cadence_days_override must be positive or None, "
                f"got {self.cadence_days_override}"
            )
