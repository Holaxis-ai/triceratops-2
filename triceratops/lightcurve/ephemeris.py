"""Ephemeris and resolved-target types for light-curve preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Ephemeris:
    """Transit ephemeris parameters.

    Attributes:
        period_days: Orbital period in days.
        t0_btjd: Transit midpoint in BTJD (BJD - 2,457,000). Must be < 10,000.
        duration_hours: Transit duration in hours, or None if unknown.
        source: Origin of this ephemeris ("manual", "exofop", etc.).
        warnings: Non-fatal issues from resolution.
    """

    period_days: float
    t0_btjd: float
    duration_hours: float | None = None
    source: str = "manual"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTarget:
    """A resolved astronomical target: identity plus optional ephemeris.

    This is the output of EphemerisResolver.resolve() and the natural
    input to LightkurveRawSource and the artifact layer. It prevents
    TIC ID and ephemeris from being passed separately and getting out of sync.
    Ephemeris may be None for catalog-only or partial-assembly flows; LC prep
    entrypoints require it to be present.
    """

    target_ref: str
    tic_id: int
    ephemeris: Ephemeris | None = None
    source: str = "unknown"


@runtime_checkable
class EphemerisResolver(Protocol):
    """Protocol for resolving a target string to a ResolvedTarget."""

    def resolve(self, target: str) -> ResolvedTarget: ...
