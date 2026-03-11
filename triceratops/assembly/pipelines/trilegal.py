"""TRILEGAL population assembly: provider query with optional cache."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from triceratops.assembly.errors import TRILEGALAcquisitionError

if TYPE_CHECKING:
    from triceratops.domain.entities import StellarField
    from triceratops.population.protocols import PopulationSynthesisProvider, TRILEGALResult


def assemble_trilegal(
    population_provider: PopulationSynthesisProvider,
    stellar_field: StellarField,
    trilegal_cache_path: str | None,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[TRILEGALResult, list[str]]:
    """Fetch TRILEGAL background population for the target star.

    Returns:
        (trilegal_result, warnings) tuple.

    Raises:
        TRILEGALAcquisitionError: If the population query fails.
    """
    warnings: list[str] = []
    cache = Path(trilegal_cache_path) if trilegal_cache_path else None
    target = stellar_field.target

    try:
        result = population_provider.query(
            ra_deg=target.ra_deg,
            dec_deg=target.dec_deg,
            target_tmag=target.tmag,
            cache_path=cache,
            status_callback=status_callback,
        )
    except Exception as exc:
        raise TRILEGALAcquisitionError(
            f"TRILEGAL query failed: {exc}"
        ) from exc

    return result, warnings
