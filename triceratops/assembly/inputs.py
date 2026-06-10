"""Assembled-inputs value objects for the data-assembly pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triceratops.domain.entities import ExternalLightCurve, LightCurve, StellarField
    from triceratops.domain.molusc import MoluscData
    from triceratops.domain.value_objects import ContrastCurveInput
    from triceratops.lightcurve.ephemeris import ResolvedTarget
    from triceratops.population.protocols import TRILEGALResult


@dataclass(frozen=True)
class AssemblyMetadata:
    """Provenance metadata produced during assembly."""

    source_labels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    created_at_utc: str | None = None
    assembler_version: str | None = None
    per_input_source: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AssembledInputs:
    """Immutable bundle of all data needed by the validation engine."""

    resolved_target: ResolvedTarget
    stellar_field: StellarField
    light_curve: LightCurve | None = None
    contrast_curve: ContrastCurveInput = None
    molusc_data: MoluscData | None = None
    trilegal_population: TRILEGALResult | None = None
    external_lcs: list[ExternalLightCurve] | None = None
    metadata: AssemblyMetadata = field(default_factory=AssemblyMetadata)

    def __post_init__(self) -> None:
        from triceratops.domain.entities import StellarField as _StellarField
        from triceratops.lightcurve.ephemeris import ResolvedTarget as _ResolvedTarget

        if not isinstance(self.resolved_target, _ResolvedTarget):
            raise TypeError(
                "resolved_target must be a ResolvedTarget, got "
                f"{type(self.resolved_target).__name__}"
            )
        if not isinstance(self.stellar_field, _StellarField):
            raise TypeError(
                f"stellar_field must be a StellarField, got {type(self.stellar_field).__name__}"
            )
