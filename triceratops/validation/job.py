"""PreparedValidationInputs and PreparedValidationMetadata — the two-phase compute boundary.

This module defines the job payload that separates the *preparation* phase (provider-backed IO,
catalog queries, population synthesis) from the *compute* phase (pure scenario orchestration).

Two-phase design
----------------
Phase 1 — Prepare (ValidationPreparer / ValidationWorkspace):
    - Query star catalog; materialise StellarField with flux ratios and transit depths.
    - Fetch TRILEGAL background population (if needed).
    - Load contrast curve and external light curves from disk.
    - Emit one fully-populated PreparedValidationInputs.

Phase 2 — Compute (ValidationEngine.compute_prepared):
    - Accept PreparedValidationInputs.
    - Execute scenarios with NO provider access, NO network calls, NO filesystem assumptions.
    - Return ValidationResult.

Remote compute boundary guarantee
----------------------------------
Any function that receives only a ``PreparedValidationInputs`` and calls
``ValidationEngine.compute_prepared()`` must not instantiate providers, query MAST,
query TRILEGAL, or depend on the current working directory.

See: working_docs/iteration/priority-3_pure-compute-boundary.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from collections.abc import Sequence

if TYPE_CHECKING:
    from triceratops.config.config import Config
    from triceratops.domain.entities import ExternalLightCurve, LightCurve, StellarField
    from triceratops.domain.scenario_id import ScenarioID
    from triceratops.domain.value_objects import ContrastCurve
    from triceratops.population.protocols import TRILEGALResult


@dataclass
class PreparedValidationInputs:
    """Fully-materialised inputs for a provider-free validation compute.

    All fields required for ``ValidationEngine.compute_prepared()`` are present
    on this object.  No network access, no filesystem IO, and no provider
    instantiation should happen during or after construction.

    Fields
    ------
    target_id:
        TIC (or KIC/EPIC) identifier for the validation target.
    stellar_field:
        Assembled field including target and neighbours, with flux_ratio and
        transit_depth_required already computed on each Star.
    light_curve:
        Phase-folded, normalised photometric time series.
    config:
        Runtime configuration (n_mc_samples, lnz_const, …).
    period_days:
        Orbital period in days.  May be a scalar or [min, max] range.
    trilegal_population:
        Materialised TRILEGAL background population.  Must be provided for
        scenarios that require it (BTP, BEB, BEBx2P).  Pass None if those
        scenarios are not being run.
    external_lcs:
        Ground-based follow-up light curves.  None if not available.
    contrast_curve:
        AO/speckle contrast curve.  None if not available.
    molusc_file:
        Local filesystem path to a MOLUSC output file.
        NOTE: This is a bare path string — not yet materialised content.
        The compute boundary is not fully clean for this field until Phase 4,
        when remote execution requires the content to be embedded rather than
        referenced by path.  Deferred to Phase 4.
    scenario_ids:
        The scenario subset that was prepared for.  ``None`` means the full
        default registry.  ``compute_prepared()`` passes this directly to
        ``compute(scenario_ids=...)``, keeping the prepare/compute contract
        consistent: the engine runs exactly the scenarios that were prepared.
    """

    target_id: int
    stellar_field: StellarField
    light_curve: LightCurve
    config: Config
    period_days: float | list[float]
    trilegal_population: TRILEGALResult | None = None
    external_lcs: list[ExternalLightCurve] | None = None
    contrast_curve: ContrastCurve | None = None
    molusc_file: str | None = None  # local path — not yet materialised; deferred to Phase 4
    scenario_ids: Sequence[ScenarioID] | None = None  # None → run full default registry


@dataclass
class PreparedValidationMetadata:
    """Optional provenance information attached to a PreparedValidationInputs.

    All fields are optional.  Consumers must not require any of these fields
    for correctness — they are for auditing, caching, and debugging only.

    Fields
    ------
    prep_timestamp:
        UTC datetime when the inputs were prepared.
    source:
        Human-readable label for where the inputs came from (e.g. "MAST/local").
    trilegal_cache_origin:
        Path or URL of the TRILEGAL cache file that was used, if any.
    warnings:
        List of non-fatal messages raised during preparation (e.g. missing
        magnitudes, fallback values used).
    """

    prep_timestamp: datetime | None = None
    source: str | None = None
    trilegal_cache_origin: str | None = None
    warnings: list[str] = field(default_factory=list)
