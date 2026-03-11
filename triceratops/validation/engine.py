"""Stateless ValidationEngine: orchestrates scenario computation and aggregation.

Replaces triceratops.py:calc_probs() with a testable, injectable design.

FPP formula (locked -- matches triceratops.py:1635):
    FPP = 1 - P(TP) - P(PTP) - P(DTP)
STP is NOT in planet_scenarios(). Do not change.

NFPP formula:
    NFPP = sum(P(s) for s in {NTP, NEB, NEBx2P})
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from triceratops.config.config import Config
from triceratops.domain.entities import (
    ExternalLightCurve,
    LightCurve,
    Star,
    StellarField,
)
from triceratops.domain.molusc import MoluscData
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.population.protocols import TRILEGALResult
from triceratops.scenarios.base import Scenario
from triceratops.scenarios.nearby_scenarios import EmptyTrilegalPeerPopulationError
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry
from triceratops.validation.job import PreparedValidationInputs


@dataclass
class ScenarioExecutionContext:
    """All inputs needed to execute a single scenario.

    Replaces the fragile (scenario, lc, params, ...) tuple previously
    passed to the parallel worker function.
    """

    scenario: Scenario
    light_curve: LightCurve
    stellar_params: StellarParameters
    period_days: float | list[float] | tuple[float, float]
    config: Config
    external_lcs: list[ExternalLightCurve] = field(default_factory=list)
    # Scenario-specific kwargs (kept as dict for backward compat with **kwargs pattern)
    contrast_curve: ContrastCurve | None = None
    trilegal_population: TRILEGALResult | None = None
    host_magnitudes: dict = field(default_factory=dict)
    target_tmag: float | None = None
    target_id: int = 0
    external_lc_bands: tuple = ()
    filt: str | None = None
    molusc_data: MoluscData | None = None


@dataclass
class ScenarioExecutionOutcome:
    """Normalized result of one scenario worker execution."""

    results: tuple[ScenarioResult, ...]
    warnings: tuple[str, ...] = ()


def _worker_initializer() -> None:
    """Limit BLAS/OMP threads to 1 per worker process.

    Called once per worker at startup by ProcessPoolExecutor.  Without this,
    each worker inherits the parent's thread settings and all workers compete
    for the same CPU cores (oversubscription), which *reduces* throughput.
    With 1 BLAS thread per worker, scenario-level parallelism is clean.
    """
    import os
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _scenario_worker(
    ctx: ScenarioExecutionContext,
) -> ScenarioExecutionOutcome:
    """Top-level worker function for ProcessPoolExecutor.

    Must be module-level (not a closure or lambda) to be picklable under
    spawn-mode multiprocessing (default on macOS / Windows).

    Each worker process receives an independent OS-entropy RNG seed on spawn,
    so no explicit np.random.seed() call is needed for correctness.
    """
    kwargs: dict = {
        "contrast_curve": ctx.contrast_curve,
        "filt": ctx.filt,
        "molusc_data": ctx.molusc_data,
        "trilegal_population": ctx.trilegal_population,
        "host_magnitudes": ctx.host_magnitudes,
        "external_lc_bands": ctx.external_lc_bands,
        "target_tmag": ctx.target_tmag,
        "target_id": ctx.target_id,
    }
    try:
        result_or_tuple = ctx.scenario.compute(
            light_curve=ctx.light_curve,
            stellar_params=ctx.stellar_params,
            period_days=ctx.period_days,
            config=ctx.config,
            external_lcs=ctx.external_lcs if ctx.external_lcs else None,
            **kwargs,
        )
    except EmptyTrilegalPeerPopulationError as exc:
        warning = (
            f"{ctx.scenario.scenario_id.value}: {exc}. "
            "Returning lnZ=-inf for this nearby scenario."
        )
        results = [_empty_scenario_result(exc.scenario_id, ctx.target_id)]
        if exc.returns_twin:
            results.append(_empty_scenario_result(ScenarioID.NEBX2P, ctx.target_id))
        return ScenarioExecutionOutcome(results=tuple(results), warnings=(warning,))
    if isinstance(result_or_tuple, tuple):
        return ScenarioExecutionOutcome(results=result_or_tuple)
    return ScenarioExecutionOutcome(results=(result_or_tuple,))


def _empty_scenario_result(
    scenario_id: ScenarioID,
    target_id: int,
) -> ScenarioResult:
    zeros = np.zeros(1, dtype=float)
    return ScenarioResult(
        scenario_id=scenario_id,
        host_star_tic_id=target_id,
        ln_evidence=float("-inf"),
        host_mass_msun=zeros.copy(),
        host_radius_rsun=zeros.copy(),
        host_u1=zeros.copy(),
        host_u2=zeros.copy(),
        period_days=zeros.copy(),
        inclination_deg=zeros.copy(),
        impact_parameter=zeros.copy(),
        eccentricity=zeros.copy(),
        arg_periastron_deg=zeros.copy(),
        planet_radius_rearth=zeros.copy(),
        eb_mass_msun=zeros.copy(),
        eb_radius_rsun=zeros.copy(),
        flux_ratio_eb_tess=zeros.copy(),
        companion_mass_msun=zeros.copy(),
        companion_radius_rsun=zeros.copy(),
        flux_ratio_companion_tess=zeros.copy(),
    )


class ValidationEngine:
    """Stateless computation core for TRICERATOPS+ candidate validation.

    All external service calls are injected and mockable. The engine holds
    no mutable state -- safe for concurrent calls.
    """

    def __init__(
        self,
        registry: ScenarioRegistry | None = None,
        catalog_provider: object | None = None,
        population_provider: object | None = None,
    ) -> None:
        # DEPRECATED: population_provider is no longer used inside compute().
        # TRILEGAL population must now be fetched by the caller (ValidationWorkspace
        # or ValidationPreparer) and passed directly via the trilegal_population
        # parameter of compute() or compute_prepared().
        # population_provider is accepted here to avoid breaking existing call sites;
        # it will be removed in a future release.
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._catalog = catalog_provider
        # _population retained for compat but no longer called internally.
        self._population = population_provider

    def compute(
        self,
        light_curve: LightCurve,
        stellar_field: StellarField,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        scenario_ids: Sequence[ScenarioID] | None = None,
        external_lcs: list[ExternalLightCurve] | None = None,
        contrast_curve: ContrastCurve | None = None,
        molusc_data: MoluscData | None = None,
        trilegal_population: TRILEGALResult | None = None,
    ) -> ValidationResult:
        """Run all requested scenarios and aggregate into a ValidationResult.

        Args:
            light_curve: Phase-folded normalised light curve.
            stellar_field: Pre-assembled field with stellar parameters.
            period_days: Orbital period in days (scalar or [min, max] range).
            config: Runtime configuration.
            scenario_ids: Which scenarios to run. If None, runs the default
                eligible set from the registry, excluding nearby-host
                scenarios when no non-target star has positive transit depth.
            external_lcs: Ground-based follow-up light curves.
            contrast_curve: AO/speckle contrast curve for companion prior.
            molusc_data: Pre-loaded MOLUSC companion population data.
            trilegal_population: Pre-materialised TRILEGAL population.

        Returns:
            ValidationResult with FPP, NFPP, and per-scenario results.
        """
        if config.seed is not None:
            np.random.seed(config.seed)

        if scenario_ids is None:
            scenarios_to_run = self._registry.all_scenarios()
            has_nearby_candidate = any(
                star.transit_depth_required is not None
                and star.transit_depth_required > 0.0
                for star in stellar_field.neighbors
            )
            if not has_nearby_candidate:
                scenarios_to_run = [
                    s for s in scenarios_to_run
                    if s.scenario_id not in ScenarioID.nearby_scenarios()
                ]
        else:
            scenarios_to_run = [self._registry.get(sid) for sid in scenario_ids]

        # trilegal_population must be pre-materialised by the caller.
        # The engine no longer fetches from any provider (Phase 2 boundary).
        # Callers: pass trilegal_population directly, or rely on ValidationWorkspace /
        # ValidationPreparer to fetch it before calling compute().

        host_magnitudes = self._extract_host_magnitudes(stellar_field.target)
        target_flux_ratio = stellar_field.target.flux_ratio
        renormed_target_lc = self._renorm_light_curve_for_host(
            light_curve, target_flux_ratio,
        )

        stellar_params = stellar_field.target.stellar_params
        if stellar_params is None:
            from triceratops.validation.errors import PreparedInputIncompleteError
            raise PreparedInputIncompleteError(
                f"Target star (TIC {stellar_field.target_id}) has no stellar_params. "
                "Stellar parameters are required for all scenario computations. "
                "Check the catalog query result or set stellar_params manually."
            )

        filt = contrast_curve.band if contrast_curve is not None else None
        shared_kwargs: dict = {
            "target_id": stellar_field.target_id,
            "contrast_curve": contrast_curve,
            "filt": filt,
            "molusc_data": molusc_data,
            "trilegal_population": trilegal_population,
            "host_magnitudes": host_magnitudes,
            "external_lc_bands": tuple(
                ext_lc.band for ext_lc in (external_lcs or [])
            ),
            "target_tmag": stellar_field.target.tmag,
        }

        nearby_ids = ScenarioID.nearby_scenarios()
        tasks = [
            ScenarioExecutionContext(
                scenario=scenario,
                light_curve=(
                    light_curve if scenario.scenario_id in nearby_ids
                    else renormed_target_lc
                ),
                stellar_params=stellar_params,
                period_days=period_days,
                config=config,
                external_lcs=external_lcs or [],
                **shared_kwargs,
            )
            for scenario in scenarios_to_run
        ]

        all_results: list[ScenarioResult] = []
        all_warnings: list[str] = []
        if config.n_workers == 0:
            for task in tasks:
                outcome = _scenario_worker(task)
                all_results.extend(outcome.results)
                all_warnings.extend(outcome.warnings)
        else:
            n_workers = (
                min(len(tasks), os.cpu_count() or 1)
                if config.n_workers < 0
                else min(config.n_workers, len(tasks))
            )
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_initializer,
            ) as pool:
                futures = [pool.submit(_scenario_worker, task) for task in tasks]
                for future in futures:
                    outcome = future.result()
                    all_results.extend(outcome.results)
                    all_warnings.extend(outcome.warnings)

        return self._aggregate(
            all_results,
            stellar_field.target_id,
            warnings=all_warnings,
            rng_seed=config.seed,
        )

    def compute_prepared(self, prepared: PreparedValidationInputs) -> ValidationResult:
        """Provider-free compute entrypoint.

        Accepts a fully-materialised PreparedValidationInputs and delegates to the
        existing compute() logic.  No providers are called; all required data must
        already be present in the prepared payload.

        This is the correct entrypoint for remote execution (e.g. Modal workers)
        where no provider access is available.

        Args:
            prepared: Fully-materialised validation inputs.  Must have
                trilegal_population already populated if any trilegal-dependent
                scenarios are being run.

        Returns:
            ValidationResult with FPP, NFPP, and per-scenario results.
        """
        # Preflight: structural field invariants + scientific preconditions
        # (stellar_params, light curve shape, period_days, TRILEGAL presence).
        # Raises PreparedInputIncompleteError / ValidationInputError / ValueError
        # before any compute work begins.
        prepared.validate()

        # Guard: target_id must match the stellar field to prevent misattributed results
        # from corrupted or mis-assembled payloads in serialized/remote job flows.
        field_target_id = prepared.stellar_field.target_id
        if prepared.target_id != field_target_id:
            raise ValueError(
                f"PreparedValidationInputs.target_id ({prepared.target_id}) does not match "
                f"stellar_field.target_id ({field_target_id}). "
                "Payload may be corrupted or incorrectly assembled."
            )

        # Guard: scenario_ids must all be registered in this engine's registry.
        # PreparedValidationInputs supports direct construction and deserialization,
        # so we cannot rely on ValidationPreparer.prepare() having validated them.
        # compute() resolves IDs with self._registry.get() which raises KeyError;
        # we raise ValueError here instead with the unknown IDs named.
        if prepared.scenario_ids is not None:
            unknown = [sid for sid in prepared.scenario_ids if sid not in self._registry]
            if unknown:
                raise ValueError(
                    f"PreparedValidationInputs.scenario_ids contains IDs not registered "
                    f"in this engine's registry: {unknown}. "
                    "Payload may be corrupted or assembled for a different registry."
                )

        # TRILEGAL presence check using the engine's actual registry.
        # prepared.validate() defers this when scenario_ids=None because
        # it cannot see the engine's registry.  Now that we have self._registry,
        # we check the full scenario set that will actually run.
        if prepared.trilegal_population is None:
            from triceratops.domain.scenario_id import ScenarioID
            from triceratops.validation.errors import PreparedInputIncompleteError
            trilegal_ids = set(ScenarioID.trilegal_scenarios())
            if prepared.scenario_ids is None:
                active = self._registry.all_scenarios()
            else:
                active = [self._registry.get(sid) for sid in prepared.scenario_ids
                          if sid in self._registry]
            missing = [s.scenario_id for s in active if s.scenario_id in trilegal_ids]
            if missing:
                raise PreparedInputIncompleteError(
                    f"trilegal_population is required for scenarios {missing} "
                    "but was not provided. "
                    "Pass a population_provider to ValidationPreparer or ValidationWorkspace, "
                    "or exclude TRILEGAL-dependent scenarios via scenario_ids."
                )

        return self.compute(
            light_curve=prepared.light_curve,
            stellar_field=prepared.stellar_field,
            period_days=prepared.period_days,
            config=prepared.config,
            scenario_ids=prepared.scenario_ids,
            external_lcs=prepared.external_lcs,
            contrast_curve=prepared.contrast_curve,
            trilegal_population=prepared.trilegal_population,
            molusc_data=prepared.molusc_data,
        )

    @staticmethod
    def _extract_host_magnitudes(target_star: Star) -> dict[str, float | None]:
        """Extract magnitude dict from the target Star object."""
        return {
            "tmag": target_star.tmag,
            "jmag": target_star.jmag,
            "hmag": target_star.hmag,
            "kmag": target_star.kmag,
            "bmag": target_star.bmag,
            "vmag": target_star.vmag,
            "gmag": target_star.gmag,
            "rmag": target_star.rmag,
            "imag": target_star.imag,
            "zmag": target_star.zmag,
        }

    @staticmethod
    def _renorm_light_curve_for_host(
        light_curve: LightCurve,
        host_flux_ratio: float | None,
    ) -> LightCurve:
        """Match the original calc_probs() host-by-host TESS renormalization."""
        if host_flux_ratio is None:
            return light_curve
        flux_ratio = float(host_flux_ratio)
        if flux_ratio <= 0.0 or flux_ratio > 1.0:
            return light_curve
        return light_curve.with_renorm(flux_ratio)

    @staticmethod
    def _aggregate(
        results: list[ScenarioResult],
        target_id: int,
        warnings: list[str] | None = None,
        rng_seed: int | None = None,
    ) -> ValidationResult:
        return _aggregate(results, target_id, warnings=warnings, rng_seed=rng_seed)


def _aggregate(
    results: list[ScenarioResult],
    target_id: int,
    warnings: list[str] | None = None,
    rng_seed: int | None = None,
) -> ValidationResult:
    """Compute relative probabilities, FPP, and NFPP.

    FPP = 1 - P(TP) - P(PTP) - P(DTP)
    NFPP = sum of NTP + NEB + NEBx2P probabilities.
    """
    if not results:
        return ValidationResult(
            target_id=target_id,
            false_positive_probability=1.0,
            nearby_false_positive_probability=0.0,
            scenario_results=[],
            warnings=[] if warnings is None else list(warnings),
            rng_seed=rng_seed,
        )

    lnZ_vals = np.array([r.ln_evidence for r in results])
    finite_mask = np.isfinite(lnZ_vals)

    if not np.any(finite_mask):
        for r in results:
            r.relative_probability = 0.0
    else:
        lnZ_max = float(np.max(lnZ_vals[finite_mask]))
        Z_vals = np.exp(np.where(finite_mask, lnZ_vals - lnZ_max, -np.inf))
        Z_total = float(np.sum(Z_vals))

        if Z_total <= 0:
            for r in results:
                r.relative_probability = 0.0
        else:
            for r, z in zip(results, Z_vals):
                r.relative_probability = float(z / Z_total)

    planet_ids = ScenarioID.planet_scenarios()
    planet_prob = sum(
        r.relative_probability for r in results if r.scenario_id in planet_ids
    )
    fpp = float(np.clip(1.0 - planet_prob, 0.0, 1.0))

    nearby_ids = ScenarioID.nearby_scenarios()
    nfpp = float(sum(
        r.relative_probability for r in results if r.scenario_id in nearby_ids
    ))

    return ValidationResult(
        target_id=target_id,
        false_positive_probability=fpp,
        nearby_false_positive_probability=nfpp,
        scenario_results=results,
        warnings=[] if warnings is None else list(warnings),
        rng_seed=rng_seed,
    )
