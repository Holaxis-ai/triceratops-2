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

import numpy as np

from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve
from triceratops.scenarios.base import Scenario
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry


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
    scenario: Scenario,
    light_curve: LightCurve,
    stellar_params: object,
    period_days: float | list[float],
    config: Config,
    external_lcs: list[ExternalLightCurve] | None,
    kwargs: dict,
) -> ScenarioResult | tuple[ScenarioResult, ScenarioResult]:
    """Top-level worker function for ProcessPoolExecutor.

    Must be module-level (not a closure or lambda) to be picklable under
    spawn-mode multiprocessing (default on macOS / Windows).

    Each worker process receives an independent OS-entropy RNG seed on spawn,
    so no explicit np.random.seed() call is needed for correctness.
    """
    return scenario.compute(
        light_curve=light_curve,
        stellar_params=stellar_params,
        period_days=period_days,
        config=config,
        external_lcs=external_lcs,
        **kwargs,
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
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._catalog = catalog_provider
        self._population = population_provider

    def compute(
        self,
        light_curve: LightCurve,
        stellar_field: StellarField,
        period_days: float | list[float],
        config: Config,
        scenario_ids: Sequence[ScenarioID] | None = None,
        external_lcs: list[ExternalLightCurve] | None = None,
        contrast_curve: ContrastCurve | None = None,
        trilegal_cache_path: str | None = None,
        molusc_file: str | None = None,
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
            trilegal_cache_path: Path to cached TRILEGAL CSV.

        Returns:
            ValidationResult with FPP, NFPP, and per-scenario results.
        """
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

        # Fetch TRILEGAL population once if any scenario needs it
        trilegal_population = None
        needs_trilegal = any(
            s.scenario_id in ScenarioID.trilegal_scenarios()
            for s in scenarios_to_run
        )
        if needs_trilegal and self._population is not None:
            target = stellar_field.target
            from pathlib import Path
            cache = Path(trilegal_cache_path) if trilegal_cache_path else None
            trilegal_population = self._population.query(  # type: ignore[union-attr]
                ra_deg=target.ra_deg,
                dec_deg=target.dec_deg,
                target_tmag=target.tmag,
                cache_path=cache,
            )

        host_magnitudes = self._extract_host_magnitudes(stellar_field.target)
        target_flux_ratio = stellar_field.target.flux_ratio
        renormed_target_lc = self._renorm_light_curve_for_host(
            light_curve, target_flux_ratio,
        )

        stellar_params = stellar_field.target.stellar_params
        if stellar_params is None:
            return self._aggregate([], stellar_field.target_id)

        filt = contrast_curve.band if contrast_curve is not None else None
        shared_kwargs: dict = {
            "contrast_curve": contrast_curve,
            "filt": filt,
            "molusc_file": molusc_file,
            "trilegal_population": trilegal_population,
            "host_magnitudes": host_magnitudes,
            "external_lc_bands": tuple(
                ext_lc.band for ext_lc in (external_lcs or [])
            ),
            "target_tmag": stellar_field.target.tmag,
        }

        nearby_ids = ScenarioID.nearby_scenarios()
        tasks = [
            (
                scenario,
                light_curve if scenario.scenario_id in nearby_ids else renormed_target_lc,
                stellar_params,
                period_days,
                config,
                external_lcs,
                shared_kwargs,
            )
            for scenario in scenarios_to_run
        ]

        all_results: list[ScenarioResult] = []
        if config.n_workers == 0:
            for task in tasks:
                result_or_tuple = _scenario_worker(*task)
                if isinstance(result_or_tuple, tuple):
                    all_results.extend(result_or_tuple)
                else:
                    all_results.append(result_or_tuple)
        else:
            n_workers = (
                min(len(tasks), os.cpu_count() or 1)
                if config.n_workers < 0
                else min(config.n_workers, len(tasks))
            )
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_initializer) as pool:
                futures = [pool.submit(_scenario_worker, *task) for task in tasks]
                for future in futures:
                    result_or_tuple = future.result()
                    if isinstance(result_or_tuple, tuple):
                        all_results.extend(result_or_tuple)
                    else:
                        all_results.append(result_or_tuple)

        return self._aggregate(all_results, stellar_field.target_id)

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
    ) -> ValidationResult:
        return _aggregate(results, target_id)


def _aggregate(
    results: list[ScenarioResult],
    target_id: int,
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
    )
