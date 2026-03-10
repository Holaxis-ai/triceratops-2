"""Scenario Protocol and BaseScenario abstract class.

The 14 lnZ_* functions in marginal_likelihoods.py share a 9-phase algorithm
skeleton. BaseScenario implements phases 1, 2, 4, 8 identically for all 14
functions. Phases 3, 5, 6, 7, 9 are abstract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from triceratops.config.config import CONST, Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.scenarios.kernels import (
    compute_lnZ,
    pack_best_indices,
    resolve_period,
)


@runtime_checkable
class Scenario(Protocol):
    """Structural protocol: any object with these members is a valid Scenario."""

    @property
    def scenario_id(self) -> ScenarioID: ...

    @property
    def is_eb(self) -> bool: ...

    @property
    def returns_twin(self) -> bool: ...

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> ScenarioResult | tuple[ScenarioResult, ScenarioResult]: ...


class BaseScenario(ABC):
    """Implements the 9-phase computation skeleton shared by all 14 scenarios.

    PHASE OVERVIEW:
    ---------------
    Phase 1: resolve_period -- identical in all 14 functions (kernel)
    Phase 2: compute scalar constants (lnsigma, logg) -- identical in all
    Phase 3: _get_host_ldc() -- ABSTRACT
    Phase 4: _resolve_external_lc_ldcs() -- shared, overridable
    Phase 5: _sample_priors() -- ABSTRACT
    Phase 6: _compute_orbital_geometry() -- ABSTRACT
    Phase 7: _evaluate_lnL() -- ABSTRACT
    Phase 8: compute_lnZ (kernel) -- identical in all 14 functions
    Phase 9: _pack_result() -- ABSTRACT
    """

    def __init__(self, ldc_catalog: object) -> None:
        """
        Args:
            ldc_catalog: Any object with .get_coefficients() -- LimbDarkeningCatalog
                         or FixedLDCCatalog for testing.
        """
        self._ldc = ldc_catalog

    # -- Identity (abstract) --

    @property
    @abstractmethod
    def scenario_id(self) -> ScenarioID:
        """Return the ScenarioID enum member for this scenario."""
        ...

    @property
    @abstractmethod
    def is_eb(self) -> bool:
        """True for all EB scenarios; False for TP scenarios."""
        ...

    @property
    def returns_twin(self) -> bool:
        """True for EB scenarios that return a (result, result_twin) tuple."""
        return self.is_eb

    # -- Main entry point --

    def compute(
        self,
        light_curve: LightCurve,
        stellar_params: StellarParameters,
        period_days: float | list[float] | tuple[float, float],
        config: Config,
        external_lcs: list[ExternalLightCurve] | None = None,
        **kwargs: object,
    ) -> ScenarioResult | tuple[ScenarioResult, ScenarioResult]:
        """Run the full 9-phase Monte Carlo marginal likelihood computation.

        For TP scenarios: returns a single ScenarioResult.
        For EB scenarios: returns (result, result_twin).
        """
        N = config.n_mc_samples

        # Phase 1: Resolve period to array of N values
        P_orb = resolve_period(period_days, N)

        # Phase 2: Scalar constants
        lnsigma = np.log(light_curve.sigma)

        # Phase 3: Resolve LDC for eclipsed star
        ldc = self._get_host_ldc(stellar_params, config.mission, P_orb, kwargs)

        # Phase 4: Resolve external LC LDCs
        resolved_ext_lcs = (
            self._resolve_external_lc_ldcs(external_lcs, stellar_params)
            if external_lcs else []
        )

        # Phase 5: Draw samples from prior distributions
        samples = self._sample_priors(N, stellar_params, P_orb, config, **kwargs)

        # Phase 6: Compute orbital geometry
        geometry = self._compute_orbital_geometry(
            samples, P_orb, stellar_params, config, **kwargs
        )

        # Phase 7: Evaluate log-likelihoods over all transiting samples
        lnL, lnL_twin = self._evaluate_lnL(
            light_curve, lnsigma, samples, geometry, ldc,
            resolved_ext_lcs, config
        )

        # Phase 8: Compute marginal likelihoods
        lnZ = compute_lnZ(lnL, config.lnz_const)
        lnZ_twin = (
            compute_lnZ(lnL_twin, config.lnz_const)
            if (lnL_twin is not None and self.returns_twin)
            else None
        )

        # Phase 9: Pack results from top n_best draws
        n_best = config.n_best_samples
        idx = pack_best_indices(lnL, n_best)
        result = self._pack_result(
            samples, geometry, ldc, lnZ, idx, stellar_params,
            resolved_ext_lcs, twin=False
        )

        if self.returns_twin and lnZ_twin is not None and lnL_twin is not None:
            idx_twin = pack_best_indices(lnL_twin, n_best)
            result_twin = self._pack_result(
                samples, geometry, ldc, lnZ_twin, idx_twin, stellar_params,
                resolved_ext_lcs, twin=True
            )
            return result, result_twin

        return result

    # -- Phase 3: LDC resolution (abstract) --

    @staticmethod
    def _stellar_logg_from_mass_radius(
        stellar_params: StellarParameters,
    ) -> float:
        """Recompute logg from M and R, matching the original code paths."""
        return float(np.log10(
            CONST.G * (stellar_params.mass_msun * CONST.Msun)
            / (stellar_params.radius_rsun * CONST.Rsun) ** 2
        ))

    @abstractmethod
    def _get_host_ldc(
        self,
        stellar_params: StellarParameters,
        mission: str,
        P_orb: np.ndarray,
        kwargs: dict,
    ) -> LimbDarkeningCoeffs:
        """Return LDC for the eclipsed star."""
        ...

    # -- Phase 4: External LC LDC resolution (shared, overridable) --

    def _resolve_external_lc_ldcs(
        self,
        external_lcs: list[ExternalLightCurve],
        stellar_params: StellarParameters,
    ) -> list[ExternalLightCurve]:
        """Fill LDC on each ExternalLightCurve using the catalog."""
        resolved = []
        for ext_lc in external_lcs:
            logg = self._stellar_logg_from_mass_radius(stellar_params)
            ldc = self._ldc.get_coefficients(  # type: ignore[union-attr]
                ext_lc.band,
                stellar_params.metallicity_dex,
                stellar_params.teff_k,
                logg,
            )
            resolved.append(ExternalLightCurve(
                light_curve=ext_lc.light_curve,
                band=ext_lc.band,
                ldc=ldc,
            ))
        return resolved

    # -- Phase 5: Prior sampling (abstract) --

    @abstractmethod
    def _sample_priors(
        self,
        n: int,
        stellar_params: StellarParameters,
        P_orb: np.ndarray,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Return a dict of named sample arrays for N Monte Carlo draws."""
        ...

    # -- Phase 6: Orbital geometry (abstract) --

    @abstractmethod
    def _compute_orbital_geometry(
        self,
        samples: dict[str, np.ndarray],
        P_orb: np.ndarray,
        stellar_params: StellarParameters,
        config: Config,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Return orbital geometry arrays for all N samples."""
        ...

    # -- Phase 7: Log-likelihood evaluation (abstract) --

    @abstractmethod
    def _evaluate_lnL(
        self,
        light_curve: LightCurve,
        lnsigma: float,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        external_lcs: list[ExternalLightCurve],
        config: Config,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate log-likelihoods for all N samples.

        Returns:
            (lnL, lnL_twin): lnL shape (N,). lnL_twin is None for TP scenarios.
        """
        ...

    # -- Phase 9: Result packing (abstract) --

    @abstractmethod
    def _pack_result(
        self,
        samples: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray],
        ldc: LimbDarkeningCoeffs,
        lnZ: float,
        idx: np.ndarray,
        stellar_params: StellarParameters,
        external_lcs: list[ExternalLightCurve],
        twin: bool = False,
    ) -> ScenarioResult:
        """Build a ScenarioResult from the top-n_best draws."""
        ...

    # -- Shared helper --

    @staticmethod
    def _sample_period(
        period_spec: float | list[float] | tuple[float, float],
        n: int,
    ) -> np.ndarray:
        """Delegate to scenarios.kernels.resolve_period()."""
        return resolve_period(period_spec, n)
