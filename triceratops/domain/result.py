"""Result types: outputs of scenario computation and validation runs."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenario_id import ScenarioID


@dataclass
class ScenarioResult:
    """Best-fit parameters and Bayesian evidence for one astrophysical scenario.

    Replaces the raw dict returned by each lnZ_* function in marginal_likelihoods.py.
    Array fields have shape (n_best_samples,) -- the top-1000 draws by likelihood.
    Scalar fields set to zeros arrays for inapplicable scenario types (e.g., planet_radius
    is zero-filled for EB scenarios).
    """

    scenario_id: ScenarioID
    host_star_tic_id: int
    ln_evidence: float                     # lnZ for this scenario

    # Best-fit arrays (shape: n_best_samples)
    host_mass_msun: np.ndarray
    host_radius_rsun: np.ndarray
    host_u1: np.ndarray
    host_u2: np.ndarray
    period_days: np.ndarray
    inclination_deg: np.ndarray
    impact_parameter: np.ndarray
    eccentricity: np.ndarray
    arg_periastron_deg: np.ndarray

    # TP fields (zeros for EB scenarios)
    planet_radius_rearth: np.ndarray

    # EB fields (zeros for TP scenarios)
    eb_mass_msun: np.ndarray
    eb_radius_rsun: np.ndarray
    flux_ratio_eb_tess: np.ndarray

    # Companion / background star fields (zeros when not applicable)
    companion_mass_msun: np.ndarray
    companion_radius_rsun: np.ndarray
    flux_ratio_companion_tess: np.ndarray

    # Per-external-LC fields (one entry per external LC; empty list when none provided)
    external_lc_u1: list[np.ndarray] = field(default_factory=list)
    external_lc_u2: list[np.ndarray] = field(default_factory=list)
    external_lc_flux_ratio_eb: list[np.ndarray] = field(default_factory=list)
    external_lc_flux_ratio_comp: list[np.ndarray] = field(default_factory=list)

    # Set by ValidationEngine._aggregate() -- not filled by individual scenarios
    relative_probability: float = 0.0


@dataclass
class ValidationResult:
    """Final output of a complete TRICERATOPS+ candidate validation run."""

    target_id: int
    false_positive_probability: float       # FPP = 1 - P(TP) - P(PTP) - P(DTP)
    nearby_false_positive_probability: float  # NFPP = sum of N-scenario probabilities
    scenario_results: list[ScenarioResult]  # all scenarios run, in order
    warnings: list[str] = field(default_factory=list)
    rng_seed: int | None = None

    @property
    def fpp(self) -> float:
        return self.false_positive_probability

    @property
    def nfpp(self) -> float:
        return self.nearby_false_positive_probability

    def get_scenario(self, scenario_id: ScenarioID) -> ScenarioResult | None:
        """Return the ScenarioResult for the given ID, or None if not found."""
        for r in self.scenario_results:
            if r.scenario_id == scenario_id:
                return r
        return None
