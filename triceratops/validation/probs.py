"""Public helpers for tutorial-style scenario probability tables."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from triceratops.domain import ValidationResult


def probs_dataframe(validation_result: ValidationResult) -> pd.DataFrame:
    """Build a tutorial-style probabilities table from a validation result."""
    rows = [
        {
            "ID": scenario.host_star_tic_id,
            "scenario": scenario.scenario_id.name,
            "M_s": _median_or_zero(scenario.host_mass_msun),
            "R_s": _median_or_zero(scenario.host_radius_rsun),
            "P_orb": _median_or_zero(scenario.period_days),
            "inc": _median_or_zero(scenario.inclination_deg),
            "b": _median_or_zero(scenario.impact_parameter),
            "ecc": _median_or_zero(scenario.eccentricity),
            "w": _median_or_zero(scenario.arg_periastron_deg),
            "R_p": _median_or_zero(scenario.planet_radius_rearth),
            "M_EB": _median_or_zero(scenario.eb_mass_msun),
            "R_EB": _median_or_zero(scenario.eb_radius_rsun),
            "M_comp": _median_or_zero(scenario.companion_mass_msun),
            "R_comp": _median_or_zero(scenario.companion_radius_rsun),
            "flux_ratio_comp_T": _median_or_zero(scenario.flux_ratio_companion_tess),
            "prob": scenario.relative_probability,
            "lnZ": scenario.ln_evidence,
        }
        for scenario in validation_result.scenario_results
    ]
    return pd.DataFrame(rows)


def _median_or_zero(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        value = float(np.nanmedian(arr))
    if math.isnan(value):
        return 0.0
    return value


__all__ = ["probs_dataframe"]
