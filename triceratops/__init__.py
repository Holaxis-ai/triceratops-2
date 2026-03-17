"""TRICERATOPS+ rewrite: astrophysical false-positive probability calculator."""
from __future__ import annotations

__version__ = "0.3.0.dev0"

from triceratops.config.config import Config, MissionConfig
from triceratops.domain import (
    ContrastCurve,
    Ephemeris,
    EphemerisResolver,
    ExternalLightCurve,
    LightCurve,
    PeriodSpec,
    ResolvedTarget,
    ScenarioResult,
    Star,
    StellarField,
    StellarParameters,
    ValidationResult,
)
from triceratops.domain.scenario_id import ScenarioID
from triceratops.io import load_contrast_curve
from triceratops.plotting import plot_field, plot_fits, plot_fits_joint, plot_fits_palomar
from triceratops.validation import (
    PreparedValidationInputs,
    PreparedValidationMetadata,
    ValidationEngine,
    ValidationPreparer,
    ValidationWorkspace,
    probs_dataframe,
)

__all__ = [
    "__version__",
    "Config",
    "ContrastCurve",
    "Ephemeris",
    "EphemerisResolver",
    "ExternalLightCurve",
    "LightCurve",
    "MissionConfig",
    "PeriodSpec",
    "PreparedValidationInputs",
    "PreparedValidationMetadata",
    "ResolvedTarget",
    "ScenarioID",
    "ScenarioResult",
    "Star",
    "StellarField",
    "StellarParameters",
    "ValidationEngine",
    "ValidationPreparer",
    "ValidationResult",
    "ValidationWorkspace",
    "load_contrast_curve",
    "plot_field",
    "plot_fits",
    "plot_fits_joint",
    "plot_fits_palomar",
    "probs_dataframe",
]
