"""TRICERATOPS+ rewrite: astrophysical false-positive probability calculator."""
from __future__ import annotations

__version__ = "0.2.0.dev0"

from triceratops.config.config import Config, MissionConfig
from triceratops.domain.scenario_id import ScenarioID
from triceratops.plotting import plot_field, plot_fits
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
    "ValidationWorkspace",
    "ValidationEngine",
    "ValidationPreparer",
    "PreparedValidationInputs",
    "PreparedValidationMetadata",
    "Config",
    "MissionConfig",
    "ScenarioID",
    "plot_field",
    "plot_fits",
    "probs_dataframe",
]
