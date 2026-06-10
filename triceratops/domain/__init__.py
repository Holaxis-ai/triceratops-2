"""Domain model: pure dataclasses, no I/O, no external dependencies."""
from triceratops.lightcurve.ephemeris import Ephemeris, EphemerisResolver, ResolvedTarget

from .entities import (
    ExternalLightCurve,
    LightCurve,
    Star,
    StellarField,
)
from .result import ScenarioResult, ValidationResult
from .scenario_id import ScenarioID
from .value_objects import (
    ContrastCurve,
    ContrastCurveInput,
    ContrastCurveSet,
    LimbDarkeningCoeffs,
    OrbitalParameters,
    PeriodSpec,
    StellarParameters,
)

__all__: list[str] = [
    "ContrastCurve",
    "ContrastCurveInput",
    "ContrastCurveSet",
    "Ephemeris",
    "EphemerisResolver",
    "ExternalLightCurve",
    "LightCurve",
    "LimbDarkeningCoeffs",
    "OrbitalParameters",
    "PeriodSpec",
    "ResolvedTarget",
    "ScenarioID",
    "ScenarioResult",
    "Star",
    "StellarField",
    "StellarParameters",
    "ValidationResult",
]
