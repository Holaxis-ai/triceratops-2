"""TRICERATOPS+ rewrite: astrophysical false-positive probability calculator."""
from __future__ import annotations

__version__ = "0.2.0.dev0"

from triceratops.config.config import Config, MissionConfig
from triceratops.domain.scenario_id import ScenarioID
from triceratops.validation import (
    ApertureProvenance,
    ApertureConfig,
    ArtifactCapabilities,
    AutoFppComputeConfig,
    AutoFppPrepareConfig,
    FilesystemPreparedArtifactStore,
    FppRunConfig,
    FppRunResult,
    PreparedArtifactStore,
    PreparedAutoFppArtifact,
    StoredArtifactRef,
    ValidationEngine,
    ValidationWorkspace,
    compute_auto_fpp,
    prepare_auto_fpp,
    run_tess_fpp,
)

__all__ = [
    "__version__",
    "ValidationWorkspace",
    "ValidationEngine",
    "PreparedAutoFppArtifact",
    "ArtifactCapabilities",
    "ApertureProvenance",
    "PreparedArtifactStore",
    "FilesystemPreparedArtifactStore",
    "StoredArtifactRef",
    "ApertureConfig",
    "AutoFppPrepareConfig",
    "AutoFppComputeConfig",
    "FppRunConfig",
    "FppRunResult",
    "prepare_auto_fpp",
    "compute_auto_fpp",
    "run_tess_fpp",
    "Config",
    "MissionConfig",
    "ScenarioID",
]
