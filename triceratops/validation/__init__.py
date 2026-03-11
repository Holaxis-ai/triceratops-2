"""Validation computation: stateless engine + stateful workspace."""
from triceratops.validation.engine import ValidationEngine
from triceratops.validation.artifacts import (
    ApertureProvenance,
    ArtifactCapabilities,
    PreparedAutoFppArtifact,
)
from triceratops.validation.store import (
    FilesystemPreparedArtifactStore,
    PreparedArtifactStore,
    StoredArtifactRef,
)
from triceratops.validation.runner import (
    ApertureConfig,
    AutoFppComputeConfig,
    AutoFppPrepareConfig,
    FppRunConfig,
    FppRunResult,
    compute_auto_fpp,
    prepare_auto_fpp,
    run_tess_fpp,
)
from triceratops.validation.workspace import ValidationWorkspace

__all__ = [
    "ValidationEngine",
    "ValidationWorkspace",
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
]
