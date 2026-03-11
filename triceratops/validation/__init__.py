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
    AutoFppPreparedField,
    AutoFppPreparedLightCurve,
    AutoFppPreparedTrilegal,
    AutoFppResolvedTarget,
    FppRunConfig,
    FppRunResult,
    assemble_auto_fpp_stellar_field,
    build_auto_fpp_artifact,
    compute_auto_fpp,
    materialize_auto_fpp_trilegal,
    prepare_auto_fpp,
    prepare_auto_fpp_lightcurve,
    resolve_auto_fpp_target,
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
    "AutoFppResolvedTarget",
    "AutoFppPreparedLightCurve",
    "AutoFppPreparedField",
    "AutoFppPreparedTrilegal",
    "FppRunConfig",
    "FppRunResult",
    "resolve_auto_fpp_target",
    "prepare_auto_fpp_lightcurve",
    "assemble_auto_fpp_stellar_field",
    "materialize_auto_fpp_trilegal",
    "build_auto_fpp_artifact",
    "prepare_auto_fpp",
    "compute_auto_fpp",
    "run_tess_fpp",
]
