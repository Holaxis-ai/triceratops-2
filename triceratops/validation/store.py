"""Artifact store abstractions for durable auto-FPP preparation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from triceratops.validation.artifacts import PreparedAutoFppArtifact


@dataclass(frozen=True)
class StoredArtifactRef:
    """Stable reference to a stored prepared artifact."""

    key: str
    location: str
    store_kind: str = "filesystem"


@runtime_checkable
class PreparedArtifactStore(Protocol):
    """Protocol for reading and writing prepared auto-FPP artifacts."""

    def put(
        self,
        artifact: PreparedAutoFppArtifact,
        *,
        key: str | None = None,
    ) -> StoredArtifactRef: ...

    def get(self, ref: StoredArtifactRef) -> PreparedAutoFppArtifact: ...


class FilesystemPreparedArtifactStore:
    """Store prepared artifacts in local directories on the filesystem."""

    def __init__(self, base_dir: str | Path = ".") -> None:
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def put(
        self,
        artifact: PreparedAutoFppArtifact,
        *,
        key: str | None = None,
    ) -> StoredArtifactRef:
        resolved_key = key or default_artifact_key(artifact)
        artifact_dir = self._base_dir / resolved_key
        artifact.to_directory(artifact_dir)
        return StoredArtifactRef(
            key=resolved_key,
            location=str(artifact_dir),
            store_kind="filesystem",
        )

    def get(self, ref: StoredArtifactRef) -> PreparedAutoFppArtifact:
        if ref.store_kind != "filesystem":
            raise ValueError(
                f"FilesystemPreparedArtifactStore cannot load store_kind={ref.store_kind!r}"
            )
        return PreparedAutoFppArtifact.from_directory(ref.location)


def default_artifact_key(artifact: PreparedAutoFppArtifact) -> str:
    """Build a default filesystem key for an artifact."""
    created = artifact.created_at_utc.replace(":", "-")
    return f"auto-fpp-tic{artifact.resolved_target.tic_id}-{created}"


__all__ = [
    "FilesystemPreparedArtifactStore",
    "PreparedArtifactStore",
    "StoredArtifactRef",
    "default_artifact_key",
]
