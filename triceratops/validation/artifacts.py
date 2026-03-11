"""Durable artifact types and local serialization for auto-FPP preparation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

import numpy as np
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.population.protocols import TRILEGALResult

ARTIFACT_VERSION = "0.1"
ARTIFACT_KIND_PREPARED = "prepared_auto_fpp"
ARTIFACT_KIND_COMPUTE_READY = "prepared_compute_inputs"


@dataclass(frozen=True)
class ArtifactCapabilities:
    """Flags describing what a prepared artifact can do without extra prep."""

    contains_trilegal_population: bool
    supports_scenarios: tuple[ScenarioID, ...]
    is_binned: bool
    bin_count: int | None = None


@dataclass(frozen=True)
class ApertureProvenance:
    """Optional aperture geometry retained for auditing or future re-analysis."""

    aperture_masks: tuple[np.ndarray, ...] = ()
    aperture_pixels_per_sector: tuple[np.ndarray, ...] = ()
    pixel_coords_per_sector: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class PreparedAutoFppArtifact:
    """Durable output of the auto-FPP preparation phase."""

    resolved_target: ResolvedTarget
    light_curve_result: LightCurvePreparationResult
    stellar_field: StellarField
    transit_depth: float
    created_at_utc: str
    code_version: str
    git_sha: str | None
    lightkurve_version: str | None
    aperture_mode: str
    aperture_threshold_sigma: float
    custom_aperture_pixels: tuple[tuple[int, int], ...]
    bin_count: int | None
    search_radius_px: int
    sigma_psf_px: float
    lightcurve_config: LightCurveConfig
    artifact_capabilities: ArtifactCapabilities
    warnings: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    aperture_provenance: ApertureProvenance | None = None
    trilegal_population: TRILEGALResult | None = None
    unbinned_light_curve: LightCurve | None = None
    artifact_version: str = ARTIFACT_VERSION
    artifact_kind: Literal["prepared_auto_fpp", "prepared_compute_inputs"] = (
        ARTIFACT_KIND_PREPARED
    )

    def to_directory(self, path: str | Path) -> Path:
        """Write the artifact to a local directory."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        light_curve_path = out_dir / "prepared_lightcurve.npz"
        np.savez(
            light_curve_path,
            phase_days=self.light_curve_result.light_curve.time_days,
            flux=self.light_curve_result.light_curve.flux,
            flux_err=np.array([self.light_curve_result.light_curve.flux_err], dtype=float),
            cadence_days=np.array([self.light_curve_result.light_curve.cadence_days], dtype=float),
            supersampling_rate=np.array(
                [self.light_curve_result.light_curve.supersampling_rate], dtype=int
            ),
            unbinned_phase_days=(
                self.unbinned_light_curve.time_days
                if self.unbinned_light_curve is not None
                else np.array([], dtype=float)
            ),
            unbinned_flux=(
                self.unbinned_light_curve.flux
                if self.unbinned_light_curve is not None
                else np.array([], dtype=float)
            ),
            unbinned_flux_err=np.array(
                [
                    self.unbinned_light_curve.flux_err
                    if self.unbinned_light_curve is not None
                    else np.nan
                ],
                dtype=float,
            ),
            unbinned_cadence_days=np.array(
                [
                    self.unbinned_light_curve.cadence_days
                    if self.unbinned_light_curve is not None
                    else np.nan
                ],
                dtype=float,
            ),
            unbinned_supersampling_rate=np.array(
                [
                    self.unbinned_light_curve.supersampling_rate
                    if self.unbinned_light_curve is not None
                    else -1
                ],
                dtype=int,
            ),
        )

        (out_dir / "stellar_field.json").write_text(
            json.dumps(_stellar_field_to_dict(self.stellar_field), indent=2, sort_keys=True)
        )

        if self.aperture_provenance is not None:
            (out_dir / "aperture_provenance.json").write_text(
                json.dumps(_aperture_provenance_to_dict(self.aperture_provenance), indent=2)
            )

        if self.trilegal_population is not None:
            np.savez(
                out_dir / "trilegal_population.npz",
                tmags=self.trilegal_population.tmags,
                masses=self.trilegal_population.masses,
                loggs=self.trilegal_population.loggs,
                teffs=self.trilegal_population.teffs,
                metallicities=self.trilegal_population.metallicities,
                jmags=self.trilegal_population.jmags,
                hmags=self.trilegal_population.hmags,
                kmags=self.trilegal_population.kmags,
                gmags=self.trilegal_population.gmags,
                rmags=self.trilegal_population.rmags,
                imags=self.trilegal_population.imags,
                zmags=self.trilegal_population.zmags,
            )

        (out_dir / "manifest.json").write_text(
            json.dumps(self.to_manifest_dict(), indent=2, sort_keys=True)
        )
        return out_dir

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return the JSON manifest for this artifact."""
        ephem = self.resolved_target.ephemeris
        return {
            "artifact_version": self.artifact_version,
            "artifact_kind": self.artifact_kind,
            "created_at_utc": self.created_at_utc,
            "code_version": self.code_version,
            "git_sha": self.git_sha,
            "lightkurve_version": self.lightkurve_version,
            "target_ref": self.resolved_target.target_ref,
            "tic_id": self.resolved_target.tic_id,
            "mission": self.stellar_field.mission,
            "resolved_target_source": self.resolved_target.source,
            "period_days": None if ephem is None else ephem.period_days,
            "t0_btjd": None if ephem is None else ephem.t0_btjd,
            "duration_hours": None if ephem is None else ephem.duration_hours,
            "ephemeris_source": None if ephem is None else ephem.source,
            "ephemeris_warnings": [] if ephem is None else list(ephem.warnings),
            "aperture_mode": self.aperture_mode,
            "aperture_threshold_sigma": self.aperture_threshold_sigma,
            "custom_aperture_pixels": [list(pixel) for pixel in self.custom_aperture_pixels],
            "bin_count": self.bin_count,
            "search_radius_px": self.search_radius_px,
            "sigma_psf_px": self.sigma_psf_px,
            "lightcurve_config": asdict(self.lightcurve_config),
            "sectors_used": list(self.light_curve_result.sectors_used),
            "cadence_used": self.light_curve_result.cadence_used,
            "warnings": list(self.warnings),
            "source_labels": list(self.source_labels),
            "transit_depth": self.transit_depth,
            "artifact_capabilities": {
                "contains_trilegal_population": (
                    self.artifact_capabilities.contains_trilegal_population
                ),
                "supports_scenarios": [
                    sid.name for sid in self.artifact_capabilities.supports_scenarios
                ],
                "is_binned": self.artifact_capabilities.is_binned,
                "bin_count": self.artifact_capabilities.bin_count,
            },
            "files": {
                "light_curve": "prepared_lightcurve.npz",
                "stellar_field": "stellar_field.json",
                "aperture_provenance": (
                    "aperture_provenance.json"
                    if self.aperture_provenance is not None
                    else None
                ),
                "trilegal_population": (
                    "trilegal_population.npz"
                    if self.trilegal_population is not None
                    else None
                ),
            },
        }

    @classmethod
    def from_directory(cls, path: str | Path) -> PreparedAutoFppArtifact:
        """Load an artifact from a local directory."""
        base = Path(path)
        manifest = json.loads((base / "manifest.json").read_text())
        light_curve_npz = np.load(base / "prepared_lightcurve.npz")
        lc = LightCurve(
            time_days=np.asarray(light_curve_npz["phase_days"], dtype=float),
            flux=np.asarray(light_curve_npz["flux"], dtype=float),
            flux_err=float(light_curve_npz["flux_err"][0]),
            cadence_days=float(light_curve_npz["cadence_days"][0]),
            supersampling_rate=int(light_curve_npz["supersampling_rate"][0]),
        )
        unbinned = None
        if light_curve_npz["unbinned_phase_days"].size > 0:
            unbinned = LightCurve(
                time_days=np.asarray(light_curve_npz["unbinned_phase_days"], dtype=float),
                flux=np.asarray(light_curve_npz["unbinned_flux"], dtype=float),
                flux_err=float(light_curve_npz["unbinned_flux_err"][0]),
                cadence_days=float(light_curve_npz["unbinned_cadence_days"][0]),
                supersampling_rate=int(
                    light_curve_npz["unbinned_supersampling_rate"][0]
                ),
            )

        ephemeris = None
        if manifest["period_days"] is not None and manifest["t0_btjd"] is not None:
            ephemeris = Ephemeris(
                period_days=float(manifest["period_days"]),
                t0_btjd=float(manifest["t0_btjd"]),
                duration_hours=(
                    None
                    if manifest["duration_hours"] is None
                    else float(manifest["duration_hours"])
                ),
                source=manifest["ephemeris_source"] or "unknown",
                warnings=tuple(manifest.get("ephemeris_warnings", [])),
            )

        aperture_provenance = None
        if manifest["files"].get("aperture_provenance"):
            aperture_provenance = _aperture_provenance_from_dict(
                json.loads((base / "aperture_provenance.json").read_text())
            )

        trilegal_population = None
        if manifest["files"].get("trilegal_population"):
            tri_npz = np.load(base / "trilegal_population.npz")
            trilegal_population = TRILEGALResult(
                tmags=np.asarray(tri_npz["tmags"], dtype=float),
                masses=np.asarray(tri_npz["masses"], dtype=float),
                loggs=np.asarray(tri_npz["loggs"], dtype=float),
                teffs=np.asarray(tri_npz["teffs"], dtype=float),
                metallicities=np.asarray(tri_npz["metallicities"], dtype=float),
                jmags=np.asarray(tri_npz["jmags"], dtype=float),
                hmags=np.asarray(tri_npz["hmags"], dtype=float),
                kmags=np.asarray(tri_npz["kmags"], dtype=float),
                gmags=np.asarray(tri_npz["gmags"], dtype=float),
                rmags=np.asarray(tri_npz["rmags"], dtype=float),
                imags=np.asarray(tri_npz["imags"], dtype=float),
                zmags=np.asarray(tri_npz["zmags"], dtype=float),
            )

        capabilities = manifest["artifact_capabilities"]
        return cls(
            artifact_version=manifest["artifact_version"],
            artifact_kind=manifest["artifact_kind"],
            created_at_utc=manifest["created_at_utc"],
            code_version=manifest["code_version"],
            git_sha=manifest["git_sha"],
            lightkurve_version=manifest["lightkurve_version"],
            resolved_target=ResolvedTarget(
                target_ref=manifest["target_ref"],
                tic_id=int(manifest["tic_id"]),
                ephemeris=ephemeris,
                source=manifest["resolved_target_source"],
            ),
            light_curve_result=LightCurvePreparationResult(
                light_curve=lc,
                ephemeris=ephemeris,
                sectors_used=tuple(int(s) for s in manifest["sectors_used"]),
                cadence_used=manifest["cadence_used"],
                warnings=list(manifest.get("warnings", [])),
            ),
            stellar_field=_stellar_field_from_dict(
                json.loads((base / "stellar_field.json").read_text())
            ),
            transit_depth=float(manifest["transit_depth"]),
            aperture_mode=manifest["aperture_mode"],
            aperture_threshold_sigma=float(manifest["aperture_threshold_sigma"]),
            custom_aperture_pixels=tuple(
                (int(pixel[0]), int(pixel[1]))
                for pixel in manifest["custom_aperture_pixels"]
            ),
            bin_count=(
                None if manifest["bin_count"] is None else int(manifest["bin_count"])
            ),
            search_radius_px=int(manifest["search_radius_px"]),
            sigma_psf_px=float(manifest["sigma_psf_px"]),
            lightcurve_config=LightCurveConfig(**manifest["lightcurve_config"]),
            artifact_capabilities=ArtifactCapabilities(
                contains_trilegal_population=bool(
                    capabilities["contains_trilegal_population"]
                ),
                supports_scenarios=tuple(
                    ScenarioID[name] for name in capabilities["supports_scenarios"]
                ),
                is_binned=bool(capabilities["is_binned"]),
                bin_count=(
                    None
                    if capabilities["bin_count"] is None
                    else int(capabilities["bin_count"])
                ),
            ),
            warnings=tuple(manifest.get("warnings", [])),
            source_labels=tuple(manifest.get("source_labels", [])),
            aperture_provenance=aperture_provenance,
            trilegal_population=trilegal_population,
            unbinned_light_curve=unbinned,
        )


def build_artifact_metadata() -> tuple[str, str | None, str | None]:
    """Return created-at timestamp, current git SHA, and lightkurve version."""
    created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_sha: str | None = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_sha = proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_sha = None

    lightkurve_version: str | None
    try:
        import lightkurve as lk

        lightkurve_version = str(lk.__version__)
    except Exception:
        lightkurve_version = None
    return created_at_utc, git_sha, lightkurve_version


def default_artifact_capabilities(
    *,
    has_trilegal: bool,
    bin_count: int | None,
) -> ArtifactCapabilities:
    """Build default capability flags for a prepared artifact."""
    if has_trilegal:
        supported = tuple(ScenarioID)
    else:
        supported = tuple(
            sid for sid in ScenarioID if sid not in ScenarioID.trilegal_scenarios()
        )
    return ArtifactCapabilities(
        contains_trilegal_population=has_trilegal,
        supports_scenarios=supported,
        is_binned=bin_count is not None,
        bin_count=bin_count,
    )


def _stellar_params_to_dict(params: StellarParameters | None) -> dict[str, float] | None:
    if params is None:
        return None
    return asdict(params)


def _star_to_dict(star: Star) -> dict[str, Any]:
    return {
        "tic_id": star.tic_id,
        "ra_deg": star.ra_deg,
        "dec_deg": star.dec_deg,
        "tmag": star.tmag,
        "jmag": star.jmag,
        "hmag": star.hmag,
        "kmag": star.kmag,
        "bmag": star.bmag,
        "vmag": star.vmag,
        "gmag": star.gmag,
        "rmag": star.rmag,
        "imag": star.imag,
        "zmag": star.zmag,
        "stellar_params": _stellar_params_to_dict(star.stellar_params),
        "separation_arcsec": star.separation_arcsec,
        "position_angle_deg": star.position_angle_deg,
        "flux_ratio": star.flux_ratio,
        "transit_depth_required": star.transit_depth_required,
    }


def _stellar_field_to_dict(field: StellarField) -> dict[str, Any]:
    return {
        "target_id": field.target_id,
        "mission": field.mission,
        "search_radius_pixels": field.search_radius_pixels,
        "stars": [_star_to_dict(star) for star in field.stars],
    }


def _star_from_dict(payload: dict[str, Any]) -> Star:
    stellar_params = payload.get("stellar_params")
    return Star(
        tic_id=int(payload["tic_id"]),
        ra_deg=float(payload["ra_deg"]),
        dec_deg=float(payload["dec_deg"]),
        tmag=float(payload["tmag"]),
        jmag=float(payload["jmag"]),
        hmag=float(payload["hmag"]),
        kmag=float(payload["kmag"]),
        bmag=float(payload["bmag"]),
        vmag=float(payload["vmag"]),
        gmag=None if payload.get("gmag") is None else float(payload["gmag"]),
        rmag=None if payload.get("rmag") is None else float(payload["rmag"]),
        imag=None if payload.get("imag") is None else float(payload["imag"]),
        zmag=None if payload.get("zmag") is None else float(payload["zmag"]),
        stellar_params=(
            None
            if stellar_params is None
            else StellarParameters(**stellar_params)
        ),
        separation_arcsec=float(payload["separation_arcsec"]),
        position_angle_deg=float(payload["position_angle_deg"]),
        flux_ratio=(
            None if payload.get("flux_ratio") is None else float(payload["flux_ratio"])
        ),
        transit_depth_required=(
            None
            if payload.get("transit_depth_required") is None
            else float(payload["transit_depth_required"])
        ),
    )


def _stellar_field_from_dict(payload: dict[str, Any]) -> StellarField:
    return StellarField(
        target_id=int(payload["target_id"]),
        mission=str(payload["mission"]),
        search_radius_pixels=int(payload["search_radius_pixels"]),
        stars=[_star_from_dict(star) for star in payload["stars"]],
    )


def _aperture_provenance_to_dict(payload: ApertureProvenance) -> dict[str, Any]:
    return {
        "aperture_masks": [mask.astype(bool).tolist() for mask in payload.aperture_masks],
        "aperture_pixels_per_sector": [
            pixels.astype(float).tolist() for pixels in payload.aperture_pixels_per_sector
        ],
        "pixel_coords_per_sector": [
            coords.astype(float).tolist() for coords in payload.pixel_coords_per_sector
        ],
    }


def _aperture_provenance_from_dict(payload: dict[str, Any]) -> ApertureProvenance:
    return ApertureProvenance(
        aperture_masks=tuple(
            np.asarray(mask, dtype=bool) for mask in payload["aperture_masks"]
        ),
        aperture_pixels_per_sector=tuple(
            np.asarray(pixels, dtype=float)
            for pixels in payload["aperture_pixels_per_sector"]
        ),
        pixel_coords_per_sector=tuple(
            np.asarray(coords, dtype=float)
            for coords in payload["pixel_coords_per_sector"]
        ),
    )


def make_prepared_artifact(
    *,
    resolved_target: ResolvedTarget,
    light_curve_result: LightCurvePreparationResult,
    stellar_field: StellarField,
    transit_depth: float,
    aperture_mode: str,
    aperture_threshold_sigma: float,
    custom_aperture_pixels: tuple[tuple[int, int], ...],
    bin_count: int | None,
    search_radius_px: int,
    sigma_psf_px: float,
    lightcurve_config: LightCurveConfig,
    warnings: tuple[str, ...] = (),
    source_labels: tuple[str, ...] = (),
    aperture_provenance: ApertureProvenance | None = None,
    trilegal_population: TRILEGALResult | None = None,
    unbinned_light_curve: LightCurve | None = None,
) -> PreparedAutoFppArtifact:
    """Construct a prepared artifact with default metadata and capabilities."""
    created_at_utc, git_sha, lightkurve_version = build_artifact_metadata()
    has_trilegal = trilegal_population is not None
    return PreparedAutoFppArtifact(
        resolved_target=resolved_target,
        light_curve_result=light_curve_result,
        stellar_field=stellar_field,
        transit_depth=transit_depth,
        created_at_utc=created_at_utc,
        code_version=_package_version(),
        git_sha=git_sha,
        lightkurve_version=lightkurve_version,
        aperture_mode=aperture_mode,
        aperture_threshold_sigma=aperture_threshold_sigma,
        custom_aperture_pixels=custom_aperture_pixels,
        bin_count=bin_count,
        search_radius_px=search_radius_px,
        sigma_psf_px=sigma_psf_px,
        lightcurve_config=lightcurve_config,
        artifact_capabilities=default_artifact_capabilities(
            has_trilegal=has_trilegal,
            bin_count=bin_count,
        ),
        warnings=warnings,
        source_labels=source_labels,
        aperture_provenance=aperture_provenance,
        trilegal_population=trilegal_population,
        unbinned_light_curve=unbinned_light_curve,
        artifact_kind=(
            ARTIFACT_KIND_COMPUTE_READY if has_trilegal else ARTIFACT_KIND_PREPARED
        ),
    )


def _package_version() -> str:
    try:
        return metadata.version("triceratops")
    except metadata.PackageNotFoundError:
        return "0.0.0+local"


__all__ = [
    "ARTIFACT_KIND_COMPUTE_READY",
    "ARTIFACT_KIND_PREPARED",
    "ARTIFACT_VERSION",
    "ApertureProvenance",
    "ArtifactCapabilities",
    "PreparedAutoFppArtifact",
    "build_artifact_metadata",
    "default_artifact_capabilities",
    "make_prepared_artifact",
]
