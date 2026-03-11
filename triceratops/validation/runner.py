"""High-level end-to-end runner for tutorial-style TESS FPP calculations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from triceratops.config.config import Config
from triceratops.domain.result import ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.convert import convert_folded_to_domain
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.errors import (
    EphemerisRequiredError,
    LightCurveError,
    LightCurveNotFoundError,
)
from triceratops.lightcurve.exofop.toi_resolution import (
    LookupStatus,
    ToiResolutionResult,
    resolve_toi_to_tic_ephemeris_depth,
)
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.lightcurve.sources.lightkurve import (
    fold_lightcurve,
    process_lightcurve_collection,
    resolve_cadence_label,
    trim_folded_lightcurve,
)
from triceratops.validation.artifacts import (
    ApertureProvenance,
    PreparedAutoFppArtifact,
    make_prepared_artifact,
)
from triceratops.validation.errors import PreparedInputIncompleteError
from triceratops.validation.job import PreparedValidationInputs
from triceratops.validation.workspace import ValidationWorkspace

if TYPE_CHECKING:
    from triceratops.domain.entities import StellarField

_CADENCE_MAP: dict[str, tuple[str, float | None]] = {
    "20sec": ("SPOC", 20.0),
    "2min": ("SPOC", 120.0),
    "10min": ("QLP", 600.0),
    "30min": ("SPOC", 1800.0),
}


@dataclass(frozen=True)
class ApertureConfig:
    """Configuration for aperture-mask selection in the lightkurve TPF path."""

    mode: Literal["default", "pipeline", "threshold", "all", "custom"] = "default"
    threshold_sigma: float = 3.0
    custom_pixels: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.threshold_sigma <= 0:
            raise ValueError(
                f"threshold_sigma must be positive, got {self.threshold_sigma}"
            )
        if self.mode == "custom" and not self.custom_pixels:
            raise ValueError("custom_pixels must be provided when mode='custom'")
        if self.mode != "custom" and self.custom_pixels:
            raise ValueError("custom_pixels are only valid when mode='custom'")
        for col, row in self.custom_pixels:
            if col < 0 or row < 0:
                raise ValueError(
                    f"custom aperture pixels must be non-negative, got {(col, row)}"
                )


@dataclass(frozen=True)
class FppRunConfig:
    """UI-facing configuration for a single tutorial-style TESS validation run."""

    aperture: ApertureConfig = field(default_factory=ApertureConfig)
    lightcurve: LightCurveConfig = field(default_factory=LightCurveConfig)
    compute: Config = field(default_factory=Config)
    bin_count: int | None = None
    transit_depth: float | None = None
    search_radius_px: int = 10
    sigma_psf_px: float = 0.75
    trilegal_cache_path: str | None = None
    exofop_cache_ttl_seconds: int = 6 * 3600
    exofop_disk_cache_dir: str | Path | None = None
    scenario_ids: tuple[ScenarioID, ...] | None = None

    def __post_init__(self) -> None:
        if self.bin_count is not None and self.bin_count < 2:
            raise ValueError(f"bin_count must be >= 2 or None, got {self.bin_count}")
        if self.transit_depth is not None and self.transit_depth <= 0:
            raise ValueError(
                f"transit_depth must be positive or None, got {self.transit_depth}"
            )
        if self.search_radius_px < 1:
            raise ValueError(
                f"search_radius_px must be >= 1, got {self.search_radius_px}"
            )
        if self.sigma_psf_px <= 0:
            raise ValueError(f"sigma_psf_px must be positive, got {self.sigma_psf_px}")
        if self.compute.mission != "TESS":
            raise ValueError(
                f"run_tess_fpp only supports compute.mission='TESS', got {self.compute.mission!r}"
            )


@dataclass(frozen=True)
class FppRunResult:
    """Outputs of the single-command tutorial-style FPP runner."""

    resolved_target: ResolvedTarget
    light_curve_result: LightCurvePreparationResult
    validation_result: ValidationResult
    workspace: ValidationWorkspace
    transit_depth: float
    aperture_masks: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class AutoFppPrepareConfig:
    """Configuration for the durable auto-FPP preparation phase."""

    aperture: ApertureConfig = field(default_factory=ApertureConfig)
    lightcurve: LightCurveConfig = field(default_factory=LightCurveConfig)
    transit_depth: float | None = None
    search_radius_px: int = 10
    sigma_psf_px: float = 0.75
    bin_count: int | None = None
    include_unbinned_folded_lightcurve: bool = False
    include_aperture_provenance: bool = True
    materialize_trilegal: bool = True
    exofop_cache_ttl_seconds: int = 6 * 3600
    exofop_disk_cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.bin_count is not None and self.bin_count < 2:
            raise ValueError(f"bin_count must be >= 2 or None, got {self.bin_count}")
        if self.transit_depth is not None and self.transit_depth <= 0:
            raise ValueError(
                f"transit_depth must be positive or None, got {self.transit_depth}"
            )
        if self.search_radius_px < 1:
            raise ValueError(
                f"search_radius_px must be >= 1, got {self.search_radius_px}"
            )
        if self.sigma_psf_px <= 0:
            raise ValueError(f"sigma_psf_px must be positive, got {self.sigma_psf_px}")


@dataclass(frozen=True)
class AutoFppComputeConfig:
    """Configuration for compute from a durable auto-FPP artifact."""

    compute: Config = field(default_factory=Config)
    scenario_ids: tuple[ScenarioID, ...] | None = None
    trilegal_cache_path: str | None = None

    def __post_init__(self) -> None:
        if self.compute.mission != "TESS":
            raise ValueError(
                "compute_auto_fpp only supports compute.mission='TESS', "
                f"got {self.compute.mission!r}"
            )


@dataclass(frozen=True)
class _PreparedApertureLightCurve:
    light_curve_result: LightCurvePreparationResult
    unbinned_light_curve: LightCurvePreparationResult | None = None
    aperture_masks: tuple[np.ndarray, ...] = ()
    tpfs: tuple[object, ...] = ()


def prepare_auto_fpp(
    target: str | int,
    *,
    config: AutoFppPrepareConfig | None = None,
    ephemeris: Ephemeris | None = None,
) -> PreparedAutoFppArtifact:
    """Prepare a durable auto-FPP artifact for later compute."""
    cfg = config or AutoFppPrepareConfig()
    resolved_target, transit_depth = _resolve_target_and_depth(
        target,
        _prepare_config_to_resolution_view(cfg),
        ephemeris,
    )
    prepare_kwargs: dict[str, object] = {}
    if cfg.include_unbinned_folded_lightcurve:
        prepare_kwargs["include_unbinned_folded"] = True
    prepared_lc = _prepare_tpf_lightcurve(
        resolved_target,
        _prepare_config_to_resolution_view(cfg),
        **prepare_kwargs,
    )

    workspace_kwargs = {
        "tic_id": resolved_target.tic_id,
        "sectors": np.asarray(prepared_lc.light_curve_result.sectors_used, dtype=int),
        "mission": "TESS",
        "search_radius": cfg.search_radius_px,
    }
    if cfg.materialize_trilegal:
        from triceratops.population.trilegal_provider import TRILEGALProvider

        workspace_kwargs["population_provider"] = TRILEGALProvider()
    try:
        workspace = ValidationWorkspace(**workspace_kwargs)
    except TypeError as exc:
        if "population_provider" not in str(exc):
            raise
        workspace_kwargs.pop("population_provider", None)
        workspace = ValidationWorkspace(**workspace_kwargs)
    workspace.set_resolved_target(resolved_target)
    field = workspace.fetch_catalog()
    pixel_coords_per_sector, aperture_pixels_per_sector = _derive_sector_geometry(
        prepared_lc.tpfs,
        prepared_lc.aperture_masks,
        field,
        cfg.search_radius_px,
    )
    workspace.calc_depths(
        transit_depth,
        pixel_coords_per_sector=pixel_coords_per_sector,
        aperture_pixels_per_sector=aperture_pixels_per_sector,
        sigma_psf_px=cfg.sigma_psf_px,
    )

    trilegal_population = None
    if cfg.materialize_trilegal:
        prepared_inputs = workspace.prepare(
            light_curve=prepared_lc.light_curve_result.light_curve,
            period_days=resolved_target.ephemeris.period_days,
        )
        trilegal_population = prepared_inputs.trilegal_population
        if trilegal_population is None:
            raise PreparedInputIncompleteError(
                "auto-FPP preparation requires TRILEGAL, but no trilegal_population "
                "was materialized. Ensure a population provider is configured and "
                "the TRILEGAL query succeeds."
            )

    aperture_provenance = None
    if cfg.include_aperture_provenance:
        aperture_provenance = ApertureProvenance(
            aperture_masks=prepared_lc.aperture_masks,
            aperture_pixels_per_sector=tuple(aperture_pixels_per_sector),
            pixel_coords_per_sector=tuple(pixel_coords_per_sector),
        )

    return make_prepared_artifact(
        resolved_target=resolved_target,
        light_curve_result=prepared_lc.light_curve_result,
        stellar_field=field,
        transit_depth=transit_depth,
        aperture_mode=cfg.aperture.mode,
        aperture_threshold_sigma=cfg.aperture.threshold_sigma,
        custom_aperture_pixels=cfg.aperture.custom_pixels,
        bin_count=cfg.bin_count,
        search_radius_px=cfg.search_radius_px,
        sigma_psf_px=cfg.sigma_psf_px,
        lightcurve_config=cfg.lightcurve,
        warnings=tuple(prepared_lc.light_curve_result.warnings),
        source_labels=("lightkurve",),
        aperture_provenance=aperture_provenance,
        trilegal_population=trilegal_population,
        unbinned_light_curve=(
            None
            if prepared_lc.unbinned_light_curve is None
            else prepared_lc.unbinned_light_curve.light_curve
        ),
    )


def compute_auto_fpp(
    prepared: PreparedAutoFppArtifact,
    *,
    config: AutoFppComputeConfig | None = None,
) -> ValidationResult:
    """Run FPP from a prepared auto-FPP artifact."""
    result, _ = _compute_prepared_artifact(
        prepared,
        config or AutoFppComputeConfig(),
    )
    return result


def run_tess_fpp(
    target: str | int,
    *,
    config: FppRunConfig | None = None,
    ephemeris: Ephemeris | None = None,
) -> FppRunResult:
    """Run a tutorial-style TESS FPP validation in one call.

    This orchestration layer owns target resolution, aperture selection,
    lightkurve photometry, transit-depth computation, and the final FPP run.
    """
    cfg = config or FppRunConfig()
    prepared = prepare_auto_fpp(
        target,
        config=_run_to_prepare_config(cfg),
        ephemeris=ephemeris,
    )
    validation_result, workspace = _compute_prepared_artifact(
        prepared,
        AutoFppComputeConfig(
            compute=cfg.compute,
            scenario_ids=cfg.scenario_ids,
            trilegal_cache_path=cfg.trilegal_cache_path,
        ),
    )
    return FppRunResult(
        resolved_target=prepared.resolved_target,
        light_curve_result=prepared.light_curve_result,
        validation_result=validation_result,
        workspace=workspace,
        transit_depth=prepared.transit_depth,
        aperture_masks=(
            ()
            if prepared.aperture_provenance is None
            else prepared.aperture_provenance.aperture_masks
        ),
    )


def _resolve_target_and_depth(
    target: str | int,
    config: FppRunConfig,
    ephemeris: Ephemeris | None,
) -> tuple[ResolvedTarget, float]:
    tic_id = _parse_tic_target(target)
    if tic_id is not None:
        if ephemeris is None:
            raise EphemerisRequiredError(
                "ephemeris must be provided when running on a TIC target"
            )
        if config.transit_depth is None:
            raise ValueError(
                "transit_depth must be provided when running on a TIC target"
            )
        return (
            ResolvedTarget(
                target_ref=f"TIC {tic_id}",
                tic_id=tic_id,
                ephemeris=ephemeris,
                source="manual",
            ),
            config.transit_depth,
        )

    toi_result = resolve_toi_to_tic_ephemeris_depth(
        str(target),
        cache_ttl_seconds=config.exofop_cache_ttl_seconds,
        disk_cache_dir=config.exofop_disk_cache_dir,
    )
    if toi_result.status != LookupStatus.OK:
        raise LightCurveError(
            f"ExoFOP resolution failed for '{target}': "
            f"{toi_result.message or toi_result.status.value}"
        )
    if (
        toi_result.tic_id is None
        or toi_result.period_days is None
        or toi_result.t0_btjd is None
    ):
        raise LightCurveError(
            f"ExoFOP resolution for '{target}' did not return a complete ephemeris"
        )
    transit_depth = _resolve_transit_depth(toi_result, config.transit_depth)
    resolved_target = ResolvedTarget(
        target_ref=str(target),
        tic_id=toi_result.tic_id,
        ephemeris=Ephemeris(
            period_days=toi_result.period_days,
            t0_btjd=toi_result.t0_btjd,
            duration_hours=toi_result.duration_hours,
        ),
        source="exofop",
    )
    return resolved_target, transit_depth


def _resolve_transit_depth(
    toi_result: ToiResolutionResult,
    configured_depth: float | None,
) -> float:
    if configured_depth is not None:
        return configured_depth
    if toi_result.depth_ppm is None or toi_result.depth_ppm <= 0:
        raise ValueError(
            "transit_depth was not provided and ExoFOP did not return a usable depth_ppm"
        )
    return toi_result.depth_ppm * 1e-6


def _prepare_tpf_lightcurve(
    target: ResolvedTarget,
    config: FppRunConfig,
    *,
    include_unbinned_folded: bool = False,
) -> _PreparedApertureLightCurve:
    import lightkurve as lk

    if target.ephemeris is None:
        raise EphemerisRequiredError(
            f"ResolvedTarget for TIC {target.tic_id} has no ephemeris"
        )

    tpf_collection = _download_tpfs(target.tic_id, config.lightcurve, config.search_radius_px)
    tpfs = tuple(tpf_collection)
    if not tpfs:
        raise LightCurveNotFoundError(f"No TPFs found for TIC {target.tic_id}")

    aperture_masks = tuple(
        _resolve_aperture_mask(tpf, config.aperture)
        for tpf in tpfs
    )
    lightcurves = [
        tpf.to_lightcurve(aperture_mask=mask)
        for tpf, mask in zip(tpfs, aperture_masks)
    ]
    lc_coll = lk.LightCurveCollection(lightcurves)
    lc = process_lightcurve_collection(lc_coll, config.lightcurve, tic_id=target.tic_id)
    folded = fold_lightcurve(lc, target.ephemeris)
    unbinned_trimmed = trim_folded_lightcurve(
        folded,
        target.ephemeris,
        config.lightcurve,
        tic_id=target.tic_id,
    )
    # Preserve lightkurve's binning implementation at the orchestration boundary.
    if config.bin_count is not None:
        folded = folded.bin(bins=config.bin_count)
    trimmed = trim_folded_lightcurve(
        folded,
        target.ephemeris,
        config.lightcurve,
        tic_id=target.tic_id,
    )

    cadence_used = resolve_cadence_label(lc, config.lightcurve.cadence)
    lc_domain = convert_folded_to_domain(trimmed, cadence=cadence_used, config=config.lightcurve)
    unbinned_domain = None
    if include_unbinned_folded:
        unbinned_domain = LightCurvePreparationResult(
            light_curve=convert_folded_to_domain(
                unbinned_trimmed,
                cadence=cadence_used,
                config=config.lightcurve,
            ),
            ephemeris=target.ephemeris,
            sectors_used=_extract_tpf_sectors(tpfs),
            cadence_used=cadence_used,
            warnings=[],
        )
    sectors_used = _extract_tpf_sectors(tpfs)
    return _PreparedApertureLightCurve(
        light_curve_result=LightCurvePreparationResult(
            light_curve=lc_domain,
            ephemeris=target.ephemeris,
            sectors_used=sectors_used,
            cadence_used=cadence_used,
            warnings=[],
        ),
        unbinned_light_curve=unbinned_domain,
        aperture_masks=aperture_masks,
        tpfs=tpfs,
    )


def _download_tpfs(
    tic_id: int,
    config: LightCurveConfig,
    search_radius_px: int,
):
    import lightkurve as lk

    search_kwargs: dict[str, object] = {
        "target": f"TIC {tic_id}",
        "mission": "TESS",
    }
    if config.cadence != "auto" and config.cadence in _CADENCE_MAP:
        author, exptime = _CADENCE_MAP[config.cadence]
        search_kwargs["author"] = author
        search_kwargs["exptime"] = exptime
    if isinstance(config.sectors, tuple):
        search_kwargs["sector"] = list(config.sectors)

    search = lk.search_targetpixelfile(**search_kwargs)
    if len(search) == 0:
        raise LightCurveNotFoundError(
            f"No TESS target pixel files found for TIC {tic_id}"
        )
    if config.sectors == "auto":
        search = search[-1:]
    cutout_size = 2 * search_radius_px + 1
    return search.download_all(
        quality_bitmask={"none": "none", "default": "default", "hard": "hard"}[
            config.quality_mask
        ],
        cutout_size=(cutout_size, cutout_size),
    )
def _resolve_aperture_mask(tpf: object, config: ApertureConfig) -> np.ndarray:
    shape = tpf.shape[1:]  # type: ignore[attr-defined]
    if config.mode == "all":
        return np.ones(shape, dtype=bool)
    if config.mode == "pipeline":
        mask = np.asarray(tpf.pipeline_mask, dtype=bool)  # type: ignore[attr-defined]
        if not np.any(mask):
            raise ValueError("pipeline aperture mask is missing or empty")
        return mask
    if config.mode == "threshold":
        return np.asarray(
            tpf.create_threshold_mask(threshold=config.threshold_sigma),  # type: ignore[attr-defined]
            dtype=bool,
        )
    if config.mode == "custom":
        mask = np.zeros(shape, dtype=bool)
        for col, row in config.custom_pixels:
            if row >= shape[0] or col >= shape[1]:
                raise ValueError(
                    f"custom aperture pixel {(col, row)} falls outside TPF shape {shape}"
                )
            mask[row, col] = True
        return mask

    pipeline_mask = np.asarray(tpf.pipeline_mask, dtype=bool)  # type: ignore[attr-defined]
    if np.any(pipeline_mask):
        return pipeline_mask
    return np.asarray(
        tpf.create_threshold_mask(threshold=config.threshold_sigma),  # type: ignore[attr-defined]
        dtype=bool,
    )


def _derive_sector_geometry(
    tpfs: tuple[object, ...],
    aperture_masks: tuple[np.ndarray, ...],
    field: StellarField,
    search_radius_px: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    star_coords = SkyCoord(
        [star.ra_deg for star in field.stars] * u.deg,
        [star.dec_deg for star in field.stars] * u.deg,
    )

    pixel_coords_per_sector: list[np.ndarray] = []
    aperture_pixels_per_sector: list[np.ndarray] = []
    for tpf, mask in zip(tpfs, aperture_masks):
        x, y = tpf.wcs.world_to_pixel(star_coords)  # type: ignore[attr-defined]
        coords = np.column_stack([x, y]).astype(float)
        local_limit_x = tpf.shape[2] + search_radius_px  # type: ignore[attr-defined]
        local_limit_y = tpf.shape[1] + search_radius_px  # type: ignore[attr-defined]
        finite = np.isfinite(coords).all(axis=1)
        inside = (
            (coords[:, 0] >= -search_radius_px)
            & (coords[:, 0] <= local_limit_x)
            & (coords[:, 1] >= -search_radius_px)
            & (coords[:, 1] <= local_limit_y)
        )
        if not np.all(finite & inside):
            raise LightCurveError(
                "Downloaded TPF does not cover all catalog stars inside the configured "
                "search radius. Reduce search_radius_px or choose a larger cutout."
            )
        pixel_coords_per_sector.append(coords)

        rows, cols = np.nonzero(mask)
        aperture_pixels_per_sector.append(
            np.column_stack([cols, rows]).astype(float)
        )
    return pixel_coords_per_sector, aperture_pixels_per_sector


def _extract_tpf_sectors(tpfs: tuple[object, ...]) -> tuple[int, ...]:
    sectors: list[int] = []
    for tpf in tpfs:
        sector = None
        if hasattr(tpf, "meta"):
            sector = tpf.meta.get("SECTOR")  # type: ignore[union-attr]
        if sector is None and hasattr(tpf, "sector"):
            sector = tpf.sector  # type: ignore[attr-defined]
        if sector is None:
            continue
        sectors.append(int(sector))
    if sectors:
        return tuple(sorted(sectors))
    return (0,)


def _parse_tic_target(target: str | int) -> int | None:
    if isinstance(target, int):
        return target
    text = str(target).strip().upper()
    if text.startswith("TIC "):
        text = text[4:].strip()
    elif text.startswith("TIC"):
        text = text[3:].strip()
    else:
        if "." in text or text.startswith("TOI"):
            return None
    return int(text) if text.isdigit() else None


def _run_to_prepare_config(config: FppRunConfig) -> AutoFppPrepareConfig:
    return AutoFppPrepareConfig(
        aperture=config.aperture,
        lightcurve=config.lightcurve,
        transit_depth=config.transit_depth,
        search_radius_px=config.search_radius_px,
        sigma_psf_px=config.sigma_psf_px,
        bin_count=config.bin_count,
        include_unbinned_folded_lightcurve=False,
        include_aperture_provenance=True,
        materialize_trilegal=True,
        exofop_cache_ttl_seconds=config.exofop_cache_ttl_seconds,
        exofop_disk_cache_dir=config.exofop_disk_cache_dir,
    )


def _prepare_config_to_resolution_view(config: AutoFppPrepareConfig) -> FppRunConfig:
    return FppRunConfig(
        aperture=config.aperture,
        lightcurve=config.lightcurve,
        bin_count=config.bin_count,
        transit_depth=config.transit_depth,
        search_radius_px=config.search_radius_px,
        sigma_psf_px=config.sigma_psf_px,
        exofop_cache_ttl_seconds=config.exofop_cache_ttl_seconds,
        exofop_disk_cache_dir=config.exofop_disk_cache_dir,
    )


def _compute_prepared_artifact(
    prepared: PreparedAutoFppArtifact,
    config: AutoFppComputeConfig,
) -> tuple[ValidationResult, ValidationWorkspace]:
    if prepared.resolved_target.ephemeris is None:
        raise EphemerisRequiredError(
            f"Prepared artifact for TIC {prepared.resolved_target.tic_id} has no ephemeris"
        )

    workspace_kwargs = {
        "tic_id": prepared.resolved_target.tic_id,
        "sectors": np.asarray(prepared.light_curve_result.sectors_used, dtype=int),
        "mission": prepared.stellar_field.mission,
        "search_radius": prepared.search_radius_px,
        "config": config.compute,
        "trilegal_cache_path": config.trilegal_cache_path,
    }
    try:
        workspace = ValidationWorkspace(
            **workspace_kwargs,
            stellar_field=deepcopy(prepared.stellar_field),
        )
    except TypeError as exc:
        if "stellar_field" not in str(exc):
            raise
        workspace = ValidationWorkspace(**workspace_kwargs)
        if hasattr(workspace, "_stellar_field"):
            workspace._stellar_field = deepcopy(prepared.stellar_field)  # type: ignore[attr-defined]
        if hasattr(workspace, "field"):
            workspace.field = deepcopy(prepared.stellar_field)  # type: ignore[attr-defined]
    workspace.set_resolved_target(prepared.resolved_target)

    requested_scenarios = (
        list(config.scenario_ids) if config.scenario_ids is not None else None
    )
    if not hasattr(workspace, "compute_prepared") and hasattr(workspace, "compute_probs"):
        result = workspace.compute_probs(
            prepared.light_curve_result.light_curve,
            prepared.resolved_target.ephemeris.period_days,
            scenario_ids=requested_scenarios,
        )
        return result, workspace

    if prepared.trilegal_population is None:
        raise PreparedInputIncompleteError(
            "Prepared auto-FPP artifact is missing trilegal_population. "
            "auto-FPP compute requires a compute-ready artifact produced with "
            "TRILEGAL materialized during preparation."
        )
    prepared_inputs = PreparedValidationInputs(
        target_id=prepared.resolved_target.tic_id,
        stellar_field=workspace.fetch_catalog(),
        light_curve=prepared.light_curve_result.light_curve,
        config=config.compute,
        period_days=prepared.resolved_target.ephemeris.period_days,
        trilegal_population=prepared.trilegal_population,
        scenario_ids=requested_scenarios,
    )
    return workspace.compute_prepared(prepared_inputs), workspace


__all__ = [
    "ApertureConfig",
    "AutoFppComputeConfig",
    "AutoFppPrepareConfig",
    "FppRunConfig",
    "FppRunResult",
    "compute_auto_fpp",
    "prepare_auto_fpp",
    "run_tess_fpp",
]
