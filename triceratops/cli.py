"""Command-line interface for high-level TRICERATOPS workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from triceratops.config.config import Config
from triceratops.domain.scenario_id import ScenarioID
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.validation.runner import (
    ApertureConfig,
    AutoFppComputeConfig,
    AutoFppPrepareConfig,
    FppRunConfig,
    compute_auto_fpp,
    prepare_auto_fpp,
    run_tess_fpp,
)
from triceratops.validation.artifacts import PreparedAutoFppArtifact
from triceratops.validation.errors import PreparedInputIncompleteError
from triceratops.validation.store import (
    FilesystemPreparedArtifactStore,
    StoredArtifactRef,
)


def main(argv: Sequence[str] | None = None) -> None:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "prepare":
        parser = _build_prepare_parser()
        args = parser.parse_args(args_in[1:])
        artifact = prepare_auto_fpp(
            args.target,
            config=_build_prepare_config(args),
            ephemeris=_build_ephemeris(args),
        )
        if artifact.trilegal_population is None:
            raise PreparedInputIncompleteError(
                "prepare did not produce a compute-ready auto-FPP artifact: "
                "trilegal_population is missing."
            )
        store = FilesystemPreparedArtifactStore(
            base_dir=_prepare_store_base_dir(args.output)
        )
        ref = store.put(
            artifact,
            key=_prepare_store_key(args.output, artifact),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "artifact_dir": ref.location,
                        "artifact_key": ref.key,
                        "artifact_kind": artifact.artifact_kind,
                        "tic_id": artifact.resolved_target.tic_id,
                        "target_ref": artifact.resolved_target.target_ref,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        print(f"artifact_dir: {ref.location}")
        print(f"target: TIC {artifact.resolved_target.tic_id}")
        print(f"artifact_kind: {artifact.artifact_kind}")
        return

    if args_in and args_in[0] == "compute":
        parser = _build_compute_parser()
        args = parser.parse_args(args_in[1:])
        store = FilesystemPreparedArtifactStore()
        artifact = store.get(
            StoredArtifactRef(
                key=Path(args.artifact_dir).name,
                location=args.artifact_dir,
            )
        )
        result = compute_auto_fpp(
            artifact,
            config=AutoFppComputeConfig(
                compute=_build_compute_config(args),
                scenario_ids=_build_scenario_ids(args),
                trilegal_cache_path=args.trilegal_cache_path,
            ),
        )
        payload = _validation_result_to_dict(artifact, result)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        print(f"target: TIC {payload['tic_id']}")
        print(f"fpp: {payload['fpp']:.8f}")
        print(f"nfpp: {payload['nfpp']:.8f}")
        return

    parser = _build_parser()
    args = parser.parse_args(args_in)

    run_config = FppRunConfig(
        aperture=_build_aperture_config(args),
        lightcurve=_build_lightcurve_config(args),
        compute=_build_compute_config(args),
        bin_count=args.bin_count,
        transit_depth=args.transit_depth,
        search_radius_px=args.search_radius_px,
        sigma_psf_px=args.sigma_psf_px,
        trilegal_cache_path=args.trilegal_cache_path,
        exofop_cache_ttl_seconds=args.exofop_cache_ttl_seconds,
        exofop_disk_cache_dir=args.exofop_disk_cache_dir,
        scenario_ids=_build_scenario_ids(args),
    )
    ephemeris = _build_ephemeris(args)
    result = run_tess_fpp(args.target, config=run_config, ephemeris=ephemeris)

    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2, sort_keys=True))
        return

    print(f"target: TIC {result.resolved_target.tic_id}")
    print(f"fpp: {result.validation_result.fpp:.8f}")
    print(f"nfpp: {result.validation_result.nfpp:.8f}")
    print(f"sectors: {','.join(str(s) for s in result.light_curve_result.sectors_used)}")
    print(f"cadence: {result.light_curve_result.cadence_used}")
    print(f"transit_depth: {result.transit_depth:.8g}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tutorial-style TESS FPP validation in one command.",
    )
    parser.add_argument("target", help="TOI target string or TIC ID")
    _add_prepare_args(parser)
    _add_compute_args(parser)
    parser.add_argument("--json", action="store_true")
    return parser


def _build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a durable auto-FPP artifact.",
    )
    parser.add_argument("target", help="TOI target string or TIC ID")
    _add_prepare_args(parser)
    parser.add_argument(
        "--materialize-trilegal",
        action="store_true",
        help="Store TRILEGAL population so compute can run without provider I/O.",
    )
    parser.add_argument(
        "--include-unbinned",
        action="store_true",
        help="Retain the unbinned folded light curve for later re-binning.",
    )
    parser.add_argument(
        "--no-aperture-provenance",
        action="store_true",
        help="Do not store aperture masks and geometry provenance.",
    )
    parser.add_argument("--output", default=None, help="Output directory for the artifact.")
    parser.add_argument("--json", action="store_true")
    return parser


def _build_compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute FPP from a prepared auto-FPP artifact.",
    )
    parser.add_argument("artifact_dir", help="Directory containing manifest.json")
    _add_compute_args(parser)
    parser.add_argument("--json", action="store_true")
    return parser


def _add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transit-depth", type=float, default=None)
    parser.add_argument("--period-days", type=float, default=None)
    parser.add_argument("--t0-btjd", type=float, default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--bin-count", type=int, default=None)
    parser.add_argument(
        "--aperture-mode",
        choices=("default", "pipeline", "threshold", "all"),
        default="default",
    )
    parser.add_argument(
        "--aperture-threshold",
        type=float,
        default=3.0,
        help="Sigma threshold for threshold apertures.",
    )
    parser.add_argument(
        "--aperture-pixel",
        action="append",
        default=[],
        metavar="COL,ROW",
        help="Custom aperture pixel. Repeat to add multiple pixels.",
    )
    parser.add_argument("--search-radius-px", type=int, default=10)
    parser.add_argument("--sigma-psf-px", type=float, default=0.75)
    parser.add_argument(
        "--cadence",
        choices=("auto", "20sec", "2min", "10min", "30min"),
        default="auto",
    )
    parser.add_argument(
        "--quality-mask",
        choices=("default", "hard", "none"),
        default="default",
    )
    parser.add_argument(
        "--detrend-method",
        choices=("flatten", "none"),
        default="flatten",
    )
    parser.add_argument("--sigma-clip", type=float, default=5.0)
    parser.add_argument("--flatten-window-length", type=int, default=401)
    parser.add_argument("--flatten-polyorder", type=int, default=3)
    parser.add_argument("--phase-window-factor", type=float, default=5.0)
    parser.add_argument(
        "--flux-type",
        choices=("pdcsap_flux", "sap_flux"),
        default="pdcsap_flux",
    )
    parser.add_argument("--cadence-days-override", type=float, default=None)
    parser.add_argument("--supersampling-rate", type=int, default=20)
    parser.add_argument("--sector", type=int, action="append", default=[])
    parser.add_argument("--all-sectors", action="store_true")
    parser.add_argument("--exofop-cache-ttl-seconds", type=int, default=6 * 3600)
    parser.add_argument("--exofop-disk-cache-dir", default=None)


def _add_compute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-mc-samples", type=int, default=20_000)
    parser.add_argument("--lnz-const", type=float, default=650.0)
    parser.add_argument("--n-best-samples", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--flat-priors", action="store_true")
    parser.add_argument("--scenario", action="append", default=[], choices=[sid.name for sid in ScenarioID])
    parser.add_argument("--trilegal-cache-path", default=None)


def _build_ephemeris(args: argparse.Namespace) -> Ephemeris | None:
    if args.period_days is None and args.t0_btjd is None and args.duration_hours is None:
        return None
    if args.period_days is None or args.t0_btjd is None:
        raise ValueError(
            "--period-days and --t0-btjd must both be provided for manual ephemeris input"
        )
    return Ephemeris(
        period_days=args.period_days,
        t0_btjd=args.t0_btjd,
        duration_hours=args.duration_hours,
    )


def _parse_pixel(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected COL,ROW for aperture pixel, got {text!r}")
    return int(parts[0]), int(parts[1])


def _build_aperture_config(args: argparse.Namespace) -> ApertureConfig:
    aperture_pixels = tuple(_parse_pixel(text) for text in args.aperture_pixel)
    aperture_mode = "custom" if aperture_pixels else args.aperture_mode
    return ApertureConfig(
        mode=aperture_mode,
        threshold_sigma=args.aperture_threshold,
        custom_pixels=aperture_pixels,
    )


def _build_lightcurve_config(args: argparse.Namespace) -> LightCurveConfig:
    sectors: tuple[int, ...] | str
    if args.all_sectors:
        sectors = "all"
    elif args.sector:
        sectors = tuple(args.sector)
    else:
        sectors = "auto"
    return LightCurveConfig(
        cadence=args.cadence,
        sectors=sectors,
        quality_mask=args.quality_mask,
        detrend_method=args.detrend_method,
        sigma_clip=args.sigma_clip,
        flatten_window_length=args.flatten_window_length,
        flatten_polyorder=args.flatten_polyorder,
        phase_window_factor=args.phase_window_factor,
        flux_type=args.flux_type,
        cadence_days_override=args.cadence_days_override,
        supersampling_rate=args.supersampling_rate,
    )


def _build_compute_config(args: argparse.Namespace) -> Config:
    return Config(
        n_mc_samples=args.n_mc_samples,
        lnz_const=args.lnz_const,
        n_best_samples=args.n_best_samples,
        parallel=not args.no_parallel,
        flat_priors=args.flat_priors,
        mission="TESS",
        n_workers=args.n_workers,
    )


def _build_scenario_ids(args: argparse.Namespace) -> tuple[ScenarioID, ...] | None:
    return tuple(ScenarioID[name] for name in args.scenario) if args.scenario else None


def _build_prepare_config(args: argparse.Namespace) -> AutoFppPrepareConfig:
    return AutoFppPrepareConfig(
        aperture=_build_aperture_config(args),
        lightcurve=_build_lightcurve_config(args),
        transit_depth=args.transit_depth,
        search_radius_px=args.search_radius_px,
        sigma_psf_px=args.sigma_psf_px,
        bin_count=args.bin_count,
        include_unbinned_folded_lightcurve=args.include_unbinned,
        include_aperture_provenance=not args.no_aperture_provenance,
        materialize_trilegal=args.materialize_trilegal,
        exofop_cache_ttl_seconds=args.exofop_cache_ttl_seconds,
        exofop_disk_cache_dir=args.exofop_disk_cache_dir,
    )


def _default_artifact_dir(artifact: PreparedAutoFppArtifact) -> Path:
    slug = str(artifact.created_at_utc).replace(":", "-")
    return Path(f"auto-fpp-tic{artifact.resolved_target.tic_id}-{slug}")


def _prepare_store_base_dir(output: str | None) -> Path:
    if output is None:
        return Path(".")
    output_path = Path(output)
    if output_path.suffix:
        return output_path.parent
    return output_path.parent if output_path.name == "" else output_path.parent


def _prepare_store_key(
    output: str | None,
    artifact: PreparedAutoFppArtifact,
) -> str | None:
    if output is None:
        return _default_artifact_dir(artifact).name
    output_path = Path(output)
    return output_path.name


def _result_to_dict(result) -> dict[str, object]:
    return {
        "target_ref": result.resolved_target.target_ref,
        "tic_id": result.resolved_target.tic_id,
        "fpp": result.validation_result.fpp,
        "nfpp": result.validation_result.nfpp,
        "transit_depth": result.transit_depth,
        "sectors_used": list(result.light_curve_result.sectors_used),
        "cadence_used": result.light_curve_result.cadence_used,
        "scenario_probabilities": {
            scenario.scenario_id.name: scenario.relative_probability
            for scenario in result.validation_result.scenario_results
        },
    }


def _validation_result_to_dict(
    artifact: PreparedAutoFppArtifact,
    result,
) -> dict[str, object]:
    return {
        "target_ref": artifact.resolved_target.target_ref,
        "tic_id": artifact.resolved_target.tic_id,
        "fpp": result.fpp,
        "nfpp": result.nfpp,
        "artifact_kind": artifact.artifact_kind,
        "scenario_probabilities": {
            scenario.scenario_id.name: scenario.relative_probability
            for scenario in result.scenario_results
        },
    }


__all__ = ["main"]
