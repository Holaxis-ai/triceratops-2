"""Tests for durable auto-FPP artifacts."""
from __future__ import annotations

import numpy as np

from auto_fpp.artifacts import (
    ARTIFACT_KIND_COMPUTE_READY,
    ARTIFACT_KIND_PREPARED,
    ApertureProvenance,
    ArtifactBundle,
    ArtifactHistoryEvent,
    ArtifactStageState,
    PreparedAutoFppArtifact,
    SectorApertureManifest,
    SectorApertureSelection,
    default_artifact_capabilities,
    make_prepared_artifact,
)
from auto_fpp.models import RepeatMetricSummary, ValidationRepeatSummary
from auto_fpp.outputs import (
    with_compute_outputs,
    with_preparation_outputs,
    with_repeat_outputs,
)
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.population.protocols import TRILEGALResult


def _stellar_field() -> StellarField:
    params = StellarParameters(
        mass_msun=1.0,
        radius_rsun=1.0,
        teff_k=5700.0,
        logg=4.4,
        metallicity_dex=0.0,
        parallax_mas=10.0,
    )
    return StellarField(
        target_id=12345,
        mission="TESS",
        search_radius_pixels=10,
        stars=[
            Star(
                tic_id=12345,
                ra_deg=10.0,
                dec_deg=20.0,
                tmag=10.0,
                jmag=9.0,
                hmag=8.9,
                kmag=8.8,
                bmag=10.2,
                vmag=10.1,
                stellar_params=params,
                flux_ratio=0.98,
                transit_depth_required=0.001,
            ),
            Star(
                tic_id=67890,
                ra_deg=10.001,
                dec_deg=20.001,
                tmag=12.5,
                jmag=11.9,
                hmag=11.7,
                kmag=11.6,
                bmag=13.0,
                vmag=12.7,
                stellar_params=params,
                separation_arcsec=5.0,
                position_angle_deg=90.0,
                flux_ratio=0.02,
                transit_depth_required=0.05,
            ),
        ],
    )


def _light_curve_result() -> LightCurvePreparationResult:
    return LightCurvePreparationResult(
        light_curve=LightCurve(
            time_days=np.linspace(-0.1, 0.1, 8),
            flux=np.linspace(0.999, 1.001, 8),
            flux_err=0.001,
            cadence_days=0.002,
            supersampling_rate=15,
        ),
        ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0, duration_hours=2.0),
        sectors_used=(14, 15),
        cadence_used="2min",
        warnings=["trimmed"],
    )


def _trilegal() -> TRILEGALResult:
    data = np.array([10.0, 11.0])
    return TRILEGALResult(
        tmags=data,
        masses=data * 0.1,
        loggs=data * 0.2,
        teffs=data * 100.0,
        metallicities=np.array([0.0, -0.1]),
        jmags=data + 1.0,
        hmags=data + 2.0,
        kmags=data + 3.0,
        gmags=data + 4.0,
        rmags=data + 5.0,
        imags=data + 6.0,
        zmags=data + 7.0,
    )


def _scenario_result() -> ScenarioResult:
    n = 4
    return ScenarioResult(
        scenario_id=ScenarioID.TP,
        host_star_tic_id=12345,
        ln_evidence=-5.0,
        host_mass_msun=np.ones(n),
        host_radius_rsun=np.ones(n),
        host_u1=np.full(n, 0.3),
        host_u2=np.full(n, 0.2),
        period_days=np.full(n, 5.0),
        inclination_deg=np.full(n, 88.0),
        impact_parameter=np.full(n, 0.1),
        eccentricity=np.zeros(n),
        arg_periastron_deg=np.full(n, 90.0),
        planet_radius_rearth=np.full(n, 2.0),
        eb_mass_msun=np.zeros(n),
        eb_radius_rsun=np.zeros(n),
        flux_ratio_eb_tess=np.zeros(n),
        companion_mass_msun=np.zeros(n),
        companion_radius_rsun=np.zeros(n),
        flux_ratio_companion_tess=np.zeros(n),
    )


def test_default_artifact_capabilities_exclude_trilegal_without_population() -> None:
    caps = default_artifact_capabilities(has_trilegal=False, bin_count=100)
    assert caps.contains_trilegal_population is False
    assert caps.is_binned is True
    assert caps.bin_count == 100
    assert ScenarioID.BTP not in caps.supports_scenarios


def test_prepared_artifact_round_trips_to_bundle() -> None:
    unbinned = LightCurve(
        time_days=np.linspace(-0.2, 0.2, 16),
        flux=np.ones(16),
        flux_err=0.002,
        cadence_days=0.001,
        supersampling_rate=20,
    )
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TOI-123.01",
            tic_id=12345,
            ephemeris=Ephemeris(
                period_days=5.0,
                t0_btjd=1000.0,
                duration_hours=2.0,
                source="exofop",
                warnings=("depth inferred",),
            ),
            source="exofop",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.0015,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        sector_aperture_overrides=(
            SectorApertureManifest(
                sector=14,
                mode="threshold",
                threshold_sigma=4.5,
            ),
        ),
        sector_aperture_selections=(
            SectorApertureSelection(
                sector=14,
                requested_mode="default",
                effective_mode="pipeline",
                threshold_sigma=3.0,
                n_pixels=2,
            ),
        ),
        bin_count=100,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        warnings=("warning",),
        source_labels=("lightkurve",),
        aperture_provenance=ApertureProvenance(
            aperture_masks=(np.array([[True, False], [False, True]]),),
            aperture_pixels_per_sector=(np.array([[0.0, 0.0], [1.0, 1.0]]),),
            pixel_coords_per_sector=(np.array([[10.0, 10.0], [11.0, 10.0]]),),
        ),
        trilegal_population=_trilegal(),
        unbinned_light_curve=unbinned,
    )

    bundle = artifact.to_bundle()
    loaded = PreparedAutoFppArtifact.from_bundle(bundle)

    assert loaded.artifact_kind == ARTIFACT_KIND_COMPUTE_READY
    assert loaded.status == "ready"
    assert loaded.resume_from is None
    assert loaded.resolved_target.target_ref == "TOI-123.01"
    assert loaded.resolved_target.ephemeris is not None
    assert loaded.resolved_target.ephemeris.warnings == ("depth inferred",)
    assert loaded.light_curve_result.sectors_used == (14, 15)
    assert loaded.sector_aperture_overrides == artifact.sector_aperture_overrides
    assert loaded.sector_aperture_selections == artifact.sector_aperture_selections
    np.testing.assert_allclose(
        loaded.light_curve_result.light_curve.time_days,
        artifact.light_curve_result.light_curve.time_days,
    )
    assert loaded.stellar_field.target.flux_ratio == artifact.stellar_field.target.flux_ratio
    assert loaded.trilegal_population is not None
    np.testing.assert_allclose(loaded.trilegal_population.tmags, np.array([10.0, 11.0]))
    assert loaded.aperture_provenance is not None
    np.testing.assert_array_equal(
        loaded.aperture_provenance.aperture_masks[0],
        np.array([[True, False], [False, True]]),
    )
    assert loaded.unbinned_light_curve is not None
    np.testing.assert_allclose(loaded.unbinned_light_curve.time_days, unbinned.time_days)
    assert next(stage for stage in loaded.stages if stage.name == "trilegal").status == (
        "completed"
    )


def test_artifact_bundle_round_trips_through_directory(tmp_path) -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
    )
    bundle = artifact.to_bundle()
    out_dir = bundle.write_directory(tmp_path / "artifact")
    loaded_bundle = ArtifactBundle.from_directory(out_dir)
    loaded = PreparedAutoFppArtifact.from_bundle(loaded_bundle)

    assert "manifest.json" in loaded_bundle.files
    assert loaded.resolved_target.tic_id == artifact.resolved_target.tic_id


def test_make_prepared_artifact_without_trilegal_is_not_compute_ready() -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        sector_aperture_overrides=(),
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
    )

    assert artifact.artifact_kind == ARTIFACT_KIND_PREPARED
    assert artifact.artifact_capabilities.contains_trilegal_population is False
    assert artifact.status == "partial"
    assert artifact.resume_from == "trilegal"
    assert next(stage for stage in artifact.stages if stage.name == "trilegal").status == (
        "skipped"
    )


def test_manifest_round_trips_stage_and_history_metadata() -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        stages=(
            ArtifactStageState(name="target_resolution", status="completed"),
            ArtifactStageState(name="lightcurve", status="completed"),
            ArtifactStageState(name="field", status="completed"),
            ArtifactStageState(name="trilegal", status="failed"),
            ArtifactStageState(name="store_upload", status="pending"),
        ),
        history=(
            ArtifactHistoryEvent(
                stage="trilegal",
                event="failed",
                at="2026-03-11T00:00:00Z",
                message="missing trilegal population",
            ),
        ),
    )

    loaded = PreparedAutoFppArtifact.from_bundle(artifact.to_bundle())

    assert loaded.status == "partial"
    assert loaded.resume_from == "trilegal"
    assert [stage.status for stage in loaded.stages] == [
        "completed",
        "completed",
        "completed",
        "failed",
        "pending",
    ]
    assert loaded.history[0].stage == "trilegal"


def test_manifest_includes_sector_aperture_overrides() -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        sector_aperture_overrides=(
            SectorApertureManifest(
                sector=14,
                mode="custom",
                threshold_sigma=3.0,
                custom_pixels=((1, 2), (3, 4)),
            ),
        ),
        sector_aperture_selections=(
            SectorApertureSelection(
                sector=14,
                requested_mode="default",
                effective_mode="pipeline",
                threshold_sigma=3.0,
                n_pixels=2,
            ),
        ),
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
    )

    manifest = artifact.to_manifest_dict()
    assert manifest["sector_aperture_overrides"] == [
        {
            "sector": 14,
            "mode": "custom",
            "threshold_sigma": 3.0,
            "custom_pixels": [[1, 2], [3, 4]],
        }
    ]
    assert manifest["sector_aperture_selections"] == [
        {
            "sector": 14,
            "requested_mode": "default",
            "effective_mode": "pipeline",
            "threshold_sigma": 3.0,
            "n_pixels": 2,
        }
    ]


def test_prepared_artifact_round_trips_extra_files() -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        extra_files={"contrast_curve_summary.json": b"{\"source\":\"exofop\"}"},
    )

    loaded = PreparedAutoFppArtifact.from_bundle(artifact.to_bundle())

    assert loaded.extra_files["contrast_curve_summary.json"] == b"{\"source\":\"exofop\"}"


def test_prepared_artifact_round_trips_contrast_curve() -> None:
    contrast_curve = ContrastCurve(
        separations_arcsec=np.array([0.2, 0.5, 1.0]),
        delta_mags=np.array([2.0, 4.0, 6.0]),
        band="J",
    )
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=100,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        trilegal_population=_trilegal(),
        contrast_curve=contrast_curve,
    )

    loaded = PreparedAutoFppArtifact.from_bundle(artifact.to_bundle())

    assert loaded.contrast_curve is not None
    assert loaded.contrast_curve.band == "J"
    np.testing.assert_allclose(
        loaded.contrast_curve.separations_arcsec,
        contrast_curve.separations_arcsec,
    )
    np.testing.assert_allclose(
        loaded.contrast_curve.delta_mags,
        contrast_curve.delta_mags,
    )


def test_preparation_outputs_are_added_and_round_trip() -> None:
    artifact = with_preparation_outputs(
        make_prepared_artifact(
            resolved_target=ResolvedTarget(
                target_ref="TIC 12345",
                tic_id=12345,
                ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
                source="manual",
            ),
            light_curve_result=_light_curve_result(),
            stellar_field=_stellar_field(),
            transit_depth=0.001,
            aperture_mode="default",
            aperture_threshold_sigma=3.0,
            custom_aperture_pixels=(),
            bin_count=100,
            search_radius_px=10,
            sigma_psf_px=0.75,
            lightcurve_config=LightCurveConfig(),
            trilegal_population=_trilegal(),
        )
    )

    assert "tables/stars.csv" in artifact.extra_files
    assert "plots/field.pdf" in artifact.extra_files
    loaded = PreparedAutoFppArtifact.from_bundle(artifact.to_bundle())
    assert "tables/stars.csv" in loaded.extra_files
    assert "plots/field.pdf" in loaded.extra_files


def test_compute_outputs_are_added_and_round_trip() -> None:
    artifact = with_preparation_outputs(
        make_prepared_artifact(
            resolved_target=ResolvedTarget(
                target_ref="TIC 12345",
                tic_id=12345,
                ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
                source="manual",
            ),
            light_curve_result=_light_curve_result(),
            stellar_field=_stellar_field(),
            transit_depth=0.001,
            aperture_mode="default",
            aperture_threshold_sigma=3.0,
            custom_aperture_pixels=(),
            bin_count=100,
            search_radius_px=10,
            sigma_psf_px=0.75,
            lightcurve_config=LightCurveConfig(),
            trilegal_population=_trilegal(),
        )
    )
    updated = with_compute_outputs(
        artifact,
        validation_result=ValidationResult(
            target_id=12345,
            false_positive_probability=0.1,
            nearby_false_positive_probability=0.02,
            scenario_results=[_scenario_result()],
        ),
        workspace=None,  # type: ignore[arg-type]
    )

    assert "tables/probs.csv" in updated.extra_files
    assert "tables/scenario_probabilities.csv" in updated.extra_files
    assert "plots/fits.pdf" in updated.extra_files
    probs_text = updated.extra_files["tables/probs.csv"].decode("utf-8")
    assert "ID" in probs_text
    assert "scenario" in probs_text
    assert "lnZ" in probs_text
    assert "flux_ratio_comp_T" in probs_text
    loaded = PreparedAutoFppArtifact.from_bundle(updated.to_bundle())
    assert "tables/probs.csv" in loaded.extra_files
    assert "tables/scenario_probabilities.csv" in loaded.extra_files
    assert "plots/fits.pdf" in loaded.extra_files


def test_repeat_outputs_are_added_and_round_trip() -> None:
    artifact = make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TIC 12345",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="manual",
        ),
        light_curve_result=_light_curve_result(),
        stellar_field=_stellar_field(),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=100,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        trilegal_population=_trilegal(),
    )
    updated = with_repeat_outputs(
        artifact,
        summary=ValidationRepeatSummary(
            repeat_n=2,
            fpp=RepeatMetricSummary(mean=0.2, std=0.1, min=0.1, max=0.3),
            nfpp=RepeatMetricSummary(mean=0.04, std=0.02, min=0.02, max=0.06),
            results=(
                ValidationResult(
                    target_id=12345,
                    false_positive_probability=0.1,
                    nearby_false_positive_probability=0.02,
                    scenario_results=[],
                ),
                ValidationResult(
                    target_id=12345,
                    false_positive_probability=0.3,
                    nearby_false_positive_probability=0.06,
                    scenario_results=[],
                ),
            ),
        ),
    )

    assert "tables/repeat_summary.json" in updated.extra_files
    assert "tables/repeat_summary.csv" in updated.extra_files
    summary_json = updated.extra_files["tables/repeat_summary.json"].decode("utf-8")
    assert "\"repeat_n\": 2" in summary_json
    summary_csv = updated.extra_files["tables/repeat_summary.csv"].decode("utf-8")
    assert "run_index,fpp,nfpp" in summary_csv
    loaded = PreparedAutoFppArtifact.from_bundle(updated.to_bundle())
    assert "tables/repeat_summary.json" in loaded.extra_files
    assert "tables/repeat_summary.csv" in loaded.extra_files
