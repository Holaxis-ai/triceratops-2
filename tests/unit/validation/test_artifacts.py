"""Tests for durable auto-FPP artifacts."""
from __future__ import annotations

import numpy as np

from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.population.protocols import TRILEGALResult
from auto_fpp.artifacts import (
    ARTIFACT_KIND_COMPUTE_READY,
    ARTIFACT_KIND_PREPARED,
    ApertureProvenance,
    PreparedAutoFppArtifact,
    default_artifact_capabilities,
    make_prepared_artifact,
)


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


def test_default_artifact_capabilities_exclude_trilegal_without_population() -> None:
    caps = default_artifact_capabilities(has_trilegal=False, bin_count=100)
    assert caps.contains_trilegal_population is False
    assert caps.is_binned is True
    assert caps.bin_count == 100
    assert ScenarioID.BTP not in caps.supports_scenarios


def test_prepared_artifact_round_trips_to_directory(tmp_path) -> None:
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

    out_dir = artifact.to_directory(tmp_path / "artifact")
    loaded = PreparedAutoFppArtifact.from_directory(out_dir)

    assert loaded.artifact_kind == ARTIFACT_KIND_COMPUTE_READY
    assert loaded.resolved_target.target_ref == "TOI-123.01"
    assert loaded.resolved_target.ephemeris is not None
    assert loaded.resolved_target.ephemeris.warnings == ("depth inferred",)
    assert loaded.light_curve_result.sectors_used == (14, 15)
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
        bin_count=None,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
    )

    assert artifact.artifact_kind == ARTIFACT_KIND_PREPARED
    assert artifact.artifact_capabilities.contains_trilegal_population is False
