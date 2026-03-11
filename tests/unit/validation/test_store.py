"""Tests for prepared artifact stores."""
from __future__ import annotations

import numpy as np

from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult
from auto_fpp.store import (
    FilesystemPreparedArtifactStore,
    StoredArtifactRef,
    default_artifact_key,
)
from auto_fpp.artifacts import make_prepared_artifact


def _artifact():
    params = StellarParameters(
        mass_msun=1.0,
        radius_rsun=1.0,
        teff_k=5700.0,
        logg=4.4,
        metallicity_dex=0.0,
        parallax_mas=10.0,
    )
    return make_prepared_artifact(
        resolved_target=ResolvedTarget(
            target_ref="TOI-123.01",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="exofop",
        ),
        light_curve_result=LightCurvePreparationResult(
            light_curve=LightCurve(
                time_days=np.linspace(-0.1, 0.1, 4),
                flux=np.ones(4),
                flux_err=0.001,
            ),
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            sectors_used=(14,),
            cadence_used="2min",
            warnings=[],
        ),
        stellar_field=StellarField(
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
                    flux_ratio=1.0,
                    transit_depth_required=0.001,
                )
            ],
        ),
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=100,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
    )


def test_filesystem_store_put_and_get_round_trip(tmp_path) -> None:
    artifact = _artifact()
    store = FilesystemPreparedArtifactStore(tmp_path)

    ref = store.put(artifact)
    loaded = store.get(ref)

    assert (tmp_path / ref.key / "manifest.json").exists()
    assert loaded.resolved_target.tic_id == artifact.resolved_target.tic_id
    np.testing.assert_allclose(
        loaded.light_curve_result.light_curve.time_days,
        artifact.light_curve_result.light_curve.time_days,
    )


def test_filesystem_store_uses_explicit_key(tmp_path) -> None:
    artifact = _artifact()
    store = FilesystemPreparedArtifactStore(tmp_path)

    ref = store.put(artifact, key="custom-key")

    assert ref == StoredArtifactRef(
        key="custom-key",
        location=str(tmp_path / "custom-key"),
        store_kind="filesystem",
    )


def test_default_artifact_key_includes_tic_id() -> None:
    artifact = _artifact()
    key = default_artifact_key(artifact)
    assert key.startswith("auto-fpp-tic12345-")
