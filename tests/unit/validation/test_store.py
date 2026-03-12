"""Tests for prepared artifact stores."""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from auto_fpp.artifacts import (
    AutoFppLightCurveCheckpoint,
    AutoFppPrepareCheckpoint,
    PreparedSectorGeometry,
    make_prepared_artifact,
)
from auto_fpp.outputs import with_preparation_outputs
from auto_fpp.store import (
    FilesystemPreparedArtifactStore,
    R2PreparedArtifactStore,
    StoredArtifactRef,
    WranglerPreparedArtifactStore,
    _bundle_file_names_from_manifest,
    _content_type,
    _endpoint_from_account_id,
    _parse_r2_locator,
    default_artifact_key,
)
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.value_objects import ContrastCurve, StellarParameters
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult


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


def _target_checkpoint() -> AutoFppPrepareCheckpoint:
    return AutoFppPrepareCheckpoint(
        resolved_target=ResolvedTarget(
            target_ref="TOI-123.01",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="exofop",
        ),
        transit_depth=0.001,
        created_at_utc="2026-03-11T12:00:00Z",
        code_version="0.0.0+local",
        git_sha="abc123",
        lightkurve_version="2.5.0",
    )


def _lightcurve_checkpoint() -> AutoFppLightCurveCheckpoint:
    return AutoFppLightCurveCheckpoint(
        resolved_target=ResolvedTarget(
            target_ref="TOI-123.01",
            tic_id=12345,
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
            source="exofop",
        ),
        light_curve_result=_artifact().light_curve_result,
        transit_depth=0.001,
        created_at_utc="2026-03-11T12:00:00Z",
        code_version="0.0.0+local",
        git_sha="abc123",
        lightkurve_version="2.5.0",
        sector_geometries=(
            PreparedSectorGeometry(
                sector=14,
                shape=(2, 2),
                wcs_header={
                    "CTYPE1": "RA---TAN",
                    "CTYPE2": "DEC--TAN",
                    "CRPIX1": 1.0,
                    "CRPIX2": 1.0,
                    "CRVAL1": 10.0,
                    "CRVAL2": 20.0,
                    "CDELT1": -0.0001,
                    "CDELT2": 0.0001,
                    "CUNIT1": "deg",
                    "CUNIT2": "deg",
                },
                aperture_mask=np.array([[True, False], [False, False]]),
                image=np.ones((2, 2)),
            ),
        ),
        requested_sectors=(14,),
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
        locator=str(tmp_path / "custom-key"),
        store_kind="filesystem",
    )


def test_filesystem_store_round_trips_target_checkpoint(tmp_path) -> None:
    checkpoint = _target_checkpoint()
    store = FilesystemPreparedArtifactStore(tmp_path)

    ref = store.put(checkpoint, key="target-checkpoint")
    loaded = store.get(ref)

    assert isinstance(loaded, AutoFppPrepareCheckpoint)
    assert loaded.resume_from == "lightcurve"
    assert loaded.resolved_target.tic_id == 12345


def test_filesystem_store_round_trips_lightcurve_checkpoint(tmp_path) -> None:
    checkpoint = _lightcurve_checkpoint()
    store = FilesystemPreparedArtifactStore(tmp_path)

    ref = store.put(checkpoint, key="lightcurve-checkpoint")
    loaded = store.get(ref)

    assert isinstance(loaded, AutoFppLightCurveCheckpoint)
    assert loaded.resume_from == "field"
    assert loaded.sector_geometries[0].sector == 14


def test_filesystem_store_replace_updates_existing_locator(tmp_path) -> None:
    artifact = _artifact()
    store = FilesystemPreparedArtifactStore(tmp_path)

    ref = store.put(artifact, key="custom-key")
    updated = with_preparation_outputs(artifact)
    replaced = store.replace(ref, updated)
    loaded = store.get(ref)

    assert replaced == ref
    assert "tables/stars.csv" in loaded.extra_files


def test_default_artifact_key_includes_tic_id() -> None:
    artifact = _artifact()
    key = default_artifact_key(artifact)
    assert key.startswith("auto-fpp-tic12345-")


class _FakeR2Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[(Bucket, Key)] = Body
        self.content_types[(Bucket, Key)] = ContentType

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_r2_store_put_and_get_round_trip() -> None:
    artifact = _artifact()
    client = _FakeR2Client()
    store = R2PreparedArtifactStore(
        bucket="science-artifacts",
        key_prefix="prepared",
        client=client,
    )

    ref = store.put(artifact, key="tic-12345")
    loaded = store.get(ref)

    assert ref == StoredArtifactRef(
        key="tic-12345",
        locator="r2://science-artifacts/prepared/tic-12345",
        store_kind="r2",
    )
    assert (
        client.content_types[
            ("science-artifacts", "prepared/tic-12345/manifest.json")
        ]
        == "application/json"
    )
    np.testing.assert_allclose(
        loaded.light_curve_result.light_curve.time_days,
        artifact.light_curve_result.light_curve.time_days,
    )
    assert loaded.resolved_target.tic_id == artifact.resolved_target.tic_id


def test_r2_store_rejects_non_r2_ref() -> None:
    store = R2PreparedArtifactStore(bucket="science-artifacts", client=_FakeR2Client())

    try:
        store.get(
            StoredArtifactRef(
                key="artifact",
                locator="/tmp/artifact",
                store_kind="filesystem",
            )
        )
    except ValueError as exc:
        assert "store_kind='filesystem'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-R2 ref")


def test_r2_store_round_trips_contrast_curve_payload() -> None:
    artifact = make_prepared_artifact(
        resolved_target=_artifact().resolved_target,
        light_curve_result=_artifact().light_curve_result,
        stellar_field=_artifact().stellar_field,
        transit_depth=0.001,
        aperture_mode="default",
        aperture_threshold_sigma=3.0,
        custom_aperture_pixels=(),
        bin_count=100,
        search_radius_px=10,
        sigma_psf_px=0.75,
        lightcurve_config=LightCurveConfig(),
        contrast_curve=ContrastCurve(
            separations_arcsec=np.array([0.5, 1.0]),
            delta_mags=np.array([4.0, 6.0]),
            band="TESS",
        ),
    )
    client = _FakeR2Client()
    store = R2PreparedArtifactStore(
        bucket="science-artifacts",
        key_prefix="prepared",
        client=client,
    )

    ref = store.put(artifact, key="tic-12345")
    loaded = store.get(ref)

    assert loaded.contrast_curve is not None
    np.testing.assert_allclose(
        loaded.contrast_curve.separations_arcsec,
        artifact.contrast_curve.separations_arcsec,
    )
    np.testing.assert_allclose(
        loaded.contrast_curve.delta_mags,
        artifact.contrast_curve.delta_mags,
    )


def test_r2_store_build_client_uses_cloudflare_env(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("CLOUDFLARE_R2_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "access-123")
    monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "secret-123")
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=_fake_client),
    )

    store = R2PreparedArtifactStore(bucket="science-artifacts")

    assert store.bucket == "science-artifacts"
    assert seen["service_name"] == "s3"
    assert seen["region_name"] == "auto"
    assert seen["endpoint_url"] == "https://acct-123.r2.cloudflarestorage.com"
    assert seen["aws_access_key_id"] == "access-123"
    assert seen["aws_secret_access_key"] == "secret-123"


def test_r2_store_build_client_rejects_partial_credentials(monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_R2_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "access-123")
    monkeypatch.delenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda **kwargs: object()),
    )

    with pytest.raises(ValueError, match="requires both access_key_id and secret_access_key"):
        R2PreparedArtifactStore(bucket="science-artifacts")


def test_r2_store_build_client_requires_endpoint_or_account(monkeypatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_R2_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("CLOUDFLARE_R2_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda **kwargs: object()),
    )

    with pytest.raises(ValueError, match="requires either endpoint_url or account_id"):
        R2PreparedArtifactStore(bucket="science-artifacts")


def test_store_helper_functions_cover_locator_manifest_and_content_type() -> None:
    assert _endpoint_from_account_id(" acct-123 ") == "https://acct-123.r2.cloudflarestorage.com"
    assert _endpoint_from_account_id(" ") is None

    assert _parse_r2_locator("r2://bucket/prefix/key") == ("bucket", "prefix/key")
    assert _parse_r2_locator("wrangler-r2://bucket/prefix/key") == ("bucket", "prefix/key")
    with pytest.raises(ValueError, match="Expected R2 locator"):
        _parse_r2_locator("/tmp/local")

    assert _bundle_file_names_from_manifest(
        {"artifact_kind": "prepare_checkpoint_target_resolution"}
    ) == ["manifest.json"]
    assert _bundle_file_names_from_manifest(
        {
            "artifact_kind": "prepare_checkpoint_lightcurve",
            "files": {"lightcurve_provenance": "lightcurve_provenance.json"},
        }
    ) == [
        "manifest.json",
        "prepared_lightcurve.npz",
        "sector_geometry.json",
        "lightcurve_provenance.json",
    ]

    names = _bundle_file_names_from_manifest(
        {
            "artifact_kind": "prepared_compute_inputs",
            "files": {
                "aperture_provenance": "aperture_provenance.json",
                "lightcurve_provenance": "lightcurve_provenance.json",
                "trilegal_population": "trilegal_population.npz",
                "contrast_curve": "contrast_curve.npz",
            },
            "lightcurve_variants": [{"file": "lightcurves/bin-100.npz"}],
            "extra_files": ["tables/stars.csv"],
        }
    )
    assert "trilegal_population.npz" in names
    assert "lightcurves/bin-100.npz" in names
    assert "tables/stars.csv" in names

    assert _content_type("manifest.json") == "application/json"
    assert _content_type("tables/stars.csv") == "text/csv"
    assert _content_type("plots/fits.pdf") == "application/pdf"
    assert _content_type("prepared_lightcurve.npz") == "application/octet-stream"
    assert _content_type("raw.bin") == "application/octet-stream"


class _FakeWranglerRunner:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.commands: list[list[str]] = []

    def __call__(
        self,
        cmd,
        *,
        check,
        cwd,
        stdout,
        stderr,
        text,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        if cmd[1:4] != ["r2", "object", "put"] and cmd[1:4] != ["r2", "object", "get"]:
            raise AssertionError(f"Unexpected wrangler command: {cmd}")
        if cmd[1:4] == ["r2", "object", "put"]:
            object_path = cmd[4]
            file_path = Path(cmd[cmd.index("--file") + 1])
            self.objects[object_path] = file_path.read_bytes()
        else:
            object_path = cmd[4]
            file_path = Path(cmd[cmd.index("--file") + 1])
            file_path.write_bytes(self.objects[object_path])
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_wrangler_store_put_and_get_round_trip() -> None:
    artifact = _artifact()
    runner = _FakeWranglerRunner()
    store = WranglerPreparedArtifactStore(
        bucket="science-artifacts",
        key_prefix="prepared",
        runner=runner,
    )

    ref = store.put(artifact, key="tic-12345")
    loaded = store.get(ref)

    assert ref == StoredArtifactRef(
        key="tic-12345",
        locator="wrangler-r2://science-artifacts/prepared/tic-12345",
        store_kind="wrangler",
    )
    assert any(
        command[:5]
        == [
            "wrangler",
            "r2",
            "object",
            "put",
            "science-artifacts/prepared/tic-12345/manifest.json",
        ]
        for command in runner.commands
    )
    np.testing.assert_allclose(
        loaded.light_curve_result.light_curve.time_days,
        artifact.light_curve_result.light_curve.time_days,
    )
    assert loaded.resolved_target.tic_id == artifact.resolved_target.tic_id


def test_wrangler_store_rejects_non_wrangler_ref() -> None:
    store = WranglerPreparedArtifactStore(bucket="science-artifacts", runner=_FakeWranglerRunner())

    try:
        store.get(
            StoredArtifactRef(
                key="artifact",
                locator="r2://science/artifact",
                store_kind="r2",
            )
        )
    except ValueError as exc:
        assert "store_kind='r2'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-Wrangler ref")


def test_wrangler_store_validates_bucket_replace_and_run_flags(tmp_path) -> None:
    runner = _FakeWranglerRunner()
    store = WranglerPreparedArtifactStore(
        bucket=None,
        key_prefix="prepared",
        wrangler_bin="npx-wrangler",
        cwd=tmp_path,
        config_path=tmp_path / "wrangler.toml",
        env_name="prod",
        runner=runner,
    )

    with pytest.raises(ValueError, match="requires a bucket for put"):
        store.put(_artifact(), key="tic-12345")

    with pytest.raises(ValueError, match="store_kind='filesystem'"):
        store.replace(
            StoredArtifactRef(
                key="artifact",
                locator="/tmp/artifact",
                store_kind="filesystem",
            ),
            _artifact(),
        )

    captured: list[list[str]] = []

    def _capture_runner(cmd, **kwargs):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    flag_store = WranglerPreparedArtifactStore(
        bucket="science-artifacts",
        wrangler_bin="npx-wrangler",
        cwd=tmp_path,
        config_path=tmp_path / "wrangler.toml",
        env_name="prod",
        runner=_capture_runner,
    )
    flag_store._run(["whoami"])
    assert captured[0][:5] == [
        "npx-wrangler",
        "--config",
        str(tmp_path / "wrangler.toml"),
        "--env",
        "prod",
    ]
