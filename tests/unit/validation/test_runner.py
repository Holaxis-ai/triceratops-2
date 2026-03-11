"""Unit tests for the high-level tutorial-style FPP runner."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.result import ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.lightcurve.exofop.toi_resolution import (
    LookupStatus,
    ToiResolutionResult,
)
from triceratops.validation import runner
from triceratops.validation.runner import ApertureConfig, FppRunConfig, run_tess_fpp


def _field() -> StellarField:
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
                jmag=10.0,
                hmag=10.0,
                kmag=10.0,
                bmag=10.0,
                vmag=10.0,
            ),
        ],
    )


def _lc_result() -> LightCurvePreparationResult:
    return LightCurvePreparationResult(
        light_curve=LightCurve(
            time_days=np.linspace(-0.1, 0.1, 20),
            flux=np.ones(20),
            flux_err=0.001,
        ),
        ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0, duration_hours=2.0),
        sectors_used=(14, 15),
        cadence_used="2min",
        warnings=[],
    )


class StubWorkspace:
    last_instance: "StubWorkspace | None" = None

    def __init__(
        self,
        tic_id: int,
        sectors: np.ndarray,
        mission: str = "TESS",
        search_radius: int = 10,
        config: Config | None = None,
        trilegal_cache_path: str | None = None,
    ) -> None:
        self.tic_id = tic_id
        self.sectors = sectors
        self.mission = mission
        self.search_radius = search_radius
        self.config = config
        self.trilegal_cache_path = trilegal_cache_path
        self._resolved_target = None
        self.field = _field()
        self.calc_depths_args = None
        self.compute_probs_args = None
        StubWorkspace.last_instance = self

    def fetch_catalog(self) -> StellarField:
        return self.field

    def set_resolved_target(self, target) -> None:
        self._resolved_target = target

    def calc_depths(
        self,
        transit_depth: float,
        pixel_coords_per_sector: list[np.ndarray],
        aperture_pixels_per_sector: list[np.ndarray],
        sigma_psf_px: float = 0.75,
    ) -> None:
        self.calc_depths_args = (
            transit_depth,
            pixel_coords_per_sector,
            aperture_pixels_per_sector,
            sigma_psf_px,
        )

    def compute_probs(
        self,
        light_curve: LightCurve,
        period_days: float,
        scenario_ids: list[ScenarioID] | None = None,
    ) -> ValidationResult:
        self.compute_probs_args = (light_curve, period_days, scenario_ids)
        return ValidationResult(
            target_id=self.tic_id,
            false_positive_probability=0.12,
            nearby_false_positive_probability=0.03,
            scenario_results=[],
        )


def test_run_tess_fpp_orchestrates_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(runner, "ValidationWorkspace", StubWorkspace)
    monkeypatch.setattr(
        runner,
        "resolve_toi_to_tic_ephemeris_depth",
        lambda target, cache_ttl_seconds, disk_cache_dir: ToiResolutionResult(
            status=LookupStatus.OK,
            toi_query=str(target),
            tic_id=12345,
            matched_toi="123.01",
            period_days=5.0,
            t0_btjd=1000.0,
            duration_hours=2.0,
            depth_ppm=2500.0,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_tpf_lightcurve",
        lambda target, config: runner._PreparedApertureLightCurve(
            light_curve_result=_lc_result(),
            aperture_masks=(np.array([[True, False], [False, True]]),),
            tpfs=("fake",),
        ),
    )
    pixel_coords = [np.array([[0.0, 0.0]])]
    aperture_pixels = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    monkeypatch.setattr(
        runner,
        "_derive_sector_geometry",
        lambda tpfs, masks, field, search_radius_px: (pixel_coords, aperture_pixels),
    )

    cfg = FppRunConfig(
        search_radius_px=12,
        sigma_psf_px=0.5,
        scenario_ids=(ScenarioID.TP, ScenarioID.PTP),
    )
    result = run_tess_fpp("TOI-123.01", config=cfg)

    workspace = StubWorkspace.last_instance
    assert workspace is not None
    assert workspace.tic_id == 12345
    assert workspace.sectors.tolist() == [14, 15]
    assert workspace.search_radius == 12
    assert workspace._resolved_target == result.resolved_target
    assert workspace.calc_depths_args is not None
    assert workspace.calc_depths_args[0] == pytest.approx(0.0025)
    assert workspace.calc_depths_args[1] == pixel_coords
    assert workspace.calc_depths_args[2] == aperture_pixels
    assert workspace.calc_depths_args[3] == pytest.approx(0.5)
    assert workspace.compute_probs_args is not None
    assert workspace.compute_probs_args[1] == pytest.approx(5.0)
    assert workspace.compute_probs_args[2] == [ScenarioID.TP, ScenarioID.PTP]
    assert result.validation_result.fpp == pytest.approx(0.12)
    assert result.transit_depth == pytest.approx(0.0025)


def test_run_tess_fpp_requires_manual_inputs_for_tic() -> None:
    with pytest.raises(Exception, match="ephemeris"):
        run_tess_fpp("TIC 12345", config=FppRunConfig(transit_depth=0.01))

    with pytest.raises(ValueError, match="transit_depth"):
        run_tess_fpp(
            "TIC 12345",
            config=FppRunConfig(),
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=1000.0),
        )


class FakeTpf:
    def __init__(self, pipeline_mask: np.ndarray) -> None:
        self.shape = (10,) + pipeline_mask.shape
        self.pipeline_mask = pipeline_mask

    def create_threshold_mask(self, threshold: float) -> np.ndarray:
        return np.array([[False, True], [True, False]])


def test_resolve_aperture_mask_uses_pipeline_for_default() -> None:
    tpf = FakeTpf(np.array([[True, False], [False, False]]))
    mask = runner._resolve_aperture_mask(tpf, ApertureConfig())
    np.testing.assert_array_equal(mask, np.array([[True, False], [False, False]]))


def test_resolve_aperture_mask_falls_back_to_threshold() -> None:
    tpf = FakeTpf(np.array([[False, False], [False, False]]))
    mask = runner._resolve_aperture_mask(tpf, ApertureConfig(mode="default"))
    np.testing.assert_array_equal(mask, np.array([[False, True], [True, False]]))


def test_resolve_aperture_mask_builds_custom_mask() -> None:
    tpf = FakeTpf(np.array([[False, False], [False, False]]))
    mask = runner._resolve_aperture_mask(
        tpf,
        ApertureConfig(mode="custom", custom_pixels=((0, 1), (1, 0))),
    )
    np.testing.assert_array_equal(mask, np.array([[False, True], [True, False]]))
