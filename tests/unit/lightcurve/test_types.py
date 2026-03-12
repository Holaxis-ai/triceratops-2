"""Tier 1 tests for lightcurve pure types: Ephemeris, ResolvedTarget,
LightCurveConfig, LightCurvePreparationResult, and the error hierarchy.

No lightkurve imports, no network calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import (
    Ephemeris,
    EphemerisResolver,
    ResolvedTarget,
)
from triceratops.lightcurve.errors import (
    DownloadTimeoutError,
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurveError,
    LightCurveNotFoundError,
    LightCurvePreparationError,
    SectorNotAvailableError,
)
from triceratops.lightcurve.result import LightCurvePreparationResult


# ═══════════════════════════════════════════════════════════════════
# Ephemeris
# ═══════════════════════════════════════════════════════════════════


class TestEphemeris:
    def test_basic_construction(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        assert eph.period_days == 3.5
        assert eph.t0_btjd == 1468.28
        assert eph.duration_hours is None
        assert eph.source == "manual"
        assert eph.warnings == ()

    def test_frozen(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        with pytest.raises(AttributeError):
            eph.period_days = 1.0  # type: ignore[misc]

    def test_with_all_fields(self):
        eph = Ephemeris(
            period_days=3.5,
            t0_btjd=1468.28,
            duration_hours=2.1,
            source="exofop",
            warnings=("low S/N",),
        )
        assert eph.duration_hours == 2.1
        assert eph.source == "exofop"
        assert eph.warnings == ("low S/N",)


# ═══════════════════════════════════════════════════════════════════
# ResolvedTarget
# ═══════════════════════════════════════════════════════════════════


class TestResolvedTarget:
    def test_basic_construction(self):
        rt = ResolvedTarget(target_ref="395.01", tic_id=395171208, source="exofop")
        assert rt.target_ref == "395.01"
        assert rt.tic_id == 395171208
        assert rt.ephemeris is None
        assert rt.source == "exofop"

    def test_with_ephemeris(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        rt = ResolvedTarget(
            target_ref="395.01", tic_id=395171208, ephemeris=eph, source="exofop"
        )
        assert rt.ephemeris is eph

    def test_frozen(self):
        rt = ResolvedTarget(target_ref="395.01", tic_id=395171208, source="exofop")
        with pytest.raises(AttributeError):
            rt.tic_id = 0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# LightCurveConfig validation
# ═══════════════════════════════════════════════════════════════════


class TestLightCurveConfig:
    def test_default_config_is_valid(self):
        cfg = LightCurveConfig()
        assert cfg.cadence == "auto"
        assert cfg.sigma_clip is None
        assert cfg.detrend_method == "none"
        assert cfg.supersampling_rate == 20

    def test_even_window_length_rejected(self):
        with pytest.raises(ValueError, match="odd integer"):
            LightCurveConfig(flatten_window_length=400)

    def test_window_length_too_small_rejected(self):
        with pytest.raises(ValueError, match="odd integer"):
            LightCurveConfig(flatten_window_length=1)

    def test_polyorder_out_of_range(self):
        with pytest.raises(ValueError, match="between 1 and 5"):
            LightCurveConfig(flatten_polyorder=0)
        with pytest.raises(ValueError, match="between 1 and 5"):
            LightCurveConfig(flatten_polyorder=6)

    def test_phase_window_factor_must_be_ge_one(self):
        with pytest.raises(ValueError, match="phase_window_factor"):
            LightCurveConfig(phase_window_factor=0.5)

    def test_supersampling_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="supersampling_rate"):
            LightCurveConfig(supersampling_rate=0)

    def test_negative_sigma_clip_rejected(self):
        with pytest.raises(ValueError, match="sigma_clip"):
            LightCurveConfig(sigma_clip=-1.0)

    def test_none_sigma_clip_is_allowed(self):
        cfg = LightCurveConfig(sigma_clip=None)
        assert cfg.sigma_clip is None

    def test_negative_cadence_days_override_rejected(self):
        with pytest.raises(ValueError, match="cadence_days_override"):
            LightCurveConfig(cadence_days_override=-0.001)

    def test_frozen(self):
        cfg = LightCurveConfig()
        with pytest.raises(AttributeError):
            cfg.cadence = "2min"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# LightCurvePreparationResult
# ═══════════════════════════════════════════════════════════════════


class TestLightCurvePreparationResult:
    def test_construction(self):
        from triceratops.domain.entities import LightCurve

        lc = LightCurve(
            time_days=np.linspace(-0.05, 0.05, 100),
            flux=np.ones(100),
            flux_err=3e-4,
            cadence_days=120 / 86400,
        )
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        result = LightCurvePreparationResult(
            light_curve=lc,
            ephemeris=eph,
            sectors_used=(14,),
            cadence_used="2min",
        )
        assert result.light_curve is lc
        assert result.ephemeris is eph
        assert result.sectors_used == (14,)
        assert result.cadence_used == "2min"
        assert result.warnings == []


# ═══════════════════════════════════════════════════════════════════
# Error hierarchy
# ═══════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    def test_all_errors_inherit_from_lightcurve_error(self):
        assert issubclass(LightCurveNotFoundError, LightCurveError)
        assert issubclass(SectorNotAvailableError, LightCurveError)
        assert issubclass(EphemerisRequiredError, LightCurveError)
        assert issubclass(DownloadTimeoutError, LightCurveError)
        assert issubclass(LightCurveEmptyError, LightCurveError)
        assert issubclass(LightCurvePreparationError, LightCurveError)

    def test_download_timeout_retryable(self):
        err = DownloadTimeoutError("failed", retryable=True)
        assert err.retryable is True

    def test_download_timeout_not_retryable(self):
        err = DownloadTimeoutError("permanent", retryable=False)
        assert err.retryable is False

    def test_lightcurve_error_is_exception(self):
        assert issubclass(LightCurveError, Exception)


# ═══════════════════════════════════════════════════════════════════
# EphemerisResolver protocol
# ═══════════════════════════════════════════════════════════════════


class TestEphemerisResolverProtocol:
    def test_custom_resolver_satisfies_protocol(self):
        class StubResolver:
            def resolve(self, target: str) -> ResolvedTarget:
                return ResolvedTarget(
                    target_ref=target, tic_id=12345, source="stub"
                )

        resolver = StubResolver()
        assert isinstance(resolver, EphemerisResolver)
