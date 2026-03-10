"""Tier 2 tests — provider contract tests for raw sources (lightkurve, no network)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
import lightkurve as lk

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.raw import RawLightCurveData
from triceratops.lightcurve.sources.array import ArrayRawSource
from triceratops.lightcurve.sources.lightkurve import LightkurveRawSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_lk_lc_collection() -> lk.LightCurveCollection:
    """Build a LightCurveCollection from numpy arrays — no MAST required."""
    n = 1000
    t0_btjd = 1468.0
    time_btjd = np.linspace(t0_btjd, t0_btjd + 7.0, n)
    rng = np.random.default_rng(42)
    flux = np.ones(n) + rng.normal(0, 3e-4, n)
    flux_err = np.full(n, 3e-4)

    time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
    single_lc = lk.LightCurve(
        time=time_obj,
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
    )
    single_lc.meta["SECTOR"] = 14
    single_lc.meta["EXPTIME"] = 120.0
    single_lc.meta["TIMEDEL"] = 120.0 / 86400.0  # days

    return lk.LightCurveCollection([single_lc])


# ---------------------------------------------------------------------------
# LightkurveRawSource contract tests
# ---------------------------------------------------------------------------


class TestLightkurveRawSource:
    def test_contract(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """Inject _override_collection, verify RawLightCurveData contract."""
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert isinstance(raw, RawLightCurveData)
        assert len(raw.time_btjd) > 0
        assert raw.flux.shape == raw.time_btjd.shape
        assert raw.flux_err.shape == raw.time_btjd.shape
        assert raw.cadence in ("20sec", "2min", "10min", "30min")

    def test_time_is_btjd_range(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """Verify returned time is in BTJD range, not full JD."""
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        # BTJD values should be ~1468, not ~2459468
        assert raw.time_btjd.min() < 10_000
        assert raw.time_btjd.max() < 10_000

    def test_flux_normalised(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """After stitch(), flux median should be approximately 1.0."""
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert abs(float(np.median(raw.flux)) - 1.0) < 0.05

    def test_all_finite(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """All arrays must be finite (NaNs removed by source)."""
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert np.all(np.isfinite(raw.time_btjd))
        assert np.all(np.isfinite(raw.flux))
        assert np.all(np.isfinite(raw.flux_err))

    def test_time_monotonically_increasing(
        self, synthetic_lk_lc_collection: lk.LightCurveCollection
    ) -> None:
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert np.all(np.diff(raw.time_btjd) > 0)

    def test_sectors_extracted(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert isinstance(raw.sectors, tuple)
        assert len(raw.sectors) >= 1
        assert 14 in raw.sectors

    def test_target_id_preserved(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        source = LightkurveRawSource(
            tic_id=99999,
            _override_collection=synthetic_lk_lc_collection,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert raw.target_id == 99999

    def test_multi_sector_collection(self) -> None:
        """Two-sector collection should stitch into a single RawLightCurveData."""
        rng = np.random.default_rng(123)
        lcs = []
        for sector, start in [(14, 1468.0), (15, 1496.0)]:
            n = 500
            t = np.linspace(start, start + 7.0, n)
            time_obj = Time(t + 2_457_000.0, format="jd", scale="tdb")
            flux = np.ones(n) + rng.normal(0, 3e-4, n)
            flux_err = np.full(n, 3e-4)
            single = lk.LightCurve(
                time=time_obj,
                flux=flux * u.dimensionless_unscaled,
                flux_err=flux_err * u.dimensionless_unscaled,
            )
            single.meta["SECTOR"] = sector
            single.meta["EXPTIME"] = 120.0
            single.meta["TIMEDEL"] = 120.0 / 86400.0
            lcs.append(single)

        coll = lk.LightCurveCollection(lcs)
        source = LightkurveRawSource(tic_id=99999, _override_collection=coll)
        raw = source.fetch_raw(LightCurveConfig(sectors="all"))
        assert isinstance(raw, RawLightCurveData)
        assert len(raw.time_btjd) > 500  # both sectors stitched
        assert 14 in raw.sectors
        assert 15 in raw.sectors


# ---------------------------------------------------------------------------
# ArrayRawSource tests
# ---------------------------------------------------------------------------


class TestArrayRawSource:
    def test_roundtrip(self) -> None:
        """ArrayRawSource wraps arrays into a valid RawLightCurveData."""
        n = 200
        t = np.linspace(1468.0, 1475.0, n)
        flux = np.ones(n)
        flux_err = np.full(n, 3e-4)

        source = ArrayRawSource(
            time=t,
            flux=flux,
            flux_err=flux_err,
            sectors=(14,),
            cadence="2min",
            exptime_seconds=120.0,
            target_id=12345,
        )
        raw = source.fetch_raw(LightCurveConfig())
        assert isinstance(raw, RawLightCurveData)
        assert len(raw.time_btjd) == n
        assert raw.flux.shape == (n,)
        assert raw.flux_err.shape == (n,)
        assert raw.sectors == (14,)
        assert raw.cadence == "2min"
        assert raw.exptime_seconds == 120.0
        assert raw.target_id == 12345
        assert raw.warnings == ()

    def test_normalisation_warning(self) -> None:
        """Flux with median far from 1.0 should produce a warning."""
        n = 200
        t = np.linspace(1468.0, 1475.0, n)
        flux = np.full(n, 1.1)  # median = 1.1, >5% from 1.0
        flux_err = np.full(n, 3e-4)

        source = ArrayRawSource(
            time=t,
            flux=flux,
            flux_err=flux_err,
            sectors=(14,),
            cadence="2min",
            exptime_seconds=120.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            raw = source.fetch_raw(LightCurveConfig())

        # Warning should be in both Python warnings and RawLightCurveData.warnings
        assert len(raw.warnings) == 1
        assert "median" in raw.warnings[0].lower() or "1.1" in raw.warnings[0]
        assert len(w) >= 1
        assert any("median" in str(warning.message).lower() for warning in w)

    def test_no_warning_when_normalised(self) -> None:
        """Flux with median ~1.0 should not produce normalisation warnings."""
        n = 200
        t = np.linspace(1468.0, 1475.0, n)
        flux = np.ones(n)
        flux_err = np.full(n, 3e-4)

        source = ArrayRawSource(
            time=t,
            flux=flux,
            flux_err=flux_err,
            sectors=(14,),
            cadence="2min",
            exptime_seconds=120.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            raw = source.fetch_raw(LightCurveConfig())

        assert raw.warnings == ()
        normalisation_warnings = [x for x in w if "median" in str(x.message).lower()]
        assert len(normalisation_warnings) == 0

    def test_config_parameter_ignored(self) -> None:
        """ArrayRawSource should accept any config without error."""
        n = 100
        t = np.linspace(1468.0, 1470.0, n)
        source = ArrayRawSource(
            time=t,
            flux=np.ones(n),
            flux_err=np.full(n, 3e-4),
            sectors=(14,),
            cadence="2min",
            exptime_seconds=120.0,
        )
        config = LightCurveConfig(
            cadence="30min",
            sigma_clip=3.0,
            detrend_method="flatten",
        )
        raw = source.fetch_raw(config)
        # Config does not affect ArrayRawSource — cadence comes from constructor
        assert raw.cadence == "2min"
