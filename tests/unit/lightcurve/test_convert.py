"""Tier 2 tests — convert_folded_to_domain() contract tests (lightkurve, no network)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
import lightkurve as lk

from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.convert import convert_folded_to_domain
from triceratops.lightcurve.errors import LightCurveEmptyError, LightCurvePreparationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_folded_lk_lc() -> lk.LightCurve:
    """Build a folded LightCurve from numpy arrays — has .phase attribute."""
    n = 800
    t0_btjd = 1468.0
    period_days = 3.5
    time_btjd = np.linspace(t0_btjd, t0_btjd + 14.0, n)

    rng = np.random.default_rng(42)
    flux = np.ones(n) + rng.normal(0, 3e-4, n)
    flux_err = np.full(n, 3e-4)

    time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
    raw_lc = lk.LightCurve(
        time=time_obj,
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
    )
    raw_lc.meta["SECTOR"] = 14
    raw_lc.meta["EXPTIME"] = 120.0

    folded = raw_lc.fold(
        period=period_days * u.day,
        epoch_time=Time(t0_btjd + 2_457_000.0, format="jd", scale="tdb"),
    )
    return folded


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvertFoldedToDomain:
    def test_basic_conversion(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """Convert folded lightkurve object to domain LightCurve."""
        lc = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min")
        assert isinstance(lc, LightCurve)
        assert np.all(np.isfinite(lc.flux))
        assert isinstance(lc.flux_err, float)
        assert lc.flux_err > 0
        assert lc.cadence_days == pytest.approx(120 / 86400)

    def test_uses_phase_not_time(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """Output time_days should be in phase-day range, not BTJD range.

        For a folded lk.FoldedLightCurve, .phase.value gives days-from-transit
        (small values near 0), NOT full BTJD values (large, ~1468+).
        """
        lc = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min")
        # Phase values should be small (within +/- period/2 = +/- 1.75)
        assert np.abs(lc.time_days).max() < 5.0
        # They should NOT be in BTJD range
        assert lc.time_days.min() > -100.0
        assert lc.time_days.max() < 100.0

    def test_scalar_flux_err(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """flux_err must be a scalar float, not an array."""
        lc = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min")
        assert isinstance(lc.flux_err, float)
        assert not hasattr(lc.flux_err, "__len__")

    def test_cadence_days_from_string(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """Cadence string should map correctly to cadence_days."""
        lc_2min = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min")
        assert lc_2min.cadence_days == pytest.approx(120 / 86400)

        lc_20sec = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="20sec")
        assert lc_20sec.cadence_days == pytest.approx(20 / 86400)

        lc_30min = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="30min")
        assert lc_30min.cadence_days == pytest.approx(1800 / 86400)

    def test_cadence_days_override(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """Config's cadence_days_override should take precedence."""
        config = LightCurveConfig(cadence_days_override=0.005)
        lc = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min", config=config)
        assert lc.cadence_days == pytest.approx(0.005)

    def test_supersampling_rate(self, synthetic_folded_lk_lc: lk.LightCurve) -> None:
        """Supersampling rate should come from config."""
        config = LightCurveConfig(supersampling_rate=10)
        lc = convert_folded_to_domain(synthetic_folded_lk_lc, cadence="2min", config=config)
        assert lc.supersampling_rate == 10

    def test_nan_removal(self) -> None:
        """NaN values should be swept out during conversion."""
        n = 200
        t0_btjd = 1468.0
        period_days = 3.5
        time_btjd = np.linspace(t0_btjd, t0_btjd + 7.0, n)

        flux = np.ones(n)
        flux[50] = np.nan  # inject NaN
        flux_err = np.full(n, 3e-4)
        flux_err[100] = np.nan  # inject NaN

        time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
        raw_lc = lk.LightCurve(
            time=time_obj,
            flux=flux * u.dimensionless_unscaled,
            flux_err=flux_err * u.dimensionless_unscaled,
        )
        raw_lc.meta["EXPTIME"] = 120.0
        folded = raw_lc.fold(
            period=period_days * u.day,
            epoch_time=Time(t0_btjd + 2_457_000.0, format="jd", scale="tdb"),
        )

        lc = convert_folded_to_domain(folded, cadence="2min")
        assert np.all(np.isfinite(lc.flux))
        assert len(lc.flux) == n - 2  # two NaN rows removed

    def test_empty_after_nan_sweep_raises(self) -> None:
        """All-NaN flux should raise LightCurveEmptyError."""
        n = 50
        t0_btjd = 1468.0
        period_days = 3.5
        time_btjd = np.linspace(t0_btjd, t0_btjd + 3.0, n)
        flux = np.full(n, np.nan)
        flux_err = np.full(n, 3e-4)

        time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
        raw_lc = lk.LightCurve(
            time=time_obj,
            flux=flux * u.dimensionless_unscaled,
            flux_err=flux_err * u.dimensionless_unscaled,
        )
        raw_lc.meta["EXPTIME"] = 120.0
        folded = raw_lc.fold(
            period=period_days * u.day,
            epoch_time=Time(t0_btjd + 2_457_000.0, format="jd", scale="tdb"),
        )

        with pytest.raises(LightCurveEmptyError):
            convert_folded_to_domain(folded, cadence="2min")

    def test_auto_cadence_from_meta(self) -> None:
        """When cadence='auto', cadence_days should come from EXPTIME metadata."""
        n = 200
        t0_btjd = 1468.0
        period_days = 3.5
        time_btjd = np.linspace(t0_btjd, t0_btjd + 7.0, n)
        flux = np.ones(n)
        flux_err = np.full(n, 3e-4)

        time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
        raw_lc = lk.LightCurve(
            time=time_obj,
            flux=flux * u.dimensionless_unscaled,
            flux_err=flux_err * u.dimensionless_unscaled,
        )
        raw_lc.meta["EXPTIME"] = 600.0  # 10-min cadence
        folded = raw_lc.fold(
            period=period_days * u.day,
            epoch_time=Time(t0_btjd + 2_457_000.0, format="jd", scale="tdb"),
        )

        lc = convert_folded_to_domain(folded, cadence="auto")
        assert lc.cadence_days == pytest.approx(600.0 / 86400)
