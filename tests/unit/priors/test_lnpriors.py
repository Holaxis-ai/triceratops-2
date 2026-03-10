"""Unit tests for log-prior functions (P1-007)."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.priors.lnpriors import (
    _compute_companion_rate,
    lnprior_background,
    lnprior_bound_companion,
    lnprior_host_mass_binary,
    lnprior_host_mass_planet,
    lnprior_period_binary,
    lnprior_period_planet,
)


@pytest.mark.unit
class TestHostMassPriors:
    def test_host_mass_planet_returns_zero(self) -> None:
        result = lnprior_host_mass_planet(np.array([1.0, 0.5]))
        assert result == 0.0

    def test_host_mass_binary_returns_zero(self) -> None:
        result = lnprior_host_mass_binary(np.array([1.0, 0.5]))
        assert result == 0.0


@pytest.mark.unit
class TestPeriodPlanet:
    def test_finite_for_typical_period(self) -> None:
        result = lnprior_period_planet(5.0)
        assert np.isfinite(result)

    def test_negative_for_large_period(self) -> None:
        result = lnprior_period_planet(45.0)
        assert result < 0

    def test_flat_priors(self) -> None:
        result = lnprior_period_planet(5.0, flat_priors=True)
        assert np.isfinite(result)


@pytest.mark.unit
class TestPeriodBinary:
    def test_finite_for_typical_period(self) -> None:
        result = lnprior_period_binary(5.0)
        assert np.isfinite(result)

    def test_negative_for_large_period(self) -> None:
        result = lnprior_period_binary(45.0)
        assert result < 0


@pytest.mark.unit
class TestBoundCompanion:
    def test_no_contrast_returns_zeros(self) -> None:
        delta_mags = np.array([1.0, 2.0, 3.0])
        result = lnprior_bound_companion(
            delta_mags=delta_mags,
            separations_arcsec=None,
            contrasts=None,
            primary_mass_msun=1.0,
            parallax_mas=10.0,
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_with_contrast_returns_array(self) -> None:
        delta_mags = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        seps = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        contrasts = np.array([0.5, 2.0, 4.0, 6.0, 8.0])
        result = lnprior_bound_companion(
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
            primary_mass_msun=1.0,
            parallax_mas=10.0,
        )
        assert result.shape == (5,)


@pytest.mark.unit
class TestBackground:
    def test_returns_array(self) -> None:
        delta_mags = np.array([1.0, 2.0, 3.0])
        seps = np.array([0.1, 0.5, 1.0, 2.0])
        contrasts = np.array([0.5, 2.0, 4.0, 6.0])
        result = lnprior_background(
            n_comp=100,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        assert result.shape == (3,)

    def test_returns_neg_inf_for_zero_separation(self) -> None:
        """When delta_mag is below the minimum contrast, separation is the
        inner working angle which may be very small, giving a very negative log-prior."""
        delta_mags = np.array([0.0])
        seps = np.array([0.5, 1.0, 2.0])
        contrasts = np.array([2.0, 4.0, 6.0])
        result = lnprior_background(
            n_comp=100,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        # Very small separation -> very negative log prior
        assert result[0] < -4


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_arrays(self) -> None:
        result = lnprior_background(
            n_comp=100,
            delta_mags=np.array([]),
            separations_arcsec=np.array([0.5, 1.0]),
            contrasts=np.array([2.0, 4.0]),
        )
        assert len(result) == 0

    def test_single_element(self) -> None:
        result = lnprior_period_planet(5.0)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Helpers shared by companion-rate tests
# ---------------------------------------------------------------------------

_SEPS = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
_CONTRASTS = np.array([0.5, 2.0, 4.0, 6.0, 8.0])
# delta_mags that interpolate into the middle of the contrast curve
_DELTA_MAGS = np.array([1.0, 2.0, 3.0, 5.0, 7.0])


@pytest.mark.unit
class TestComputeCompanionRate:
    def test_nan_parallax_fallback_finite(self) -> None:
        # plx=nan should substitute 0.1 without crashing; result must be finite
        # or -inf (log of 0 is allowed), but no NaN.
        result = _compute_companion_rate(
            primary_mass=1.0,
            parallax_mas=np.nan,
            delta_mags=_DELTA_MAGS.copy(),
            separations=_SEPS,
            contrasts=_CONTRASTS,
            include_short_period=False,
        )
        assert not np.any(np.isnan(result))

    def test_nan_parallax_fallback_same_as_explicit_01(self) -> None:
        # Substituted plx=0.1 must produce identical results to explicit plx=0.1
        result_nan = _compute_companion_rate(
            primary_mass=1.0,
            parallax_mas=np.nan,
            delta_mags=_DELTA_MAGS.copy(),
            separations=_SEPS,
            contrasts=_CONTRASTS,
            include_short_period=False,
        )
        result_01 = _compute_companion_rate(
            primary_mass=1.0,
            parallax_mas=0.1,
            delta_mags=_DELTA_MAGS.copy(),
            separations=_SEPS,
            contrasts=_CONTRASTS,
            include_short_period=False,
        )
        np.testing.assert_array_equal(result_nan, result_01)

    def test_low_mass_branch_differs_from_high_mass(self) -> None:
        # M_s < 1.0 applies the scaling factor 0.65*f_comp + 0.35*f_comp*M_act,
        # so its log-prior values should differ from the M_s >= 1.0 branch.
        kwargs = dict(
            parallax_mas=10.0,
            delta_mags=_DELTA_MAGS.copy(),
            separations=_SEPS,
            contrasts=_CONTRASTS,
            include_short_period=True,
        )
        result_low = _compute_companion_rate(primary_mass=0.5, **kwargs)
        result_high = _compute_companion_rate(primary_mass=1.5, **kwargs)
        # At least one finite element should differ between the two branches
        finite = np.isfinite(result_low) & np.isfinite(result_high)
        assert np.any(finite), "Expected some finite values to compare"
        assert not np.allclose(result_low[finite], result_high[finite])

    def test_low_mass_scaling_factor_formula(self) -> None:
        # For M_s < 1.0, the code applies f_act = 0.65*f_comp + 0.35*f_comp*M_act
        # This should give a smaller result than M_s=1.0 when M_act < 1.0.
        # We run with include_short_period=True so more bins are populated.
        kwargs = dict(
            parallax_mas=10.0,
            delta_mags=_DELTA_MAGS.copy(),
            separations=_SEPS,
            contrasts=_CONTRASTS,
            include_short_period=True,
        )
        result_m1 = _compute_companion_rate(primary_mass=1.0, **kwargs)
        # M_act=0.5 => scale = 0.65 + 0.35*0.5 = 0.825 < 1.0
        result_m05 = _compute_companion_rate(primary_mass=0.5, **kwargs)
        finite = np.isfinite(result_m1) & np.isfinite(result_m05)
        assert np.any(finite)
        # ln(0.825*f) = ln(f) + ln(0.825), so the low-mass result is smaller
        np.testing.assert_array_less(result_m05[finite], result_m1[finite])

    def test_period_breakpoints_continuous_high_mass(self) -> None:
        # For M_s >= 1.0 with include_short_period=True, the piecewise function
        # must be roughly continuous at each breakpoint (logP = 1.0, 2.0, 3.4, 5.5, 8.0).
        # We test this by converting logP breakpoints to separations and checking
        # that adjacent values are close.
        from triceratops.config.config import CONST

        M_s = 1.0
        plx = 10.0  # d = 100 pc
        d = 1000 / plx  # AU

        # Convert a logP value to separation in AU: d = (G*M*P^2/(4pi^2))^(1/3)
        def logP_to_sep_au(logP: float) -> float:
            P_sec = (10**logP) * 86400.0
            a_cm = ((CONST.G * M_s * CONST.Msun) / (4 * np.pi**2) * P_sec**2) ** (1 / 3)
            return a_cm / CONST.au

        for logP_break in [1.0, 2.0, 3.4, 5.5, 8.0]:
            sep_break = logP_to_sep_au(logP_break)
            # Separations just below and just above the breakpoint (in AU)
            sep_lo = logP_to_sep_au(logP_break - 0.01)
            sep_hi = logP_to_sep_au(logP_break + 0.01)
            # Convert AU separations to arcsec for the contrast curve
            sep_arcsec_lo = sep_lo / d
            sep_arcsec_hi = sep_hi / d
            sep_arcsec_break = sep_break / d
            # Build single-element contrast curves that map any delta_mag -> this sep
            delta = np.array([3.0])
            seps_cc = np.array([0.0, sep_arcsec_lo, sep_arcsec_break, sep_arcsec_hi, 1e6])
            contrasts_cc = np.array([0.0, 1.0, 2.0, 4.0, 5.0])

            lo_result = _compute_companion_rate(
                primary_mass=M_s,
                parallax_mas=plx,
                delta_mags=delta,
                separations=np.array([0.0, sep_arcsec_lo, 1e6]),
                contrasts=np.array([0.0, 1.0, 5.0]),
                include_short_period=True,
            )
            hi_result = _compute_companion_rate(
                primary_mass=M_s,
                parallax_mas=plx,
                delta_mags=delta,
                separations=np.array([0.0, sep_arcsec_hi, 1e6]),
                contrasts=np.array([0.0, 1.0, 5.0]),
                include_short_period=True,
            )
            # Both should be finite or both -inf; if finite, should be close
            lo_fin = np.isfinite(lo_result[0])
            hi_fin = np.isfinite(hi_result[0])
            if lo_fin and hi_fin:
                assert abs(lo_result[0] - hi_result[0]) < 2.0, (
                    f"Discontinuity at logP={logP_break}: lo={lo_result[0]:.4f}, "
                    f"hi={hi_result[0]:.4f}"
                )

    def test_f_comp_zero_returns_neg_inf(self) -> None:
        # When f_comp <= 0 (e.g. logP < 3.4 for TP variant), log(0) = -inf.
        # Use a very small separation so logP is tiny, forcing the zero branch.
        from triceratops.config.config import CONST

        M_s = 1.5
        plx = 1000.0  # d = 1 pc => tiny seps in AU
        d = 1000 / plx
        # Very small arcsec separation => very small AU => very short period
        tiny_sep_arcsec = 1e-6
        result = _compute_companion_rate(
            primary_mass=M_s,
            parallax_mas=plx,
            delta_mags=np.array([3.0]),
            separations=np.array([0.0, tiny_sep_arcsec, 1.0]),
            contrasts=np.array([0.0, 2.0, 8.0]),
            include_short_period=False,  # TP: short-period excluded
        )
        assert result[0] == -np.inf


@pytest.mark.unit
class TestBackgroundEdgeCases:
    def test_zero_separations_returns_neg_inf(self) -> None:
        # delta_mags below the minimum contrast -> interpolation returns the
        # smallest separation, which is 0.0 -> log(0) = -inf
        delta_mags = np.array([-1.0, -2.0])  # below min contrast
        seps = np.array([0.0, 0.5, 1.0])
        contrasts = np.array([1.0, 3.0, 6.0])
        result = lnprior_background(
            n_comp=100,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        np.testing.assert_array_equal(result, -np.inf)

    def test_mixed_zero_and_positive_separations(self) -> None:
        # First delta_mag maps to sep=0, second maps to a real positive sep
        seps = np.array([0.0, 0.5, 1.0, 2.0])
        contrasts = np.array([1.0, 3.0, 5.0, 8.0])
        delta_mags = np.array([-1.0, 6.0])  # first below min, second > max
        result = lnprior_background(
            n_comp=100,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        assert result[0] == -np.inf
        assert np.isfinite(result[1])

    def test_monotonically_increasing_with_separation(self) -> None:
        # For fixed n_comp, log-prior = log(C * sep^2) is strictly increasing in sep.
        # Map linearly-spaced delta_mags to monotonically increasing separations.
        seps = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        contrasts = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        delta_mags = np.array([1.5, 2.5, 4.0, 6.5])  # increasing -> increasing seps
        result = lnprior_background(
            n_comp=50,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        assert np.all(np.diff(result) > 0), "Expected monotonically increasing log-prior"

    def test_n_comp_1_no_crash(self) -> None:
        delta_mags = np.array([3.0, 5.0])
        seps = np.array([0.1, 0.5, 1.0])
        contrasts = np.array([1.0, 3.0, 6.0])
        result = lnprior_background(
            n_comp=1,
            delta_mags=delta_mags,
            separations_arcsec=seps,
            contrasts=contrasts,
        )
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
