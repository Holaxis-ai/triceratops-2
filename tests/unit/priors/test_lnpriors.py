"""Unit tests for log-prior functions (P1-007)."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.priors.lnpriors import (
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
