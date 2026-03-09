"""Tests for triceratops_new.domain.value_objects."""
from __future__ import annotations

import math

import numpy as np
import pytest

from triceratops_new.domain.value_objects import (
    ContrastCurve,
    LimbDarkeningCoeffs,
    StellarParameters,
)


class TestStellarParameters:
    def test_stellar_parameters_frozen(self) -> None:
        sp = StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        )
        with pytest.raises(AttributeError):
            sp.mass_msun = 2.0  # type: ignore[misc]

    def test_from_tic_row_defaults(self) -> None:
        row: dict = {"mass": None, "rad": float("nan"), "Teff": None, "plx": None}
        sp = StellarParameters.from_tic_row(row)
        assert sp.mass_msun == 1.0
        assert sp.radius_rsun == 1.0
        assert sp.teff_k == 5778.0
        assert sp.parallax_mas == 10.0
        assert sp.metallicity_dex == 0.0
        assert math.isfinite(sp.logg)

    def test_from_tic_row_computes_logg(self) -> None:
        from astropy.constants import G as _G
        from astropy.constants import M_sun, R_sun

        Msun = M_sun.cgs.value
        Rsun = R_sun.cgs.value
        G = _G.cgs.value

        row = {"mass": 1.5, "rad": 1.2, "Teff": 6000.0, "plx": 5.0}
        sp = StellarParameters.from_tic_row(row)
        expected_logg = math.log10(G * 1.5 * Msun / (1.2 * Rsun) ** 2)
        assert abs(sp.logg - expected_logg) < 1e-10


class TestContrastCurve:
    @pytest.fixture()
    def cc(self) -> ContrastCurve:
        return ContrastCurve(
            separations_arcsec=np.array([0.1, 0.5, 1.0, 2.0]),
            delta_mags=np.array([1.0, 3.0, 5.0, 7.0]),
            band="K",
        )

    def test_contrast_curve_below_inner_working_angle(self, cc: ContrastCurve) -> None:
        assert cc.max_detectable_delta_mag(0.05) == 0.0

    def test_contrast_curve_above_outer_boundary(self, cc: ContrastCurve) -> None:
        assert cc.max_detectable_delta_mag(5.0) == 7.0

    def test_contrast_curve_interpolation_interior(self, cc: ContrastCurve) -> None:
        val = cc.max_detectable_delta_mag(0.75)
        expected = np.interp(0.75, cc.separations_arcsec, cc.delta_mags)
        assert abs(val - expected) < 1e-10


class TestLimbDarkeningCoeffs:
    def test_ldc_as_ldc_array_shape(self) -> None:
        ldc = LimbDarkeningCoeffs(u1=0.3, u2=0.1, band="TESS")
        arr = ldc.as_ldc_array
        assert arr.shape == (1, 2)
        assert arr[0, 0] == pytest.approx(0.3)
        assert arr[0, 1] == pytest.approx(0.1)
