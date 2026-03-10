"""Tests for triceratops.domain.entities."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.entities import LightCurve, Star, StellarField


def _make_star(tic_id: int = 12345678, **kwargs) -> Star:
    defaults = dict(
        tic_id=tic_id, ra_deg=83.82, dec_deg=-5.39,
        tmag=10.5, jmag=9.8, hmag=9.5, kmag=9.4,
        bmag=11.2, vmag=10.8,
    )
    defaults.update(kwargs)
    return Star(**defaults)  # type: ignore[arg-type]


def _make_field() -> StellarField:
    target = _make_star(tic_id=100)
    neighbor = _make_star(tic_id=101, separation_arcsec=2.0)
    return StellarField(
        target_id=100, mission="TESS", search_radius_pixels=10,
        stars=[target, neighbor],
    )


class TestStellarField:
    def test_stellar_field_target_is_first_star(self) -> None:
        field = _make_field()
        assert field.target.tic_id == 100

    def test_stellar_field_neighbors(self) -> None:
        field = _make_field()
        assert len(field.neighbors) == 1
        assert field.neighbors[0].tic_id == 101


class TestStellarFieldAddNeighbor:
    def test_add_neighbor_appends(self) -> None:
        field = _make_field()
        field.add_neighbor(_make_star(tic_id=999))
        assert len(field.stars) == 3
        assert field.stars[-1].tic_id == 999

    def test_target_remains_at_index_zero(self) -> None:
        field = _make_field()
        field.add_neighbor(_make_star(tic_id=999))
        assert field.target.tic_id == 100

    def test_duplicate_tic_raises(self) -> None:
        field = _make_field()
        with pytest.raises(ValueError, match="already exists"):
            field.add_neighbor(_make_star(tic_id=101))  # 101 already in field

    def test_duplicate_target_tic_raises(self) -> None:
        field = _make_field()
        with pytest.raises(ValueError, match="already exists"):
            field.add_neighbor(_make_star(tic_id=100))  # target TIC ID


class TestStellarFieldRemoveNeighbor:
    def test_remove_neighbor_succeeds(self) -> None:
        field = _make_field()
        field.remove_neighbor(101)
        assert len(field.stars) == 1
        assert field.target.tic_id == 100

    def test_remove_nonexistent_raises(self) -> None:
        field = _make_field()
        with pytest.raises(ValueError, match="not found"):
            field.remove_neighbor(9999)

    def test_remove_target_raises(self) -> None:
        field = _make_field()
        with pytest.raises(ValueError, match="Cannot remove the target"):
            field.remove_neighbor(100)

    def test_target_invariant_preserved_after_remove(self) -> None:
        field = _make_field()
        field.add_neighbor(_make_star(tic_id=102))
        field.remove_neighbor(101)
        assert field.target.tic_id == 100
        assert len(field.stars) == 2


class TestStellarFieldUpdateStar:
    def test_update_direct_attribute(self) -> None:
        field = _make_field()
        field.update_star(100, tmag=15.0)
        assert field.target.tmag == pytest.approx(15.0)

    def test_update_target_attribute(self) -> None:
        """Target attributes can be updated (just not removed)."""
        field = _make_field()
        field.update_star(100, tmag=9.0)
        assert field.target.tmag == pytest.approx(9.0)

    def test_update_alias_teff(self) -> None:
        from triceratops.domain.value_objects import StellarParameters
        sp = StellarParameters(mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                               logg=4.4, metallicity_dex=0.0, parallax_mas=10.0)
        star = _make_star(tic_id=100, stellar_params=sp)
        field = StellarField(target_id=100, mission="TESS", search_radius_pixels=10,
                             stars=[star])
        field.update_star(100, Teff=6000.0)
        assert field.target.stellar_params.teff_k == pytest.approx(6000.0)

    def test_update_alias_without_stellar_params_raises_typeerror(self) -> None:
        field = _make_field()  # stars have no stellar_params
        with pytest.raises(TypeError, match="stellar_params is None"):
            field.update_star(100, Teff=6000.0)

    def test_update_unknown_attribute_raises(self) -> None:
        field = _make_field()
        with pytest.raises(AttributeError, match="no attribute"):
            field.update_star(100, nonexistent=42)

    def test_update_nonexistent_star_raises(self) -> None:
        field = _make_field()
        with pytest.raises(ValueError, match="not found"):
            field.update_star(9999, tmag=10.0)


class TestStellarFieldValidate:
    def test_valid_field_passes(self) -> None:
        field = _make_field()
        field.validate()  # must not raise

    def test_empty_stars_raises(self) -> None:
        field = StellarField(target_id=100, mission="TESS",
                             search_radius_pixels=10, stars=[])
        with pytest.raises(ValueError, match="empty"):
            field.validate()

    def test_wrong_target_at_index_zero_raises(self) -> None:
        neighbor = _make_star(tic_id=101)
        target = _make_star(tic_id=100)
        field = StellarField(target_id=100, mission="TESS",
                             search_radius_pixels=10, stars=[neighbor, target])
        with pytest.raises(ValueError, match=r"stars\[0\]"):
            field.validate()

    def test_duplicate_tic_id_raises(self) -> None:
        target = _make_star(tic_id=100)
        dup = _make_star(tic_id=101)
        dup2 = _make_star(tic_id=101)
        field = StellarField(target_id=100, mission="TESS",
                             search_radius_pixels=10, stars=[target, dup, dup2])
        with pytest.raises(ValueError, match="duplicate"):
            field.validate()

    def test_validate_message_includes_target_id(self) -> None:
        neighbor = _make_star(tic_id=999)
        field = StellarField(target_id=100, mission="TESS",
                             search_radius_pixels=10, stars=[neighbor])
        with pytest.raises(ValueError) as exc_info:
            field.validate()
        assert "100" in str(exc_info.value)


class TestLightCurve:
    @pytest.fixture()
    def lc(self) -> LightCurve:
        time = np.linspace(-0.2, 0.2, 100)
        flux = np.ones(100) * 0.99
        return LightCurve(time_days=time, flux=flux, flux_err=0.001)

    def test_light_curve_with_renorm_flux_ratio_1(self, lc: LightCurve) -> None:
        renormed = lc.with_renorm(1.0)
        np.testing.assert_array_almost_equal(renormed.flux, lc.flux)
        assert renormed.flux_err == pytest.approx(lc.flux_err)

    def test_light_curve_with_renorm_scales_flux_and_err(self, lc: LightCurve) -> None:
        renormed = lc.with_renorm(0.5)
        # flux_renormed = (0.99 - 0.5) / 0.5 = 0.98
        expected_flux = (0.99 - 0.5) / 0.5
        np.testing.assert_array_almost_equal(renormed.flux, expected_flux)
        assert renormed.flux_err == pytest.approx(0.001 / 0.5)


class TestStar:
    def test_star_mag_for_band_all_bands(self) -> None:
        star = _make_star(gmag=12.0, rmag=11.5, imag=11.0, zmag=10.5)
        expected = {
            "TESS": 10.5, "J": 9.8, "H": 9.5, "K": 9.4,
            "B": 11.2, "V": 10.8, "g": 12.0, "r": 11.5,
            "i": 11.0, "z": 10.5,
        }
        for band, mag in expected.items():
            assert star.mag_for_band(band) == pytest.approx(mag), f"Failed for band {band}"

    def test_star_mag_for_band_unknown(self) -> None:
        star = _make_star()
        assert star.mag_for_band("U") is None
