"""Direct unit tests for triceratops/scenarios/_companion_helpers.py.

These helpers were previously tested only indirectly via scenario integration
tests.  This module gives them focused, fast coverage independent of the
scenario classes that use them.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from triceratops.scenarios._companion_helpers import (
    _compute_companion_properties,
    _flux_ratio_in_band,
    _load_molusc_qs,
)


# ---------------------------------------------------------------------------
# _flux_ratio_in_band
# ---------------------------------------------------------------------------

class TestFluxRatioInBand:
    """Tests for _flux_ratio_in_band(masses, primary_mass_msun, band)."""

    def test_return_in_open_unit_interval(self) -> None:
        """For physical masses the result must lie strictly in (0, 1)."""
        masses = np.array([0.5, 0.8, 1.0, 1.2])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_equal_mass_gives_half(self) -> None:
        """companion mass == primary mass → flux ratio == 0.5."""
        masses = np.array([1.0])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_fainter_companion_below_half(self) -> None:
        """companion mass < primary mass → ratio < 0.5."""
        masses = np.array([0.3])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        assert result[0] < 0.5

    def test_brighter_companion_above_half(self) -> None:
        """companion mass > primary mass → ratio > 0.5."""
        masses = np.array([2.0])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        assert result[0] > 0.5

    def test_tess_and_j_give_different_results(self) -> None:
        """Different filter bands produce different flux ratios for the same masses."""
        masses = np.array([0.5, 1.0, 1.5])
        tess = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        j_band = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="J")
        assert not np.allclose(tess, j_band), (
            "TESS and J-band flux ratios should differ"
        )

    def test_output_shape_matches_input(self) -> None:
        masses = np.array([0.4, 0.8, 1.2, 1.6])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        assert result.shape == masses.shape

    def test_monotone_increasing_with_mass(self) -> None:
        """More massive companion → higher flux ratio (more luminous)."""
        masses = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
        result = _flux_ratio_in_band(masses, primary_mass_msun=1.0, band="TESS")
        assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# _compute_companion_properties
# ---------------------------------------------------------------------------

class TestComputeCompanionProperties:
    """Tests for _compute_companion_properties(qs_comp, M_s, R_s, Teff, n, filt)."""

    def _solar_params(self):
        return dict(M_s=1.0, R_s=1.0, Teff=5778.0)

    def test_returns_four_arrays(self) -> None:
        n = 10
        qs = np.full(n, 0.5)
        result = _compute_companion_properties(qs, n=n, **self._solar_params())
        assert len(result) == 4
        masses, radii, teffs, flux_ratios = result
        for arr in (masses, radii, teffs, flux_ratios):
            assert arr.shape == (n,)

    def test_positive_radii_for_valid_inputs(self) -> None:
        n = 20
        qs = np.random.default_rng(1).uniform(0.1, 1.0, n)
        _, radii, _, _ = _compute_companion_properties(qs, n=n, **self._solar_params())
        assert np.all(radii > 0.0)

    def test_unit_mass_ratio_gives_equal_masses(self) -> None:
        """mass_ratio = 1.0 → companion mass equals primary mass."""
        n = 5
        M_s = 1.0
        qs = np.ones(n)
        masses, _, _, _ = _compute_companion_properties(
            qs, M_s=M_s, R_s=1.0, Teff=5778.0, n=n,
        )
        np.testing.assert_allclose(masses, np.full(n, M_s), rtol=1e-10)

    def test_flux_ratios_in_unit_interval(self) -> None:
        n = 30
        qs = np.random.default_rng(2).uniform(0.1, 1.0, n)
        _, _, _, flux_ratios = _compute_companion_properties(
            qs, n=n, **self._solar_params(),
        )
        assert np.all(flux_ratios > 0.0)
        assert np.all(flux_ratios < 1.0)

    def test_zero_mass_ratio_does_not_crash(self) -> None:
        """mass_ratio=0 produces zero companion masses; must not raise."""
        n = 4
        qs = np.zeros(n)
        masses, radii, _, _ = _compute_companion_properties(
            qs, n=n, **self._solar_params(),
        )
        np.testing.assert_array_equal(masses, np.zeros(n))

    def test_different_bands_give_different_flux_ratios(self) -> None:
        """filt='TESS' vs filt='J' should produce different flux ratios."""
        n = 10
        qs = np.full(n, 0.5)
        params = self._solar_params()
        _, _, _, fr_tess = _compute_companion_properties(qs, n=n, filt="TESS", **params)
        _, _, _, fr_j = _compute_companion_properties(qs, n=n, filt="J", **params)
        assert not np.allclose(fr_tess, fr_j)

    def test_masses_scale_with_primary_mass(self) -> None:
        """companion mass = mass_ratio * primary_mass."""
        n = 5
        qs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        M_s = 1.5
        masses, _, _, _ = _compute_companion_properties(
            qs, M_s=M_s, R_s=1.3, Teff=6000.0, n=n,
        )
        np.testing.assert_allclose(masses, qs * M_s, rtol=1e-10)


# ---------------------------------------------------------------------------
# _load_molusc_qs
# ---------------------------------------------------------------------------

class TestLoadMoluscQs:
    """Tests for _load_molusc_qs(molusc_file, n, primary_mass)."""

    _HEADER = "semi-major axis(AU),eccentricity,mass ratio"

    def _write_csv(self, tmp_path, rows: list[tuple[float, float, float]]) -> str:
        """Write a minimal MOLUSC CSV and return the path as a string."""
        path = tmp_path / "molusc.csv"
        lines = [self._HEADER]
        for a, e, q in rows:
            lines.append(f"{a},{e},{q}")
        path.write_text("\n".join(lines))
        return str(path)

    def test_valid_file_returns_array_of_length_n(self, tmp_path) -> None:
        """A valid MOLUSC file with enough wide rows returns an array of length n."""
        # a*(1-e) must be > 10 AU to pass the filter
        rows = [(20.0, 0.0, 0.4), (30.0, 0.1, 0.5), (50.0, 0.0, 0.6)]
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=2, primary_mass=1.0)
        assert result.shape == (2,)

    def test_pads_with_zeros_when_shorter_than_n(self, tmp_path) -> None:
        """When fewer rows survive filtering than n, result is padded with zeros."""
        rows = [(20.0, 0.0, 0.4)]  # only one wide-separation row
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=5, primary_mass=1.0)
        assert result.shape == (5,)
        assert result[1] == pytest.approx(0.0)

    def test_truncates_when_longer_than_n(self, tmp_path) -> None:
        """When more rows survive filtering than n, result is truncated to n."""
        rows = [(a, 0.0, 0.5) for a in [20, 30, 40, 50, 60]]
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=3, primary_mass=1.0)
        assert result.shape == (3,)

    def test_filters_out_short_separation_rows(self, tmp_path) -> None:
        """Rows where a*(1-e) <= 10 must be excluded."""
        # Only the 50 AU row is wide enough; the 5 AU row is excluded.
        rows = [(5.0, 0.0, 0.3), (50.0, 0.0, 0.7)]
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=1, primary_mass=1.0)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.7)

    def test_file_not_found_raises(self, tmp_path) -> None:
        """A missing file must raise (FileNotFoundError or similar)."""
        missing = str(tmp_path / "does_not_exist.csv")
        with pytest.raises(Exception):
            _load_molusc_qs(missing, n=5, primary_mass=1.0)

    def test_minimum_mass_ratio_enforced(self, tmp_path) -> None:
        """Rows with q < 0.1/primary_mass are clamped to 0.1/primary_mass."""
        primary_mass = 1.0
        min_q = 0.1 / primary_mass
        # Set a very small mass ratio that should get clamped.
        rows = [(20.0, 0.0, 0.001)]
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=1, primary_mass=primary_mass)
        assert result[0] == pytest.approx(min_q)

    def test_empty_file_after_filtering_pads_zeros(self, tmp_path) -> None:
        """When all rows are filtered out, result is all zeros of length n."""
        # All rows have a*(1-e) = 2 <= 10, so none survive.
        rows = [(2.0, 0.0, 0.5), (3.0, 0.0, 0.6)]
        path = self._write_csv(tmp_path, rows)
        result = _load_molusc_qs(path, n=4, primary_mass=1.0)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, np.zeros(4))
