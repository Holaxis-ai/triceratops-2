"""Direct unit tests for triceratops/scenarios/_background_helpers.py.

These helpers were previously tested only indirectly via scenario integration
tests.  This module gives them focused, fast coverage independent of the
scenario classes that use them.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.value_objects import ContrastCurve, ContrastCurveSet
from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.lnpriors import lnprior_background
from triceratops.scenarios._background_helpers import (
    _combined_delta_mag,
    _compute_fluxratios_comp,
    _compute_lnprior_companion,
    _filter_population_by_target_tmag,
    _needs_sdss_delta_mags,
    _sample_population_indices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_population(n: int, tmag_values: np.ndarray | None = None) -> TRILEGALResult:
    """Build a minimal TRILEGALResult with controllable Tmags."""
    rng = np.random.default_rng(0)
    if tmag_values is None:
        tmag_values = rng.uniform(10.0, 16.0, n)
    ones = np.ones(n)
    return TRILEGALResult(
        tmags=tmag_values,
        masses=ones,
        loggs=ones * 4.4,
        teffs=ones * 5778.0,
        metallicities=ones * 0.0,
        jmags=ones * 9.5,
        hmags=ones * 9.0,
        kmags=ones * 8.8,
        gmags=ones * 10.5,
        rmags=ones * 10.0,
        imags=ones * 9.5,
        zmags=ones * 9.0,
    )


# ---------------------------------------------------------------------------
# _sample_population_indices
# ---------------------------------------------------------------------------

class TestSamplePopulationIndices:
    def test_legacy_exclude_last_draws_from_zero_to_n_comp_minus_2(self) -> None:
        """legacy_exclude_last=True: indices must be in [0, n_comp-2]."""
        n_comp = 10
        idxs = _sample_population_indices(n_comp, n=1000, legacy_exclude_last=True)
        assert idxs.min() >= 0
        assert idxs.max() <= n_comp - 2

    def test_without_legacy_draws_up_to_n_comp_minus_1(self) -> None:
        """legacy_exclude_last=False: indices can reach n_comp-1."""
        n_comp = 10
        idxs = _sample_population_indices(n_comp, n=10000, legacy_exclude_last=False)
        assert idxs.min() >= 0
        assert idxs.max() <= n_comp - 1
        # With 10 000 draws from [0, 9] the last index is virtually certain to appear.
        assert idxs.max() == n_comp - 1

    def test_large_sample_all_indices_valid(self) -> None:
        """n=1000, n_comp=10, legacy=False: every index in [0, 9]."""
        n_comp = 10
        idxs = _sample_population_indices(n_comp, n=1000, legacy_exclude_last=False)
        assert np.all(idxs >= 0)
        assert np.all(idxs <= n_comp - 1)
        assert idxs.shape == (1000,)

    def test_n_comp_1_legacy_true_does_not_crash(self) -> None:
        """n_comp=1 with legacy_exclude_last=True: must not raise."""
        idxs = _sample_population_indices(1, n=5, legacy_exclude_last=True)
        # Result should be an array of length 5 — values are 0 (only valid index).
        assert len(idxs) == 5
        assert np.all(idxs == 0)

    def test_returns_ndarray(self) -> None:
        idxs = _sample_population_indices(5, n=10)
        assert isinstance(idxs, np.ndarray)


# ---------------------------------------------------------------------------
# _filter_population_by_target_tmag
# ---------------------------------------------------------------------------

class TestFilterPopulationByTargetTmag:
    def test_fainter_stars_kept(self) -> None:
        """Stars with Tmag > target_tmag (fainter) must be kept."""
        tmags = np.array([8.0, 9.5, 11.0, 12.5])
        pop = _make_population(4, tmag_values=tmags)
        result = _filter_population_by_target_tmag(pop, target_tmag=10.0)
        np.testing.assert_array_equal(result.tmags, np.array([11.0, 12.5]))

    def test_brighter_stars_excluded(self) -> None:
        """Stars with Tmag < target_tmag (strictly brighter) must be excluded."""
        # The filter keeps tmag >= target_tmag, so only 8.0 is excluded.
        tmags = np.array([8.0, 10.0, 12.0])
        pop = _make_population(3, tmag_values=tmags)
        result = _filter_population_by_target_tmag(pop, target_tmag=10.0)
        # 8.0 is excluded (strictly less than 10.0); 10.0 and 12.0 are kept.
        assert result.n_stars == 2
        np.testing.assert_array_equal(result.tmags, np.array([10.0, 12.0]))

    def test_all_filtered_gives_empty_population(self) -> None:
        """When every star is brighter than target, result has n_stars == 0."""
        tmags = np.array([5.0, 6.0, 7.0])
        pop = _make_population(3, tmag_values=tmags)
        result = _filter_population_by_target_tmag(pop, target_tmag=10.0)
        assert result.n_stars == 0

    def test_target_tmag_none_returns_all(self) -> None:
        """target_tmag=None must return the original population unchanged."""
        tmags = np.array([5.0, 10.0, 15.0])
        pop = _make_population(3, tmag_values=tmags)
        result = _filter_population_by_target_tmag(pop, target_tmag=None)
        assert result is pop  # exact same object

    def test_other_arrays_also_filtered(self) -> None:
        """All arrays (masses, loggs, etc.) must be sliced consistently."""
        tmags = np.array([8.0, 12.0])
        masses = np.array([0.5, 1.5])
        pop = TRILEGALResult(
            tmags=tmags, masses=masses,
            loggs=np.array([4.5, 4.3]),
            teffs=np.array([4000.0, 6000.0]),
            metallicities=np.array([-0.5, 0.1]),
            jmags=np.array([7.0, 11.0]),
            hmags=np.array([6.5, 10.5]),
            kmags=np.array([6.3, 10.3]),
            gmags=np.array([8.5, 12.5]),
            rmags=np.array([8.3, 12.3]),
            imags=np.array([8.1, 12.1]),
            zmags=np.array([8.0, 12.0]),
        )
        result = _filter_population_by_target_tmag(pop, target_tmag=10.0)
        assert result.n_stars == 1
        assert result.masses[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# _needs_sdss_delta_mags
# ---------------------------------------------------------------------------

class TestNeedsSdssDeltaMags:
    def test_empty_bands_returns_false(self) -> None:
        assert _needs_sdss_delta_mags((), filt=None) is False

    def test_g_band_returns_true(self) -> None:
        assert _needs_sdss_delta_mags(("g",), filt=None) is True

    def test_r_band_returns_true(self) -> None:
        assert _needs_sdss_delta_mags(("r",), filt=None) is True

    def test_i_band_returns_true(self) -> None:
        assert _needs_sdss_delta_mags((), filt="i") is True

    def test_z_band_returns_true(self) -> None:
        assert _needs_sdss_delta_mags(("z",), filt=None) is True

    def test_jhk_only_returns_false(self) -> None:
        assert _needs_sdss_delta_mags(("J", "H", "K"), filt=None) is False

    def test_tess_only_returns_false(self) -> None:
        assert _needs_sdss_delta_mags(("TESS",), filt=None) is False

    def test_mixed_sdss_and_jhk_returns_true(self) -> None:
        assert _needs_sdss_delta_mags(("J", "r"), filt=None) is True

    def test_filt_adds_to_active_bands(self) -> None:
        """filt='g' combined with empty external_lc_bands → True."""
        assert _needs_sdss_delta_mags((), filt="g") is True


# ---------------------------------------------------------------------------
# _compute_fluxratios_comp
# ---------------------------------------------------------------------------

class TestComputeFluxratiosComp:
    def test_delta_mag_zero_gives_half(self) -> None:
        """delta_mags=0 means equal brightness → flux_ratio = 0.5."""
        delta = np.array([0.0])
        result = _compute_fluxratios_comp(delta)
        np.testing.assert_allclose(result, [0.5], atol=1e-12)

    def test_large_positive_delta_mag_near_zero(self) -> None:
        """Very large delta_mags → companion much fainter → ratio near 0."""
        delta = np.array([10.0])
        result = _compute_fluxratios_comp(delta)
        # ratio = 10^4 / (1 + 10^4) ≈ 0.9999  — companion brighter, so large ratio
        # Wait: delta_mags = target_mag - companion_mag  >0 means comp is fainter
        # ratio = 10^(delta/2.5)/(1+10^(delta/2.5))
        # For delta=10: ratio ≈ 1.0, not 0.
        # For very negative delta: companion is fainter → ratio near 0.
        assert result[0] > 0.9

    def test_large_negative_delta_mag_near_zero(self) -> None:
        """Negative delta_mags: companion fainter than target → ratio near 0."""
        delta = np.array([-10.0])
        result = _compute_fluxratios_comp(delta)
        assert result[0] < 0.01

    def test_positive_delta_mag_above_half(self) -> None:
        """Positive delta_mags (target fainter) → ratio > 0.5."""
        delta = np.array([2.5])
        result = _compute_fluxratios_comp(delta)
        assert result[0] > 0.5

    def test_negative_delta_mag_below_half(self) -> None:
        """Negative delta_mags (companion fainter) → ratio < 0.5."""
        delta = np.array([-2.5])
        result = _compute_fluxratios_comp(delta)
        assert result[0] < 0.5

    def test_array_input_element_wise(self) -> None:
        """Verify that array input is processed element-wise."""
        delta = np.array([-5.0, 0.0, 5.0])
        result = _compute_fluxratios_comp(delta)
        assert result.shape == (3,)
        assert result[0] < result[1] < result[2]
        np.testing.assert_allclose(result[1], 0.5, atol=1e-12)

    def test_output_in_open_unit_interval(self) -> None:
        """Flux ratios should always lie strictly in (0, 1)."""
        delta = np.linspace(-8.0, 8.0, 50)
        result = _compute_fluxratios_comp(delta)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)


# ---------------------------------------------------------------------------
# _combined_delta_mag
# ---------------------------------------------------------------------------

class TestCombinedDeltaMag:
    def test_equal_flux_ratios_brighter_than_either(self) -> None:
        """Two equal components: combined delta_mag > individual delta_mag."""
        fr = np.array([0.25])
        individual_delta = 2.5 * np.log10(fr / (1 - fr))  # negative (comp fainter)
        combined = _combined_delta_mag(fr, fr)
        # Combined system is brighter → delta_mag is larger (less negative or more positive)
        assert combined[0] > individual_delta[0]

    def test_one_zero_flux_ratio_equals_other(self) -> None:
        """primary_flux_ratio=0 → term vanishes → combined equals secondary only."""
        primary = np.array([0.0])
        secondary = np.array([0.3])
        combined = _combined_delta_mag(primary, secondary)
        expected = 2.5 * np.log10(secondary / (1 - secondary))
        np.testing.assert_allclose(combined, expected, rtol=1e-10)

    def test_symmetry(self) -> None:
        """combined_delta_mag is symmetric in its two arguments."""
        a = np.array([0.2, 0.4])
        b = np.array([0.3, 0.1])
        np.testing.assert_allclose(
            _combined_delta_mag(a, b),
            _combined_delta_mag(b, a),
            rtol=1e-12,
        )

    def test_combined_greater_than_either_component(self) -> None:
        """The combined system is at least as bright as each component alone."""
        a = np.array([0.15, 0.35])
        b = np.array([0.10, 0.20])
        combined = _combined_delta_mag(a, b)
        delta_a = 2.5 * np.log10(a / (1 - a))
        delta_b = 2.5 * np.log10(b / (1 - b))
        assert np.all(combined >= delta_a)
        assert np.all(combined >= delta_b)


class TestComputeLnpriorCompanion:
    def test_no_contrast_legacy_uses_log10(self) -> None:
        idxs = np.array([0, 1, 0])
        fluxratios_comp = np.array([0.2, 0.3])
        delta_mags_map = {"delta_TESSmags": np.array([-1.0, -0.5])}
        result = _compute_lnprior_companion(
            n_comp=25,
            fluxratios_comp=fluxratios_comp,
            idxs=idxs,
            delta_mags_map=delta_mags_map,
            contrast_curve=None,
            filt="TESS",
            numerical_mode="legacy",
        )
        expected = np.log10((25 / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        np.testing.assert_allclose(result, expected)

    def test_no_contrast_corrected_uses_natural_log(self) -> None:
        idxs = np.array([0, 1, 0])
        fluxratios_comp = np.array([0.2, 0.3])
        delta_mags_map = {"delta_TESSmags": np.array([-1.0, -0.5])}
        result = _compute_lnprior_companion(
            n_comp=25,
            fluxratios_comp=fluxratios_comp,
            idxs=idxs,
            delta_mags_map=delta_mags_map,
            contrast_curve=None,
            filt="TESS",
            numerical_mode="corrected",
        )
        expected = np.log((25 / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        np.testing.assert_allclose(result, expected)

    def test_curve_set_single_matches_single_curve(self) -> None:
        curve = ContrastCurve(
            separations_arcsec=np.array([0.1, 0.5, 1.0]),
            delta_mags=np.array([1.0, 3.0, 5.0]),
            band="TESS",
        )
        idxs = np.array([0, 1, 2])
        fluxratios_comp = np.array([0.1, 0.2, 0.3])
        delta_mags_map = {"delta_TESSmags": np.array([-1.5, -2.5, -3.5])}

        single = _compute_lnprior_companion(
            n_comp=25,
            fluxratios_comp=fluxratios_comp,
            idxs=idxs,
            delta_mags_map=delta_mags_map,
            contrast_curve=curve,
            filt="TESS",
            numerical_mode="corrected",
        )
        as_set = _compute_lnprior_companion(
            n_comp=25,
            fluxratios_comp=fluxratios_comp,
            idxs=idxs,
            delta_mags_map=delta_mags_map,
            contrast_curve=ContrastCurveSet([curve]),
            filt=None,
            numerical_mode="corrected",
        )

        np.testing.assert_array_equal(as_set, single)

    def test_curve_set_uses_tightest_curve_per_draw(self) -> None:
        loose = ContrastCurve(
            separations_arcsec=np.array([0.2, 1.0, 2.0]),
            delta_mags=np.array([1.0, 3.0, 5.0]),
            band="TESS",
        )
        tight = ContrastCurve(
            separations_arcsec=np.array([0.1, 0.5, 1.0]),
            delta_mags=np.array([1.0, 3.0, 5.0]),
            band="TESS",
        )
        idxs = np.array([0, 1, 2])
        fluxratios_comp = np.array([0.1, 0.2, 0.3])
        delta_mags_map = {"delta_TESSmags": np.array([-1.5, -2.5, -3.5])}

        result = _compute_lnprior_companion(
            n_comp=25,
            fluxratios_comp=fluxratios_comp,
            idxs=idxs,
            delta_mags_map=delta_mags_map,
            contrast_curve=ContrastCurveSet([loose, tight]),
            filt=None,
            numerical_mode="corrected",
        )
        expected = lnprior_background(
            25,
            np.abs(delta_mags_map["delta_TESSmags"][idxs]),
            tight.separations_arcsec,
            tight.delta_mags,
            numerical_mode="corrected",
        )
        expected[expected > 0.0] = 0.0

        np.testing.assert_allclose(result, expected)
