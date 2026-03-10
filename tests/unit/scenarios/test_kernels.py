"""Tests for triceratops.scenarios.kernels -- highest priority tests in the project."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.kernels import (
    build_transit_mask,
    compute_lnZ,
    load_external_lcs,
    pack_best_indices,
    resolve_period,
)

# ---------------------------------------------------------------------------
# resolve_period
# ---------------------------------------------------------------------------

class TestResolvePeriod:
    def test_scalar_float(self) -> None:
        result = resolve_period(5.0, 100)
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, 5.0)

    def test_numpy_float64(self) -> None:
        result = resolve_period(np.float64(5.0), 100)
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, 5.0)

    def test_numpy_int32(self) -> None:
        result = resolve_period(np.int32(5), 100)
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, 5.0)

    def test_range_values_in_bounds(self) -> None:
        result = resolve_period([1.0, 10.0], 10000)
        assert result.shape == (10000,)
        assert np.all(result >= 1.0)
        assert np.all(result <= 10.0)

    def test_range_3_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            resolve_period([1.0, 2.0, 3.0], 100)

    def test_range_1_element_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            resolve_period([1.0], 100)

    def test_zero_period_scalar(self) -> None:
        """Zero period scalar: returns array of zeros (no raise expected)."""
        result = resolve_period(0.0, 50)
        assert result.shape == (50,)
        np.testing.assert_array_equal(result, 0.0)

    def test_negative_period_scalar(self) -> None:
        """Negative scalar period: silently returns negative array (no raise)."""
        result = resolve_period(-1.0, 50)
        assert result.shape == (50,)
        np.testing.assert_array_equal(result, -1.0)

    def test_period_min_equals_period_max(self) -> None:
        """[5.0, 5.0] range: uniform sample from [5,5) => constant 5.0."""
        result = resolve_period([5.0, 5.0], 200)
        assert result.shape == (200,)
        np.testing.assert_array_equal(result, 5.0)

    def test_large_range_within_bounds(self) -> None:
        """Large range [0.1, 1000.0]: all N=10000 draws stay within bounds."""
        result = resolve_period([0.1, 1000.0], 10000)
        assert result.shape == (10000,)
        assert np.all(result >= 0.1)
        assert np.all(result <= 1000.0)

    def test_two_element_list_draws_in_range(self) -> None:
        """[1.0, 2.0]: draws in [1.0, 2.0)."""
        result = resolve_period([1.0, 2.0], 5000)
        assert result.shape == (5000,)
        assert np.all(result >= 1.0)
        assert np.all(result < 2.0)

    def test_tuple_same_as_list(self) -> None:
        """Tuple (1.0, 2.0) behaves identically to list [1.0, 2.0]."""
        np.random.seed(0)
        result_list = resolve_period([1.0, 2.0], 500)
        np.random.seed(0)
        result_tuple = resolve_period((1.0, 2.0), 500)
        np.testing.assert_array_equal(result_list, result_tuple)


# ---------------------------------------------------------------------------
# compute_lnZ
# ---------------------------------------------------------------------------

class TestComputeLnZ:
    def test_all_neg_inf(self) -> None:
        lnL = np.full(100, -np.inf)
        assert compute_lnZ(lnL, 650.0) == -np.inf

    def test_all_zero_lnL(self) -> None:
        lnL = np.zeros(100)
        result = compute_lnZ(lnL, 650.0)
        # Z = mean(exp(0 + 650)) = exp(650), lnZ = 650
        assert result == pytest.approx(650.0, rel=1e-10)

    def test_known_value(self) -> None:
        # If lnL = [-650] for all N, then exp(lnL + 650) = exp(0) = 1
        # Z = mean(1) = 1, lnZ = log(1) = 0
        lnL = np.full(50, -650.0)
        result = compute_lnZ(lnL, 650.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_single_good_sample_among_neg_inf(self) -> None:
        lnL = np.full(100, -np.inf)
        lnL[42] = -650.0  # one good sample
        result = compute_lnZ(lnL, 650.0)
        # Z = mean of [0, ..., 1, ..., 0] = 1/100
        expected = np.log(1.0 / 100.0)
        assert result == pytest.approx(expected, rel=1e-6)

    # --- Numerical stability tests ---

    def test_large_positive_lnL_no_overflow(self) -> None:
        """Large positive lnL values must not overflow; result should be finite."""
        lnL = np.full(100, 5000.0)
        lnz_const = 0.0
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result), "compute_lnZ overflowed for large positive lnL"
        # lnL_max=5000, log(sum_exp/N)=log(1)=0; result = 5000 + lnz_const
        assert result == pytest.approx(5000.0 + lnz_const, rel=1e-10)

    def test_large_positive_lnL_with_lnz_const(self) -> None:
        """Large positive lnL + nonzero lnz_const: result ≈ lnL_max + lnz_const."""
        lnL = np.full(50, 5000.0)
        lnz_const = 650.0
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result)
        assert result == pytest.approx(5000.0 + lnz_const, rel=1e-10)

    def test_extreme_spread_underflows_gracefully(self) -> None:
        """Values far below max underflow to 0; result is still finite."""
        N = 200
        lnL_max = 0.0
        lnL = np.full(N, lnL_max - 800.0)
        lnL[0] = lnL_max  # single value at max
        result = compute_lnZ(lnL, 0.0)
        assert np.isfinite(result), "result is not finite for extreme spread"
        # Only index 0 contributes; exp(-800) underflows to 0 for others.
        # result = lnL_max + log(1/N) = log(1/N)
        expected = lnL_max + np.log(1.0 / N)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_identical_values_equal_that_value(self) -> None:
        """All lnL identical: log-sum-exp gives exactly that value (+ lnz_const)."""
        lnL = np.full(300, -300.0)
        lnz_const = 10.0
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result)
        # lnL_max=-300, log(sum_exp/N)=log(N/N)=0; result=-300+lnz_const
        assert result == pytest.approx(-300.0 + lnz_const, rel=1e-10)

    def test_single_finite_among_many_neg_inf(self) -> None:
        """One finite value X among N-1 -inf entries: result ≈ X + lnz_const - log(N)."""
        N = 500
        X = -123.0
        lnz_const = 50.0
        lnL = np.full(N, -np.inf)
        lnL[7] = X
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result)
        expected = X + lnz_const + np.log(1.0 / N)
        assert result == pytest.approx(expected, rel=1e-8)

    def test_large_N_all_zeros(self) -> None:
        """N=1_000_000 draws all at 0.0: result ≈ lnz_const."""
        N = 1_000_000
        lnL = np.zeros(N)
        lnz_const = 42.0
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result)
        # lnL_max=0, log(N/N)=0; result=0+lnz_const
        assert result == pytest.approx(lnz_const, rel=1e-10)

    def test_mixed_extreme_range_correct_result(self) -> None:
        """Mixed lnL from -10000 to +100: max-anchored computation is correct."""
        N = 1000
        rng = np.random.default_rng(42)
        lnL = rng.uniform(-10000.0, 100.0, size=N)
        lnz_const = 0.0
        result = compute_lnZ(lnL, lnz_const)
        assert np.isfinite(result), "mixed extreme range produced non-finite result"
        # Reference: only values near the max contribute; the max itself is ~100
        lnL_max = np.max(lnL)
        assert result <= lnL_max + lnz_const
        assert result > -np.inf


# ---------------------------------------------------------------------------
# pack_best_indices
# ---------------------------------------------------------------------------

class TestPackBestIndices:
    def test_top_n(self) -> None:
        lnL = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = pack_best_indices(lnL, 3)
        assert len(idx) == 3
        # Smallest -lnL means largest lnL: indices 1,4,2
        assert set(idx) == {1, 4, 2}

    def test_n_larger_than_array(self) -> None:
        lnL = np.array([1.0, 2.0, 3.0])
        idx = pack_best_indices(lnL, 10)
        assert len(idx) == 3


# ---------------------------------------------------------------------------
# load_external_lcs
# ---------------------------------------------------------------------------

class TestLoadExternalLcs:
    def test_mismatch_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            load_external_lcs(
                ["a.txt", "b.txt"], ["TESS"],
                FixedLDCCatalog(), 0.0, 5778.0, 4.44,
            )

    def test_too_many_raises(self) -> None:
        files = [f"f{i}.txt" for i in range(8)]
        filters = ["TESS"] * 8
        with pytest.raises(ValueError, match="Maximum 7"):
            load_external_lcs(files, filters, FixedLDCCatalog(), 0.0, 5778.0, 4.44)

    def test_returns_list_of_correct_length(self, tmp_path) -> None:
        # Create two stub LC files
        for i in range(2):
            data = np.column_stack([
                np.linspace(-0.1, 0.1, 50),
                np.ones(50),
                np.full(50, 0.001),
            ])
            np.savetxt(tmp_path / f"lc_{i}.txt", data)

        result = load_external_lcs(
            [str(tmp_path / "lc_0.txt"), str(tmp_path / "lc_1.txt")],
            ["TESS", "J"],
            FixedLDCCatalog(u1=0.3, u2=0.1),
            0.0, 5778.0, 4.44,
        )
        assert len(result) == 2

    def test_ldc_populated(self, tmp_path) -> None:
        data = np.column_stack([
            np.linspace(-0.1, 0.1, 50),
            np.ones(50),
            np.full(50, 0.001),
        ])
        np.savetxt(tmp_path / "lc.txt", data)

        result = load_external_lcs(
            [str(tmp_path / "lc.txt")],
            ["J"],
            FixedLDCCatalog(u1=0.35, u2=0.15),
            0.0, 5778.0, 4.44,
        )
        assert result[0].ldc is not None
        assert result[0].ldc.u1 == pytest.approx(0.35)
        assert result[0].ldc.band == "J"


# ---------------------------------------------------------------------------
# build_transit_mask
# ---------------------------------------------------------------------------

class TestBuildTransitMask:
    def test_all_transiting(self) -> None:
        n = 10
        inc = np.full(n, 89.0)
        ptra = np.full(n, 0.5)  # arccos(1/0.5) = arccos(2) is invalid, but ptra <= 1
        # ptra=0.5 -> inc_min = arccos(2.0) which is nan... let me use ptra=0.9
        ptra = np.full(n, 0.9)  # inc_min = arccos(1/0.9) ~ 83.6 deg
        coll = np.full(n, False)
        mask = build_transit_mask(inc, ptra, coll)
        assert mask.shape == (n,)
        assert np.all(mask)

    def test_collision_excluded(self) -> None:
        n = 5
        inc = np.full(n, 89.0)
        ptra = np.full(n, 0.9)
        coll = np.array([False, True, False, True, False])
        mask = build_transit_mask(inc, ptra, coll)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_ptra_gt_1_excluded(self) -> None:
        n = 3
        inc = np.full(n, 89.0)
        ptra = np.array([0.9, 1.5, 0.8])  # middle one has ptra > 1
        coll = np.full(n, False)
        mask = build_transit_mask(inc, ptra, coll)
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.True_

    def test_extra_mask_applied(self) -> None:
        n = 4
        inc = np.full(n, 89.0)
        ptra = np.full(n, 0.9)
        coll = np.full(n, False)
        extra = np.array([True, True, False, True])
        mask = build_transit_mask(inc, ptra, coll, extra_mask=extra)
        assert mask[2] is np.False_
