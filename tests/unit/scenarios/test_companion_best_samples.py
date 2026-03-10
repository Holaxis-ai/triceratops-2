"""Tests for BUG-07: companion scenarios must rank best-fit samples by
posterior score (lnL + lnprior_comp), not raw likelihood (lnL).

These tests are written to FAIL against the buggy code (pack_best_indices(lnL,
n_best)) and PASS after the fix (pack_best_indices(lnL + lnprior_comp, n_best)).

BUG-07 reference: working_docs/Original_bugs/BUG-07_companion_best_sample_ranking.md
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.kernels import pack_best_indices

_LNL_MOD = "triceratops.scenarios.companion_scenarios"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stellar_params() -> StellarParameters:
    return StellarParameters(
        mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
        logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
    )


@pytest.fixture()
def transit_lc() -> LightCurve:
    time = np.linspace(-0.1, 0.1, 80)
    flux = np.ones(80)
    flux[30:50] = 0.998
    return LightCurve(time_days=time, flux=flux, flux_err=5e-4)


@pytest.fixture()
def small_config() -> Config:
    # n_best_samples=5 so the divergence between lnL-ranked and posterior-ranked
    # is easy to detect (top-5 by lnL vs top-5 by lnL + lnprior_comp)
    return Config(n_mc_samples=300, n_best_samples=5)


# ---------------------------------------------------------------------------
# Pure unit tests on pack_best_indices — independent of any scenario
# ---------------------------------------------------------------------------

class TestPackBestIndicesWithPrior:
    """Direct verification that adding a non-flat prior changes selected indices.

    These tests do not depend on any scenario; they test the fix objective in
    isolation. Tests 6 and 7 in the required suite.
    """

    def test_non_flat_prior_changes_top_k_selection(self) -> None:
        """BUG-07: non-flat prior must change top-k selection."""
        rng = np.random.default_rng(123)
        n = 100
        lnL = rng.standard_normal(n) * 3.0

        # Build a prior that inverts the ranking of the top 10 draws
        lnprior = np.zeros(n)
        top10_by_lnL = np.argsort(-lnL)[:10]
        lnprior[top10_by_lnL] = -200.0  # severely penalise top-10 draws

        n_best = 5
        idx_lnL_only = set(pack_best_indices(lnL, n_best).tolist())
        idx_posterior = set(pack_best_indices(lnL + lnprior, n_best).tolist())

        assert idx_lnL_only != idx_posterior, (
            "Non-flat prior must change the top-k selection when the prior "
            "strongly penalises the top likelihood draws."
        )

    def test_posterior_ranked_set_avoids_heavily_penalised_draws(self) -> None:
        """Draws with severe prior penalty must not appear in posterior top-k."""
        n = 50
        lnL = np.zeros(n)
        lnL[0:5] = 10.0   # top-5 by likelihood

        lnprior = np.zeros(n)
        lnprior[0:5] = -1000.0  # heavy penalty on those same 5 draws

        n_best = 5
        posterior_idx = set(pack_best_indices(lnL + lnprior, n_best).tolist())
        penalised_idx = {0, 1, 2, 3, 4}

        assert penalised_idx.isdisjoint(posterior_idx), (
            "Posterior-ranked top-5 must not include heavily-penalised draws. "
            f"Got: {posterior_idx}"
        )

    def test_flat_prior_produces_same_ranking_as_raw_lnL(self) -> None:
        """When lnprior_comp is constant, lnL+C gives identical ranking to lnL."""
        rng = np.random.default_rng(99)
        n = 200
        lnL = rng.standard_normal(n) * 5.0
        lnprior_const = -3.7  # constant prior — must not change ranking

        n_best = 20
        idx_raw = pack_best_indices(lnL, n_best)
        idx_posterior = pack_best_indices(lnL + lnprior_const, n_best)

        np.testing.assert_array_equal(
            idx_raw, idx_posterior,
            err_msg=(
                "Flat prior should not change sample ranking. "
                "pack_best_indices(lnL, k) != pack_best_indices(lnL+C, k)"
            ),
        )


# ---------------------------------------------------------------------------
# Scenario-level tests using argument-capture approach
#
# Strategy: patch compute_lnZ and pack_best_indices to capture the score
# arrays they receive. The lnL function is patched to return finite values
# for ALL draws (not just masked ones) — this ensures pack_best_indices is
# always called and the captured arrays have meaningful content.
# ---------------------------------------------------------------------------

def _make_finite_lnL_planet(*args, **kwargs):
    """Mock lnL_planet_p: return finite values for all draws (ignores mask)."""
    # Positional signature: time, flux, sigma, rps, periods, incs, as_, rss,
    #                       u1s, u2s, eccs, argps, companion_flux_ratios, mask, ...
    rps = args[3] if len(args) > 3 else kwargs["rps"]
    n = len(rps)
    rng = np.random.default_rng(42)
    return rng.standard_normal(n) - 3.0


def _make_finite_lnL_eb(*args, **kwargs):
    """Mock lnL_eb_p: return finite values for all draws."""
    # Positional signature: time, flux, sigma, rss, rcomps, eb_flux_ratios, ...
    rss = args[3] if len(args) > 3 else kwargs["rss"]
    n = len(rss)
    rng = np.random.default_rng(43)
    return rng.standard_normal(n) - 4.0


def _make_finite_lnL_eb_twin(*args, **kwargs):
    """Mock lnL_eb_twin_p: return finite values for all draws."""
    rss = args[3] if len(args) > 3 else kwargs["rss"]
    n = len(rss)
    rng = np.random.default_rng(44)
    return rng.standard_normal(n) - 4.5


class TestPTPConsistency:
    """BUG-07: PTPScenario pack_best_indices must receive lnL + lnprior_comp.

    Test 1 in the required suite.
    """

    def test_ptp_pack_best_indices_arg_equals_lnz_arg(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """compute_lnZ and pack_best_indices must receive the SAME score array."""
        from triceratops.scenarios.companion_scenarios import PTPScenario

        lnZ_args: list[np.ndarray] = []
        pack_args: list[np.ndarray] = []

        def capture_lnZ(score_array, lnz_const):
            lnZ_args.append(score_array.copy())
            return float(-np.inf)

        def capture_pack(score_array, n_best):
            pack_args.append(score_array.copy())
            # Must return a valid index array
            return np.array([0], dtype=int)

        scenario = PTPScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_make_finite_lnL_planet), \
             patch(f"{_LNL_MOD}.compute_lnZ", side_effect=capture_lnZ), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert lnZ_args, "compute_lnZ was not called — PTP compute() did not reach lnZ step"
        assert pack_args, "pack_best_indices was not called — PTP compute() did not reach idx step"

        np.testing.assert_array_equal(
            lnZ_args[0], pack_args[0],
            err_msg=(
                "BUG-07 PTP: compute_lnZ and pack_best_indices received different "
                "score arrays. pack_best_indices is missing the lnprior_comp term."
            ),
        )


class TestPTPDivergence:
    """BUG-07: PTP divergence test — posterior top-5 != likelihood top-5.

    Test 2 in the required suite. The critical test: proves the fix actually
    changes behaviour when lnprior_comp is non-flat.
    """

    def test_ptp_best_indices_use_posterior_score(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """pack_best_indices must receive lnL + lnprior_comp, not raw lnL."""
        from triceratops.scenarios.companion_scenarios import PTPScenario

        n = small_config.n_mc_samples
        rng = np.random.default_rng(7)

        # Controlled lnL: draws 0..4 have highest lnL (top-5 by likelihood)
        lnL_base = rng.standard_normal(n) - 20.0
        lnL_base[:5] = -1.0   # top-5 by raw lnL

        # Controlled lnprior_comp: draws 0..4 are penalised heavily → draws 5..9 win
        lnprior_base = np.zeros(n)
        lnprior_base[:5] = -100.0
        lnprior_base[5:10] = 0.0

        posterior_scores = lnL_base + lnprior_base
        posterior_top5 = set(np.argsort(-posterior_scores)[:5].tolist())
        likelihood_top5 = set(np.argsort(-lnL_base)[:5].tolist())
        # Sanity: must be different for the divergence test to be meaningful
        assert posterior_top5 != likelihood_top5, (
            "Test setup error: posterior and likelihood top-5 are identical"
        )

        captured_pack_scores: list[np.ndarray] = []

        def mock_lnL_planet(*args, **kwargs):
            # Return our controlled lnL values for all draws
            return lnL_base.copy()

        def mock_companion_prior(*args, **kwargs):
            return lnprior_base.copy()

        def capture_pack(score_array, n_best):
            captured_pack_scores.append(score_array.copy())
            return pack_best_indices(score_array, n_best)

        scenario = PTPScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=mock_lnL_planet), \
             patch(f"{_LNL_MOD}._compute_companion_prior", side_effect=mock_companion_prior), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert captured_pack_scores, (
            "pack_best_indices was not called — PTP compute() did not reach idx step"
        )

        captured = captured_pack_scores[0]
        captured_top5 = set(np.argsort(-captured)[:5].tolist())

        assert captured_top5 == posterior_top5, (
            f"BUG-07: pack_best_indices received score whose top-5 = {captured_top5}, "
            f"but posterior top-5 = {posterior_top5}. "
            f"Raw lnL was passed instead of lnL + lnprior_comp."
        )
        assert captured_top5 != likelihood_top5, (
            "Fix did not propagate the prior into the score: "
            "captured top-5 still matches likelihood-only top-5."
        )


class TestPEBTwinBranch:
    """BUG-07 twin branch: PEBScenario idx_twin must use lnL_twin + lnprior_comp.

    Test 3 in the required suite.
    """

    def test_peb_both_branches_use_posterior_score(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """Both primary and twin pack_best_indices calls must match compute_lnZ."""
        from triceratops.scenarios.companion_scenarios import PEBScenario

        lnZ_args: list[np.ndarray] = []
        pack_args: list[np.ndarray] = []

        def capture_lnZ(score_array, lnz_const):
            lnZ_args.append(score_array.copy())
            return float(-np.inf)

        def capture_pack(score_array, n_best):
            pack_args.append(score_array.copy())
            return np.array([0], dtype=int)

        scenario = PEBScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_make_finite_lnL_eb), \
             patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_make_finite_lnL_eb_twin), \
             patch(f"{_LNL_MOD}.compute_lnZ", side_effect=capture_lnZ), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert len(lnZ_args) >= 2, (
            f"PEB must call compute_lnZ twice (primary + twin), got {len(lnZ_args)}"
        )
        assert len(pack_args) >= 2, (
            f"PEB must call pack_best_indices twice, got {len(pack_args)}"
        )

        # Primary branch: lnZ call 0 must match pack call 0
        np.testing.assert_array_equal(
            lnZ_args[0], pack_args[0],
            err_msg="BUG-07 PEB primary: pack_best_indices score != compute_lnZ score",
        )
        # Twin branch: lnZ call 1 must match pack call 1
        np.testing.assert_array_equal(
            lnZ_args[1], pack_args[1],
            err_msg="BUG-07 PEB twin: pack_best_indices score != compute_lnZ score",
        )


class TestSEBTwinBranch:
    """BUG-07 twin branch: SEBScenario idx and idx_twin both use posterior score.

    Test 4 in the required suite.
    """

    def test_seb_both_branches_use_posterior_score(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """Both primary and twin pack_best_indices calls must match compute_lnZ."""
        from triceratops.scenarios.companion_scenarios import SEBScenario

        lnZ_args: list[np.ndarray] = []
        pack_args: list[np.ndarray] = []

        def capture_lnZ(score_array, lnz_const):
            lnZ_args.append(score_array.copy())
            return float(-np.inf)

        def capture_pack(score_array, n_best):
            pack_args.append(score_array.copy())
            return np.array([0], dtype=int)

        scenario = SEBScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_make_finite_lnL_eb), \
             patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_make_finite_lnL_eb_twin), \
             patch(f"{_LNL_MOD}.compute_lnZ", side_effect=capture_lnZ), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert len(lnZ_args) >= 2, (
            f"SEB must call compute_lnZ twice (primary + twin), got {len(lnZ_args)}"
        )
        assert len(pack_args) >= 2, (
            f"SEB must call pack_best_indices twice, got {len(pack_args)}"
        )

        np.testing.assert_array_equal(
            lnZ_args[0], pack_args[0],
            err_msg="BUG-07 SEB primary branch: pack score != lnZ score",
        )
        np.testing.assert_array_equal(
            lnZ_args[1], pack_args[1],
            err_msg="BUG-07 SEB twin branch: pack score != lnZ score",
        )


class TestFlatPriorEdgeCase:
    """Edge case: flat (constant) lnprior_comp must not change sample ranking.

    Test 5 in the required suite — regression guard.
    """

    def test_ptp_flat_prior_does_not_change_ranking(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """When lnprior_comp is constant, lnZ arg and pack arg differ only by
        that constant — so both must be array-equal (same shape & pattern)."""
        from triceratops.scenarios.companion_scenarios import PTPScenario

        lnZ_args: list[np.ndarray] = []
        pack_args: list[np.ndarray] = []

        def capture_lnZ(score_array, lnz_const):
            lnZ_args.append(score_array.copy())
            return float(-np.inf)

        def capture_pack(score_array, n_best):
            pack_args.append(score_array.copy())
            return np.array([0], dtype=int)

        def flat_prior(*args, **kwargs):
            n_ = small_config.n_mc_samples
            return np.full(n_, -2.5)

        scenario = PTPScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_make_finite_lnL_planet), \
             patch(f"{_LNL_MOD}._compute_companion_prior", side_effect=flat_prior), \
             patch(f"{_LNL_MOD}.compute_lnZ", side_effect=capture_lnZ), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert lnZ_args, "compute_lnZ was not called"
        assert pack_args, "pack_best_indices was not called"

        # With flat prior: lnZ_arg = lnL + C, pack_arg should also = lnL + C.
        # They must be identical arrays (same values at same indices).
        np.testing.assert_array_equal(
            lnZ_args[0], pack_args[0],
            err_msg=(
                "BUG-07 flat-prior regression: even with constant lnprior_comp, "
                "pack_best_indices must still receive lnL + lnprior_comp "
                "(not raw lnL)."
            ),
        )

        # Also verify that ranking by lnL+C = ranking by lnL (flat prior invariant)
        n_best = small_config.n_best_samples
        n_ = small_config.n_mc_samples
        lnL_synthetic = np.random.default_rng(0).standard_normal(n_) - 3.0
        const = -2.5
        idx_lnL = pack_best_indices(lnL_synthetic, n_best)
        idx_posterior = pack_best_indices(lnL_synthetic + const, n_best)
        np.testing.assert_array_equal(
            idx_lnL, idx_posterior,
            err_msg="Flat prior must not change ranking.",
        )


class TestSTPConsistency:
    """BUG-07: STPScenario pack_best_indices must receive lnL + lnprior_comp.

    Test 6 in the required suite.
    """

    def test_stp_pack_best_indices_arg_equals_lnz_arg(
        self, transit_lc, stellar_params, small_config,
    ) -> None:
        """compute_lnZ and pack_best_indices must receive the SAME score array."""
        from triceratops.scenarios.companion_scenarios import STPScenario

        lnZ_args: list[np.ndarray] = []
        pack_args: list[np.ndarray] = []

        def capture_lnZ(score_array, lnz_const):
            lnZ_args.append(score_array.copy())
            return float(-np.inf)

        def capture_pack(score_array, n_best):
            pack_args.append(score_array.copy())
            return np.array([0], dtype=int)

        scenario = STPScenario(FixedLDCCatalog())

        with patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_make_finite_lnL_planet), \
             patch(f"{_LNL_MOD}.compute_lnZ", side_effect=capture_lnZ), \
             patch(f"{_LNL_MOD}.pack_best_indices", side_effect=capture_pack):
            try:
                scenario.compute(transit_lc, stellar_params, 5.0, small_config)
            except Exception:
                pass

        assert lnZ_args, "compute_lnZ was not called — STP compute() did not reach lnZ step"
        assert pack_args, "pack_best_indices was not called — STP compute() did not reach idx step"

        np.testing.assert_array_equal(
            lnZ_args[0], pack_args[0],
            err_msg=(
                "BUG-07 STP: compute_lnZ and pack_best_indices received different "
                "score arrays. pack_best_indices is missing the lnprior_comp term."
            ),
        )
