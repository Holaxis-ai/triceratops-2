"""Unit tests for triceratops.scenarios._eb_branching.

Tests cover:
  - q<0.95 draws land in the standard branch (mask=True, mask_twin=False)
  - q>=0.95 draws land in the twin branch (mask=False, mask_twin=True)
  - Threshold boundary (q=0.95 exactly) goes to the twin branch
  - Collision-flagged draws are excluded from both branches
  - Draws with ptra>1 (non-transiting geometry) are excluded
  - extra_mask gates both branches independently
  - Custom q_threshold is respected
  - All-False masks are handled (no samples in a branch)
  - Mixed population produces the correct split
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.scenarios._eb_branching import build_eb_branch_masks
from triceratops.scenarios.constants import EB_Q_TWIN_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(N: int, q_values=None, inc_deg=None, ptra=None, coll=None,
                 ptra_twin=None, coll_twin=None, extra_mask=None):
    """Return arrays suitable for build_eb_branch_masks with sane defaults."""
    rng = np.random.default_rng(42)
    if q_values is None:
        q_values = np.full(N, 0.5)
    qs = np.asarray(q_values, dtype=float)

    if inc_deg is None:
        inc_deg = np.full(N, 89.0)  # near-edge-on: transit likely
    incs = np.asarray(inc_deg, dtype=float)

    # ptra small enough that arccos(ptra) < 89 deg so inclination passes
    if ptra is None:
        ptra = np.full(N, 0.1)
    if coll is None:
        coll = np.zeros(N, dtype=bool)
    if ptra_twin is None:
        ptra_twin = np.full(N, 0.08)
    if coll_twin is None:
        coll_twin = np.zeros(N, dtype=bool)

    return dict(
        qs=qs, incs=incs, ptra=ptra, coll=coll,
        ptra_twin=ptra_twin, coll_twin=coll_twin,
        extra_mask=extra_mask,
    )


# ---------------------------------------------------------------------------
# Test: basic q-branching
# ---------------------------------------------------------------------------

class TestQBranching:
    def test_q_below_threshold_activates_standard_branch(self) -> None:
        """All q < 0.95 → mask all True, mask_twin all False."""
        N = 50
        kw = _make_inputs(N, q_values=np.full(N, 0.90))
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert np.all(mask), "q<0.95 draws should be in standard branch"
        assert not np.any(mask_twin), "q<0.95 draws must not be in twin branch"

    def test_q_at_threshold_activates_twin_branch(self) -> None:
        """q == 0.95 exactly should go to twin branch (>=)."""
        N = 20
        kw = _make_inputs(N, q_values=np.full(N, EB_Q_TWIN_THRESHOLD))
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask), "q=0.95 must not activate standard branch"
        assert np.all(mask_twin), "q=0.95 must activate twin branch"

    def test_q_above_threshold_activates_twin_branch(self) -> None:
        """All q > 0.95 → mask all False, mask_twin all True."""
        N = 30
        kw = _make_inputs(N, q_values=np.full(N, 0.99))
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask), "q>0.95 must not activate standard branch"
        assert np.all(mask_twin), "q>0.95 must activate twin branch"

    def test_mixed_q_population_split(self) -> None:
        """Mixed q splits draws correctly across both branches."""
        N = 100
        qs = np.concatenate([
            np.full(60, 0.5),   # standard branch
            np.full(40, 0.97),  # twin branch
        ])
        kw = _make_inputs(N, q_values=qs)
        mask, mask_twin = build_eb_branch_masks(**kw)
        # Standard branch active for first 60
        assert np.all(mask[:60])
        assert not np.any(mask[60:])
        # Twin branch active for last 40
        assert not np.any(mask_twin[:60])
        assert np.all(mask_twin[60:])

    def test_no_draw_in_both_branches_simultaneously(self) -> None:
        """A given draw must not appear in both branches at once."""
        N = 200
        rng = np.random.default_rng(7)
        qs = rng.uniform(0.0, 1.5, size=N)
        kw = _make_inputs(N, q_values=qs)
        mask, mask_twin = build_eb_branch_masks(**kw)
        both = mask & mask_twin
        assert not np.any(both), "No draw may be in both branches"


# ---------------------------------------------------------------------------
# Test: collision exclusion
# ---------------------------------------------------------------------------

class TestCollisionExclusion:
    def test_colliding_draws_excluded_from_standard_branch(self) -> None:
        """coll=True draws with q<0.95 must be excluded."""
        N = 10
        coll = np.ones(N, dtype=bool)  # all collide
        kw = _make_inputs(N, q_values=np.full(N, 0.5), coll=coll)
        mask, _ = build_eb_branch_masks(**kw)
        assert not np.any(mask), "Colliding draws must not pass standard mask"

    def test_colliding_draws_excluded_from_twin_branch(self) -> None:
        """coll_twin=True draws with q>=0.95 must be excluded."""
        N = 10
        coll_twin = np.ones(N, dtype=bool)
        kw = _make_inputs(N, q_values=np.full(N, 0.97), coll_twin=coll_twin)
        _, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask_twin), "Colliding twin draws must not pass twin mask"

    def test_partial_collision_flags(self) -> None:
        """Only non-colliding draws pass."""
        N = 6
        qs = np.full(N, 0.5)
        coll = np.array([False, True, False, True, False, False])
        kw = _make_inputs(N, q_values=qs, coll=coll)
        mask, _ = build_eb_branch_masks(**kw)
        expected = ~coll  # colliding draws excluded
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Test: non-transiting geometry (ptra > 1.0)
# ---------------------------------------------------------------------------

class TestNonTransitingGeometry:
    def test_ptra_gt_one_excluded_from_standard_branch(self) -> None:
        """ptra > 1.0 means geometry is invalid; inclination check excludes draw."""
        N = 5
        ptra = np.full(N, 1.5)  # invalid: transit probability > 1
        # build_transit_mask sets inc_min=90 for ptra>1, so only inc>=90 passes
        kw = _make_inputs(N, q_values=np.full(N, 0.5), ptra=ptra,
                         inc_deg=np.full(N, 89.0))  # inc<90 → excluded
        mask, _ = build_eb_branch_masks(**kw)
        assert not np.any(mask)

    def test_ptra_gt_one_excluded_from_twin_branch(self) -> None:
        """ptra_twin > 1.0 excludes draw from twin branch."""
        N = 5
        ptra_twin = np.full(N, 2.0)
        kw = _make_inputs(N, q_values=np.full(N, 0.99), ptra_twin=ptra_twin,
                         inc_deg=np.full(N, 89.0))
        _, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask_twin)


# ---------------------------------------------------------------------------
# Test: extra_mask
# ---------------------------------------------------------------------------

class TestExtraMask:
    def test_extra_mask_gates_standard_branch(self) -> None:
        """extra_mask=False excludes draw from standard branch regardless of q."""
        N = 10
        extra = np.zeros(N, dtype=bool)  # all blocked
        kw = _make_inputs(N, q_values=np.full(N, 0.5), extra_mask=extra)
        mask, _ = build_eb_branch_masks(**kw)
        assert not np.any(mask)

    def test_extra_mask_gates_twin_branch(self) -> None:
        """extra_mask=False excludes draw from twin branch regardless of q."""
        N = 10
        extra = np.zeros(N, dtype=bool)
        kw = _make_inputs(N, q_values=np.full(N, 0.99), extra_mask=extra)
        _, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask_twin)

    def test_extra_mask_partial(self) -> None:
        """extra_mask only passes allowed draws into each branch."""
        N = 6
        qs = np.array([0.5, 0.5, 0.5, 0.99, 0.99, 0.99])
        extra = np.array([True, False, True, True, False, True])
        kw = _make_inputs(N, q_values=qs, extra_mask=extra)
        mask, mask_twin = build_eb_branch_masks(**kw)
        # Standard: draws 0,2 pass (q<0.95 AND extra); draw 1 blocked by extra
        assert mask[0] and not mask[1] and mask[2]
        assert not mask[3] and not mask[4] and not mask[5]
        # Twin: draws 3,5 pass (q>=0.95 AND extra); draw 4 blocked by extra
        assert not mask_twin[0] and not mask_twin[1] and not mask_twin[2]
        assert mask_twin[3] and not mask_twin[4] and mask_twin[5]

    def test_none_extra_mask_is_accepted(self) -> None:
        """extra_mask=None (default) must not raise and must work."""
        N = 10
        kw = _make_inputs(N, q_values=np.full(N, 0.5))
        # Remove extra_mask from dict so it uses the default None
        kw.pop("extra_mask")
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert mask.shape == (N,)
        assert mask_twin.shape == (N,)


# ---------------------------------------------------------------------------
# Test: custom q_threshold
# ---------------------------------------------------------------------------

class TestCustomThreshold:
    def test_custom_threshold_respected(self) -> None:
        """A non-default threshold partitions draws correctly."""
        N = 10
        qs = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 0.6, 0.8, 0.85, 0.95, 0.99])
        kw = _make_inputs(N, q_values=qs)
        # Use threshold of 0.8 instead of 0.95
        mask, mask_twin = build_eb_branch_masks(
            q_threshold=0.8, **kw,
        )
        q_lt = qs < 0.8
        q_ge = qs >= 0.8
        # Draws in standard branch must have q<0.8 (and pass geometry)
        assert not np.any(mask & ~q_lt), "Standard mask must not include q>=0.8 draws"
        assert not np.any(mask_twin & ~q_ge), "Twin mask must not include q<0.8 draws"

    def test_default_threshold_is_eb_q_twin_threshold(self) -> None:
        """Default threshold equals EB_Q_TWIN_THRESHOLD (0.95)."""
        N = 4
        qs = np.array([0.94, 0.95, 0.96, 0.80])
        kw = _make_inputs(N, q_values=qs)
        mask, mask_twin = build_eb_branch_masks(**kw)
        # 0.94 → standard; 0.95, 0.96, 0.80 treated below separately
        assert mask[0] and not mask_twin[0]   # q=0.94 → standard
        assert not mask[1] and mask_twin[1]   # q=0.95 → twin (boundary)
        assert not mask[2] and mask_twin[2]   # q=0.96 → twin
        assert mask[3] and not mask_twin[3]   # q=0.80 → standard


# ---------------------------------------------------------------------------
# Test: output shapes and types
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_output_shapes(self) -> None:
        N = 77
        kw = _make_inputs(N)
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert mask.shape == (N,)
        assert mask_twin.shape == (N,)

    def test_output_dtypes_are_bool(self) -> None:
        N = 20
        kw = _make_inputs(N)
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert mask.dtype == bool
        assert mask_twin.dtype == bool

    def test_all_masked_out_returns_all_false(self) -> None:
        """When nothing can transit (all coll=True), both masks are all False."""
        N = 15
        coll = np.ones(N, dtype=bool)
        coll_twin = np.ones(N, dtype=bool)
        kw = _make_inputs(N, coll=coll, coll_twin=coll_twin)
        # Mix of q values
        kw["qs"] = np.concatenate([np.full(8, 0.5), np.full(7, 0.99)])
        mask, mask_twin = build_eb_branch_masks(**kw)
        assert not np.any(mask)
        assert not np.any(mask_twin)
