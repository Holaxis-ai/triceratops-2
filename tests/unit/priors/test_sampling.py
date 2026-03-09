"""Unit tests for prior sampling functions (P1-007)."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_companion_mass_ratio,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
    sample_planet_radius,
)


@pytest.mark.unit
class TestArgPeriastron:
    def test_known_values(self) -> None:
        u = np.array([0.0, 0.5, 1.0])
        result = sample_arg_periastron(u)
        np.testing.assert_allclose(result, [0.0, 180.0, 360.0])


@pytest.mark.unit
class TestInclination:
    def test_range(self) -> None:
        u = np.random.default_rng(42).random(1000)
        result = sample_inclination(u, lower=0.0, upper=90.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 90.0)

    def test_cosine_distribution(self) -> None:
        """cos(inc) should be roughly uniform when inc is sampled correctly."""
        u = np.random.default_rng(42).random(10000)
        inc = sample_inclination(u)
        cos_inc = np.cos(np.deg2rad(inc))
        # Bin cos(inc) into 10 equal bins; expect roughly equal counts
        counts, _ = np.histogram(cos_inc, bins=10, range=(0.0, 1.0))
        # Each bin should have ~1000 samples; allow 30% deviation
        assert np.all(counts > 700), f"Distribution not uniform in cos(inc): {counts}"
        assert np.all(counts < 1300), f"Distribution not uniform in cos(inc): {counts}"

    def test_near_zero_for_u_zero(self) -> None:
        result = sample_inclination(np.zeros(10))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_near_90_for_u_near_1(self) -> None:
        result = sample_inclination(np.ones(10) * 0.9999)
        assert np.all(result > 80.0)


@pytest.mark.unit
class TestEccentricity:
    def test_planet_in_0_1(self) -> None:
        u = np.random.default_rng(42).random(1000)
        e = sample_eccentricity(u, planet=True)
        assert np.all(e >= 0.0)
        assert np.all(e < 1.0)

    def test_planet_mean_range(self) -> None:
        u = np.random.default_rng(42).random(10000)
        e = sample_eccentricity(u, planet=True)
        assert 0.05 <= e.mean() <= 0.25

    def test_eb_in_0_1(self) -> None:
        u = np.random.default_rng(42).random(1000)
        e = sample_eccentricity(u, planet=False, period=5.0)
        assert np.all(e >= 0.0)
        assert np.all(e < 1.0)


@pytest.mark.unit
class TestPlanetRadius:
    def test_flat_uniform(self) -> None:
        u = np.random.default_rng(42).random(10000)
        r = sample_planet_radius(u, host_mass=1.0, flat=True)
        assert np.all(r >= 0.5)
        assert np.all(r <= 20.0)
        # Should be roughly uniform
        assert abs(r.mean() - 10.25) < 0.5

    def test_nonflat_shape_matches_u(self) -> None:
        u = np.random.default_rng(42).random(500)
        r = sample_planet_radius(u, host_mass=1.0, flat=False)
        assert r.shape == (500,)

    def test_nonflat_in_valid_range(self) -> None:
        u = np.random.default_rng(42).random(10000)
        r = sample_planet_radius(u, host_mass=1.0, flat=False)
        assert np.all(r >= 0.5)
        assert np.all(r <= 20.0)


@pytest.mark.unit
class TestMassRatio:
    def test_positive(self) -> None:
        u = np.random.default_rng(42).random(1000)
        q = sample_mass_ratio(u, primary_mass=1.0)
        assert np.all(q > 0)
        assert np.all(q <= 1.0)

    def test_low_mass_primary(self) -> None:
        u = np.random.default_rng(42).random(100)
        q = sample_mass_ratio(u, primary_mass=0.05)
        np.testing.assert_array_equal(q, 1.0)


@pytest.mark.unit
class TestCompanionMassRatio:
    def test_positive(self) -> None:
        u = np.random.default_rng(42).random(1000)
        q = sample_companion_mass_ratio(u, primary_mass=1.0)
        assert np.all(q > 0)
        assert np.all(q <= 1.0)


@pytest.mark.unit
class TestDeterminism:
    def test_all_functions_deterministic(self) -> None:
        u = np.random.default_rng(42).random(100)

        funcs = [
            lambda x: sample_arg_periastron(x),
            lambda x: sample_inclination(x),
            lambda x: sample_eccentricity(x, planet=True),
            lambda x: sample_eccentricity(x, planet=False, period=5.0),
            lambda x: sample_planet_radius(x.copy(), host_mass=1.0, flat=False),
            lambda x: sample_planet_radius(x.copy(), host_mass=0.3, flat=False),
            lambda x: sample_mass_ratio(x.copy(), primary_mass=1.0),
            lambda x: sample_companion_mass_ratio(x.copy(), primary_mass=1.0),
        ]

        for fn in funcs:
            r1 = fn(u.copy())
            r2 = fn(u.copy())
            np.testing.assert_array_equal(r1, r2, err_msg=f"{fn} not deterministic")
