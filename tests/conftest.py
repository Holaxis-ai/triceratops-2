"""Shared pytest fixtures for all test levels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.stubs import StubPopulationSynthesisProvider, StubStarCatalogProvider
from tests.fixtures.synthetic import make_flat_lightcurve, make_transit_lightcurve
from triceratops_new.config.config import Config, MissionConfig
from triceratops_new.domain.entities import LightCurve, StellarField

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"


@pytest.fixture
def default_config() -> Config:
    return Config(n_mc_samples=500, n_best_samples=50)  # small N for speed in unit tests


@pytest.fixture
def tess_mission_config() -> MissionConfig:
    return MissionConfig.for_mission("TESS")


@pytest.fixture
def transit_lc() -> LightCurve:
    return make_transit_lightcurve(R_p_rearth=2.0, P_orb_days=5.0, rng_seed=42)


@pytest.fixture
def flat_lc() -> LightCurve:
    return make_flat_lightcurve(rng_seed=99)


@pytest.fixture
def stub_catalog() -> StubStarCatalogProvider:
    return StubStarCatalogProvider()


@pytest.fixture
def stub_population() -> StubPopulationSynthesisProvider:
    return StubPopulationSynthesisProvider()


@pytest.fixture
def stellar_field(stub_catalog: StubStarCatalogProvider) -> StellarField:
    return stub_catalog.query_nearby_stars(12345678, 10, "TESS")


def load_golden(name: str) -> dict:
    """Load a golden-value dict from tests/fixtures/golden/{name}.json.

    Args:
        name: Basename without extension, e.g. "toi4051_fpp".

    Raises:
        FileNotFoundError: If the golden file has not been captured yet.
    """
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Golden value file not found: {path}\n"
            f"Run scripts/capture_golden_values.py to generate it."
        )
    with open(path) as f:
        return json.load(f)
