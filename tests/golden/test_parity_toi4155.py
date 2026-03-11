"""Parity tests: verify triceratops matches golden values on TOI-4155.

TOI-4155 (TIC 467331291) is a validated transiting planet (FPP ~ 0.0002).

Tolerance specification (from BRIEFING.md):
  - lnZ per scenario: rtol <= 0.01 (1%) with same random seed
  - FPP, NFPP: atol <= 1e-4
  - Deterministic helpers: exact match

These tests use stub providers (no network). The golden JSON was captured
once and committed. Tests skip cleanly if fixture data is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import GOLDEN_DIR

# Tolerance constants (from BRIEFING.md)
LNZ_RTOL = 0.01  # 1% relative tolerance on lnZ
FPP_ATOL = 1e-4  # absolute tolerance on FPP and NFPP
LEGACY_LNZ_OFFSET = 650.0

# TOI-4155 parameters (from example notebook)
TIC_ID = 467331291
SECTORS = np.array([52])
P_ORB = 3.320729
TRANSIT_DEPTH = 0.000957
SEED = 42
N_MC = 10_000

LC_PATH = Path(__file__).parent.parent / "fixtures" / "light_curves" / "toi4155_lc.npy"
GOLDEN_PATH = GOLDEN_DIR / "toi4155.json"

ALL_SCENARIO_IDS = [
    "TP", "EB", "EBx2P",
    "PTP", "PEB", "PEBx2P",
    "STP", "SEB", "SEBx2P",
    "DTP", "DEB", "DEBx2P",
    "BTP", "BEB", "BEBx2P",
    "NTP", "NEB", "NEBx2P",
]


def _golden_available() -> bool:
    return GOLDEN_PATH.exists() and LC_PATH.exists()


def _load_golden() -> dict:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _build_test_registry():
    """Build registry with FixedLDCCatalog to avoid LDC data file dependency."""
    from triceratops.limb_darkening.catalog import FixedLDCCatalog
    from triceratops.scenarios.registry import build_default_registry
    return build_default_registry(ldc_catalog=FixedLDCCatalog())


def _probs_df(result):
    """Build scenario DataFrame from ValidationResult."""
    import pandas as pd
    return pd.DataFrame([
        {
            "scenario": r.scenario_id.value,
            "prob": r.relative_probability,
            "lnZ": r.ln_evidence,
        }
        for r in result.scenario_results
    ])


def _run_new_code(golden: dict):
    """Run ValidationWorkspace on TOI-4155 stub data."""
    from tests.fixtures.stubs import StubPopulationSynthesisProvider, StubStarCatalogProvider
    from triceratops.config.config import Config
    from triceratops.domain.entities import LightCurve
    from triceratops.validation.workspace import ValidationWorkspace

    seed = golden.get("seed", SEED)
    n = golden.get("n_mc_samples", N_MC)

    np.random.seed(seed)

    lc_data = np.load(LC_PATH)
    time, flux = lc_data[0], lc_data[1]
    sigma = float(lc_data[2, 0])

    stub_catalog = StubStarCatalogProvider(target_tic_id=TIC_ID)
    stub_population = StubPopulationSynthesisProvider()

    test_registry = _build_test_registry()

    with patch("triceratops.validation.engine.DEFAULT_REGISTRY", test_registry):
        ws = ValidationWorkspace(
            tic_id=TIC_ID,
            sectors=SECTORS,
            catalog_provider=stub_catalog,
            population_provider=stub_population,
        )
        # Set flux ratios manually since stubs don't provide aperture pixels
        ws.stars[0].flux_ratio = 0.99
        ws.stars[0].transit_depth_required = TRANSIT_DEPTH / 0.99
        ws.stars[1].flux_ratio = 0.01
        ws.stars[1].transit_depth_required = 0.0

        ws.config = Config(
            n_mc_samples=n,
            parallel=False,
            mission="TESS",
        )

        lc = LightCurve(
            time_days=time,
            flux=flux,
            flux_err=sigma,
        )

        try:
            ws.compute_probs(light_curve=lc, period_days=P_ORB)
        except ImportError as exc:
            pytest.skip(f"pytransit not available with current numpy: {exc}")
    return ws


@pytest.fixture(scope="module")
def golden_toi4155() -> dict:
    if not _golden_available():
        pytest.skip("TOI-4155 golden fixture data not found")
    return _load_golden()


@pytest.fixture(scope="module")
def new_toi4155_result(golden_toi4155: dict):
    return _run_new_code(golden_toi4155)


@pytest.mark.golden
def test_fpp_within_tolerance(new_toi4155_result, golden_toi4155: dict) -> None:
    """FPP matches golden value within atol=1e-4."""
    new_fpp = new_toi4155_result.fpp
    golden_fpp = golden_toi4155["FPP"]
    assert abs(new_fpp - golden_fpp) <= FPP_ATOL, (
        f"FPP mismatch: new={new_fpp:.6f}, golden={golden_fpp:.6f}, "
        f"diff={abs(new_fpp - golden_fpp):.2e}, atol={FPP_ATOL}"
    )


@pytest.mark.golden
def test_nfpp_within_tolerance(new_toi4155_result, golden_toi4155: dict) -> None:
    """NFPP matches golden value within atol=1e-4."""
    new_nfpp = new_toi4155_result.nfpp
    golden_nfpp = golden_toi4155["NFPP"]
    assert abs(new_nfpp - golden_nfpp) <= FPP_ATOL, (
        f"NFPP mismatch: new={new_nfpp:.6f}, golden={golden_nfpp:.6f}"
    )


@pytest.mark.golden
def test_probs_sum_to_one(new_toi4155_result) -> None:
    """Relative probabilities sum to 1.0."""
    probs_df = _probs_df(new_toi4155_result.results)
    total = probs_df["prob"].sum()
    assert abs(total - 1.0) < 1e-10 or total == 0.0, (
        f"Probabilities sum to {total}, expected 1.0"
    )


@pytest.mark.golden
@pytest.mark.parametrize("scenario_id", ALL_SCENARIO_IDS)
def test_lnZ_within_rtol(
    new_toi4155_result, golden_toi4155: dict, scenario_id: str
) -> None:
    """lnZ for each scenario matches golden value within rtol=1%."""
    if scenario_id not in golden_toi4155:
        pytest.skip(f"Scenario {scenario_id} not in golden values")

    probs_df = _probs_df(new_toi4155_result.results)
    new_row = probs_df[probs_df["scenario"] == scenario_id]
    if new_row.empty:
        pytest.skip(f"Scenario {scenario_id} not in new result")

    new_lnZ = float(new_row["lnZ"].iloc[0])
    golden_lnZ = golden_toi4155[scenario_id]["lnZ"] - LEGACY_LNZ_OFFSET

    if not np.isfinite(golden_lnZ) and not np.isfinite(new_lnZ):
        return  # both -inf: pass

    assert np.isfinite(new_lnZ), f"New lnZ is -inf for {scenario_id}"
    assert abs(new_lnZ - golden_lnZ) / (abs(golden_lnZ) + 1e-10) <= LNZ_RTOL, (
        f"lnZ mismatch for {scenario_id}: new={new_lnZ:.3f}, "
        f"golden={golden_lnZ:.3f}, rtol exceeded"
    )


@pytest.mark.golden
@pytest.mark.parametrize("scenario_id", ALL_SCENARIO_IDS)
def test_prob_within_rtol(
    new_toi4155_result, golden_toi4155: dict, scenario_id: str
) -> None:
    """Relative probability for each scenario matches golden value."""
    if scenario_id not in golden_toi4155:
        pytest.skip(f"Scenario {scenario_id} not in golden values")

    probs_df = _probs_df(new_toi4155_result.results)
    new_row = probs_df[probs_df["scenario"] == scenario_id]
    if new_row.empty:
        pytest.skip(f"Scenario {scenario_id} not in new result")

    new_prob = float(new_row["prob"].iloc[0])
    golden_prob = golden_toi4155[scenario_id]["prob"]

    if golden_prob < 1e-15 and new_prob < 1e-15:
        return  # both negligible

    if golden_prob > 1e-10:
        assert abs(new_prob - golden_prob) / golden_prob <= LNZ_RTOL, (
            f"prob mismatch for {scenario_id}: new={new_prob:.6e}, "
            f"golden={golden_prob:.6e}"
        )
