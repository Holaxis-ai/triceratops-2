"""Tests for MoluscData domain object and load_molusc_file IO function."""
from __future__ import annotations

import os
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.molusc import MoluscData
from triceratops.domain.result import ValidationResult
from triceratops.domain.value_objects import StellarParameters
from triceratops.io.molusc import load_molusc_file
from triceratops.scenarios.registry import ScenarioRegistry
from triceratops.validation.engine import ValidationEngine
from triceratops.validation.job import PreparedValidationInputs


# ---------------------------------------------------------------------------
# MoluscData construction and validation
# ---------------------------------------------------------------------------

class TestMoluscData:
    def test_construct_matching_lengths(self) -> None:
        md = MoluscData(
            semi_major_axis_au=np.array([1.0, 2.0]),
            eccentricity=np.array([0.0, 0.1]),
            mass_ratio=np.array([0.5, 0.6]),
        )
        assert len(md.semi_major_axis_au) == 2

    def test_mismatched_semi_major_axis(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MoluscData(
                semi_major_axis_au=np.array([1.0]),
                eccentricity=np.array([0.0, 0.1]),
                mass_ratio=np.array([0.5, 0.6]),
            )

    def test_mismatched_eccentricity(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MoluscData(
                semi_major_axis_au=np.array([1.0, 2.0]),
                eccentricity=np.array([0.0]),
                mass_ratio=np.array([0.5, 0.6]),
            )

    def test_mismatched_mass_ratio(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MoluscData(
                semi_major_axis_au=np.array([1.0, 2.0]),
                eccentricity=np.array([0.0, 0.1]),
                mass_ratio=np.array([0.5]),
            )

    def test_frozen(self) -> None:
        md = MoluscData(
            semi_major_axis_au=np.array([1.0]),
            eccentricity=np.array([0.0]),
            mass_ratio=np.array([0.5]),
        )
        with pytest.raises(FrozenInstanceError):
            md.semi_major_axis_au = np.array([9.0])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_molusc_file
# ---------------------------------------------------------------------------

class TestLoadMoluscFile:
    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "molusc.csv"
        csv.write_text(
            "semi-major axis(AU),eccentricity,mass ratio\n"
            "10.0,0.1,0.4\n"
            "20.0,0.2,0.5\n"
        )
        md = load_molusc_file(csv)
        np.testing.assert_array_almost_equal(md.semi_major_axis_au, [10.0, 20.0])
        np.testing.assert_array_almost_equal(md.eccentricity, [0.1, 0.2])
        np.testing.assert_array_almost_equal(md.mass_ratio, [0.4, 0.5])

    def test_missing_column(self, tmp_path: Path) -> None:
        csv = tmp_path / "bad.csv"
        csv.write_text("semi-major axis(AU),eccentricity\n1.0,0.1\n")
        with pytest.raises(ValueError, match=str(csv)):
            load_molusc_file(csv)

    def test_empty_file(self, tmp_path: Path) -> None:
        csv = tmp_path / "empty.csv"
        csv.write_text("semi-major axis(AU),eccentricity,mass ratio\n")
        with pytest.raises(ValueError, match=str(csv)):
            load_molusc_file(csv)

    def test_non_numeric_column_raises_with_path(self, tmp_path: Path) -> None:
        csv = tmp_path / "bad_numeric.csv"
        csv.write_text(
            "semi-major axis(AU),eccentricity,mass ratio\n"
            "foo,0.1,0.4\n"
        )
        with pytest.raises(ValueError, match=str(csv)):
            load_molusc_file(csv)

    def test_nan_in_column_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "nan.csv"
        csv.write_text(
            "semi-major axis(AU),eccentricity,mass ratio\n"
            "10.0,NaN,0.4\n"
        )
        with pytest.raises(ValueError, match="non-finite"):
            load_molusc_file(csv)

    def test_inf_in_column_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "inf.csv"
        csv.write_text(
            "semi-major axis(AU),eccentricity,mass ratio\n"
            "inf,0.1,0.4\n"
        )
        with pytest.raises(ValueError, match="non-finite"):
            load_molusc_file(csv)


class TestMoluscDataDtypeValidation:
    def test_integer_array_raises(self) -> None:
        """Integer arrays must be rejected — numpy ints are not floating-point."""
        with pytest.raises(ValueError, match="floating-point"):
            MoluscData(
                semi_major_axis_au=np.array([10, 20], dtype=int),
                eccentricity=np.array([0.0, 0.1]),
                mass_ratio=np.array([0.5, 0.6]),
            )

    def test_object_array_raises(self) -> None:
        """Object arrays (e.g. strings coerced by np.asarray) must be rejected."""
        with pytest.raises(ValueError, match="floating-point"):
            MoluscData(
                semi_major_axis_au=np.array(["foo", "bar"]),
                eccentricity=np.array([0.0, 0.1]),
                mass_ratio=np.array([0.5, 0.6]),
            )

    def test_nan_in_array_raises(self) -> None:
        """Directly constructed MoluscData with NaN must be rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            MoluscData(
                semi_major_axis_au=np.array([float("nan"), 20.0]),
                eccentricity=np.array([0.0, 0.1]),
                mass_ratio=np.array([0.5, 0.6]),
            )

    def test_inf_in_array_raises(self) -> None:
        """Directly constructed MoluscData with inf must be rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            MoluscData(
                semi_major_axis_au=np.array([20.0, 30.0]),
                eccentricity=np.array([float("inf"), 0.1]),
                mass_ratio=np.array([0.5, 0.6]),
            )


# ---------------------------------------------------------------------------
# CWD independence with molusc_data
# ---------------------------------------------------------------------------

class TestCwdIndependenceWithMolusc:
    def test_compute_prepared_with_molusc_from_foreign_cwd(self) -> None:
        """compute_prepared() with molusc_data does not crash from a foreign CWD."""
        star = Star(
            tic_id=99999,
            ra_deg=100.0, dec_deg=20.0,
            tmag=11.0, jmag=10.3, hmag=10.1, kmag=10.0,
            bmag=11.5, vmag=11.2,
            stellar_params=StellarParameters(
                mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
                logg=4.4, metallicity_dex=0.0, parallax_mas=12.0,
            ),
            flux_ratio=1.0,
            transit_depth_required=0.01,
        )
        sf = StellarField(
            target_id=99999, mission="TESS",
            search_radius_pixels=10, stars=[star],
        )
        t = np.linspace(-0.1, 0.1, 50)
        flux = np.ones(50)
        flux[20:30] = 0.999
        lc = LightCurve(time_days=t, flux=flux, flux_err=0.001)
        cfg = Config(n_mc_samples=100, n_best_samples=10, n_workers=0)
        molusc_data = MoluscData(
            semi_major_axis_au=np.array([20.0, 30.0, 50.0]),
            eccentricity=np.array([0.0, 0.1, 0.2]),
            mass_ratio=np.array([0.4, 0.5, 0.6]),
        )

        # Empty registry: no scenarios run, but engine plumbing still executes.
        registry = ScenarioRegistry()
        engine = ValidationEngine(registry=registry)
        prepared = PreparedValidationInputs(
            target_id=99999,
            stellar_field=sf,
            light_curve=lc,
            config=cfg,
            period_days=5.0,
            molusc_data=molusc_data,
        )

        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                result = engine.compute_prepared(prepared)
        finally:
            os.chdir(original_cwd)

        assert isinstance(result, ValidationResult)
