"""Tests for priority-7 mission scope rationalization.

TESS is the only supported mission for prepared compute.
Kepler/K2 payloads must be rejected at both the prep boundary
(ValidationPreparer.prepare()) and the compute boundary
(PreparedValidationInputs.validate()).
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.validation.errors import UnsupportedComputeModeError
from triceratops.validation.job import PreparedValidationInputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sp() -> StellarParameters:
    return StellarParameters(
        mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
        logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
    )


def _star(tic_id: int = 12345) -> Star:
    return Star(
        tic_id=tic_id, ra_deg=10.0, dec_deg=5.0,
        tmag=11.0, jmag=10.5, hmag=10.3, kmag=10.2,
        bmag=11.5, vmag=11.2,
        stellar_params=_sp(),
        flux_ratio=1.0,
        transit_depth_required=0.01,
    )


def _field(mission: str = "TESS") -> StellarField:
    return StellarField(
        target_id=12345, mission=mission, search_radius_pixels=10,
        stars=[_star()],
    )


def _lc() -> LightCurve:
    t = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=t, flux=flux, flux_err=0.001)


def _cfg() -> Config:
    return Config(n_mc_samples=100, n_best_samples=10)


def _payload(mission: str = "TESS") -> PreparedValidationInputs:
    return PreparedValidationInputs(
        target_id=12345,
        stellar_field=_field(mission=mission),
        light_curve=_lc(),
        config=_cfg(),
        period_days=5.0,
        scenario_ids=[],
    )


# ---------------------------------------------------------------------------
# Compute boundary: PreparedValidationInputs.validate()
# ---------------------------------------------------------------------------


class TestComputeBoundaryMissionGate:
    def test_tess_payload_validate_passes(self) -> None:
        """TESS mission is accepted at the compute boundary."""
        _payload(mission="TESS").validate()

    def test_kepler_payload_validate_raises(self) -> None:
        with pytest.raises(UnsupportedComputeModeError):
            _payload(mission="Kepler").validate()

    def test_k2_payload_validate_raises(self) -> None:
        with pytest.raises(UnsupportedComputeModeError):
            _payload(mission="K2").validate()

    def test_error_message_names_mission(self) -> None:
        with pytest.raises(UnsupportedComputeModeError) as exc_info:
            _payload(mission="Kepler").validate()
        assert "Kepler" in str(exc_info.value)

    def test_error_message_names_tess_as_supported(self) -> None:
        with pytest.raises(UnsupportedComputeModeError) as exc_info:
            _payload(mission="K2").validate()
        assert "TESS" in str(exc_info.value)

    def test_compute_prepared_rejects_non_tess(self) -> None:
        """compute_prepared() propagates the UnsupportedComputeModeError."""
        from triceratops.scenarios.registry import ScenarioRegistry
        from triceratops.validation.engine import ValidationEngine

        engine = ValidationEngine(registry=ScenarioRegistry())
        with pytest.raises(UnsupportedComputeModeError):
            engine.compute_prepared(_payload(mission="Kepler"))


# ---------------------------------------------------------------------------
# Workspace: mission gate fires before any provider IO
# ---------------------------------------------------------------------------


class TestWorkspaceMissionGate:
    def _make_workspace(self, mission: str) -> object:
        from unittest.mock import MagicMock
        from triceratops.validation.workspace import ValidationWorkspace

        catalog = MagicMock()
        catalog.query_nearby_stars.return_value = _field(mission=mission)
        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            mission=mission,
            catalog_provider=catalog,
        )
        return ws

    def test_tess_workspace_passes_gate(self) -> None:
        """TESS workspace compute_probs() does not raise at the mission gate."""
        from triceratops.scenarios.registry import ScenarioRegistry
        from triceratops.validation.engine import ValidationEngine

        ws = self._make_workspace("TESS")
        ws._engine._registry = ScenarioRegistry()  # empty — no scenarios run
        # Should reach compute_prepared() without raising UnsupportedComputeModeError
        result = ws.compute_probs(_lc(), period_days=5.0)
        assert result is not None

    def test_kepler_workspace_raises_before_trilegal(self) -> None:
        """Non-TESS compute_probs() must fail before triggering provider IO."""
        from unittest.mock import MagicMock
        from triceratops.validation.workspace import ValidationWorkspace

        catalog = MagicMock()
        catalog.query_nearby_stars.return_value = _field(mission="Kepler")
        population = MagicMock()

        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            mission="Kepler",
            catalog_provider=catalog,
            population_provider=population,
        )

        with pytest.raises(UnsupportedComputeModeError):
            ws.compute_probs(_lc(), period_days=5.0)

        population.query.assert_not_called()

    def test_k2_workspace_raises(self) -> None:
        ws = self._make_workspace("K2")
        with pytest.raises(UnsupportedComputeModeError):
            ws.compute_probs(_lc(), period_days=5.0)

    def test_field_mission_is_checked_not_construction_arg(self) -> None:
        """Gate uses stellar_field.mission; a TESS-arg workspace with a
        Kepler-returning provider must still be rejected."""
        from unittest.mock import MagicMock
        from triceratops.validation.workspace import ValidationWorkspace

        catalog = MagicMock()
        # Provider returns a Kepler field even though mission="TESS" was passed
        catalog.query_nearby_stars.return_value = _field(mission="Kepler")

        ws = ValidationWorkspace(
            tic_id=12345,
            sectors=np.array([1]),
            mission="TESS",          # construction arg says TESS...
            catalog_provider=catalog,
        )
        # ...but the assembled field says Kepler — must be rejected
        with pytest.raises(UnsupportedComputeModeError):
            ws.compute_probs(_lc(), period_days=5.0)


# ---------------------------------------------------------------------------
# Prep boundary: ValidationPreparer.prepare()
# ---------------------------------------------------------------------------


class TestPrepBoundaryMissionGate:
    def _make_preparer(self) -> object:
        from unittest.mock import MagicMock
        from triceratops.validation.preparer import ValidationPreparer

        preparer = ValidationPreparer(
            catalog_provider=MagicMock(),
            aperture_provider=MagicMock(),
        )
        return preparer

    def test_tess_is_accepted(self) -> None:
        """prepare() with mission='TESS' does not raise at the mission gate."""
        from unittest.mock import MagicMock, patch
        from triceratops.validation.preparer import ValidationPreparer

        catalog = MagicMock()
        catalog.query_nearby_stars.return_value = _field(mission="TESS")

        preparer = ValidationPreparer(catalog_provider=catalog)
        # Patch downstream IO so the test stops after the mission gate passes
        with patch.object(preparer, "_population", None):
            try:
                preparer.prepare(
                    target_id=12345,
                    sectors=np.array([1]),
                    light_curve=_lc(),
                    config=_cfg(),
                    period_days=5.0,
                    mission="TESS",
                    scenario_ids=[],
                )
            except UnsupportedComputeModeError:
                pytest.fail("TESS should not be rejected by mission gate")
            except Exception:
                pass  # other errors from stub providers are fine

    def test_kepler_raises_at_mission_gate(self) -> None:
        preparer = self._make_preparer()
        with pytest.raises(UnsupportedComputeModeError):
            preparer.prepare(
                target_id=12345,
                sectors=np.array([1]),
                light_curve=_lc(),
                config=_cfg(),
                period_days=5.0,
                mission="Kepler",
            )

    def test_k2_raises_at_mission_gate(self) -> None:
        preparer = self._make_preparer()
        with pytest.raises(UnsupportedComputeModeError):
            preparer.prepare(
                target_id=12345,
                sectors=np.array([1]),
                light_curve=_lc(),
                config=_cfg(),
                period_days=5.0,
                mission="K2",
            )

    def test_error_message_names_rejected_mission(self) -> None:
        preparer = self._make_preparer()
        with pytest.raises(UnsupportedComputeModeError) as exc_info:
            preparer.prepare(
                target_id=12345,
                sectors=np.array([1]),
                light_curve=_lc(),
                config=_cfg(),
                period_days=5.0,
                mission="Kepler",
            )
        assert "Kepler" in str(exc_info.value)

    def test_mission_gate_fires_before_io(self) -> None:
        """Mission gate must fire before any catalog query (no provider calls)."""
        from unittest.mock import MagicMock
        from triceratops.validation.preparer import ValidationPreparer

        catalog = MagicMock()
        preparer = ValidationPreparer(catalog_provider=catalog)

        with pytest.raises(UnsupportedComputeModeError):
            preparer.prepare(
                target_id=12345,
                sectors=np.array([1]),
                light_curve=_lc(),
                config=_cfg(),
                period_days=5.0,
                mission="K2",
            )

        catalog.query_nearby_stars.assert_not_called()

    def test_provider_returning_wrong_mission_field_is_rejected(self) -> None:
        """If the catalog returns a Kepler field despite mission='TESS' being passed,
        the field-level mission check catches it before TRILEGAL fetch."""
        from unittest.mock import MagicMock
        from triceratops.validation.preparer import ValidationPreparer

        catalog = MagicMock()
        # Provider returns a Kepler field even though TESS was requested
        catalog.query_nearby_stars.return_value = _field(mission="Kepler")
        population = MagicMock()

        preparer = ValidationPreparer(
            catalog_provider=catalog,
            population_provider=population,
        )
        with pytest.raises(UnsupportedComputeModeError) as exc_info:
            preparer.prepare(
                target_id=12345,
                sectors=np.array([1]),
                light_curve=_lc(),
                config=_cfg(),
                period_days=5.0,
                mission="TESS",
                scenario_ids=[],
            )
        # Error names the field mission, not the argument
        assert "Kepler" in str(exc_info.value)
        # TRILEGAL was never fetched
        population.query.assert_not_called()
