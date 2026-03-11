"""Tests for assemble_light_curve() pipeline function."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from triceratops.assembly.errors import AssemblyLightCurveError
from triceratops.assembly.pipelines.lightcurve import assemble_light_curve
from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.lightcurve.result import LightCurvePreparationResult


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_light_curve() -> LightCurve:
    t = np.linspace(-0.1, 0.1, 50)
    flux = np.ones_like(t)
    return LightCurve(time_days=t, flux=flux, flux_err=0.001)


def _make_ephemeris() -> Ephemeris:
    return Ephemeris(period_days=5.0, t0_btjd=1000.0)


def _make_result(
    warnings: list[str] | None = None,
) -> LightCurvePreparationResult:
    return LightCurvePreparationResult(
        light_curve=_make_light_curve(),
        ephemeris=_make_ephemeris(),
        sectors_used=(1, 2),
        cadence_used="2min",
        warnings=warnings or [],
    )


class StubLcSource:
    """Returns a fixed LightCurvePreparationResult."""

    def __init__(
        self,
        result: LightCurvePreparationResult | None = None,
    ) -> None:
        self._result = result or _make_result()
        self.call_count = 0
        self.last_ephemeris: object = None
        self.last_config: object = None

    def prepare(
        self, ephemeris: object, config: LightCurveConfig,
    ) -> LightCurvePreparationResult:
        self.call_count += 1
        self.last_ephemeris = ephemeris
        self.last_config = config
        return self._result


class BoomLcSource:
    """Raises on prepare()."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("lc exploded")

    def prepare(self, ephemeris: object, config: object) -> object:
        raise self._exc


class StubArtifactStore:
    """Records put_prepared_lc calls and returns a fixed artifact ID."""

    def __init__(self, artifact_id: str = "art-lc-001") -> None:
        self._artifact_id = artifact_id
        self.stored: list[LightCurve] = []

    def put_raw_lc(self, data: object) -> str:
        return "raw-id"

    def put_prepared_lc(self, lc: LightCurve) -> str:
        self.stored.append(lc)
        return self._artifact_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAssembleLightCurveHappyPath:
    def test_returns_light_curve_from_source(self) -> None:
        source = StubLcSource()
        eph = _make_ephemeris()
        lc, label, warnings, artifact_ids = assemble_light_curve(
            source, None, eph, LightCurveConfig(), require=True,
        )
        assert lc is not None
        assert label == "lc_source"
        assert warnings == []
        assert artifact_ids == []
        assert source.call_count == 1

    def test_passes_ephemeris_and_config_to_source(self) -> None:
        source = StubLcSource()
        eph = _make_ephemeris()
        cfg = LightCurveConfig(cadence="2min")
        assemble_light_curve(source, None, eph, cfg, require=True)
        assert source.last_ephemeris is eph
        assert source.last_config is cfg

    def test_propagates_warnings_from_result(self) -> None:
        result = _make_result(warnings=["sector 3 missing", "gap detected"])
        source = StubLcSource(result=result)
        eph = _make_ephemeris()
        _, _, warnings, _ = assemble_light_curve(
            source, None, eph, LightCurveConfig(), require=True,
        )
        assert warnings == ["sector 3 missing", "gap detected"]

    def test_lc_config_none_uses_default(self) -> None:
        source = StubLcSource()
        eph = _make_ephemeris()
        assemble_light_curve(source, None, eph, None, require=True)
        assert source.call_count == 1
        assert isinstance(source.last_config, LightCurveConfig)


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class TestAssembleLightCurveArtifactStore:
    def test_stores_prepared_lc_when_store_provided(self) -> None:
        source = StubLcSource()
        store = StubArtifactStore(artifact_id="art-42")
        eph = _make_ephemeris()
        lc, _, _, artifact_ids = assemble_light_curve(
            source, store, eph, LightCurveConfig(), require=True,
        )
        assert artifact_ids == ["art-42"]
        assert len(store.stored) == 1
        assert store.stored[0] is lc

    def test_no_artifact_ids_when_store_is_none(self) -> None:
        source = StubLcSource()
        eph = _make_ephemeris()
        _, _, _, artifact_ids = assemble_light_curve(
            source, None, eph, LightCurveConfig(), require=True,
        )
        assert artifact_ids == []


# ---------------------------------------------------------------------------
# Error handling: require=True
# ---------------------------------------------------------------------------


class TestAssembleLightCurveRequireTrue:
    def test_raises_assembly_error_when_source_fails(self) -> None:
        source = BoomLcSource(RuntimeError("download failed"))
        eph = _make_ephemeris()
        with pytest.raises(AssemblyLightCurveError, match="download failed"):
            assemble_light_curve(
                source, None, eph, LightCurveConfig(), require=True,
            )

    def test_chains_original_exception(self) -> None:
        original = ValueError("bad sector")
        source = BoomLcSource(original)
        eph = _make_ephemeris()
        with pytest.raises(AssemblyLightCurveError) as exc_info:
            assemble_light_curve(
                source, None, eph, LightCurveConfig(), require=True,
            )
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Error handling: require=False
# ---------------------------------------------------------------------------


class TestAssembleLightCurveRequireFalse:
    def test_returns_none_on_failure(self) -> None:
        source = BoomLcSource(RuntimeError("network timeout"))
        eph = _make_ephemeris()
        lc, label, warnings, artifact_ids = assemble_light_curve(
            source, None, eph, LightCurveConfig(), require=False,
        )
        assert lc is None
        assert label == "lc_source"
        assert len(warnings) == 1
        assert "network timeout" in warnings[0]
        assert artifact_ids == []

    def test_does_not_store_artifact_on_failure(self) -> None:
        source = BoomLcSource()
        store = StubArtifactStore()
        eph = _make_ephemeris()
        assemble_light_curve(
            source, store, eph, LightCurveConfig(), require=False,
        )
        assert store.stored == []
