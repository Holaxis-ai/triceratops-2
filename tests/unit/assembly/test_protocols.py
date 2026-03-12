from __future__ import annotations

import numpy as np

from triceratops.assembly.protocols import (
    ArtifactStore,
    ContrastCurveSource,
    ExternalLcSource,
    LightCurveSource,
    MoluscSource,
)
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.molusc import MoluscData
from triceratops.domain.value_objects import ContrastCurve
from triceratops.lightcurve.config import LightCurveConfig


class _LightCurveSourceImpl:
    def prepare(self, ephemeris: object, config: LightCurveConfig) -> object:
        return {"ok": True, "config": config}


class _ContrastCurveSourceImpl:
    def load(self, band: str) -> ContrastCurve:
        return ContrastCurve(
            separations_arcsec=np.array([0.2, 0.5]),
            delta_mags=np.array([2.0, 5.0]),
            band=band,
        )


class _MoluscSourceImpl:
    def load(self) -> MoluscData:
        return MoluscData(
            semi_major_axis_au=np.array([1.0]),
            eccentricity=np.array([0.1]),
            mass_ratio=np.array([0.5]),
        )


class _ExternalLcSourceImpl:
    def load(self) -> list[ExternalLightCurve]:
        return [
            ExternalLightCurve(
                light_curve=LightCurve(
                    time_days=np.array([0.0, 0.1]),
                    flux=np.array([1.0, 0.99]),
                    flux_err=0.001,
                ),
                band="J",
            )
        ]


class _ArtifactStoreImpl:
    def put_raw_lc(self, data: object) -> str:
        return "raw-1"

    def put_prepared_lc(self, lc: LightCurve) -> str:
        return "prepared-1"


def test_runtime_checkable_assembly_protocols_accept_conforming_implementations() -> None:
    assert isinstance(_LightCurveSourceImpl(), LightCurveSource)
    assert isinstance(_ContrastCurveSourceImpl(), ContrastCurveSource)
    assert isinstance(_MoluscSourceImpl(), MoluscSource)
    assert isinstance(_ExternalLcSourceImpl(), ExternalLcSource)
    assert isinstance(_ArtifactStoreImpl(), ArtifactStore)
