from __future__ import annotations

import numpy as np
import pytest

from triceratops.assembly.config import AssemblyConfig
from triceratops.assembly.errors import AcquisitionError, CatalogAcquisitionError
from triceratops.assembly.pipelines.external_lcs import assemble_external_lcs
from triceratops.assembly.pipelines.stellar_field import assemble_stellar_field
from triceratops.domain.entities import ExternalLightCurve, LightCurve, Star, StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget


def _resolved_target() -> ResolvedTarget:
    return ResolvedTarget(
        target_ref="TOI-123.01",
        tic_id=12345,
        ephemeris=Ephemeris(period_days=4.2, t0_btjd=1001.25),
        source="exofop",
    )


def _stellar_field() -> StellarField:
    params = StellarParameters(
        mass_msun=1.0,
        radius_rsun=1.0,
        teff_k=5700.0,
        logg=4.4,
        metallicity_dex=0.0,
        parallax_mas=10.0,
    )
    return StellarField(
        target_id=12345,
        mission="TESS",
        search_radius_pixels=10,
        stars=[
            Star(
                tic_id=12345,
                ra_deg=10.0,
                dec_deg=20.0,
                tmag=10.0,
                jmag=9.0,
                hmag=8.9,
                kmag=8.8,
                bmag=10.2,
                vmag=10.1,
                stellar_params=params,
            )
        ],
    )


def _external_lc() -> ExternalLightCurve:
    return ExternalLightCurve(
        light_curve=LightCurve(
            time_days=np.array([0.0, 0.1]),
            flux=np.array([1.0, 0.99]),
            flux_err=0.001,
        ),
        band="J",
    )


class _ExternalSource:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def load(self):
        if self.fail:
            raise RuntimeError("boom")
        return [_external_lc()]


class _CatalogProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def query_nearby_stars(self, *, tic_id: int, search_radius_px: int, mission: str):
        if self.fail:
            raise RuntimeError("catalog down")
        return _stellar_field()


def test_assemble_external_lcs_returns_curves_and_wraps_errors() -> None:
    lcs, warnings = assemble_external_lcs(_ExternalSource())
    assert len(lcs) == 1
    assert warnings == []

    with pytest.raises(AcquisitionError, match="Failed to load external light curves"):
        assemble_external_lcs(_ExternalSource(fail=True))


def test_assemble_stellar_field_queries_catalog_and_computes_fluxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "triceratops.catalog.flux_contributions.compute_flux_ratios",
        lambda field, pixel_coords_per_sector, aperture_pixels_per_sector, sigma_psf_px: [0.8],
    )
    monkeypatch.setattr(
        "triceratops.catalog.flux_contributions.compute_transit_depths",
        lambda flux_ratios, transit_depth: [transit_depth / flux_ratios[0]],
    )

    field, warnings = assemble_stellar_field(
        _CatalogProvider(),
        _resolved_target(),
        AssemblyConfig(),
        transit_depth=0.01,
        pixel_coords_per_sector=[np.array([[0.0, 0.0]])],
        aperture_pixels_per_sector=[np.array([[0.0, 0.0]])],
        sigma_psf_px=0.75,
    )

    assert warnings == []
    assert field.target.flux_ratio == pytest.approx(0.8)
    assert field.target.transit_depth_required == pytest.approx(0.0125)


def test_assemble_stellar_field_warns_without_geometry_and_wraps_catalog_errors() -> None:
    field, warnings = assemble_stellar_field(
        _CatalogProvider(),
        _resolved_target(),
        AssemblyConfig(),
        transit_depth=0.01,
        pixel_coords_per_sector=None,
        aperture_pixels_per_sector=None,
        sigma_psf_px=0.75,
    )

    assert len(warnings) == 1
    assert "flux ratios not computed" in warnings[0]
    assert field.target.flux_ratio is None

    with pytest.raises(CatalogAcquisitionError, match="Catalog query failed"):
        assemble_stellar_field(
            _CatalogProvider(fail=True),
            _resolved_target(),
            AssemblyConfig(),
            transit_depth=None,
            pixel_coords_per_sector=None,
            aperture_pixels_per_sector=None,
            sigma_psf_px=0.75,
        )
