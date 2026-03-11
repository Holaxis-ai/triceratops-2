"""Tests for lazy workspace construction (deferred catalog query)."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.entities import Star, StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.validation.workspace import ValidationWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_field() -> StellarField:
    target = Star(
        tic_id=12345678,
        ra_deg=83.82,
        dec_deg=-5.39,
        tmag=10.5,
        jmag=9.8,
        hmag=9.5,
        kmag=9.4,
        bmag=11.2,
        vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        ),
    )
    return StellarField(
        target_id=12345678,
        mission="TESS",
        search_radius_pixels=10,
        stars=[target],
    )


class _CountingCatalogProvider:
    """Catalog provider that counts how many times query_nearby_stars is called."""

    def __init__(self) -> None:
        self.call_count = 0

    def query_nearby_stars(
        self, tic_id: int, search_radius_px: int, mission: str,
    ) -> StellarField:
        self.call_count += 1
        return _make_field()


# ---------------------------------------------------------------------------
# Lazy construction tests
# ---------------------------------------------------------------------------


class TestLazyConstruction:
    def test_init_does_not_query_catalog(self) -> None:
        """Constructing ValidationWorkspace must NOT call the catalog provider."""
        provider = _CountingCatalogProvider()
        _ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        assert provider.call_count == 0

    def test_stars_triggers_catalog_query(self) -> None:
        """Accessing .stars must trigger exactly one catalog query."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        assert provider.call_count == 0

        _ = ws.stars
        assert provider.call_count == 1

    def test_target_triggers_catalog_query(self) -> None:
        """Accessing .target must trigger exactly one catalog query."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        assert provider.call_count == 0

        _ = ws.target
        assert provider.call_count == 1

    def test_fetch_catalog_triggers_catalog_query(self) -> None:
        """Calling fetch_catalog() must trigger the catalog query."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        assert provider.call_count == 0

        field = ws.fetch_catalog()
        assert provider.call_count == 1
        assert isinstance(field, StellarField)

    def test_repeated_stars_access_no_requery(self) -> None:
        """Accessing .stars multiple times must NOT re-query the catalog."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        _ = ws.stars
        _ = ws.stars
        _ = ws.stars
        assert provider.call_count == 1

    def test_stars_then_target_no_requery(self) -> None:
        """Accessing .stars then .target must NOT re-query."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        _ = ws.stars
        _ = ws.target
        assert provider.call_count == 1

    def test_fetch_catalog_then_stars_no_requery(self) -> None:
        """fetch_catalog() followed by .stars must NOT re-query."""
        provider = _CountingCatalogProvider()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
        )
        ws.fetch_catalog()
        _ = ws.stars
        _ = ws.target
        assert provider.call_count == 1


# ---------------------------------------------------------------------------
# Pre-loaded stellar field tests
# ---------------------------------------------------------------------------


class TestPreLoadedStellarField:
    def test_preloaded_field_skips_catalog(self) -> None:
        """Passing stellar_field= skips the catalog provider entirely."""
        provider = _CountingCatalogProvider()
        field = _make_field()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
            stellar_field=field,
        )
        # Access stars -- should use pre-loaded field, not query
        stars = ws.stars
        assert provider.call_count == 0
        assert len(stars) == 1
        assert stars[0].tic_id == 12345678

    def test_preloaded_field_is_same_object(self) -> None:
        """The pre-loaded field should be the exact same object."""
        field = _make_field()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_CountingCatalogProvider(),
            stellar_field=field,
        )
        assert ws._stellar_field is field

    def test_preloaded_field_target_accessible(self) -> None:
        """Target property works with pre-loaded field."""
        field = _make_field()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_CountingCatalogProvider(),
            stellar_field=field,
        )
        assert ws.target.tic_id == 12345678

    def test_fetch_catalog_returns_preloaded_field(self) -> None:
        """fetch_catalog() on a workspace with pre-loaded field returns the same field."""
        provider = _CountingCatalogProvider()
        field = _make_field()
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=provider,
            stellar_field=field,
        )
        result = ws.fetch_catalog()
        assert result is field
        assert provider.call_count == 0
