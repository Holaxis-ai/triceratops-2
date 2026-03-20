from __future__ import annotations

from astropy.table import Table

from triceratops.catalog.mast_provider import MASTCatalogProvider


def test_query_nearby_stars_reorders_exact_target_row_first(monkeypatch) -> None:
    rows = Table(
        rows=[
            (
                620298691,
                None,
                None,
                13.4937,
                12.0,
                11.9,
                11.8,
                None,
                None,
                None,
                None,
                None,
                36.1366830090692,
                24.1016409975529,
                1.0,
                1.0,
                5500.0,
                100.0,
                10.0,
            ),
            (
                21119973,
                None,
                None,
                13.4937,
                12.0,
                11.9,
                11.8,
                None,
                None,
                None,
                None,
                None,
                36.1366830090692,
                24.1016409975529,
                1.0,
                1.0,
                5500.0,
                100.0,
                10.0,
            ),
            (
                620298692,
                None,
                None,
                14.2326,
                12.3,
                12.2,
                12.1,
                None,
                None,
                None,
                None,
                None,
                36.1372056935116,
                24.1013579136851,
                0.9,
                0.9,
                5200.0,
                110.0,
                9.5,
            ),
        ],
        names=[
            "ID",
            "Bmag",
            "Vmag",
            "Tmag",
            "Jmag",
            "Hmag",
            "Kmag",
            "gmag",
            "rmag",
            "imag",
            "zmag",
            "unused",
            "ra",
            "dec",
            "mass",
            "rad",
            "Teff",
            "d",
            "plx",
        ],
    )

    def _fake_query_object(*args, **kwargs):
        _ = (args, kwargs)
        return rows

    monkeypatch.setattr(
        "triceratops.catalog.mast_provider.Catalogs.query_object",
        _fake_query_object,
    )

    field = MASTCatalogProvider().query_nearby_stars(
        tic_id=21119973,
        search_radius_px=10,
        mission="TESS",
    )

    assert field.target_id == 21119973
    assert field.stars[0].tic_id == 21119973
    assert field.stars[1].tic_id == 620298691
    assert field.stars[0].separation_arcsec == 0.0
