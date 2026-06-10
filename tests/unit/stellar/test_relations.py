"""Tests for stellar relation filter aliases."""
from __future__ import annotations

import numpy as np

from triceratops.stellar.relations import StellarRelations, canonicalize_filter_name


def test_common_imaging_filter_aliases_are_canonicalized() -> None:
    assert canonicalize_filter_name("Jcont") == "J"
    assert canonicalize_filter_name("Hcont") == "H"
    assert canonicalize_filter_name("Ks") == "K"
    assert canonicalize_filter_name("Kcont") == "K"
    assert canonicalize_filter_name("Brgamma") == "K"
    assert canonicalize_filter_name("LP600") == "Vis"


def test_existing_optical_aliases_are_canonicalized() -> None:
    assert canonicalize_filter_name("562nm") == "Vis"
    assert canonicalize_filter_name("832nm") == "Vis"


def test_imaging_filter_aliases_are_used_by_flux_ratio_lookup() -> None:
    relations = StellarRelations()
    masses = np.array([0.3, 0.8, 1.2])

    np.testing.assert_allclose(
        relations.get_flux_ratio(masses, "Jcont"),
        relations.get_flux_ratio(masses, "J"),
    )
    np.testing.assert_allclose(
        relations.get_flux_ratio(masses, "Hcont"),
        relations.get_flux_ratio(masses, "H"),
    )
    np.testing.assert_allclose(
        relations.get_flux_ratio(masses, "Brgamma"),
        relations.get_flux_ratio(masses, "K"),
    )
    np.testing.assert_allclose(
        relations.get_flux_ratio(masses, "LP600"),
        relations.get_flux_ratio(masses, "Vis"),
    )
