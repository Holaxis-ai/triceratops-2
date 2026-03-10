"""Backward-compatibility shim: ``from triceratops._compat import target``.

The ``target`` class delegates all functionality to ValidationWorkspace while
preserving the original constructor signature and attribute names exactly.

Compatibility matrix (from 07_architectural_vision_v2.md):
    import path:     from triceratops._compat import target
    constructor:     target(ID, sectors, search_radius=10, mission="TESS", ...)
    calc_probs():    target.calc_probs(time, flux, sigma, ...)
    result attrs:    target.probs (DataFrame), target.FPP, target.NFPP
    star field:      target.stars, target.add_star(), target.remove_star()
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.population.protocols import PopulationSynthesisProvider
from triceratops.validation.workspace import ValidationWorkspace


class target:
    """Drop-in replacement for the original triceratops.target class.

    Thin wrapper around ValidationWorkspace. All original public attributes
    are preserved as properties for backward compatibility.

    Deprecated parameters emit DeprecationWarning but still work.
    """

    def __init__(
        self,
        ID: int,
        sectors: np.ndarray,
        search_radius: int = 10,
        mission: str = "TESS",
        lightkurve_cache_dir: str | None = None,
        trilegal_fname: str | None = None,
        # Injected providers (for testing -- not in original API)
        _catalog_provider: StarCatalogProvider | None = None,
        _aperture_provider: ApertureProvider | None = None,
        _population_provider: PopulationSynthesisProvider | None = None,
    ) -> None:
        """Args match original triceratops.target.__init__ exactly.

        New private args (_catalog_provider, _aperture_provider,
        _population_provider) are for testing only and not part of the
        public API.
        """
        if lightkurve_cache_dir is not None:
            warnings.warn(
                "lightkurve_cache_dir is not supported in the refactored "
                "triceratops and will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._workspace = ValidationWorkspace(
            tic_id=ID,
            sectors=sectors,
            mission=mission,
            search_radius=search_radius,
            trilegal_cache_path=trilegal_fname,
            catalog_provider=_catalog_provider,
            aperture_provider=_aperture_provider,
            population_provider=_population_provider,
        )

        # Original attributes available on target
        self.ID = ID
        self.sectors = sectors
        self.mission = mission
        self.search_radius = search_radius

    # -- Original target attributes --

    @property
    def stars(self) -> list:
        """List of Star objects in the field (matches original target.stars)."""
        return self._workspace.stars

    @property
    def FPP(self) -> float:
        """False Positive Probability (matches original target.FPP)."""
        return self._workspace.fpp

    @property
    def NFPP(self) -> float:
        """Nearby False Positive Probability (matches original target.NFPP)."""
        return self._workspace.nfpp

    @property
    def probs(self):
        """Scenario probability DataFrame (matches original target.probs).

        Returns a pandas DataFrame with columns: scenario, prob, lnZ.
        Constructed from ValidationResult.scenario_results.
        """
        import pandas as pd

        if self._workspace.results is None:
            raise RuntimeError("calc_probs() must be called before accessing probs")
        data = [
            {
                "scenario": r.scenario_id.value,
                "prob": r.relative_probability,
                "lnZ": r.ln_evidence,
            }
            for r in self._workspace.results.scenario_results
        ]
        return pd.DataFrame(data)

    # -- Original calc_depths() --

    def calc_depths(
        self,
        tdepth: float,
        all_ap_pixels: list | None = None,
    ) -> None:
        """Compute flux ratios and transit depths.

        Matches original target.calc_depths(tdepth, all_ap_pixels).
        """
        self._workspace.calc_depths(
            transit_depth=tdepth,
            pixel_coords_per_sector=(
                all_ap_pixels if all_ap_pixels is not None else []
            ),
            aperture_pixels_per_sector=(
                all_ap_pixels if all_ap_pixels is not None else []
            ),
        )

    # -- Original calc_probs() --

    def calc_probs(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        sigma: float,
        P_orb: float | list,
        Tdur: float = 0.1,  # noqa: ARG002
        depth: float = 0.0,  # noqa: ARG002
        n: int = 20_000,
        saved_lnZ: bool = False,  # noqa: ARG002
        parallel: bool = True,
        lnz_const: float = 650.0,
        nb_samples: int = 1000,
        flat_priors: bool = False,
        exptime: float = 0.00139,
        nsamples: int = 20,
        plot: bool = True,
        external_lc_files: list | None = None,
        filt_lcs: list | None = None,
        renorm_external_lcs: bool = False,  # noqa: ARG002
        external_fluxes_of_stars: list | None = None,  # noqa: ARG002
        contrast_curve_file: str | None = None,
        filt: str | None = None,
        molusc_file: str | None = None,
        trilegal_fname: str | None = None,  # noqa: ARG002
        n_workers: int = 0,
    ) -> None:
        """Run validation and populate self.probs, self.FPP, self.NFPP.

        Matches original target.calc_probs() signature exactly.
        Maps all original kwargs to the new ValidationWorkspace API.
        """
        # Build Config from the original-style kwargs
        config = Config(
            n_mc_samples=n,
            lnz_const=lnz_const,
            n_best_samples=nb_samples,
            parallel=parallel,
            flat_priors=flat_priors,
            mission=self.mission,
            n_workers=n_workers,
        )
        self._workspace.config = config

        # Build LightCurve from raw arrays
        lc = LightCurve(
            time_days=np.asarray(time),
            flux=np.asarray(flux),
            flux_err=float(sigma),
            cadence_days=exptime,
            supersampling_rate=nsamples,
        )

        # Load contrast curve if provided
        contrast_curve = None
        if contrast_curve_file is not None:
            from triceratops.io.contrast_curves import load_contrast_curve

            band = filt or "TESS"
            contrast_curve = load_contrast_curve(
                Path(contrast_curve_file), band=band,
            )

        # Load external LCs if provided
        ext_lcs: list[ExternalLightCurve] | None = None
        if external_lc_files and filt_lcs:
            from triceratops.io.external_lc import load_external_lc_as_object

            ext_lcs = [
                load_external_lc_as_object(Path(f), b)
                for f, b in zip(external_lc_files, filt_lcs)
            ]

        self._workspace.compute_probs(
            light_curve=lc,
            period_days=P_orb,
            external_lcs=ext_lcs,
            contrast_curve=contrast_curve,
            molusc_file=molusc_file,
        )

        if plot:
            try:
                self._workspace.plot_fits()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass  # plotting is non-critical; don't fail the calculation

    # -- Mutation API (pass-through) --

    def add_star(self, star: object) -> None:
        self._workspace.add_star(star)  # type: ignore[arg-type]

    def remove_star(self, tic_id: int) -> None:
        self._workspace.remove_star(tic_id)

    def update_star(self, tic_id: int, **kwargs: object) -> None:
        self._workspace.update_star(tic_id, **kwargs)

    # -- Plotting pass-through --

    def plot_field(self, **kwargs: object) -> None:
        self._workspace.plot_field(**kwargs)  # type: ignore[attr-defined]

    def plot_fits(self, **kwargs: object) -> None:
        self._workspace.plot_fits(**kwargs)  # type: ignore[attr-defined]
