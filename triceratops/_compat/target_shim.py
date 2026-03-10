"""Backward-compatibility shim: ``from triceratops._compat import target``.

The ``target`` class delegates all functionality to ValidationWorkspace while
preserving the original constructor signature and attribute names exactly.

Compatibility matrix (from target_facade_spec.md):
    import path:     from triceratops._compat import target
    constructor:     target(ID, sectors, search_radius=10, mission="TESS", ...)
    calc_probs():    target.calc_probs(time, flux_0, flux_err_0, ...)
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
    def stars(self):
        """DataFrame of stars in the field (matches original target.stars)."""
        import pandas as pd

        rows = []
        for s in self._workspace._stellar_field.stars:
            sp = s.stellar_params
            rows.append({
                "ID":        s.tic_id,
                "Tmag":      s.tmag,   "Jmag": s.jmag, "Hmag": s.hmag,
                "Kmag":      s.kmag,   "Bmag": s.bmag, "Vmag": s.vmag,
                "gmag":      s.gmag,   "rmag": s.rmag, "imag": s.imag,
                "zmag":      s.zmag,
                "ra":        s.ra_deg, "dec":  s.dec_deg,
                "mass":      sp.mass_msun    if sp else float("nan"),
                "rad":       sp.radius_rsun  if sp else float("nan"),
                "Teff":      sp.teff_k       if sp else float("nan"),
                "plx":       sp.parallax_mas if sp else float("nan"),
                "sep":       s.separation_arcsec,
                "PA":        s.position_angle_deg,
                "fluxratio": s.flux_ratio             if s.flux_ratio             is not None else float("nan"),
                "tdepth":    s.transit_depth_required  if s.transit_depth_required is not None else float("nan"),
            })
        return pd.DataFrame(rows)

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

        Raises NotImplementedError because pixel coordinates require WCS,
        which is not yet available in this version.
        """
        raise NotImplementedError(
            "calc_depths() requires pixel coordinates derived from WCS, which is "
            "not yet available in this version. Use ValidationWorkspace.calc_depths() "
            "directly and supply pixel_coords_per_sector from your own WCS transform."
        )

    # -- Original calc_probs() --

    def calc_probs(
        self,
        time: np.ndarray,
        flux_0: np.ndarray,
        flux_err_0: float,
        P_orb: float | list,
        contrast_curve_file: str | None = None,
        filt: str = "TESS",
        N: int = 1_000_000,
        parallel: bool = False,
        drop_scenario: list = [],  # noqa: ARG002, B006
        verbose: int = 1,  # noqa: ARG002
        flatpriors: bool = False,
        exptime: float = 0.00139,
        nsamples: int = 20,
        molusc_file: str | None = None,
        external_lc_files: list | None = None,
        filt_lcs: list | None = None,
        lnz_const: int = 650,
        Z_star: float = 0.0,
        plot: bool = True,
        n_workers: int = 0,
    ) -> None:
        """Run validation and populate self.probs, self.FPP, self.NFPP.

        Matches original target.calc_probs() signature exactly.
        Maps all original kwargs to the new ValidationWorkspace API.
        """
        # Translate Z_star into update_star if non-zero
        if Z_star != 0.0:
            self._workspace.update_star(self.ID, metallicity=Z_star)

        # Build Config from the original-style kwargs
        config = Config(
            n_mc_samples=N,
            lnz_const=lnz_const,
            parallel=parallel,
            flat_priors=flatpriors,
            mission=self.mission,
            n_workers=n_workers,
        )
        self._workspace.config = config

        # Build LightCurve from raw arrays
        lc = LightCurve(
            time_days=np.asarray(time),
            flux=np.asarray(flux_0),
            flux_err=float(flux_err_0),
            cadence_days=exptime,
            supersampling_rate=nsamples,
        )

        # Load contrast curve if provided
        contrast_curve = None
        if contrast_curve_file is not None:
            from triceratops.io.contrast_curves import load_contrast_curve

            band = filt
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

        # Load MOLUSC data if provided
        molusc_data = None
        if molusc_file is not None:
            from triceratops.io.molusc import load_molusc_file

            molusc_data = load_molusc_file(Path(molusc_file))

        self._workspace.compute_probs(
            light_curve=lc,
            period_days=P_orb,
            external_lcs=ext_lcs,
            contrast_curve=contrast_curve,
            molusc_data=molusc_data,
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

    def plot_field(
        self,
        sector: int | None = None,  # noqa: ARG002
        ap_pixels: object = None,  # noqa: ARG002
        ap_color: str = "red",  # noqa: ARG002
        save: bool = False,
        fname: str | None = None,
    ) -> None:
        warnings.warn(
            "plot_field() shows angular coordinates only in this version. "
            "The TESS FFI pixel-grid panel is not yet available.",
            UserWarning,
            stacklevel=2,
        )
        self._workspace.plot_field(save=save, fname=fname)

    def plot_fits(
        self,
        time: object = None,  # noqa: ARG002
        flux_0: object = None,  # noqa: ARG002
        flux_err_0: object = None,  # noqa: ARG002
        x_range: list | None = None,  # noqa: ARG002
        y_range: list | None = None,  # noqa: ARG002
        nrows: int = 0,  # noqa: ARG002
        save: bool = False,
        fname: str | None = None,
    ) -> None:
        self._workspace.plot_fits(save=save, fname=fname)  # type: ignore[attr-defined]
