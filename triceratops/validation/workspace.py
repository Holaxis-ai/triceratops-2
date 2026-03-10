"""ValidationWorkspace: stateful analysis session for a single validation target.

Replaces the stateful aspects of the original ``target`` class:
- Owns assembled StellarField (queryable and mutatable)
- Caches the most recent ValidationResult
- Provides add_star / remove_star / update_star mutation API
- Delegates computation to ValidationEngine
- Delegates plotting to plotting module
"""
from __future__ import annotations

import numpy as np

from triceratops.catalog.flux_contributions import compute_flux_ratios, compute_transit_depths
from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve, Star, StellarField
from triceratops.domain.result import ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import ContrastCurve
from triceratops.population.protocols import PopulationSynthesisProvider
from triceratops.validation.engine import ValidationEngine


class ValidationWorkspace:
    """Stateful analysis session for a single TRICERATOPS+ validation target.

    Constructor fires catalog queries (via injected providers) to assemble
    the stellar field. Use stub providers in tests to avoid network calls.
    """

    def __init__(
        self,
        tic_id: int,
        sectors: np.ndarray,
        mission: str = "TESS",
        search_radius: int = 10,
        config: Config | None = None,
        catalog_provider: StarCatalogProvider | None = None,
        aperture_provider: ApertureProvider | None = None,
        population_provider: PopulationSynthesisProvider | None = None,
        trilegal_cache_path: str | None = None,
    ) -> None:
        self.tic_id = tic_id
        self.sectors = sectors
        self.mission = mission
        self.search_radius = search_radius
        self.config = config or Config(mission=mission)
        self._trilegal_cache_path = trilegal_cache_path

        if catalog_provider is None:
            from triceratops.catalog.mast_provider import MASTCatalogProvider
            catalog_provider = MASTCatalogProvider()
        if aperture_provider is None:
            from triceratops.catalog.mast_provider import TesscutApertureProvider
            aperture_provider = TesscutApertureProvider()
        self._catalog_provider = catalog_provider
        self._aperture_provider = aperture_provider
        self._population_provider = population_provider

        # Fire catalog query during construction (matches original target.__init__)
        self._stellar_field: StellarField = self._catalog_provider.query_nearby_stars(
            tic_id=tic_id,
            search_radius_px=search_radius,
            mission=mission,
        )

        self._last_result: ValidationResult | None = None
        self._last_light_curve: LightCurve | None = None

        self._engine = ValidationEngine(
            catalog_provider=self._catalog_provider,
            population_provider=self._population_provider,
        )

    # -- Star field access and mutation --

    @property
    def stars(self) -> list[Star]:
        """All stars in the stellar field (target at index 0)."""
        return self._stellar_field.stars

    @property
    def target(self) -> Star:
        """The target star."""
        return self._stellar_field.target

    def add_star(self, star: Star) -> None:
        """Add a star to the stellar field. Invalidates cached results."""
        self._stellar_field.stars.append(star)
        self._last_result = None

    def remove_star(self, tic_id: int) -> None:
        """Remove a star by TIC ID. Raises ValueError if not found."""
        original_len = len(self._stellar_field.stars)
        self._stellar_field.stars[:] = [
            s for s in self._stellar_field.stars if s.tic_id != tic_id
        ]
        if len(self._stellar_field.stars) == original_len:
            raise ValueError(f"Star with TIC ID {tic_id} not found in stellar field.")
        self._last_result = None

    def update_star(self, tic_id: int, **kwargs: object) -> None:
        """Update fields on a star by TIC ID. Raises ValueError if not found.

        Accepts both direct Star attribute names and TIC-style aliases:
          Teff → stellar_params.teff_k
          mass → stellar_params.mass_msun
          logg → stellar_params.logg
          metallicity → stellar_params.metallicity_dex
        """
        from dataclasses import replace

        # TIC catalog name → (StellarParameters field name,)
        _STELLAR_PARAMS_ALIASES: dict[str, str] = {
            "Teff": "teff_k",
            "mass": "mass_msun",
            "logg": "logg",
            "metallicity": "metallicity_dex",
        }

        for star in self._stellar_field.stars:
            if star.tic_id == tic_id:
                stellar_updates: dict[str, object] = {}
                for attr, value in kwargs.items():
                    if attr in _STELLAR_PARAMS_ALIASES:
                        stellar_updates[_STELLAR_PARAMS_ALIASES[attr]] = value
                    elif hasattr(star, attr):
                        object.__setattr__(star, attr, value)
                    else:
                        raise AttributeError(f"Star has no attribute {attr!r}")
                if stellar_updates:
                    object.__setattr__(
                        star, "stellar_params",
                        replace(star.stellar_params, **stellar_updates),
                    )
                self._last_result = None
                return
        raise ValueError(f"Star with TIC ID {tic_id} not found.")

    # -- Flux/depth computation --

    def calc_depths(
        self,
        transit_depth: float,
        pixel_coords_per_sector: list[np.ndarray],
        aperture_pixels_per_sector: list[np.ndarray],
        sigma_psf_px: float = 0.75,
    ) -> None:
        """Compute flux ratios and intrinsic transit depths for all stars.

        Args:
            transit_depth: Observed transit depth (fractional; 0.01 = 1%).
            pixel_coords_per_sector: Per-sector star pixel positions,
                each shape (N_stars, 2) as (col, row).
            aperture_pixels_per_sector: Per-sector aperture pixel positions,
                each shape (N_pixels, 2) as (col, row).
            sigma_psf_px: PSF sigma in pixels (default 0.75).
        """
        flux_ratios = compute_flux_ratios(
            self._stellar_field,
            pixel_coords_per_sector,
            aperture_pixels_per_sector,
            sigma_psf_px,
        )
        transit_depths = compute_transit_depths(flux_ratios, transit_depth)

        for star, fr, td in zip(self._stellar_field.stars, flux_ratios, transit_depths):
            star.flux_ratio = fr
            star.transit_depth_required = td

        self._last_result = None

    # -- Probability computation --

    def compute_probs(
        self,
        light_curve: LightCurve,
        period_days: float | list[float],
        scenario_ids: list[ScenarioID] | None = None,
        external_lcs: list[ExternalLightCurve] | None = None,
        contrast_curve: ContrastCurve | None = None,
        molusc_file: str | None = None,
    ) -> ValidationResult:
        """Run validation computation and cache the result.

        Provider-backed IO (TRILEGAL population fetch) is performed here, before
        calling the engine via PreparedValidationInputs.  The engine receives
        only materialised data.

        Returns:
            ValidationResult, also stored internally for property access.
        """
        from triceratops.validation.job import PreparedValidationInputs

        self._last_light_curve = light_curve

        # Materialise TRILEGAL population here (workspace owns provider IO).
        # The engine is provider-free: it only accepts pre-materialised data.
        trilegal_population = None
        if self._population_provider is not None:
            from pathlib import Path
            from triceratops.domain.scenario_id import ScenarioID as _SID
            from triceratops.scenarios.registry import DEFAULT_REGISTRY

            eligible = DEFAULT_REGISTRY.all_scenarios()
            if scenario_ids is not None:
                eligible = [DEFAULT_REGISTRY.get(sid) for sid in scenario_ids]
            needs_trilegal = any(
                s.scenario_id in _SID.trilegal_scenarios() for s in eligible
            )
            if needs_trilegal:
                cache = Path(self._trilegal_cache_path) if self._trilegal_cache_path else None
                target = self._stellar_field.target
                trilegal_population = self._population_provider.query(
                    ra_deg=target.ra_deg,
                    dec_deg=target.dec_deg,
                    target_tmag=target.tmag,
                    cache_path=cache,
                )

        # Build a PreparedValidationInputs and route through compute_prepared().
        # This keeps the engine's input contract explicit and testable.
        # scenario_ids filtering happens inside engine.compute() via scenario_ids arg.
        # For now we pass scenario_ids=None in the prepared path (engine default logic
        # applies) and fall back to compute() for scenario_id filtering.
        if scenario_ids is None:
            prepared = PreparedValidationInputs(
                target_id=self.tic_id,
                stellar_field=self._stellar_field,
                light_curve=light_curve,
                config=self.config,
                period_days=period_days,
                trilegal_population=trilegal_population,
                external_lcs=external_lcs,
                contrast_curve=contrast_curve,
                molusc_file=molusc_file,
            )
            result = self._engine.compute_prepared(prepared)
        else:
            # scenario_ids filtering path: use compute() directly until compute_prepared
            # supports scenario_ids selection (planned for a follow-up).
            result = self._engine.compute(
                light_curve=light_curve,
                stellar_field=self._stellar_field,
                period_days=period_days,
                config=self.config,
                scenario_ids=scenario_ids,
                external_lcs=external_lcs,
                contrast_curve=contrast_curve,
                trilegal_population=trilegal_population,
                molusc_file=molusc_file,
            )

        self._last_result = result
        return result

    # -- Result access properties --

    @property
    def results(self) -> ValidationResult | None:
        """Most recent validation result, or None if not yet computed."""
        return self._last_result

    @property
    def fpp(self) -> float:
        """False Positive Probability from the most recent run."""
        if self._last_result is None:
            raise RuntimeError("compute_probs() must be called before accessing fpp")
        return self._last_result.fpp

    @property
    def nfpp(self) -> float:
        """Nearby False Positive Probability from the most recent run."""
        if self._last_result is None:
            raise RuntimeError("compute_probs() must be called before accessing nfpp")
        return self._last_result.nfpp

    # -- Plotting --

    def plot_field(self, **kwargs: object) -> None:
        """Plot the stellar field around the target.

        Delegates to :func:`triceratops.plotting.plot_field`.

        Args:
            **kwargs: Forwarded to ``plot_field`` (e.g. ``save``, ``fname``).
        """
        from triceratops.plotting import plot_field

        plot_field(self._stellar_field, self.search_radius, **kwargs)

    def plot_fits(self, **kwargs: object) -> None:
        """Plot best-fit model light curves for each non-negligible scenario.

        Delegates to :func:`triceratops.plotting.plot_fits`.

        Raises:
            RuntimeError: If ``compute_probs()`` has not been called yet.

        Args:
            **kwargs: Forwarded to ``plot_fits`` (e.g. ``save``, ``fname``).
        """
        from triceratops.plotting import plot_fits

        if self._last_result is None:
            raise RuntimeError(
                "compute_probs() must be called before plot_fits()"
            )
        if self._last_light_curve is None:
            raise RuntimeError(
                "No light curve stored; re-run compute_probs() to populate it."
            )
        plot_fits(self._last_light_curve, self._last_result, **kwargs)
