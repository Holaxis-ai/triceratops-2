"""ValidationWorkspace: stateful analysis session for a single validation target.

Replaces the stateful aspects of the original ``target`` class:
- Owns assembled StellarField (queryable and mutatable)
- Caches the most recent ValidationResult
- Provides add_star / remove_star / update_star mutation API
- Delegates computation to ValidationEngine
- Delegates plotting to plotting module
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from triceratops.domain.molusc import MoluscData


class ValidationWorkspace:
    """Stateful analysis session for a single TRICERATOPS+ validation target.

    Constructor fires catalog queries (via injected providers) to assemble
    the stellar field. Use stub providers in tests to avoid network calls.

    Only ``mission="TESS"`` is supported for prepared compute.  Kepler/K2
    support is experimental; passing a non-TESS mission will raise
    ``UnsupportedComputeModeError`` when ``compute_probs()`` is called.
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
        from triceratops.assembly.config import AssemblyConfig
        from triceratops.assembly.pipelines.stellar_field import assemble_stellar_field
        from triceratops.lightcurve.ephemeris import ResolvedTarget

        _init_target = ResolvedTarget(
            target_ref=str(tic_id),
            tic_id=tic_id,
            ephemeris=None,
            source="workspace",
        )
        _init_config = AssemblyConfig(
            mission=mission,
            catalog_search_radius_px=search_radius,
            include_light_curve=False,
        )
        self._stellar_field: StellarField
        self._stellar_field, _ = assemble_stellar_field(
            catalog_provider=catalog_provider,
            target=_init_target,
            config=_init_config,
            transit_depth=None,
            pixel_coords_per_sector=None,
            aperture_pixels_per_sector=None,
            sigma_psf_px=0.75,
        )

        self._last_result: ValidationResult | None = None
        self._last_light_curve: LightCurve | None = None

        self._engine = ValidationEngine(
            catalog_provider=self._catalog_provider,
            population_provider=self._population_provider,
        )

        # Assembly-layer delegation objects
        from triceratops.assembly.orchestrator import DataAssemblyOrchestrator
        from triceratops.lightcurve.ephemeris import ResolvedTarget
        from triceratops.validation.preparer import ValidationPreparer

        self._orchestrator = DataAssemblyOrchestrator(
            catalog_provider=self._catalog_provider,
            population_provider=self._population_provider,
            aperture_provider=self._aperture_provider,
        )
        self._resolved_target = ResolvedTarget(
            target_ref=str(tic_id),
            tic_id=tic_id,
            ephemeris=None,
            source="workspace",
        )
        self._preparer = ValidationPreparer(
            registry=self._engine._registry,
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
        """Add a neighbor star to the stellar field. Invalidates cached results.

        Raises:
            ValueError: If a star with the same TIC ID already exists.
        """
        self._stellar_field.add_neighbor(star)
        self._last_result = None

    def remove_star(self, tic_id: int) -> None:
        """Remove a star by TIC ID. Invalidates cached results.

        Raises:
            ValueError: If tic_id is the target star.
            ValueError: If no star with tic_id is found.
        """
        self._stellar_field.remove_neighbor(tic_id)
        self._last_result = None

    def update_star(self, tic_id: int, **kwargs: object) -> None:
        """Update fields on a star by TIC ID. Invalidates cached results.

        Accepts both direct Star attribute names and TIC-style aliases:
          Teff → stellar_params.teff_k
          mass → stellar_params.mass_msun
          logg → stellar_params.logg
          metallicity → stellar_params.metallicity_dex

        Raises:
            ValueError: If no star with tic_id is found.
            TypeError: If an alias update is requested and stellar_params is None.
            AttributeError: If an unknown attribute name is given.
        """
        self._stellar_field.update_star(tic_id, **kwargs)
        self._last_result = None

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
        period_days: float | list[float] | tuple[float, float],
        scenario_ids: list[ScenarioID] | None = None,
        external_lcs: list[ExternalLightCurve] | None = None,
        contrast_curve: ContrastCurve | None = None,
        molusc_data: MoluscData | None = None,
    ) -> ValidationResult:
        """Run validation computation and cache the result.

        Delegates acquisition (TRILEGAL) to the orchestrator, then routes
        through prepare() and compute_prepared() so all validation gates
        apply uniformly.

        Returns:
            ValidationResult, also stored internally for property access.
        """
        import dataclasses

        from triceratops.assembly.config import AssemblyConfig
        from triceratops.validation.errors import UnsupportedComputeModeError

        # Mission gate — fail before any provider IO.
        if self._stellar_field.mission != "TESS":
            raise UnsupportedComputeModeError(
                f"compute_probs() only supports mission='TESS'. "
                f"Stellar field has mission={self._stellar_field.mission!r}. "
                "Kepler/K2 support is experimental and not available for "
                "prepared compute jobs."
            )

        self._last_light_curve = light_curve

        # Assemble via orchestrator — TRILEGAL fetch happens here if needed.
        # Pass stellar_field to skip catalog re-query.
        reassembly_config = AssemblyConfig(
            mission=self.mission,
            include_light_curve=False,
            trilegal_cache_path=self._trilegal_cache_path,
        )
        assembled = self._orchestrator.assemble(
            target=self._resolved_target,
            config=reassembly_config,
            scenario_ids=scenario_ids,
            stellar_field=self._stellar_field,
        )

        # Supplement with workspace-local inputs
        assembled = dataclasses.replace(
            assembled,
            light_curve=light_curve,
            contrast_curve=contrast_curve,
            molusc_data=molusc_data,
            external_lcs=external_lcs,
        )

        prepared = self._preparer.prepare(
            assembled, self.config, period_days, scenario_ids=scenario_ids,
        )
        result = self._engine.compute_prepared(prepared)

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
