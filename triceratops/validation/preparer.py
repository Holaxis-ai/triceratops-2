"""ValidationPreparer: owns provider-backed data acquisition and materialisation.

This module is the intended home for all provider-backed IO when using the
two-phase prepare/compute split.  ValidationWorkspace still performs a catalog
query in its constructor (for compatibility with the interactive local workflow),
but any code that needs a clean, testable prepare boundary should use this class.
Its output (PreparedValidationInputs) is the clean boundary for provider-free compute.

Two-phase design
----------------
Preparer (this module):
    - Query star catalog via StarCatalogProvider.
    - Compute flux ratios and transit depths.
    - Fetch TRILEGAL background population via PopulationSynthesisProvider.
    - Load contrast curve from disk.
    - Load external light curves from disk.
    - Emit a fully-populated PreparedValidationInputs.

Compute (ValidationEngine.compute_prepared):
    - Accept PreparedValidationInputs.
    - Execute scenarios with no provider access.
    - Return ValidationResult.

Remote compute boundary guarantee
----------------------------------
Code that runs in a remote worker (e.g. Modal) should only ever call
``ValidationEngine.compute_prepared()``.  It must never instantiate a
``ValidationPreparer`` or any provider.

See: working_docs/iteration/priority-3_pure-compute-boundary.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from triceratops.catalog.flux_contributions import compute_flux_ratios, compute_transit_depths
from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve, StellarField
from triceratops.domain.value_objects import ContrastCurve
from triceratops.population.protocols import PopulationSynthesisProvider, TRILEGALResult
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry
from triceratops.validation.job import PreparedValidationInputs, PreparedValidationMetadata


class ValidationPreparer:
    """Owns all provider-backed data acquisition and materialisation.

    Produces a PreparedValidationInputs that is ready for provider-free
    compute via ValidationEngine.compute_prepared().

    All network I/O, filesystem I/O, and provider calls are contained in
    this class.  Nothing downstream of the returned PreparedValidationInputs
    should require provider access.
    """

    def __init__(
        self,
        catalog_provider: StarCatalogProvider,
        population_provider: PopulationSynthesisProvider | None = None,
        aperture_provider: ApertureProvider | None = None,
        registry: ScenarioRegistry | None = None,
    ) -> None:
        """Construct with injected providers.

        Args:
            catalog_provider: Required.  Queries the star catalog.
            population_provider: Optional.  Fetches TRILEGAL background population.
                If None, trilegal-dependent scenarios will receive no population data.
            aperture_provider: Optional.  Provides pixel-level aperture data for
                flux-ratio computation.  If None, flux ratios must be set via
                calc_depths() / set directly on stars before compute.
            registry: Scenario registry used to validate scenario_ids and determine
                TRILEGAL eligibility.  Must be the same registry passed to the
                ValidationEngine that will call compute_prepared().  Defaults to
                DEFAULT_REGISTRY.
        """
        self._catalog = catalog_provider
        self._population = population_provider
        self._aperture = aperture_provider
        self._registry = registry if registry is not None else DEFAULT_REGISTRY

    def prepare(
        self,
        target_id: int,
        sectors: np.ndarray,
        light_curve: LightCurve,
        config: Config,
        period_days: float | list[float] | tuple[float, float],
        mission: str = "TESS",
        search_radius: int = 10,
        transit_depth: float | None = None,
        pixel_coords_per_sector: list[np.ndarray] | None = None,
        aperture_pixels_per_sector: list[np.ndarray] | None = None,
        sigma_psf_px: float = 0.75,
        trilegal_cache_path: str | None = None,
        contrast_curve_file: str | None = None,
        contrast_curve_band: str = "TESS",
        external_lc_files: list[str] | None = None,
        filt_lcs: list[str] | None = None,
        scenario_ids: list | None = None,
        molusc_file: str | None = None,
        # molusc_file: local path — not yet materialised; deferred to Phase 4
        # if remote execution requires embedded content rather than a path.
    ) -> PreparedValidationInputs:
        """Fetch catalog and population data, load artifacts, return prepared inputs.

        Performs all provider-backed and filesystem IO.  The returned
        PreparedValidationInputs is self-contained for provider-free compute.

        Args:
            target_id: TIC (or KIC/EPIC) identifier.
            sectors: Sector/quarter/campaign numbers as a numpy array.
            light_curve: Phase-folded, normalised photometric time series.
            config: Runtime configuration.
            period_days: Orbital period in days (scalar or [min, max] range).
            mission: Survey mission name.  Only ``"TESS"`` is supported for
                prepared compute.  Kepler/K2 support is experimental and not
                yet available for prepared or remote compute jobs.
            search_radius: Search radius in pixels for catalog query.
            transit_depth: Observed transit depth (fractional).  Required if
                flux ratios should be computed from pixel data.
            pixel_coords_per_sector: Per-sector star pixel positions for flux
                ratio computation.  Required together with transit_depth.
            aperture_pixels_per_sector: Per-sector aperture pixel positions.
            sigma_psf_px: PSF sigma in pixels for flux ratio computation.
            trilegal_cache_path: Optional path to a cached TRILEGAL CSV.
            contrast_curve_file: Optional path to contrast curve file.
            contrast_curve_band: Filter band label for the contrast curve.
            external_lc_files: Optional list of paths to external LC files.
            filt_lcs: Filter band labels for external_lc_files (same length).
                Must be provided together with external_lc_files; raises if
                one is non-empty and the other is empty or None.
            scenario_ids: Optional list of ScenarioIDs that will be run.
                Used to determine whether a TRILEGAL population fetch is
                needed.  If None, the full default registry is assumed.
            molusc_file: Local path to MOLUSC output file.
                NOTE: bare path — not yet materialised; deferred to Phase 4.

        Returns:
            PreparedValidationInputs ready for ValidationEngine.compute_prepared().
        """
        from datetime import datetime, timezone

        warnings: list[str] = []

        # ---- 0. Mission gate ----
        # Only TESS is fully supported for prepared compute.  Kepler/K2 prep
        # paths are incomplete (no cutout support, no K2 LDC data) and are not
        # eligible for prepared or remote compute jobs.
        if mission != "TESS":
            from triceratops.validation.errors import UnsupportedComputeModeError
            raise UnsupportedComputeModeError(
                f"prepare() only supports mission='TESS'. Got {mission!r}. "
                "Kepler/K2 support is experimental and not available for "
                "prepared compute jobs."
            )

        # ---- 1. Validate scenario_ids against the registry ----
        # Do this before any IO so callers get a clear error for unregistered IDs.
        # Unregistered IDs are silently tolerated by nothing downstream — they would
        # crash with KeyError inside compute().  Fail here instead with the ID named.
        if scenario_ids is not None:
            unknown = [sid for sid in scenario_ids if self._registry.get_or_none(sid) is None]
            if unknown:
                raise ValueError(
                    f"scenario_ids contains IDs not registered in the registry: {unknown}. "
                    "Remove them or register the corresponding scenario before calling prepare()."
                )

        # ---- 2. Catalog query ----
        stellar_field: StellarField = self._catalog.query_nearby_stars(
            tic_id=target_id,
            search_radius_px=search_radius,
            mission=mission,
        )
        # Validate immediately after assembly — catch catalog bugs before any
        # downstream IO (TRILEGAL fetch, file loads).
        stellar_field.validate()

        # Mission integrity check on the returned field — the catalog provider
        # is the authoritative source of stellar_field.mission and may disagree
        # with the requested mission argument (misbehaving stub or real provider).
        # This check uses the same source (stellar_field.mission) as the compute
        # boundary in PreparedValidationInputs.validate(), so both layers are
        # consistent even if the field is assembled by a custom provider.
        if stellar_field.mission != "TESS":
            from triceratops.validation.errors import UnsupportedComputeModeError
            raise UnsupportedComputeModeError(
                f"Catalog query returned a field with mission={stellar_field.mission!r}. "
                "Prepared compute only supports mission='TESS'."
            )

        # ---- 3. Flux ratios and transit depths (if depth data provided) ----
        if (
            transit_depth is not None
            and pixel_coords_per_sector is not None
            and aperture_pixels_per_sector is not None
        ):
            flux_ratios = compute_flux_ratios(
                stellar_field,
                pixel_coords_per_sector,
                aperture_pixels_per_sector,
                sigma_psf_px,
            )
            transit_depths = compute_transit_depths(flux_ratios, transit_depth)
            for star, fr, td in zip(stellar_field.stars, flux_ratios, transit_depths):
                star.flux_ratio = fr
                star.transit_depth_required = td
        else:
            if transit_depth is not None:
                warnings.append(
                    "transit_depth provided but pixel_coords_per_sector or "
                    "aperture_pixels_per_sector missing — flux ratios not computed."
                )

        # ---- 4. TRILEGAL population fetch (only if needed by active scenarios) ----
        trilegal_population: TRILEGALResult | None = None
        trilegal_cache_origin: str | None = None
        if self._population is not None:
            from triceratops.domain.scenario_id import ScenarioID

            # scenario_ids already validated in step 0; all IDs are registered.
            if scenario_ids is not None:
                eligible = [self._registry.get(sid) for sid in scenario_ids]
            else:
                eligible = self._registry.all_scenarios()
            needs_trilegal = any(
                s.scenario_id in ScenarioID.trilegal_scenarios()
                for s in eligible
            )
            if needs_trilegal:
                cache = Path(trilegal_cache_path) if trilegal_cache_path else None
                target = stellar_field.target
                trilegal_population = self._population.query(
                    ra_deg=target.ra_deg,
                    dec_deg=target.dec_deg,
                    target_tmag=target.tmag,
                    cache_path=cache,
                )
                trilegal_cache_origin = trilegal_cache_path

        # ---- 5. Contrast curve loading ----
        contrast_curve: ContrastCurve | None = None
        if contrast_curve_file is not None:
            from triceratops.io.contrast_curves import load_contrast_curve
            contrast_curve = load_contrast_curve(
                Path(contrast_curve_file), band=contrast_curve_band,
            )

        # ---- 6. External light curves loading ----
        external_lcs: list[ExternalLightCurve] | None = None
        _have_files = bool(external_lc_files)
        _have_filts = bool(filt_lcs)
        if _have_files or _have_filts:
            if not (_have_files and _have_filts):
                raise ValueError(
                    "external_lc_files and filt_lcs must both be provided together; "
                    f"got external_lc_files={external_lc_files!r} and filt_lcs={filt_lcs!r}."
                )
            if len(external_lc_files) != len(filt_lcs):  # type: ignore[arg-type]
                raise ValueError(
                    f"external_lc_files and filt_lcs must have the same length, "
                    f"got {len(external_lc_files)} files and {len(filt_lcs)} filters."  # type: ignore[arg-type]
                )
            from triceratops.io.external_lc import load_external_lc_as_object
            external_lcs = [
                load_external_lc_as_object(Path(f), b)
                for f, b in zip(external_lc_files, filt_lcs)  # type: ignore[arg-type]
            ]

        # ---- Metadata ----
        _metadata = PreparedValidationMetadata(
            prep_timestamp=datetime.now(tz=timezone.utc),
            source="ValidationPreparer",
            trilegal_cache_origin=trilegal_cache_origin,
            warnings=warnings,
        )

        return PreparedValidationInputs(
            target_id=target_id,
            stellar_field=stellar_field,
            light_curve=light_curve,
            config=config,
            period_days=period_days,
            trilegal_population=trilegal_population,
            external_lcs=external_lcs,
            contrast_curve=contrast_curve,
            molusc_file=molusc_file,
            scenario_ids=scenario_ids,
        )
