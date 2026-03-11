"""DataAssemblyOrchestrator: coordinates all input assembly for one run."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from triceratops.assembly.config import AssemblyConfig
from triceratops.assembly.errors import AssemblyConfigError, AssemblyError, AssemblyLightCurveError
from triceratops.assembly.inputs import AssembledInputs, AssemblyMetadata
from triceratops.domain.scenario_id import ScenarioID
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry

if TYPE_CHECKING:
    from triceratops.assembly.protocols import (
        ArtifactStore,
        ContrastCurveSource,
        ExternalLcSource,
        MoluscSource,
        LightCurveSource,
    )
    from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
    from triceratops.domain.entities import (
        ExternalLightCurve,
        LightCurve,
        StellarField,
    )
    from triceratops.domain.molusc import MoluscData
    from triceratops.domain.value_objects import ContrastCurve
    from triceratops.lightcurve.ephemeris import EphemerisResolver, ResolvedTarget
    from triceratops.population.protocols import PopulationSynthesisProvider, TRILEGALResult


class DataAssemblyOrchestrator:
    """Coordinates all input assembly for one validation run.

    Invokes input-specific sub-pipelines, merges their outputs into
    AssembledInputs, and carries provenance in AssemblyMetadata.

    The orchestrator is a coordination layer — it does not implement
    source-specific transformation logic itself. Each input family has
    its own sub-pipeline in triceratops/assembly/pipelines/.

    All providers and sources are injected at construction time.
    """

    def __init__(
        self,
        *,
        catalog_provider: StarCatalogProvider,
        ephemeris_resolver: EphemerisResolver | None = None,
        lc_source: LightCurveSource | None = None,
        population_provider: PopulationSynthesisProvider | None = None,
        aperture_provider: ApertureProvider | None = None,
        contrast_source: ContrastCurveSource | None = None,
        molusc_source: MoluscSource | None = None,
        external_lc_source: ExternalLcSource | None = None,
        artifact_store: ArtifactStore | None = None,
        registry: ScenarioRegistry | None = None,
    ) -> None:
        self._catalog = catalog_provider
        self._ephemeris_resolver = ephemeris_resolver
        self._lc_source = lc_source
        self._population = population_provider
        self._aperture = aperture_provider
        self._contrast = contrast_source
        self._molusc = molusc_source
        self._external_lc = external_lc_source
        self._artifact_store = artifact_store
        self._registry = registry if registry is not None else DEFAULT_REGISTRY

    def assemble(
        self,
        target: ResolvedTarget,
        config: AssemblyConfig | None = None,
        *,
        scenario_ids: Sequence[ScenarioID] | None = None,
        transit_depth: float | None = None,
        pixel_coords_per_sector: list[np.ndarray] | None = None,
        aperture_pixels_per_sector: list[np.ndarray] | None = None,
        sigma_psf_px: float = 0.75,
        stellar_field: StellarField | None = None,
    ) -> AssembledInputs:
        """Assemble all inputs for a validation run.

        Args:
            target: Resolved target with TIC ID and optional ephemeris.
            config: Assembly configuration. Defaults to AssemblyConfig().
            scenario_ids: Restrict to these scenarios. None means all.
            transit_depth: Observed transit depth for flux-ratio computation.
            pixel_coords_per_sector: Per-sector pixel coordinates for flux ratios.
            aperture_pixels_per_sector: Per-sector aperture masks for flux ratios.
            sigma_psf_px: PSF standard deviation in pixels.
            stellar_field: Pre-existing stellar field to use instead of querying.

        Returns:
            AssembledInputs with all assembled data and provenance metadata.
        """
        if config is None:
            config = AssemblyConfig()

        # Validate scenario_ids before any I/O
        if scenario_ids is not None:
            unknown = [sid for sid in scenario_ids if sid not in self._registry]
            if unknown:
                raise AssemblyConfigError(
                    f"Unknown scenario IDs: {unknown!r}. "
                    f"Registered: {sorted(self._registry)}"
                )

        all_warnings: list[str] = []
        all_source_labels: list[str] = []
        all_artifact_ids: list[str] = []
        per_input_source: dict[str, str] = {}

        # Step 1: Stellar field
        if stellar_field is None:
            stellar_field, sf_warnings = self._assemble_stellar_field(
                target, config, transit_depth,
                pixel_coords_per_sector, aperture_pixels_per_sector,
                sigma_psf_px,
            )
            all_warnings.extend(sf_warnings)
            all_source_labels.append("catalog")
            per_input_source["stellar_field"] = "catalog"
        else:
            per_input_source["stellar_field"] = "provided"

        # Step 2: Light curve
        light_curve: LightCurve | None = None
        if config.include_light_curve and self._lc_source is not None:
            if target.ephemeris is None:
                raise AssemblyLightCurveError(
                    "Light-curve assembly requested but target has no ephemeris. "
                    "Set include_light_curve=False or provide an ephemeris."
                )
            light_curve, lc_label, lc_warnings, lc_artifacts = (
                self._assemble_light_curve(target, config)
            )
            all_warnings.extend(lc_warnings)
            all_artifact_ids.extend(lc_artifacts)
            all_source_labels.append(lc_label)
            per_input_source["light_curve"] = lc_label

        # Enforce require_light_curve after Step 2
        if config.require_light_curve and config.include_light_curve and light_curve is None:
            raise AssemblyLightCurveError(
                "config.require_light_curve is True but no light curve was assembled. "
                "Provide an lc_source or set require_light_curve=False."
            )

        # Step 3: Contrast curve
        contrast_curve: ContrastCurve | None = None
        if config.include_contrast_curve and self._contrast is not None:
            contrast_curve, cc_warnings = self._assemble_contrast_curve(config)
            all_warnings.extend(cc_warnings)
            all_source_labels.append("contrast_curve")
            per_input_source["contrast_curve"] = "contrast_source"

        # Step 4: MOLUSC
        molusc_data: MoluscData | None = None
        if config.include_molusc and self._molusc is not None:
            molusc_data, mol_warnings = self._assemble_molusc()
            all_warnings.extend(mol_warnings)
            all_source_labels.append("molusc")
            per_input_source["molusc_data"] = "molusc_source"

        # Step 5: TRILEGAL
        trilegal_population: TRILEGALResult | None = None
        if (
            config.include_trilegal
            and self._population is not None
            and self._scenarios_need_trilegal(scenario_ids)
        ):
            trilegal_population, tri_warnings = self._assemble_trilegal(
                stellar_field, config,
            )
            all_warnings.extend(tri_warnings)
            all_source_labels.append("trilegal")
            per_input_source["trilegal_population"] = "trilegal"

        # Step 6: External light curves
        external_lcs: list[ExternalLightCurve] | None = None
        if config.include_external_lcs and self._external_lc is not None:
            external_lcs, elc_warnings = self._assemble_external_lcs()
            all_warnings.extend(elc_warnings)
            all_source_labels.append("external_lcs")
            per_input_source["external_lcs"] = "external_lc_source"

        metadata = AssemblyMetadata(
            source_labels=tuple(all_source_labels),
            warnings=tuple(all_warnings),
            artifact_ids=tuple(all_artifact_ids),
            created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
            per_input_source=tuple(sorted(per_input_source.items())),
        )

        return AssembledInputs(
            resolved_target=target,
            stellar_field=stellar_field,
            light_curve=light_curve,
            contrast_curve=contrast_curve,
            molusc_data=molusc_data,
            trilegal_population=trilegal_population,
            external_lcs=external_lcs,
            metadata=metadata,
        )

    # -- Private delegation methods --

    def _scenarios_need_trilegal(
        self, scenario_ids: Sequence[ScenarioID] | None,
    ) -> bool:
        """Check whether any of the requested scenarios need TRILEGAL data."""
        if scenario_ids is not None:
            ids_to_check = [sid for sid in scenario_ids if sid in self._registry]
        else:
            ids_to_check = [s.scenario_id for s in self._registry.all_scenarios()]
        trilegal_ids = ScenarioID.trilegal_scenarios()
        return any(sid in trilegal_ids for sid in ids_to_check)

    def _assemble_stellar_field(
        self,
        target: ResolvedTarget,
        config: AssemblyConfig,
        transit_depth: float | None,
        pixel_coords_per_sector: list[np.ndarray] | None,
        aperture_pixels_per_sector: list[np.ndarray] | None,
        sigma_psf_px: float,
    ) -> tuple[StellarField, list[str]]:
        from triceratops.assembly.pipelines.stellar_field import assemble_stellar_field

        return assemble_stellar_field(
            self._catalog, target, config,
            transit_depth, pixel_coords_per_sector,
            aperture_pixels_per_sector, sigma_psf_px,
        )

    def _assemble_light_curve(
        self,
        target: ResolvedTarget,
        config: AssemblyConfig,
    ) -> tuple[LightCurve | None, str, list[str], list[str]]:
        from triceratops.assembly.pipelines.lightcurve import assemble_light_curve

        assert target.ephemeris is not None  # guarded by caller
        return assemble_light_curve(
            self._lc_source,  # type: ignore[arg-type]
            self._artifact_store,
            target.ephemeris,
            config.lc_config,
            config.require_light_curve,
        )

    def _assemble_contrast_curve(
        self, config: AssemblyConfig,
    ) -> tuple[ContrastCurve, list[str]]:
        from triceratops.assembly.pipelines.contrast import assemble_contrast_curve

        return assemble_contrast_curve(
            self._contrast,  # type: ignore[arg-type]
            config.contrast_curve_band,
        )

    def _assemble_molusc(self) -> tuple[MoluscData, list[str]]:
        from triceratops.assembly.pipelines.molusc import assemble_molusc

        return assemble_molusc(self._molusc)  # type: ignore[arg-type]

    def _assemble_trilegal(
        self,
        stellar_field: StellarField,
        config: AssemblyConfig,
    ) -> tuple[TRILEGALResult, list[str]]:
        from triceratops.assembly.pipelines.trilegal import assemble_trilegal

        return assemble_trilegal(
            self._population,  # type: ignore[arg-type]
            stellar_field,
            config.trilegal_cache_path,
        )

    def _assemble_external_lcs(
        self,
    ) -> tuple[list[ExternalLightCurve], list[str]]:
        from triceratops.assembly.pipelines.external_lcs import assemble_external_lcs

        return assemble_external_lcs(self._external_lc)  # type: ignore[arg-type]
