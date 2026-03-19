"""Config, MissionConfig, and PhysicalConstants frozen dataclasses.

All magic numbers from the original TRICERATOPS+ codebase are named here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import astropy.constants as _const  # noqa: I001

_Msun: float = _const.M_sun.cgs.value  # type: ignore[attr-defined]
_Rsun: float = _const.R_sun.cgs.value  # type: ignore[attr-defined]
_Rearth: float = _const.R_earth.cgs.value  # type: ignore[attr-defined]
_G: float = _const.G.cgs.value  # type: ignore[attr-defined]
_au: float = _const.au.cgs.value  # type: ignore[attr-defined]


@dataclass(frozen=True)
class PhysicalConstants:
    """CGS physical constants sourced from astropy.constants.

    All values in CGS units (g, cm, s). Named here so that the rest of the
    codebase never imports astropy.constants directly -- changes to astropy
    version or unit system are isolated to this file.
    """

    Msun: float = _Msun
    Rsun: float = _Rsun
    Rearth: float = _Rearth
    G: float = _G
    au: float = _au

    @property
    def pi(self) -> float:
        """Mathematical constant pi."""
        return math.pi


# Module-level singleton -- import and use directly
CONST = PhysicalConstants()


@dataclass(frozen=True)
class MissionConfig:
    """Pixel geometry and timing parameters for a specific survey mission.

    Do not instantiate directly -- use the ``for_mission()`` class method.
    """

    mission: str
    pixel_size_arcsec: float  # angular size of one pixel in arcsec
    default_exptime_days: float  # default cadence for LC fitting
    default_nsamples: int  # pytransit supersampling rate

    # Sentinel values used in catalog queries
    search_radius_default_px: int = 10

    _MISSIONS: ClassVar[dict[str, dict]] = {
        "TESS": {
            "pixel_size_arcsec": 20.25,
            "default_exptime_days": 0.00139,  # 2-minute cadence
            "default_nsamples": 20,
        },
        "Kepler": {
            "pixel_size_arcsec": 4.0,
            "default_exptime_days": 0.02083,  # 30-minute cadence
            "default_nsamples": 10,
        },
        "K2": {
            "pixel_size_arcsec": 4.0,
            "default_exptime_days": 0.02083,
            "default_nsamples": 10,
        },
    }

    @classmethod
    def for_mission(cls, mission: str) -> MissionConfig:
        """Return the MissionConfig for the named mission.

        Args:
            mission: One of "TESS", "Kepler", or "K2".

        Raises:
            ValueError: If mission is not a recognized name.
        """
        if mission not in cls._MISSIONS:
            raise ValueError(
                f"Unknown mission {mission!r}. "
                f"Must be one of: {sorted(cls._MISSIONS)}"
            )
        return cls(mission=mission, **cls._MISSIONS[mission])

    @property
    def pixel_size_deg(self) -> float:
        """Pixel size converted to degrees."""
        return self.pixel_size_arcsec / 3600.0


@dataclass(frozen=True)
class Config:
    """Runtime parameters for a single TRICERATOPS+ validation run.

    All parameters have scientifically-motivated defaults matching the original
    codebase. Override by constructing with explicit keyword arguments.

    Attributes:
        n_mc_samples: Number of Monte Carlo draws per scenario (was ``N`` in original).
        n_best_samples: Number of top-likelihood draws to retain for best-fit
            parameter reporting (was ``N_samples = 1000`` in the accumulation block).
        seed: Optional NumPy RNG seed. When set, compute runs are reproducible.
        parallel: If True, use vectorized (masked array) likelihood evaluation.
            If False, use the serial per-sample loop. Original default: True.
        flat_priors: If True, draw planet radii uniformly instead of using the
            broken-power-law distribution. Original parameter name: ``flatpriors``.
        mission: Default mission when not overridden by MissionConfig.
        numerical_mode: Internal compatibility mode for parity investigations.
            ``"corrected"`` keeps the refactor's numerical fixes; ``"legacy"``
            restores the original evidence/background-prior behavior.
    """

    n_mc_samples: int = 1_000_000
    n_best_samples: int = 1000
    seed: int | None = None
    parallel: bool = True
    flat_priors: bool = False
    mission: str = "TESS"
    n_workers: int = 0
    numerical_mode: str = "corrected"
    """Number of worker processes for scenario-level parallelism.

    0  — serial execution (default; fully reproducible with np.random.seed).
    -1 — one worker per scenario up to os.cpu_count().
    N  — exactly N worker processes.

    Note: scenario-level multiprocessing (``n_workers != 0``) is not supported
    together with an explicit ``seed``. Use ``n_workers=0`` when you need
    seeded/reproducible runs; ``parallel=True`` still enables intra-scenario
    vectorisation in that mode.
    """

    def __post_init__(self) -> None:
        if self.n_mc_samples < 1:
            raise ValueError(f"n_mc_samples must be >= 1, got {self.n_mc_samples}")
        if self.n_best_samples < 1:
            raise ValueError(f"n_best_samples must be >= 1, got {self.n_best_samples}")
        if self.n_best_samples > self.n_mc_samples:
            raise ValueError(
                f"n_best_samples ({self.n_best_samples}) cannot exceed "
                f"n_mc_samples ({self.n_mc_samples})"
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0 or None, got {self.seed}")
        if self.mission not in ("TESS", "Kepler", "K2"):
            raise ValueError(f"Unknown mission {self.mission!r}")
        if self.n_workers < -1:
            raise ValueError(f"n_workers must be >= -1, got {self.n_workers}")
        if self.seed is not None and self.n_workers != 0:
            raise ValueError(
                "seeded scenario-level multiprocessing is not supported: "
                "use n_workers=0 for reproducible runs"
            )
        if self.numerical_mode not in ("corrected", "legacy"):
            raise ValueError(
                "numerical_mode must be 'corrected' or 'legacy', "
                f"got {self.numerical_mode!r}"
            )

    @property
    def mission_config(self) -> MissionConfig:
        """Convenience access to the MissionConfig for self.mission."""
        return MissionConfig.for_mission(self.mission)
