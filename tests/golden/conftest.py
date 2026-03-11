"""Conftest for golden parity tests.

Mocks pytransit with a simple box-transit model when the real pytransit
is not available (e.g., numpy version incompatibility). This allows
golden tests to run in any environment. The mock produces deterministic
results consistent with the golden JSON values captured under the same mock.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np


def _ensure_pytransit_numpy_compat() -> None:
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid


class _MockQuadraticModel:
    """Minimal pytransit QuadraticModel replacement using box transits."""

    def __init__(self, interpolate: bool = False) -> None:
        self._time = np.array([0.0])

    def set_data(
        self,
        time: np.ndarray,
        exptimes: np.ndarray | None = None,
        nsamples: list | None = None,
    ) -> None:
        self._time = np.asarray(time)

    def evaluate_ps(
        self,
        k: float,
        ldc: np.ndarray,
        t0: float,
        p: float,
        a: float,
        i: float,
        e: float = 0.0,
        w: float = 0.0,
    ) -> np.ndarray:
        k, p, t0 = float(k), float(p), float(t0)
        if p == 0:
            return np.ones_like(self._time)
        phase = ((self._time - t0) / p + 0.5) % 1.0 - 0.5
        flux = np.ones_like(self._time)
        flux[np.abs(phase) < 0.025] = 1.0 - min(k**2, 0.99)
        return flux

    def evaluate_pv(self, pvp: np.ndarray, ldc: np.ndarray) -> np.ndarray:
        pvp = np.atleast_2d(pvp)
        n_samples = pvp.shape[0]
        n_time = len(self._time)
        flux = np.ones((n_samples, n_time))
        for j in range(n_samples):
            k, t0, p = pvp[j, 0], pvp[j, 1], pvp[j, 2]
            if p == 0:
                continue
            phase = ((self._time - t0) / p + 0.5) % 1.0 - 0.5
            flux[j, np.abs(phase) < 0.025] = 1.0 - min(k**2, 0.99)
        return flux


def _install_mock_pytransit() -> None:
    """Install mock pytransit if the real one is not importable."""
    _ensure_pytransit_numpy_compat()
    try:
        import pytransit  # noqa: F401
    except (ImportError, Exception):
        mock_module = MagicMock()
        mock_module.QuadraticModel = _MockQuadraticModel
        sys.modules["pytransit"] = mock_module

        # Clear thread-local transit model cache so the mock is picked up
        try:
            import triceratops.likelihoods.transit_model as tm_mod

            if hasattr(tm_mod._local, "tm"):
                del tm_mod._local.tm
            if hasattr(tm_mod._local, "tm_sec"):
                del tm_mod._local.tm_sec
        except ImportError:
            pass


_install_mock_pytransit()
