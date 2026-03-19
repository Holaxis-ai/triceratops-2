from __future__ import annotations

import pytest

from triceratops.config.config import Config


def test_config_rejects_seeded_scenario_level_multiprocessing() -> None:
    with pytest.raises(
        ValueError,
        match="seeded scenario-level multiprocessing is not supported",
    ):
        Config(
            n_mc_samples=100,
            n_best_samples=10,
            seed=17,
            n_workers=1,
        )


def test_config_allows_seeded_intra_scenario_parallelism_without_process_pool() -> None:
    cfg = Config(
        n_mc_samples=100,
        n_best_samples=10,
        seed=17,
        parallel=True,
        n_workers=0,
    )

    assert cfg.seed == 17
    assert cfg.parallel is True
    assert cfg.n_workers == 0
