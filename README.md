# TRICERATOPS

Bayesian false-positive probability (FPP) calculator for candidate exoplanets.

Determines whether a transit signal detected by TESS is a genuine planet or a false positive
(eclipsing binary, background blend, etc.) by computing the relative probabilities of ~34
astrophysical scenarios.

This is a modernized rewrite building on two prior codebases:

- **[TRICERATOPS](https://github.com/stevengiacalone/triceratops)** — the original implementation by Giacalone & Dressing (2020) and Giacalone et al. (2021)
- **[TRICERATOPS+](https://github.com/JGB276/TRICERATOPS-plus)** — extended fork by J.G. Barraza with additional scenario handling and bug fixes

TRICERATOPS-2 refactors the core algorithms into a clean API with dependency-injected I/O and a provider-free compute boundary suitable for local or remote execution, while preserving numerical parity with the original implementations.

## Installation

> **Note:** This package is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/Holaxis-ai/triceratops-2.git
cd triceratops-2
pip install -e .
```

## Quick start

```python
import numpy as np
from triceratops import ValidationWorkspace, Config, LightCurve

# Create a workspace for a TESS target
ws = ValidationWorkspace(
    tic_id=237101326,
    sectors=np.array([1, 2]),
)

# Inspect the stellar field
print(ws.stars_df)

# Run FPP computation
result = ws.compute_probs(
    light_curve=light_curve,   # phase-folded LightCurve
    period_days=5.0,
)

print(f"FPP  = {result.fpp:.4f}")
print(f"NFPP = {result.nfpp:.4f}")
```

## Two-phase compute boundary

The library separates I/O from computation:

```python
# Phase 1: Prepare (fetches TRILEGAL, catalog data)
prepared = ws.prepare(light_curve, period_days=5.0)

# Phase 2: Compute (pure math, no network calls)
result = ws.compute_prepared(prepared)
```

`PreparedValidationInputs` is fully serializable — ship it to a remote worker
(e.g., Modal) for scaled execution.

## Numerical parity

This implementation reproduces the original TRICERATOPS results with four
documented bug fixes in the underlying numerics. All corrections are
verified by deterministic golden regression tests at fixed random seeds.

The runtime config also supports two numerical modes:

- `numerical_mode="corrected"`: the default scientific mode; keeps the
  documented numerical fixes enabled
- `numerical_mode="legacy"`: an internal compatibility mode for parity
  investigations against vendor `TRICERATOPS+`

`legacy` is intended for parity validation and migration checks, not normal
forward use. `corrected` remains the default mode for new computations.

`legacy` only restores the documented numerical behavior needed for parity
checks. It does not imply blanket emulation of every historical code path or
every optional input mode. In particular, nearby-host execution now follows the
validated per-neighbor parity path used by this rewrite, and external follow-up
light-curve parity for nearby hosts is still a deferred edge case rather than a
supported compatibility guarantee.

See [PARITY.md](PARITY.md) for the full numerical changes log with
formulas, affected scenarios, and before/after comparisons.

## Documentation

**[API Reference](https://Holaxis-ai.github.io/triceratops/)** — auto-generated from docstrings.

To serve locally:

```bash
pip install "holaxis-triceratops[docs]"
mkdocs serve
```

## Acknowledgments

This project would not exist without the foundational work of:

- **Steven Giacalone & Courtney Dressing** — creators of the original [TRICERATOPS](https://github.com/stevengiacalone/triceratops) (Giacalone & Dressing 2020, *ApJ*; Giacalone et al. 2021, *AJ*)
- **J.G. Barraza** — author of [TRICERATOPS+](https://github.com/JGB276/TRICERATOPS-plus), which extended the original with additional scenario handling and corrections

If you use this software in published research, please cite the original TRICERATOPS papers:

> Giacalone, S. & Dressing, C. D. 2020, *The Astrophysical Journal*, 900, 24
>
> Giacalone, S. et al. 2021, *The Astronomical Journal*, 161, 24

## License

MIT
