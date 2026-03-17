# TRICERATOPS

Bayesian false-positive probability (FPP) calculator for candidate exoplanets.

Determines whether a transit signal detected by TESS is a genuine planet or a false positive
(eclipsing binary, background blend, etc.) by computing the relative probabilities of ~34
astrophysical scenarios.

This is a modernized rewrite of the original
[TRICERATOPS](https://github.com/stevengiacalone/triceratops) codebase
(Giacalone & Dressing 2020, Giacalone et al. 2021), with a clean API,
dependency-injected I/O, and a provider-free compute boundary suitable for
local or remote execution.

## Installation

```bash
pip install holaxis-triceratops
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
conservative (FPP decreases slightly) and are verified by deterministic
golden regression tests at fixed random seeds.

See [PARITY.md](PARITY.md) for the full numerical changes log with
formulas, affected scenarios, and before/after comparisons.

## Documentation

**[API Reference](https://Holaxis-ai.github.io/triceratops/)** — auto-generated from docstrings.

To serve locally:

```bash
pip install "holaxis-triceratops[docs]"
mkdocs serve
```

## License

MIT
