# TRICERATOPS

Bayesian false-positive probability calculator for candidate exoplanets.

## Quick start

```python
from triceratops import ValidationWorkspace, LightCurve

ws = ValidationWorkspace(tic_id=237101326, sectors=[1, 2])

result = ws.compute_probs(light_curve=lc, period_days=5.0)
print(f"FPP = {result.fpp:.4f}")
```

## API Reference

See the [API Reference](api/index.md) for full documentation of the public interface.
