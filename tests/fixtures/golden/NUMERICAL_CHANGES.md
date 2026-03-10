# Numerical Changes Log

Append-only record of every intentional change to TRICERATOPS+ output values.

Each entry documents: what changed, why the old behaviour was wrong, what the
correct formula is, which scenarios and metrics are affected, and the direction
of the FPP shift.  Entries are ordered chronologically.  The golden JSON files
in this directory are always updated at the same commit as the entry that
describes them.

The target `golden/*.json` files track **triceratops-fast output** after all
applied fixes.  The vendor TRICERATOPS+ output is preserved in comments below
for reference; it differs from the golden values because it contains the bugs
that are fixed here.

---

## NC-01 — compute_lnZ: log-sum-exp fix for marginal likelihood underflow

**Commit:** `366e443d`
**Date:** 2026-03-09
**Status:** Applied
**Affects FPP:** Yes (decreases — background false-positive scenarios gain
probability mass)
**Affects NFPP:** Yes (can change, direction depends on target)

### Affected scenarios

All scenarios are nominally affected because `compute_lnZ` is used by every
scenario.  In practice the change is largest for scenarios whose likelihood
distributions have heavy negative tails — i.e. bright-background and diluted-
background scenarios (BTP, BEB, BEBx2P, DTP) and high-mass EB scenarios.

### The bug

The original TRICERATOPS+ code computes the marginal log-likelihood (log
evidence) as:

```python
Z     = mean(exp(lnL + lnz_const))   # lnz_const = 650
lnZ   = log(Z)
```

For any scenario where a significant fraction of MC draws produce very poor
transit fits, `lnL` can be extremely negative (e.g. -10 000 per sample).
`exp(-10 000 + 650) = exp(-9 350)` underflows to exactly `0.0` in IEEE 754
float64 (underflow threshold ~exp(-745)).  When all or most draws underflow,
`Z = 0` and `lnZ = -inf`.

A scenario with `lnZ = -inf` is assigned **zero relative probability**,
regardless of how much evidence actually supports it.  The original code
therefore systematically underweights background and EB scenarios whose
typical likelihoods are far from the maximum, pushing FPP artificially
toward 1.0.

### The fix

The log-sum-exp identity rewrites the same quantity in a form that is immune
to underflow:

```
log(mean(exp(lnL)))
  = lnL_max + log( sum(exp(lnL - lnL_max)) / N )
```

Because every term `lnL[i] - lnL_max <= 0`, `exp(lnL[i] - lnL_max)` is
always in `(0, 1]` and never underflows.  The result is mathematically
identical to the original formula but numerically stable for any lnL
magnitude.

The `lnz_const` offset is preserved in the returned value for backward
compatibility with callers that inspect raw lnZ numbers (it cancels in all
relative-probability comparisons).

### Observed FPP shift (TOI-4051, n=10 000, seed=42)

| Metric | Vendor (buggy) | triceratops-fast (fixed) |
|--------|---------------|--------------------------|
| FPP    | 0.999173      | 0.996334                 |
| NFPP   | 0.000146      | 0.000000                 |

Note: NFPP reaching 0.0 reflects that for this target the nearby-star
scenarios (NEB, NEBx2P) were also affected by the underflow — their
lnZ values were -inf in the original code.

---

## NC-02 — calc_depths: analytic PSF integral replaces adaptive quadrature

**Commit:** `6e7bbc1d`
**Date:** 2026-03-09
**Status:** Applied
**Affects FPP:** Yes (changes flux ratios and transit depths used by all
scenarios, therefore shifts lnZ values for all scenarios)
**Affects NFPP:** Yes

### Affected scenarios

All scenarios that depend on per-star flux ratios and transit depths computed
by `calc_depths()`.  In practice this is every scenario, because the flux
ratio of the target star enters the light-curve renormalisation passed to
every scenario.

### The bug

The original TRICERATOPS+ `calc_depths()` computes the fraction of aperture
flux contributed by each star by numerically integrating a 2-D Gaussian PSF
over each aperture pixel using `scipy.integrate.dblquad` with a Python loop:

```python
for star in stars:
    for pixel in aperture:
        flux[star] += dblquad(gauss2d, ...)
```

`dblquad` uses adaptive Gaussian quadrature and is slow, but more
importantly the numerical tolerance it achieves is not uniform across the
parameter space.  For stars far from the aperture centre the integrand is
nearly zero and the adaptive scheme can return values that differ by up to
~1e-6 from the true integral.

### The fix

The 2-D Gaussian integral over a pixel box is separable and has an exact
closed form in terms of the standard normal CDF Φ:

```
∫∫ G(x,y) dx dy  =
    A * [Φ((px + 0.5 - μ_x)/σ) - Φ((px - 0.5 - μ_x)/σ)]
      * [Φ((py + 0.5 - μ_y)/σ) - Φ((py - 0.5 - μ_y)/σ)]
```

`scipy.special.ndtr` evaluates this to machine precision in a single
vectorised call.  The result agrees with `dblquad` at `rtol=1e-6` across all
tested configurations (verified in `tests/unit/catalog/test_flux_contributions.py`).

### Observed FPP shift

Flux ratio changes are typically < 1e-4 fractional for well-centred stars.
The FPP shift is scenario-dependent and is dominated by NC-01 for most
targets.  The two changes were applied together; their joint effect is
recorded in the golden files.

---

## NC-03 — lnprior_background: np.log10 → np.log (PENDING)

**Commit:** (pending)
**Date:** (pending)
**Status:** Pending — not yet applied.  Tracked here so the rationale is
written before the code change, not after.
**Affects FPP:** Yes (decreases — background false-positive scenarios gain
probability mass)
**Affects NFPP:** No (nearby-star scenarios are unaffected)

### Affected scenarios

BTP, BEB, BEBx2P, DTP — the four scenarios whose priors call
`lnprior_background()` or its inline equivalent.

### The bug

The occurrence-rate prior for a background star companion is proportional to
the area of an annulus at separation `s` arcsec:

```
p(s) ∝ n_comp × (1/3600)² × s²
```

The log of this prior should therefore be:

```python
lnprior = np.log( n_comp / 0.1 * (1/3600)**2 * s**2 )
```

The original TRICERATOPS+ code uses `np.log10` at three call sites:

```python
lnprior = np.log10( n_comp / 0.1 * (1/3600)**2 * s**2 )   # BUG
```

Because `np.log10(x) = np.log(x) / log(10) ≈ np.log(x) / 2.303`, using
`log10` reduces every log-prior value by a factor of `1/log(10) ≈ 0.434`.
This means background scenarios are assigned a prior that is `~2.3×` smaller
(in log space) than the correct value, underweighting them in the evidence
sum and inflating FPP.

For TESS-typical separations (~2–10 arcsec) the prior difference is
`log(s²) × (1 - 1/log10) ≈ 1.3–3.0` log-units per sample, accumulating
over 1M draws to a systematic lnZ bias of hundreds of units.

### Expected FPP shift

FPP will **decrease** further after this fix (background scenarios gain
additional probability mass beyond the NC-01 shift).  The exact magnitude
depends on the target's neighbour density.  Golden files will be updated
at the commit that applies this fix.

### Call sites

- `triceratops/priors/lnpriors.py` — `lnprior_background()`, line ~378
- `triceratops/scenarios/background_scenarios.py` — BTPScenario prior fallback
- `triceratops/scenarios/background_scenarios.py` — BEBScenario prior fallback

---

*To add a new entry: copy the template below, fill in all fields, and commit
the updated golden JSON files in the same PR.*

<!--
## NC-XX — Short title

**Commit:** (hash or "pending")
**Date:** YYYY-MM-DD
**Status:** Applied | Pending
**Affects FPP:** Yes/No (direction and brief reason)
**Affects NFPP:** Yes/No

### Affected scenarios
List which of TP/EB/EBx2P/PTP/.../BEBx2P are affected and why.

### The bug
Describe the original code behaviour, why it is incorrect, and the
mathematical formula that should be used.

### The fix
Describe the corrected formula or algorithm.

### Observed FPP shift (Target, n=X, seed=Y)
Table showing before/after values for FPP (and NFPP if applicable).
-->
