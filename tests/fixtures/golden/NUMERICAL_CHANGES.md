# Numerical Changes Log

Append-only record of every intentional change to TRICERATOPS+ output values.

Each entry documents: what changed, why the old behaviour was wrong, what the
correct formula is, which scenarios and metrics are affected, and the expected
scenario-family shift. Net headline FPP/NFPP movement is often target-dependent
because all scenario probabilities renormalize together after the fix. Entries
are ordered chronologically. The golden JSON files
in this directory are always updated at the same commit as the entry that
describes them.

The target `golden/*.json` files track **triceratops-fast output** after all
applied fixes.  The vendor TRICERATOPS+ output is preserved in comments below
for reference; it differs from the golden values because it contains the bugs
that are fixed here. For parity investigations against vendor
TRICERATOPS+, use `numerical_mode="legacy"`; the golden fixtures remain the
corrected-mode reference.

---

## NC-01 — compute_lnZ: log-sum-exp fix for marginal likelihood underflow

**Commit:** `366e443d`
**Date:** 2026-03-09
**Status:** Applied
**Affects FPP:** Yes (net direction is target-dependent; tends to increase the
relative weight of heavy-tail false-positive scenarios)
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
Z     = mean(exp(lnL + const))
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

The initial rewrite preserved a legacy additive offset in returned `lnZ`
values for compatibility. That offset was later removed once the stable
log-sum-exp implementation was adopted everywhere and no caller needed the
old shifted scale.

### Observed shift (target-dependent)

| Metric | Vendor (buggy) | triceratops-fast (fixed) |
|--------|---------------|--------------------------|
| FPP    | 0.999173      | 0.996334                 |
| NFPP   | 0.000146      | 0.000000                 |

This target moved downward in FPP because previously underweighted
false-positive scenarios regained probability mass. That direction is not
universal. For example, on `TOI-205.01` at `n=1,000,000`, `seed=17`, the
corrected-mode run shifts probability mass out of `DTP` and into
`TP/PTP/STP`, raising FPP from `0.016659227756992467` (`legacy`) to
`0.01874191364346589` (`corrected`).

Note: NFPP reaching 0.0 for `TOI-4051` reflects that nearby-star scenarios
(`NEB`, `NEBx2P`) were also affected by the underflow — their lnZ values were
`-inf` in the original code.

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

## NC-03 — lnprior_background: np.log10 → np.log

**Commit:** `931230f2`
**Date:** 2026-03-09
**Status:** Applied
**Affects FPP:** Yes (net direction is target-dependent; tends to increase the
relative weight of background scenarios)
**Affects NFPP:** No (nearby-star scenarios are unaffected)

### Affected scenarios

BTP, BEB, BEBx2P, DTP, DEB, DEBx2P — the six scenarios whose priors call
`lnprior_background()` or its inline equivalent via `_compute_lnprior_companion()`.

DEB and DEBx2P are affected because they use `_compute_lnprior_companion()`,
which calls `lnprior_background()` (when a contrast curve is present) or uses
the same inline `np.log10` formula (without contrast curve).  The no-contrast
path in `_compute_lnprior_companion()` was also corrected from `np.log10` to
`np.log`.

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

### Net FPP interpretation

This fix increases background-scenario priors, especially for
`BTP/BEB/BEBx2P/DTP/DEB/DEBx2P`. In isolation that tends to push headline FPP
down, but the final net direction is still target-dependent after all scenario
families renormalize together.

`TOI-205.01` is a concrete example where the corrected mode still ends up with
higher FPP overall at `n=1,000,000`, `seed=17`:

- `legacy`: `FPP = 0.016659227756992467`
- `corrected`: `FPP = 0.01874191364346589`

For that target, corrected mode reduces `DTP` but increases `TP`, `PTP`, and
`STP` enough that the net FPP rises.

### Call sites

- `triceratops/priors/lnpriors.py` — `lnprior_background()`, line ~378
- `triceratops/scenarios/background_scenarios.py` — `_compute_lnprior_companion()` no-contrast fallback (DTP, DEB, DEBx2P)
- `triceratops/scenarios/background_scenarios.py` — `_compute_bright_background_lnprior()` no-contrast fallback (BTP, BEB, BEBx2P)

---

## NC-04 — BEBScenario: coll_twin → coll for q < 0.95 collision mask

**Commit:** `ebcca028`
**Date:** 2026-03-09
**Status:** Applied
**Affects FPP:** Yes in principle (BEB/BEBx2P lnZ decreases), but the
magnitude is below Monte Carlo noise at n=10,000 — FPP is unchanged
at 4 significant figures for both TOI-4051 and TOI-4155.
**Affects NFPP:** No (nearby-star scenarios are unaffected)

### Affected scenarios

BEB, BEBx2P — the two scenarios produced by `BEBScenario._evaluate_lnL`.
The q < 0.95 branch (standard period EB) uses an incorrect collision check;
the q >= 0.95 branch (twin, 2x period) was already correct.

### The bug

`BEBScenario._evaluate_lnL` has two branches:

- **q < 0.95** — standard EB at the observed period `P_orb`
- **q >= 0.95** — equal-mass twin EB at `2 × P_orb`

Each branch needs a separate collision check because the orbital semi-major
axis `a` differs between the two periods (Kepler's third law: `a ∝ P^(2/3)`).

The correct checks are:

```python
coll      = collision_check(a,      R_EB, R_comp, eccs)  # standard-period orbit
coll_twin = (2 * R_comp * Rsun) > a_twin * (1 - eccs)    # twin-period orbit
```

The original TRICERATOPS+ code (marginal_likelihoods.py line ~3492) and
the triceratops-fast code (before this fix) both apply `coll_twin` to the
q < 0.95 branch:

```python
# BUG: q < 0.95 branch uses coll_twin instead of coll
mask = (incs >= inc_min) & (coll_twin == False) & (qs < 0.95)
```

`coll_twin` tests whether the *twin* orbit (at `2 × P_orb`, larger `a_twin`)
would collide, which is a **less restrictive** criterion than `coll` for the
actual standard-period orbit.  Because `a_twin > a`, `coll_twin` triggers
less often, so the q < 0.95 mask admits draws that would be excluded by the
correct collision criterion.  This over-counts physically impossible BEB
configurations, inflating BEB and BEBx2P lnZ.

Note: `DEBScenario._evaluate_lnL` (line ~774) already uses `geometry["coll"]`
for the q < 0.95 branch and is correct.

### The fix

Change `geometry["coll_twin"]` to `geometry["coll"]` in the
`BEBScenario._evaluate_lnL` q < 0.95 block:

```python
# Fixed: q < 0.95 uses the standard-period collision check
mask = build_transit_mask(
    samples["incs"], geometry["Ptra"], geometry["coll"],  # coll, not coll_twin
    extra_mask=q_lt_mask,
)
```

No other logic changes required.

### Observed FPP shift

The number of BEB draws with `coll == True` but `coll_twin == False` (the
only draws affected by this fix) is zero for both test targets at n=10,000,
seed=42.  FPP is unchanged at 4 significant figures:

| Target   | FPP before | FPP after |
|----------|-----------|-----------|
| TOI-4051 | 0.9961    | 0.9961    |
| TOI-4155 | 0.01550   | 0.01550   |

The fix is still correct — it removes a logically wrong criterion — but the
magnitude is below Monte Carlo noise at n=10,000 for these targets.  At
n=1,000,000 draws or in highly crowded fields with short-period BEB
candidates a measurable shift may emerge.  Golden JSON files are unchanged.

### Call site

- `triceratops/scenarios/background_scenarios.py` — `BEBScenario._evaluate_lnL`,
  q < 0.95 block, `build_transit_mask(... geometry["coll_twin"] ...)` → `geometry["coll"]`

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
