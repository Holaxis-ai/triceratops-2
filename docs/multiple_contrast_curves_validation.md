# Multiple Contrast Curves: Local Validation Report

Date: 2026-06-09

Branch: `codex/multiple-contrast-curves`

Implementation commits:

- `5380d97 Support multiple contrast curves`
- `cb58c05 Document local multi-curve validation`

Follow-up validation changes:

- `scripts/validate_multiple_contrast_curves.py`
- generator-safe contrast-curve input canonicalization at the validation boundary

## Scope

This report validates the local multiple-contrast-curve implementation with
realistic Monte Carlo settings and matched seeds.  It replaces the earlier
`n_mc_samples=300` smoke test; the low-sample run was useful only for plumbing.

The validation uses the compare-harness curve manifest and prepared artifacts
read-only from:

`/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both`

The harness helper `lib.compute_at_seed()` was not used because it hardcodes
Modal.  The local driver calls `auto_fpp.compute.compute_prepared_artifact()`
with `AutoFppComputeConfig(compute_backend="local", bin_count=200)`.

## Artifacts

Generated outputs live under:

`docs/multiple_contrast_curves_validation/`

- `runs_local_multicc.jsonl`: 145 full local runs, one row per target/variant/seed.
- `selected_curves_local_multicc.csv`: predeclared curve selection by geometry.
- `summary_local_multicc.csv`: per-target/variant means over five seeds.
- `impact_local_multicc.csv`: mean deltas versus the best single-band variant.
- `seed_paired_impact_local_multicc.csv`: matched-seed deltas for every multi run.
- `ln_evidence_summary_local_multicc.csv`: per-scenario raw `lnZ` means for
  rerun rows with raw-evidence capture.  Currently this covers TOI-6067.

## Environment

Command shape:

```bash
PYTHONPATH=/Users/collier/projects/Holaxis/astro/triceratops:/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both:/Users/collier/projects/Holaxis/astro/triceratops-auto \
  /Users/collier/projects/Holaxis/astro/triceratops-auto/.venv/bin/python \
  scripts/validate_multiple_contrast_curves.py
```

Compute config:

- backend: `local`
- `bin_count=200`
- `n_mc_samples=1_000_000`
- `n_best_samples=1000`
- `n_workers=0`
- seeds: `101`, `211`, `323`, `437`, `541`

Curve selection was predeclared before FPP evaluation.  For each target and
band, the selected curve minimizes `sep_at_dmag_3`, then maximizes
`dmag_at_0.5`, then maximizes maximum depth.

## Selected Curves

| target | band | basename | sep at dmag 3 | dmag at 0.5" |
|---|---:|---|---:|---:|
| TOI-1738.01 | Vis | `TOI1738I-ef20211022-832_sensitivity.dat` | 0.068820 | 6.273821 |
| TOI-1738.01 | J | `TOI1738I-sg20210305-J_plot.tbl` | 0.865706 | 1.844874 |
| TOI-1738.01 | K | `TOI1738I-dc20210224-Brgamma_plot.tbl` | 0.131586 | 7.146643 |
| TOI-5961.01 | Vis | `TOI5961I-cc20230704-832_sensitivity.dat` | 0.066709 | 6.552230 |
| TOI-5961.01 | K | `TOI5961I-dc20240727-Kcont_plot.tbl` | 0.189737 | 6.675441 |
| TOI-1703.01 | Vis | `TOI1703I-ef20201202-562_sensitivity.dat` | 0.074278 | 5.375845 |
| TOI-1703.01 | J | `TOI1703I-sg20201230-J_plot.tbl` | 1.039675 | 1.548214 |
| TOI-1703.01 | K | `TOI1703I-dc20201105-Brgamma_plot.tbl` | 0.186547 | 6.125734 |
| TOI-6067.01 | Vis | `TOI6067I-cc20240525-562_sensitivity.dat` | 0.069990 | 5.515095 |
| TOI-6067.01 | H | `TOI6067I-dc20240727-Hcont_plot.tbl` | 0.197668 | 6.713709 |
| TOI-6067.01 | K | `TOI6067I-dc20240727-Kcont_plot.tbl` | 0.182869 | 6.729394 |

## Mean Results

### TOI-1738.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.0679776 | 0.00552843 | 0.016885 | 0.0452204 | 0.00154752 | 0.932022 |
| `single_vis` | Vis | 0.0343707 | 0.00200695 | 0.0197772 | 0.00952768 | 2.93906e-06 | 0.965629 |
| `single_j` | J | 0.0485524 | 0.00347718 | 0.017822 | 0.025424 | 0.000743247 | 0.951448 |
| `single_k` | K | 0.033494 | 0.00206223 | 0.0193179 | 0.00921015 | 2.10054e-05 | 0.966506 |
| `multi_k_vis` | K+Vis | 0.0326307 | 0.00200917 | 0.0198598 | 0.00768402 | 2.9499e-06 | 0.967369 |
| `multi_j_k` | J+K | 0.0338101 | 0.0021242 | 0.0195107 | 0.00928362 | 2.12175e-05 | 0.96619 |
| `multi_j_k_vis` | J+K+Vis | 0.0326307 | 0.00200917 | 0.0198598 | 0.00768402 | 2.9499e-06 | 0.967369 |

Interpretation: `K+Vis` improves on the best single-band result.  `J` is not
limiting once `K` and `Vis` are included.

### TOI-5961.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.0856582 | 0.00830487 | 4.47352e-07 | 0.0856517 | 6.03403e-06 | 0.914342 |
| `single_vis` | Vis | 5.84218e-07 | 1.04897e-07 | 5.61662e-07 | 2.18622e-12 | 2.2554e-08 | 0.999999 |
| `single_k` | K | 0.000298427 | 0.000205655 | 5.46402e-07 | 0.000297585 | 2.94971e-07 | 0.999702 |
| `multi_k_vis` | K+Vis | 5.84216e-07 | 1.04898e-07 | 5.61667e-07 | 2.18624e-12 | 2.25469e-08 | 0.999999 |

Interpretation: the selected `Vis` curve dominates; adding `K` is effectively
neutral at this precision.

### TOI-1703.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.128313 | 0.0184356 | 3.3188e-06 | 0.127909 | 0.000351116 | 0.871687 |
| `single_vis` | Vis | 0.0760069 | 0.0201182 | 4.07832e-06 | 0.0759313 | 9.76218e-06 | 0.923993 |
| `single_j` | J | 0.114562 | 0.0198201 | 3.43836e-06 | 0.114237 | 0.000270842 | 0.885438 |
| `single_k` | K | 0.0794148 | 0.0186227 | 4.01551e-06 | 0.0793438 | 7.46733e-06 | 0.920585 |
| `multi_k_vis` | K+Vis | 0.0742372 | 0.0190142 | 4.17324e-06 | 0.0741644 | 6.20711e-06 | 0.925763 |
| `multi_j_k` | J+K | 0.0802638 | 0.0191041 | 4.05448e-06 | 0.0801919 | 7.55472e-06 | 0.919736 |

Interpretation: `K+Vis` improves on both selected `K` and selected `Vis`.  `J`
again does not add a limiting constraint after `K+Vis`.

### TOI-6067.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.184354 | 0.0110895 | 0.00301467 | 0.143231 | 0.037071 | 0.815646 |
| `single_vis` | Vis | 0.1348 | 0.00829831 | 0.00383668 | 0.118077 | 0.0115535 | 0.8652 |
| `single_h` | H | 0.0937395 | 0.00606196 | 0.00314967 | 0.0875255 | 0.00197807 | 0.906261 |
| `single_k` | K | 0.0820877 | 0.0044755 | 0.00272845 | 0.0763424 | 0.00208329 | 0.917912 |
| `multi_k_vis` | K+Vis | 0.115122 | 0.00662524 | 0.00401074 | 0.106698 | 0.00302373 | 0.884878 |
| `multi_h_k` | H+K | 0.0931765 | 0.00602282 | 0.00315529 | 0.0869528 | 0.00197988 | 0.906823 |
| `multi_h_k_vis` | H+K+Vis | 0.113689 | 0.00654828 | 0.00403756 | 0.105742 | 0.00251124 | 0.886311 |

Interpretation: `H+K` is slightly better than `H` alone but worse than `K`
alone on total FPP. Adding `Vis` raises total FPP relative to `K`.  This is not
Monte Carlo noise; it occurs for all five matched seeds.  The raw-evidence
diagnostic below shows that the multi curve lowers false-positive scenario
evidence relative to `single_k`, but it lowers planet-channel companion
scenario evidence as well.  Therefore total FPP is not a monotonic invariant for
this validation.

## Impact Summary

| target | multi variant | best single | delta FPP | delta bound | delta background | delta FPP vs no curve |
|---|---|---|---:|---:|---:|---:|
| TOI-1738.01 | `multi_k_vis` | `single_k` | -0.000863 | -0.001526 | -0.0000181 | -0.035347 |
| TOI-5961.01 | `multi_k_vis` | `single_vis` | -2.12e-12 | 2.32e-17 | -7.09e-12 | -0.085658 |
| TOI-1703.01 | `multi_k_vis` | `single_vis` | -0.001770 | -0.001767 | -0.00000356 | -0.054076 |
| TOI-6067.01 | `multi_h_k` | `single_k` | +0.011089 | +0.010610 | -0.000103 | -0.091178 |
| TOI-6067.01 | `multi_h_k_vis` | `single_k` | +0.031601 | +0.029399 | +0.000428 | -0.070665 |
| TOI-6067.01 | `multi_k_vis` | `single_k` | +0.033035 | +0.030355 | +0.000940 | -0.069232 |

The seed-paired diagnostic found no cases where a multi-curve channel
probability exceeded the loosest component single-curve channel probability.
That check is intentionally weak.  It does not prove monotonicity relative to
the strongest component single curve; in fact, 40 of 70 multi rows exceed the
minimum component-single bound posterior and 48 of 70 exceed the minimum
component-single background posterior.  These are normalized posterior
probabilities, not raw priors.

## Raw Evidence Diagnostic

TOI-6067 was rerun after adding `ln_evidence` capture to
`runs_local_multicc.jsonl`.  The rerun reproduces the same FPP values and adds
`ln_evidence_summary_local_multicc.csv`.

Mean `lnZ` values for selected TOI-6067 scenarios:

| scenario | single K | single Vis | multi K+Vis | multi K+Vis - single K |
|---|---:|---:|---:|---:|
| TP | -72.607 | -72.607 | -72.607 | +0.000 |
| PTP | -75.050 | -74.949 | -75.161 | -0.112 |
| DTP | -73.100 | -75.735 | -75.877 | -2.776 |
| STP | -76.637 | -76.827 | -76.892 | -0.255 |
| SEB | -74.898 | -74.749 | -74.909 | -0.011 |
| SEBx2P | -76.525 | -76.432 | -76.563 | -0.037 |
| PEBx2P | -82.512 | -82.516 | -82.613 | -0.100 |
| BTP | -91.360 | -91.979 | -92.548 | -1.188 |
| BEB | -78.647 | -77.252 | -78.669 | -0.023 |
| BEBx2P | -79.175 | -77.854 | -79.179 | -0.004 |

This resolves the main TOI-6067 ambiguity at the scenario-evidence level:
`multi_k_vis` is not increasing raw false-positive evidence relative to
`single_k`.  It lowers raw FP evidence, but it also lowers DTP/PTP, which
TRICERATOPS counts as planet scenarios.  After normalization, the planet
denominator shrinks enough that total FPP rises.  The validation should
therefore use raw `lnZ`/prior diagnostics for monotonicity questions, not final
FPP alone.

## Review Findings

Round-one subagent review found:

- One-shot iterable inputs were unsafe: `_single_contrast_curve_band()` consumed
  generators before scenario workers saw them.  This is fixed by
  `canonicalize_contrast_curve_input()` and covered by unit tests.
- Engine-level generator dispatch is also covered: `_compute()` materializes a
  generator to `ContrastCurveSet` before scenario execution.
- MOLUSC companion paths still bypass contrast-curve constraints.  This is a
  pre-existing model limitation and should remain a documented caveat until
  MOLUSC carries per-draw projected separations through the prior calculation.
- `Vis`, `562nm`, and `832nm` use the existing TESS/visible flux spline
  approximation, not a separately calibrated speckle bandpass.
- SDSS `g/r/i/z` curves fail clearly if the needed target/population
  magnitudes are unavailable; partial SDSS backfill is not implemented.

## Scientific Caveats

- Multiple curves are interpreted as joint radial non-detections.  The code
  does not model position angle, azimuthal completeness, image footprint,
  epoch differences, probabilistic non-detections, or orbit-motion effects.
- Total FPP is an aggregate after scenario evidence normalization.  It can move
  differently from a single channel because TRICERATOPS counts PTP and DTP as
  planet scenarios, not false positives.
- A single-vs-multi posterior comparison does not isolate only the contrast
  prior when the single curve changes the active `filt` path.  Raw `lnZ` and
  helper-level prior checks are the safer diagnostics.
- Multi-band support is only as physical as the available flux-relation spline
  for each band.  J/H/K and SDSS bands use hardcoded legacy splines; Vis aliases
  map to the TESS/visible approximation.

## Commands

Focused tests and lint after the generator fix:

```bash
.venv/bin/pytest -q tests/unit/domain/test_value_objects.py tests/unit/validation/test_job.py
.venv/bin/pytest -q tests/unit/domain/test_value_objects.py tests/unit/validation/test_job.py tests/unit/validation/test_engine.py::TestEngineCompute::test_compute_materializes_generator_contrast_curve
.venv/bin/ruff check triceratops/domain/value_objects.py triceratops/validation/engine.py triceratops/validation/job.py tests/unit/domain/test_value_objects.py tests/unit/validation/test_job.py scripts/validate_multiple_contrast_curves.py
.venv/bin/pytest -q
```

Results:

- Focused tests: `36 passed`
- Generator-dispatch focused tests: `37 passed`
- Ruff on touched files: passed
- Full test suite: `878 passed, 14 skipped`
