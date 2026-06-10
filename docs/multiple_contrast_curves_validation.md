# Multiple Contrast Curves: Local Validation Report

Date: 2026-06-10

Branch: `codex/multiple-contrast-curves`

## Scope

This report validates the Phase 1 multiple-contrast-curve implementation after
the blind-curve guard:

- A curve that is blind to a draw contributes no constraint to the multi-curve
  min-combine.
- If all curves are blind for a draw, the multi-curve path falls back to the
  largest component outer working angle.
- Single-curve behavior remains canonical/TRICERATOPS-compatible, including
  the inherited faint-end OWA clamp.
- Band aliases now include `Jcont`, `Hcont`, `Brgamma`, and `LP600`; existing
  `562nm` and `832nm` labels still route to the visible/TESS approximation.

The validation uses prepared artifacts and contrast-curve manifests read-only
from:

`/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both`

The local driver calls `auto_fpp.compute.compute_prepared_artifact()` with
`AutoFppComputeConfig(compute_backend="local", bin_count=200)`.

## Artifacts

Generated outputs live under:

`docs/multiple_contrast_curves_validation/`

- `runs_local_multicc.jsonl`: 145 local runs, one row per target/variant/seed;
  component OWAs are serialized with each contrast-curve row, and non-finite
  `lnZ` values use strict-JSON-safe tags.
- `selected_curves_local_multicc.csv`: predeclared curve selection by geometry.
- `summary_local_multicc.csv`: per-target/variant means over five seeds.
- `impact_local_multicc.csv`: mean deltas versus the best single-band variant.
- `seed_paired_impact_local_multicc.csv`: matched-seed deltas for every multi run.
- `ln_evidence_summary_local_multicc.csv`: per-scenario raw `lnZ` means for all
  four validation targets.
- `ln_evidence_monotonicity_check_local_multicc.csv`: raw `lnZ` monotonicity
  diagnostic with a strict check and the analytically allowed all-blind OWA
  envelope.
- `toi6067_fpp_distribution_interpretation.md`: detailed explanation of the
  remaining TOI-6067 normalized-FPP behavior.

## Environment

Command:

```bash
/Users/collier/projects/Holaxis/astro/triceratops-auto/.venv/bin/python \
  scripts/validate_multiple_contrast_curves.py --force
```

Compute config:

- backend: `local`
- `bin_count=200`
- `n_mc_samples=1_000_000`
- `n_best_samples=1000`
- `n_workers=0`
- seeds: `101`, `211`, `323`, `437`, `541`

Curve selection was predeclared before FPP evaluation. For each target and
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
| `multi_k_vis` | K+Vis | 0.032194 | 0.0019282 | 0.019578 | 0.00760159 | 2.90769e-06 | 0.967806 |
| `multi_j_k` | J+K | 0.0338101 | 0.0021242 | 0.0195107 | 0.00928362 | 2.12175e-05 | 0.96619 |
| `multi_j_k_vis` | J+K+Vis | 0.0324989 | 0.00198626 | 0.0197746 | 0.00765924 | 2.93717e-06 | 0.967501 |
| `multi_all_selected_bands` | Vis+J+K | 0.0324989 | 0.00198626 | 0.0197746 | 0.00765924 | 2.93717e-06 | 0.967501 |

Interpretation: `K+Vis` improves on the best single-band result. `J` is not
limiting once `K` and `Vis` are included.

### TOI-5961.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.0856582 | 0.00830487 | 4.47352e-07 | 0.0856517 | 6.03403e-06 | 0.914342 |
| `single_vis` | Vis | 5.84218e-07 | 1.04897e-07 | 5.61662e-07 | 2.18622e-12 | 2.2554e-08 | 0.999999 |
| `single_k` | K | 0.000298427 | 0.000205655 | 5.46402e-07 | 0.000297585 | 2.94971e-07 | 0.999702 |
| `multi_k_vis` | K+Vis | 5.72145e-07 | 1.01993e-07 | 5.50053e-07 | 2.13643e-12 | 2.20893e-08 | 0.999999 |
| `multi_all_selected_bands` | Vis+K | 5.72145e-07 | 1.01993e-07 | 5.50053e-07 | 2.13643e-12 | 2.20893e-08 | 0.999999 |

Interpretation: the selected `Vis` curve dominates; adding `K` is effectively
neutral at this precision.

### TOI-1703.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.128313 | 0.0184356 | 3.3188e-06 | 0.127909 | 0.000351116 | 0.871687 |
| `single_vis` | Vis | 0.0760069 | 0.0201182 | 4.07832e-06 | 0.0759313 | 9.76218e-06 | 0.923993 |
| `single_j` | J | 0.114562 | 0.0198201 | 3.43836e-06 | 0.114237 | 0.000270842 | 0.885438 |
| `single_k` | K | 0.0794148 | 0.0186227 | 4.01551e-06 | 0.0793438 | 7.46733e-06 | 0.920585 |
| `multi_k_vis` | K+Vis | 0.0730492 | 0.0182992 | 4.11237e-06 | 0.0729779 | 6.09542e-06 | 0.926951 |
| `multi_j_k` | J+K | 0.0802638 | 0.0191041 | 4.05448e-06 | 0.0801919 | 7.55472e-06 | 0.919736 |
| `multi_j_k_vis` | J+K+Vis | 0.0738519 | 0.0187806 | 4.15335e-06 | 0.0737797 | 6.1707e-06 | 0.926148 |
| `multi_all_selected_bands` | Vis+J+K | 0.0738519 | 0.0187806 | 4.15335e-06 | 0.0737797 | 6.1707e-06 | 0.926148 |

Interpretation: `K+Vis` improves on both selected `K` and selected `Vis`.
`J` again does not add a limiting constraint after `K+Vis`.

### TOI-6067.01

| variant | bands | FPP mean | FPP std | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_curve` | none | 0.184354 | 0.0110895 | 0.00301467 | 0.143231 | 0.037071 | 0.815646 |
| `single_vis` | Vis | 0.1348 | 0.00829831 | 0.00383668 | 0.118077 | 0.0115535 | 0.8652 |
| `single_h` | H | 0.0937395 | 0.00606196 | 0.00314967 | 0.0875255 | 0.00197807 | 0.906261 |
| `single_k` | K | 0.0820877 | 0.0044755 | 0.00272845 | 0.0763424 | 0.00208329 | 0.917912 |
| `multi_k_vis` | K+Vis | 0.0794544 | 0.00425059 | 0.00275104 | 0.0736941 | 0.00206712 | 0.920546 |
| `multi_h_k` | H+K | 0.0931765 | 0.00602282 | 0.00315529 | 0.0869528 | 0.00197988 | 0.906823 |
| `multi_h_k_vis` | H+K+Vis | 0.0903103 | 0.00579781 | 0.00318467 | 0.0840515 | 0.00197433 | 0.90969 |
| `multi_all_selected_bands` | Vis+H+K | 0.0903103 | 0.00579781 | 0.00318467 | 0.0840515 | 0.00197433 | 0.90969 |

Interpretation: after the blind-curve guard, `K+Vis` is now better than `K`
alone on total FPP. The remaining non-monotonic normalized-FPP behavior is in
the H-containing combinations: `H+K` and `H+K+Vis` lower background evidence
relative to `K`, but they also reduce the planet-side total enough that
normalized FPP is higher than `single_k`.

## Impact Summary

| target | multi variant | best single | delta FPP | delta bound | delta background | delta FPP vs no curve |
|---|---|---|---:|---:|---:|---:|
| TOI-1703.01 | `multi_all_selected_bands` | `single_vis` | -0.00215492 | -0.00215165 | -3.59148e-06 | -0.0544612 |
| TOI-1703.01 | `multi_j_k` | `single_vis` | +0.00425697 | +0.00426056 | -2.20746e-06 | -0.0480494 |
| TOI-1703.01 | `multi_j_k_vis` | `single_vis` | -0.00215492 | -0.00215165 | -3.59148e-06 | -0.0544612 |
| TOI-1703.01 | `multi_k_vis` | `single_vis` | -0.00295768 | -0.00295338 | -3.66676e-06 | -0.055264 |
| TOI-1738.01 | `multi_all_selected_bands` | `single_k` | -0.000995045 | -0.00155091 | -1.80682e-05 | -0.0354787 |
| TOI-1738.01 | `multi_j_k` | `single_k` | +0.000316131 | +7.34703e-05 | +2.12106e-07 | -0.0341675 |
| TOI-1738.01 | `multi_j_k_vis` | `single_k` | -0.000995045 | -0.00155091 | -1.80682e-05 | -0.0354787 |
| TOI-1738.01 | `multi_k_vis` | `single_k` | -0.00129994 | -0.00160856 | -1.80977e-05 | -0.0357836 |
| TOI-5961.01 | `multi_all_selected_bands` | `single_vis` | -1.20735e-08 | -4.9786e-14 | -4.64693e-10 | -0.0856576 |
| TOI-5961.01 | `multi_k_vis` | `single_vis` | -1.20735e-08 | -4.9786e-14 | -4.64693e-10 | -0.0856576 |
| TOI-6067.01 | `multi_all_selected_bands` | `single_k` | +0.00822265 | +0.00770911 | -0.000108957 | -0.0940439 |
| TOI-6067.01 | `multi_h_k` | `single_k` | +0.0110888 | +0.0106104 | -0.000103409 | -0.0911778 |
| TOI-6067.01 | `multi_h_k_vis` | `single_k` | +0.00822265 | +0.00770911 | -0.000108957 | -0.0940439 |
| TOI-6067.01 | `multi_k_vis` | `single_k` | -0.00263331 | -0.0026483 | -1.61693e-05 | -0.1049 |

## Raw Evidence Diagnostic

Raw `lnZ` capture now covers all four validation targets.

The strict diagnostic checks whether each multi-curve FP scenario has
`lnZ(multi) <= min(lnZ(component singles))`. That strict check is useful, but
it is too strong for the agreed Phase 1 semantics because single-curve paths
retain the canonical faint-end OWA clamp while multi-curve all-blind draws fall
back to the largest component OWA. Therefore the validation also computes the
maximum possible all-blind margin:

```text
2 * ln(max(component OWA) / min(component OWA))
```

Results:

- Diagnostic rows: `1050`
- Strict monotonic failures: `87`
- Strict failures by scenario: `DEBx2P` = `53`, `DEB` = `34`
- Bounded failures after the all-blind OWA envelope: `0`
- Largest margin: `4.257480653976728`
- Largest margin minus bound: `1.95e-14`

The largest strict margin is exactly the TOI-5961 Vis/K all-blind area ratio:

```text
2 * ln(9.833 / 1.17) = 4.2574806539767085
```

So the strict violations are consistent with the intentional all-blind fallback,
not unexplained evidence growth. All raw FP evidence margins stay within the
analytic all-blind envelope.

Selected TOI-6067 raw `lnZ` means:

| scenario | single K | single H | single Vis | K+Vis | H+K+Vis | K+Vis - K | H+K+Vis - K |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP | -72.607 | -72.607 | -72.607 | -72.607 | -72.607 | +0.000 | +0.000 |
| PTP | -75.050 | -75.060 | -74.949 | -75.161 | -75.192 | -0.112 | -0.142 |
| DTP | -73.100 | -73.621 | -75.735 | -73.100 | -73.621 | -0.000 | -0.521 |
| STP | -76.637 | -76.609 | -76.827 | -76.892 | -76.899 | -0.255 | -0.262 |
| SEB | -74.898 | -74.912 | -74.749 | -74.909 | -74.927 | -0.011 | -0.029 |
| SEBx2P | -76.525 | -76.532 | -76.432 | -76.563 | -76.576 | -0.037 | -0.050 |
| PEB | -95.980 | -95.967 | -95.671 | -96.045 | -96.053 | -0.065 | -0.073 |
| PEBx2P | -82.512 | -82.500 | -82.516 | -82.613 | -82.623 | -0.100 | -0.110 |
| BTP | -91.360 | -91.559 | -91.979 | -92.548 | -92.630 | -1.188 | -1.270 |
| BEB | -78.647 | -78.849 | -77.252 | -78.669 | -78.868 | -0.023 | -0.221 |
| BEBx2P | -79.175 | -79.357 | -77.854 | -79.179 | -79.361 | -0.004 | -0.186 |
| DEB | -97.146 | -97.490 | -99.076 | -97.673 | -97.967 | -0.527 | -0.821 |
| DEBx2P | -82.127 | -82.877 | -84.247 | -82.127 | -82.877 | -0.000 | -0.750 |

## Caveats

- Multiple curves are interpreted as joint radial non-detections. The code does
  not model position angle, azimuthal completeness, image footprint, epoch
  differences, probabilistic non-detections, or orbit-motion effects.
- Total FPP is an aggregate after scenario evidence normalization. It can move
  differently from a single channel because TRICERATOPS counts PTP and DTP as
  planet scenarios, not false positives.
- The Phase 1 PR intentionally does not change canonical single-curve clamp
  behavior. The all-blind multi-curve fallback is the only clamp-related change
  in this PR.
- Multi-band support is only as physical as the available flux-relation spline
  for each band. J/H/K and SDSS bands use hardcoded legacy splines; Vis aliases
  map to the TESS/visible approximation.

## Commands

Validation:

```bash
/Users/collier/projects/Holaxis/astro/triceratops-auto/.venv/bin/python \
  scripts/validate_multiple_contrast_curves.py --force
/Users/collier/projects/Holaxis/astro/triceratops-auto/.venv/bin/python \
  scripts/validate_multiple_contrast_curves.py --summary-only
```

Code checks:

```bash
python -m pytest tests/unit/scenarios/test_background_helpers.py \
  tests/unit/scenarios/test_btp_beb.py \
  tests/unit/scenarios/test_ptp_peb.py \
  tests/unit/stellar/test_relations.py \
  tests/unit/priors/test_lnpriors.py -q
python -m pytest -q
python -m ruff check scripts/validate_multiple_contrast_curves.py \
  triceratops/priors/lnpriors.py \
  triceratops/scenarios/_background_helpers.py \
  triceratops/scenarios/_companion_helpers.py \
  triceratops/stellar/relations.py \
  tests/unit/scenarios/test_background_helpers.py \
  tests/unit/scenarios/test_btp_beb.py \
  tests/unit/scenarios/test_ptp_peb.py \
  tests/unit/stellar/test_relations.py
git diff --check
```
