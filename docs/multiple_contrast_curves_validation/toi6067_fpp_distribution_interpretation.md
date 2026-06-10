# TOI-6067: Why H-Containing Curve Sets Can Raise FPP Relative to K Alone

Date: 2026-06-10

Source artifacts:

- `docs/multiple_contrast_curves_validation/runs_local_multicc.jsonl`
- `docs/multiple_contrast_curves_validation/summary_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/ln_evidence_summary_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/ln_evidence_monotonicity_check_local_multicc.csv`

This note explains the TOI-6067 result from the refreshed Phase 1 local
multiple-contrast-curve validation matrix. The blind-curve guard changed the
previous interpretation: `K+Vis` now slightly improves total FPP relative to
`K` alone. The remaining non-monotonic normalized-FPP behavior is in the
H-containing curve sets.

| variant | FPP mean |
|---|---:|
| no curve | 0.184354 |
| single Vis | 0.134800 |
| single H | 0.093739 |
| single K | 0.082088 |
| K+Vis | 0.079454 |
| H+K | 0.093177 |
| H+K+Vis | 0.090310 |

So `K+Vis` is now the lowest-FPP TOI-6067 variant in this matrix. `H+K` and
`H+K+Vis` still have higher normalized FPP than `K` alone.

## Definitions

TRICERATOPS computes FPP as:

```text
FPP = 1 - P(TP) - P(PTP) - P(DTP)
```

where:

- `TP`: planet transiting the target star.
- `PTP`: planet transiting the target/primary, with dilution from a physically
  bound companion.
- `DTP`: planet transiting the target star, diluted by an unresolved/background
  star.

`DTP` is planet-side for published TRICERATOPS FPP because the signal is still
planetary rather than an eclipsing binary. It is not the same model as `TP`: it
requires extra third-light flux from an unresolved/background source.

## Probability Distribution

Mean normalized scenario probabilities over five matched 1M-draw seeds:

| scenario/group | no curve | single K | single H | single Vis | K+Vis | H+K+Vis |
|---|---:|---:|---:|---:|---:|---:|
| TP | 0.596519 | 0.540139 | 0.623776 | 0.758455 | 0.544621 | 0.630726 |
| PTP | 0.120458 | 0.047061 | 0.053760 | 0.073289 | 0.042518 | 0.047741 |
| DTP | 0.098668 | 0.330713 | 0.228725 | 0.033456 | 0.333407 | 0.231223 |
| planet total | 0.815646 | 0.917912 | 0.906261 | 0.865200 | 0.920546 | 0.909690 |
| bound FP total | 0.143231 | 0.076342 | 0.087525 | 0.118077 | 0.073694 | 0.084052 |
| background FP total | 0.037071 | 0.002083 | 0.001978 | 0.011553 | 0.002067 | 0.001974 |
| nearby total | 0.003015 | 0.002728 | 0.003150 | 0.003837 | 0.002751 | 0.003185 |
| FPP | 0.184354 | 0.082088 | 0.093739 | 0.134800 | 0.079454 | 0.090310 |

The important comparison is `single K` to `H+K+Vis`:

| quantity | H+K+Vis - single K |
|---|---:|
| TP | +0.090588 |
| PTP | +0.000680 |
| DTP | -0.099490 |
| planet total | -0.008223 |
| FPP | +0.008223 |

Some probability does move into `TP`; normalized `TP` rises from `0.540139` to
`0.630726`. But the reduction in `DTP` is larger than the gain in `TP + PTP`,
so the total planet-side bucket shrinks:

```text
single K planet total    = TP + PTP + DTP = 0.917912
H+K+Vis planet total     = TP + PTP + DTP = 0.909690
difference               = -0.008223
```

Since:

```text
FPP = 1 - planet total
```

FPP rises by the same amount:

```text
0.090310 - 0.082088 = +0.008223
```

## Why It Does Not Simply Redistribute to TP

The model does not conserve probability mass within the planet bucket. Even
though DTP and TP are both target-planet interpretations, TRICERATOPS does not
apply a rule like:

```text
if DTP dilution is ruled out, add that probability to TP
```

Instead, every scenario has an evidence value `Z`, and final probabilities are
normalized across all scenarios:

```text
P(scenario_i) = Z_i / sum(Z_all_scenarios)
```

When a contrast curve suppresses `DTP`, the raw `DTP` evidence is lowered. The
raw `TP` evidence is not recomputed upward to absorb the excluded DTP parameter
volume; `TP` is a separate model with its own likelihood integral and prior
volume. The only transfer is indirect normalization across all scenarios. Some
relative probability can land in `TP`, some can land in `PTP`, and some can
land in false-positive scenarios. There is no mechanism that transfers all
disallowed DTP dilution probability specifically to `TP`.

For TOI-6067, adding H to K increases normalized `TP`, but it also reduces
normalized `DTP`. The increase in `TP + PTP` is not enough to offset the DTP
loss.

## Raw Evidence Check

The raw `lnZ` diagnostics show that the H-containing multi-curve run is not
raising raw false-positive evidence relative to `single K` outside the
documented all-blind OWA envelope. For selected scenarios:

| scenario | single K lnZ | H+K+Vis lnZ | H+K+Vis - single K |
|---|---:|---:|---:|
| DTP | -73.100 | -73.621 | -0.521 |
| PTP | -75.050 | -75.192 | -0.142 |
| STP | -76.637 | -76.899 | -0.262 |
| SEB | -74.898 | -74.927 | -0.029 |
| SEBx2P | -76.525 | -76.576 | -0.050 |
| PEBx2P | -82.512 | -82.623 | -0.110 |
| BTP | -91.360 | -92.630 | -1.270 |
| BEB | -78.647 | -78.868 | -0.221 |
| BEBx2P | -79.175 | -79.361 | -0.186 |
| DEB | -97.146 | -97.967 | -0.821 |
| DEBx2P | -82.127 | -82.877 | -0.750 |

The multi-curve run lowers raw evidence for DTP and for the listed
false-positive scenarios. The final FPP still rises because final FPP is a
normalized probability ratio, and DTP is counted as planet-side probability.

## Why K+Vis Changed After The Guard

Before the blind-curve guard, a curve that was too shallow to see a draw still
contributed its OWA through the canonical interpolation clamp. In a multi-curve
minimum, that could make a blind curve artificially restrictive. The refreshed
Phase 1 logic makes a blind curve contribute `inf` instead; only curves that can
actually see a draw participate in the minimum. If all curves are blind, the set
falls back to the largest component OWA.

That fix changes the TOI-6067 K+Vis behavior:

| quantity | K+Vis - single K |
|---|---:|
| TP | +0.004482 |
| PTP | -0.004543 |
| DTP | +0.002694 |
| planet total | +0.002633 |
| FPP | -0.002633 |

K+Vis now slightly increases the total planet-side bucket and lowers FPP. This
is the expected direction after removing the blind-curve overconstraint.

## Takeaway

The remaining TOI-6067 caveat is not that multi-curve support ignores a curve or
that probability should mechanically move from DTP into TP. It is that published
TRICERATOPS FPP is a normalized aggregate over several planet-side and
false-positive submodels.

For TOI-6067, H-containing curve sets reduce DTP and FP raw evidence relative
to K, while increasing TP. The DTP reduction is still large enough that the
total planet-side normalized probability drops, so reported FPP rises relative
to `single_k`. K+Vis no longer shows that issue in the refreshed Phase 1 matrix.
