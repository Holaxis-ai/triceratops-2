# TOI-6067: Why K+Vis Can Raise FPP Relative to K Alone

Date: 2026-06-09

Source artifacts:

- `docs/multiple_contrast_curves_validation/runs_local_multicc.jsonl`
- `docs/multiple_contrast_curves_validation/summary_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/ln_evidence_summary_local_multicc.csv`

This note explains the TOI-6067 result from the full local multiple-contrast
curve validation matrix.  The confusing result is:

| variant | FPP mean |
|---|---:|
| no curve | 0.184354 |
| single Vis | 0.134800 |
| single K | 0.082088 |
| multi K+Vis | 0.115122 |

So `Vis` alone improves FPP relative to no curve, but adding `Vis` to `K`
raises FPP relative to `K` alone.

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
planetary rather than an eclipsing binary.  It is also still a target-star
planet interpretation, but it is not the same model as `TP`: it requires extra
third-light flux from an unresolved/background source.

## Probability Distribution

Mean normalized scenario probabilities over five matched 1M-draw seeds:

| scenario/group | no curve | single K | single Vis | K+Vis |
|---|---:|---:|---:|---:|
| TP | 0.596519 | 0.540139 | 0.758455 | 0.792660 |
| PTP | 0.120458 | 0.047061 | 0.073289 | 0.061803 |
| DTP | 0.098668 | 0.330713 | 0.033456 | 0.030415 |
| planet total | 0.815646 | 0.917912 | 0.865200 | 0.884878 |
| bound FP total | 0.143231 | 0.076342 | 0.118077 | 0.106698 |
| background FP total | 0.037071 | 0.002083 | 0.011553 | 0.003024 |
| nearby total | 0.003015 | 0.002728 | 0.003837 | 0.004011 |
| FPP | 0.184354 | 0.082088 | 0.134800 | 0.115122 |

The important comparison is `single K` to `K+Vis`:

| quantity | K+Vis - single K |
|---|---:|
| TP | +0.252522 |
| PTP | +0.014742 |
| DTP | -0.300298 |
| planet total | -0.033035 |
| FPP | +0.033035 |

So some probability does move into `TP` after adding `Vis`; normalized `TP`
rises from `0.540139` to `0.792660`.  But the lost `DTP` probability is larger
than the gain in `TP` and `PTP`, so the total planet-side bucket shrinks:

```text
single K planet total    = TP + PTP + DTP = 0.917912
K+Vis planet total       = TP + PTP + DTP = 0.884878
difference               = -0.033035
```

Since:

```text
FPP = 1 - planet total
```

FPP rises by the same amount:

```text
0.115122 - 0.082088 = +0.033035
```

## Why It Does Not Simply Redistribute to TP

The model does not conserve probability mass within the planet bucket.  Even
though DTP and TP are both target-planet interpretations, TRICERATOPS does not
apply a rule like:

```text
if DTP dilution is ruled out, add that probability to TP
```

Instead, every scenario has an evidence value `Z`, and the final probabilities
are normalized across all scenarios:

```text
P(scenario_i) = Z_i / sum(Z_all_scenarios)
```

When a contrast curve suppresses `DTP`, the raw `DTP` evidence is lowered.  The
raw `TP` evidence is not recomputed upward to absorb the excluded DTP parameter
volume; `TP` is a separate model with its own likelihood integral and prior
volume.  The only transfer is indirect normalization across all scenarios.  Some
relative probability can land in `TP`, some can land in `PTP`, and some can land
in false-positive scenarios.  There is no mechanism that transfers all
disallowed DTP dilution probability specifically to `TP`.

For TOI-6067, adding `Vis` to `K` does increase normalized `TP`, but it also
sharply reduces normalized `DTP`.  The increase in `TP + PTP` is not enough to
offset the DTP loss.

## Raw Evidence Check

The raw `lnZ` diagnostics show that `K+Vis` is not increasing raw
false-positive evidence relative to `single K`.  For selected scenarios:

| scenario | single K lnZ | K+Vis lnZ | K+Vis - single K |
|---|---:|---:|---:|
| DTP | -73.100 | -75.877 | -2.776 |
| PTP | -75.050 | -75.161 | -0.112 |
| STP | -76.637 | -76.892 | -0.255 |
| SEB | -74.898 | -74.909 | -0.011 |
| BTP | -91.360 | -92.548 | -1.188 |
| BEB | -78.647 | -78.669 | -0.023 |
| BEBx2P | -79.175 | -79.179 | -0.004 |

The multi-curve run lowers raw evidence for DTP and for the listed
false-positive scenarios.  The final FPP still rises because final FPP is a
normalized probability ratio, and DTP is counted as planet-side probability.

## Why K Behaves Differently From Vis

For TOI-6067, `K` does not constrain DTP the way `Vis` does:

| DTP lnZ | value |
|---|---:|
| no curve | -74.412 |
| single K | -73.100 |
| single Vis | -75.735 |
| K+Vis | -75.877 |

The selected `K` curve is wide and deep, with an outer working angle near
`9.9"`; the selected `Vis` curve has a much smaller footprint, around `1.17"`.
In TRICERATOPS' DTP/background prior, no curve uses a fixed default radius, and
a supplied contrast curve maps each draw's delta magnitude to an allowed
separation.  A wide AO curve can leave more DTP parameter space viable than the
default no-curve radius for relevant draws, while the tighter Vis/speckle curve
strongly suppresses DTP.

That is why `single K` reports the lowest FPP: it constrains many false-positive
channels while leaving a large target-planet-with-background-dilution (`DTP`)
channel alive.  Adding `Vis` suppresses that DTP channel.  Scientifically, this
argues against the need for a background-dilution explanation, and the
normalized `TP` probability does rise.  But under the published TRICERATOPS FPP
convention it can still increase FPP because the suppressed DTP evidence had
been counted as planet evidence.

## Takeaway

The TOI-6067 result is not evidence that the K curve is ignored.  It is evidence
that final FPP mixes several planet-side submodels rather than forcing ruled-out
third-light/dilution probability back into the undiluted TP model.

For this target, `Vis` helps rule down background-diluted target-planet
interpretations (`DTP`).  But because TRICERATOPS counts DTP as planet-side
probability, suppressing DTP can raise the reported FPP even while the
normalized undiluted `TP` probability rises.

The practical lesson is that published FPP alone is not a pure measure of
support for undiluted `TP`.  For TOI-6067, the probability distribution matters:
`K+Vis` raises `TP` from `0.540139` to `0.792660`, while reducing `DTP` from
`0.330713` to `0.030415`.
