# Multiple Contrast Curves PR Review Guidance

Date: 2026-06-10

PR: <https://github.com/Holaxis-ai/triceratops-2/pull/2>

Branch: `codex/multiple-contrast-curves`

## Review Goal

This PR adds support for passing more than one contrast curve into the
refactored TRICERATOPS validation path. Review should answer two questions:

1. Is the implementation a correct extension of the current TRICERATOPS model?
2. Are the scientific limits clear enough that users will not overinterpret the
   result as a full imaging-completeness model?

The PR should not be reviewed as a complete rewrite of contrast-curve science.
It intentionally preserves the canonical/refactored TRICERATOPS assumptions:
radial contrast limits, no position-angle dependence, no image-footprint model,
no probabilistic completeness surface, and no epoch/orbit-motion modeling.

## Suggested Review Order

1. Read the public behavior and science summary:
   `docs/multiple_contrast_curves_validation.md`.
2. Read the TOI-6067 interpretation note:
   `docs/multiple_contrast_curves_validation/toi6067_fpp_distribution_interpretation.md`.
3. Review the input model and validation boundary:
   `triceratops/domain/value_objects.py`,
   `triceratops/validation/job.py`, and
   `triceratops/validation/engine.py`.
4. Review prior combination logic:
   `triceratops/priors/lnpriors.py`,
   `triceratops/scenarios/_companion_helpers.py`, and
   `triceratops/scenarios/_background_helpers.py`.
5. Review scenario call sites:
   `triceratops/scenarios/companion_scenarios.py` and
   `triceratops/scenarios/background_scenarios.py`.
6. Review validation tooling and artifacts:
   `scripts/validate_multiple_contrast_curves.py` and
   `docs/multiple_contrast_curves_validation/`.

## Implementation Checklist

- Single-curve behavior remains backward compatible.
- `ContrastCurveSet` accepts multiple `ContrastCurve` objects and rejects empty
  or invalid members.
- Generic iterables and one-shot generators are canonicalized before scenario
  dispatch, so they are not consumed by band detection and are safe for prepared
  inputs.
- Exactly one active curve still exposes its band through `filt`; multiple
  curves intentionally use per-curve bands and leave `filt=None`.
- Companion paths normalize `filt=None` to the existing TESS baseline when a
  single scalar filter is still needed.
- Multiple curves combine by taking the tightest allowed separation per Monte
  Carlo draw.
- Background/D-scenario paths evaluate each curve in that curve's band before
  combining radial constraints.
- MOLUSC-backed companion paths still bypass TRICERATOPS contrast-curve priors,
  matching original TRICERATOPS behavior; this PR now emits a validation warning
  when MOLUSC data and contrast curves are both supplied.

## Science Checklist

- Do not require total FPP to monotonically decrease relative to the best single
  curve. FPP is a normalized aggregate over scenario evidences.
- Check raw `lnZ` diagnostics when a final FPP result looks counterintuitive.
- Remember that `TP`, `PTP`, and `DTP` all count as planet-side probability in
  published TRICERATOPS FPP.
- `DTP` means a target-star transiting planet diluted by an
  unresolved/background star. It is not a planet transiting the background star.
- Suppressing `DTP` does not force the lost probability into `TP`; the scenarios
  are separate evidence integrals.
- `Vis`, `562nm`, and `832nm` still use the existing TESS/visible approximation,
  not independently calibrated speckle bandpasses.
- This PR models joint radial non-detections only. It does not model position
  angle, anisotropic completeness, field of view, image footprint, observing
  epoch, or non-detections as probability surfaces.

## TOI-6067 Review Notes

TOI-6067 is the main non-monotonic case and should be treated as a review focus.

Mean probabilities over five matched 1M-draw seeds:

| variant | FPP | TP | PTP | DTP | planet total |
|---|---:|---:|---:|---:|---:|
| no curve | 0.184354 | 0.596519 | 0.120458 | 0.098668 | 0.815646 |
| single K | 0.082088 | 0.540139 | 0.047061 | 0.330713 | 0.917912 |
| single Vis | 0.134800 | 0.758455 | 0.073289 | 0.033456 | 0.865200 |
| K+Vis | 0.115122 | 0.792660 | 0.061803 | 0.030415 | 0.884878 |

`K+Vis` raises normalized FPP relative to `single K` because DTP drops by more
than TP/PTP rise. The raw evidence check shows that `K+Vis` lowers raw
false-positive evidence relative to `single K`; the FPP increase is a
normalization effect after a planet-side channel is suppressed.

Reviewers should not treat this as evidence that the K curve is ignored. The
more precise conclusion is that multiple curves reshape the scenario
distribution, and published FPP alone is not a pure measure of support for
undiluted TP.

## Validation Artifacts

The local validation matrix used:

- 4 multi-band targets: TOI-1738.01, TOI-5961.01, TOI-1703.01, TOI-6067.01.
- 5 matched seeds: `101`, `211`, `323`, `437`, `541`.
- 1,000,000 Monte Carlo draws per run.
- 145 local runs.

Primary artifacts:

- `docs/multiple_contrast_curves_validation/runs_local_multicc.jsonl`
- `docs/multiple_contrast_curves_validation/summary_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/impact_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/seed_paired_impact_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/ln_evidence_summary_local_multicc.csv`
- `docs/multiple_contrast_curves_validation/selected_curves_local_multicc.csv`

The full validation run is expensive. Reviewers should usually inspect the
artifacts and rerun focused tests first. Re-run the full validation matrix only
if implementation changes alter the prior math or scenario dispatch.

## Suggested Commands

Focused tests:

```bash
python -m pytest tests/unit/domain/test_value_objects.py \
  tests/unit/validation/test_job.py \
  tests/unit/validation/test_engine.py \
  tests/unit/scenarios/test_background_helpers.py \
  tests/unit/scenarios/test_ptp_peb.py \
  tests/unit/scenarios/test_btp_beb.py -q
```

Latest MOLUSC warning regression:

```bash
python -m pytest tests/unit/validation/test_engine.py -q
```

Lint and whitespace checks:

```bash
python -m ruff check triceratops/validation/engine.py \
  triceratops/validation/job.py \
  triceratops/domain/value_objects.py \
  triceratops/priors/lnpriors.py \
  triceratops/scenarios/_companion_helpers.py \
  triceratops/scenarios/_background_helpers.py \
  triceratops/scenarios/background_scenarios.py \
  triceratops/scenarios/companion_scenarios.py \
  scripts/validate_multiple_contrast_curves.py \
  tests/unit/domain/test_value_objects.py \
  tests/unit/validation/test_job.py \
  tests/unit/validation/test_engine.py \
  tests/unit/scenarios/test_background_helpers.py \
  tests/unit/scenarios/test_ptp_peb.py \
  tests/unit/scenarios/test_btp_beb.py

git diff --check
```

Optional full suite:

```bash
python -m pytest -q
```

## Questions For Reviewers

- Is the per-draw minimum allowed separation the correct implementation of
  joint radial non-detections under the current TRICERATOPS model?
- Are the band mappings and fallback behavior explicit enough, especially for
  `Vis`/speckle aliases and unsupported bands?
- Is the MOLUSC warning enough, or should future work add a richer MOLUSC data
  model with projected separations and per-band brightness?
- Does the TOI-6067 documentation make clear why FPP can rise even when raw FP
  evidence falls?
- Are the validation artifacts sufficient for review without rerunning the
  full 145-run matrix?

## Approval Criteria

This PR is ready for approval if:

- Backward-compatible single-curve behavior is preserved.
- Multiple-curve inputs are canonicalized safely at public/prepared boundaries.
- Prior combination is correct for radial non-detections.
- The test suite covers generator safety, single-vs-set equivalence, multi-curve
  combination, and MOLUSC warning behavior.
- The documentation clearly states the model limits and the non-monotonic FPP
  interpretation.
