# Multiple Contrast Curves: Local Validation Notes

Date: 2026-06-09

Branch: `codex/multiple-contrast-curves`

Implementation commit: `5380d97 Support multiple contrast curves`

## Scope

These checks validate that TRICERATOPS can accept multiple contrast curves in
one compute run and apply them locally, without Modal. They are implementation
and plumbing checks, not publication-quality FPP estimates. The Monte Carlo
sample count was intentionally small to keep local validation fast.

The compare-harness data came from:

`/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both`

The harness data-flow note used for setup was:

`/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both/working_docs/fpp_data_flow.md`

## Environment

The local run used the auto-fpp virtualenv because it has the auto-fpp loader
dependencies, while `PYTHONPATH` put this TRICERATOPS checkout first:

```bash
PYTHONPATH=/Users/collier/projects/Holaxis/astro/triceratops:/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both:/Users/collier/projects/Holaxis/astro/triceratops-auto \
  /Users/collier/projects/Holaxis/astro/triceratops-auto/.venv/bin/python
```

Compute config:

- `compute_backend="local"`
- `bin_count=200`
- `n_mc_samples=300`
- `n_best_samples=30`
- `n_workers=0`
- seeds: `123`, `456`, `789`

## Unit Validation

Commands:

```bash
.venv/bin/ruff check triceratops/__init__.py triceratops/domain/__init__.py triceratops/domain/value_objects.py triceratops/priors/lnpriors.py triceratops/scenarios/_companion_helpers.py triceratops/scenarios/_background_helpers.py triceratops/scenarios/background_scenarios.py triceratops/validation/engine.py triceratops/validation/job.py triceratops/validation/workspace.py triceratops/assembly/inputs.py triceratops/assembly/orchestrator.py tests/unit/domain/test_value_objects.py tests/unit/scenarios/test_ptp_peb.py tests/unit/scenarios/test_background_helpers.py tests/unit/scenarios/test_btp_beb.py tests/unit/validation/test_job.py
.venv/bin/pytest -q
```

Results:

- Ruff on touched files: passed
- Full test suite: `872 passed, 14 skipped`

## Local TOI Comparisons

### TOI-1738.01

Curves:

- `single_k`: `TOI1738I-dc20201205-Brgamma_plot.tbl` (`K`)
- `single_vis`: `TOI1738I-ef20211022-562_sensitivity.dat` (`Vis`)
- `multi_k_vis`: both curves above

Mean over matched seeds:

| variant | FPP mean | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|
| `single_k` | 0.100126362951 | 0.011832998574 | 0.088293142821 | 2.215557488e-7 | 0.899873637049 |
| `single_vis` | 0.091761597104 | 0.012093004016 | 0.079666728497 | 1.864591053e-6 | 0.908238402896 |
| `multi_k_vis` | 0.089261689561 | 0.012127712424 | 0.077133755351 | 2.217856724e-7 | 0.910738310439 |

Observation: the multi-curve run completed locally and produced a combined
constraint that was at least as restrictive as the weaker single-curve variant
for this low-sample run. The background channel stayed close to the K run
because the K curve was more restrictive for the relevant background draws,
while the bound channel benefited from the per-band intersection.

### TOI-5961.01

Curves:

- `single_k`: `TOI5961I-dc20240727-Kcont_plot.tbl` (`K`)
- `single_vis`: `TOI5961I-cc20230704-562_sensitivity.dat` (`Vis`)
- `multi_k_vis`: both curves above

Mean over matched seeds:

| variant | FPP mean | NFPP mean | bound mean | background mean | planet mean |
|---|---:|---:|---:|---:|---:|
| `single_k` | 0.020699542315 | 0.020699542315 | 1.974513996e-47 | 4.856674015e-23 | 0.979300457685 |
| `single_vis` | 0.040916376869 | 0.040916376869 | 1.867901286e-79 | 2.644523641e-22 | 0.959083623131 |
| `multi_k_vis` | 0.040916376869 | 0.040916376869 | 1.639464341e-79 | 4.894246198e-23 | 0.959083623131 |

Observation: the multi-curve result matched the Vis-dominated single-curve FPP
at this sample size, while the background mean moved toward the tighter K
constraint. This is consistent with the implemented rule: each Monte Carlo draw
uses the smallest allowed separation after evaluating every curve in its own
band.

## Interpretation

The local checks confirm:

- A single `ContrastCurve` remains supported.
- A `ContrastCurveSet` remains supported.
- A plain iterable/list of `ContrastCurve` objects is also supported.
- Multi-band curve sets run through the prepared compute boundary locally.
- The result is a per-draw intersection of radial non-detection constraints,
  not a global choice of one curve.

The checks do not claim final science-grade FPP values. A production comparison
should use the same local path with a larger `n_mc_samples` and the full target
set or a paper-selected representative subset.
