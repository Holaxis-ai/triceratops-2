#!/usr/bin/env python3
"""Run local, matched-seed validation for multiple contrast curves.

This script is intentionally separate from the compare harness's Modal-oriented
compute helper.  It reuses the harness only for read-only artifact loading,
manifest parsing helpers, and curve parsing.
"""
# ruff: noqa: E402,I001
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path("/Users/collier/projects/Holaxis/astro/analysis/contrast-curve-best-of-both")
AUTO_REPO = Path("/Users/collier/projects/Holaxis/astro/triceratops-auto")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "multiple_contrast_curves_validation"
DEFAULT_TARGETS = ("toi-1738-01", "toi-5961-01", "toi-1703-01", "toi-6067-01")
DEFAULT_SEEDS = (101, 211, 323, 437, 541)
DEFAULT_N_MC = 1_000_000

for path in (AUTO_REPO, HARNESS_ROOT, REPO_ROOT):
    sys.path.insert(0, str(path))

import lib  # type: ignore  # noqa: E402

# lib.py prepends AUTO_REPO.  Put this checkout back in front before any later
# triceratops imports happen through auto-fpp.
sys.path.insert(0, str(REPO_ROOT))

from auto_fpp.compute import compute_prepared_artifact  # type: ignore  # noqa: E402
from auto_fpp.models import AutoFppComputeConfig  # type: ignore  # noqa: E402
from triceratops.config import Config  # noqa: E402
from triceratops.domain.value_objects import ContrastCurveSet  # noqa: E402

PLANET = ("TP", "PTP", "DTP")
BOUND = ("STP", "SEB", "SEBx2P", "PEB", "PEBx2P", "DEB", "DEBx2P")
BACKGROUND = ("BTP", "BEB", "BEBx2P")
NEARBY = ("NTP", "NEB", "NEBx2P")
TARGET_EB = ("EB", "EBx2P")
FP_SCENARIOS = BOUND + BACKGROUND + NEARBY + TARGET_EB
_NONFINITE_FLOAT_KEY = "__nonfinite_float__"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row for row in rows
        if row.get("is_curve") == "1" and row.get("use") == "1"
    ]


def _target_rows(rows: list[dict[str, str]], slug: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["slug"] == slug]
    if not selected:
        raise ValueError(f"No usable contrast curves found for {slug!r}")
    return selected


def _curve_metric(row: dict[str, str]) -> tuple[float, float, float, str]:
    curve = lib.parse_curve(row["file"], row["band"])
    props = lib.curve_properties(curve)
    reach = props.get("sep_at_dmag_3")
    dmag_05 = props.get("dmag_at_0.5")
    dmag_max = props.get("dmag_max")
    reach_val = float(reach) if reach is not None and math.isfinite(float(reach)) else float("inf")
    dmag_05_val = (
        float(dmag_05)
        if dmag_05 is not None and math.isfinite(float(dmag_05))
        else -float("inf")
    )
    dmag_max_val = (
        float(dmag_max)
        if dmag_max is not None and math.isfinite(float(dmag_max))
        else -float("inf")
    )
    return reach_val, -dmag_05_val, -dmag_max_val, row["basename"]


def _best_rows_by_band(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_band: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_band[row["band"]].append(row)
    return {
        band: sorted(band_rows, key=_curve_metric)[0]
        for band, band_rows in by_band.items()
    }


def _preferred_band_order(bands: set[str]) -> list[str]:
    preferred = ["Vis", "J", "H", "K", "TESS", "Kepler", "g", "r", "i", "z"]
    return [band for band in preferred if band in bands] + sorted(bands - set(preferred))


def _variants_for_target(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    best = _best_rows_by_band(rows)
    bands = _preferred_band_order(set(best))
    variants: list[dict[str, Any]] = [{
        "variant": "no_curve",
        "rows": [],
    }]
    for band in bands:
        variants.append({
            "variant": f"single_{band.lower()}",
            "rows": [best[band]],
        })

    combo_order = [
        ("K", "Vis"),
        ("J", "K", "Vis"),
        ("H", "K", "Vis"),
        ("J", "K"),
        ("H", "K"),
    ]
    seen = {variant["variant"] for variant in variants}
    for combo in combo_order:
        if all(band in best for band in combo):
            name = "multi_" + "_".join(band.lower() for band in combo)
            if name not in seen:
                variants.append({
                    "variant": name,
                    "rows": [best[band] for band in combo],
                })
                seen.add(name)

    if len(bands) >= 2:
        name = "multi_all_selected_bands"
        if name not in seen:
            variants.append({
                "variant": name,
                "rows": [best[band] for band in bands],
            })
    return variants


def _curve_from_rows(rows: list[dict[str, str]]):
    curves = [lib.parse_curve(row["file"], row["band"]) for row in rows]
    if not curves:
        return None
    if len(curves) == 1:
        return curves[0]
    return ContrastCurveSet(curves)


def _component_owas_from_rows(rows: list[dict[str, str]]) -> list[float]:
    return [
        float(np.max(lib.parse_curve(row["file"], row["band"]).separations_arcsec))
        for row in rows
    ]


def _scenario_totals(result) -> dict[str, float]:
    totals: dict[str, float] = {}
    for scenario in result.scenario_results:
        key = scenario.scenario_id.value
        totals[key] = totals.get(key, 0.0) + float(scenario.relative_probability)
    return totals


def _scenario_ln_evidence(result) -> dict[str, float]:
    lnz: dict[str, float] = {}
    for scenario in result.scenario_results:
        key = scenario.scenario_id.value
        value = float(scenario.ln_evidence)
        if key in lnz:
            lnz[key] = float(np.logaddexp(lnz[key], value))
        else:
            lnz[key] = value
    return lnz


def _sum_group(scenarios: dict[str, float], keys: tuple[str, ...]) -> float:
    return sum(scenarios.get(key, 0.0) for key in keys)


def _result_row(
    *,
    slug: str,
    target_ref: str,
    variant: dict[str, Any],
    seed: int,
    n_mc: int,
    result: Any,
    seconds: float,
) -> dict[str, Any]:
    scenarios = _scenario_totals(result)
    ln_evidence = _scenario_ln_evidence(result)
    rows = variant["rows"]
    return {
        "target": target_ref,
        "slug": slug,
        "variant": variant["variant"],
        "seed": seed,
        "n_mc": n_mc,
        "backend": "local",
        "seconds": seconds,
        "bands": [row["band"] for row in rows],
        "component_basenames": [row["basename"] for row in rows],
        "component_owas_arcsec": _component_owas_from_rows(rows),
        "fpp": float(result.false_positive_probability),
        "nfpp": float(result.nearby_false_positive_probability),
        "planet": _sum_group(scenarios, PLANET),
        "bound": _sum_group(scenarios, BOUND),
        "background": _sum_group(scenarios, BACKGROUND),
        "nearby": _sum_group(scenarios, NEARBY),
        "target_eb": _sum_group(scenarios, TARGET_EB),
        "scenarios": scenarios,
        "ln_evidence": ln_evidence,
    }


def _completed_keys(jsonl_path: Path) -> set[tuple[str, str, int, int]]:
    keys: set[tuple[str, str, int, int]] = set()
    for row in _read_jsonl(jsonl_path):
        keys.add((row["slug"], row["variant"], int(row["seed"]), int(row["n_mc"])))
    return keys


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return {_NONFINITE_FLOAT_KEY: "NaN"}
        if value == math.inf:
            return {_NONFINITE_FLOAT_KEY: "Infinity"}
        if value == -math.inf:
            return {_NONFINITE_FLOAT_KEY: "-Infinity"}
        return value
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(inner) for inner in value]
    return value


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_NONFINITE_FLOAT_KEY}:
            tag = value[_NONFINITE_FLOAT_KEY]
            if tag == "NaN":
                return float("nan")
            if tag == "Infinity":
                return math.inf
            if tag == "-Infinity":
                return -math.inf
        return {key: _json_restore(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_restore(inner) for inner in value]
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def _rewrite_without_keys(
    jsonl_path: Path,
    keys_to_remove: set[tuple[str, str, int, int]],
) -> None:
    if not jsonl_path.exists() or not keys_to_remove:
        return
    rows = _read_jsonl(jsonl_path)
    kept = [
        row for row in rows
        if (
            row["slug"], row["variant"], int(row["seed"]), int(row["n_mc"])
        ) not in keys_to_remove
    ]
    _write_jsonl(jsonl_path, kept)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [_json_restore(json.loads(line)) for line in f if line.strip()]


def _summary_mean(values: list[float]) -> float:
    if all(math.isfinite(value) for value in values):
        return statistics.fmean(values)
    if all(value == values[0] for value in values):
        return values[0]
    return float("nan")


def _summary_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    if all(math.isfinite(value) for value in values):
        return statistics.stdev(values)
    if all(value == values[0] for value in values):
        return 0.0
    return float("nan")


def _selection_owa_lookup(selection_rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {
        (row["slug"], row["band"]): float(row["owa_arcsec"])
        for row in selection_rows
    }


def _backfill_component_owas(
    rows: list[dict[str, Any]],
    selected_owas: dict[tuple[str, str], float],
) -> None:
    for row in rows:
        if row.get("component_owas_arcsec") is not None:
            continue
        row["component_owas_arcsec"] = [
            selected_owas[(row["slug"], band)]
            for band in row["bands"]
        ]


def _write_summaries(
    jsonl_path: Path,
    output_dir: Path,
    *,
    selected_owas: dict[tuple[str, str], float] | None = None,
) -> None:
    rows = _read_jsonl(jsonl_path)
    if selected_owas is not None:
        _backfill_component_owas(rows, selected_owas)
        _write_jsonl(jsonl_path, rows)
    _write_seed_paired_impact(rows, output_dir)
    _write_ln_evidence_summary(rows, output_dir)
    _write_ln_evidence_monotonicity_check(
        rows,
        output_dir,
        selected_owas=selected_owas,
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["slug"], row["variant"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (slug, variant), group in sorted(grouped.items()):
        def mean(key: str) -> float:
            return statistics.fmean(float(row[key]) for row in group)

        def stdev(key: str) -> float:
            values = [float(row[key]) for row in group]
            return statistics.stdev(values) if len(values) > 1 else 0.0

        first = group[0]
        summary_rows.append({
            "target": first["target"],
            "slug": slug,
            "variant": variant,
            "bands": "+".join(first["bands"]) if first["bands"] else "none",
            "n_seeds": len(group),
            "n_mc": first["n_mc"],
            "fpp_mean": mean("fpp"),
            "fpp_std": stdev("fpp"),
            "nfpp_mean": mean("nfpp"),
            "bound_mean": mean("bound"),
            "background_mean": mean("background"),
            "planet_mean": mean("planet"),
            "nearby_mean": mean("nearby"),
            "target_eb_mean": mean("target_eb"),
            "seconds_mean": mean("seconds"),
            "component_basenames": ";".join(first["component_basenames"]),
        })

    _write_csv(
        output_dir / "summary_local_multicc.csv",
        summary_rows,
        [
            "target", "slug", "variant", "bands", "n_seeds", "n_mc",
            "fpp_mean", "fpp_std", "nfpp_mean", "bound_mean",
            "background_mean", "planet_mean", "nearby_mean", "target_eb_mean",
            "seconds_mean", "component_basenames",
        ],
    )

    by_slug = defaultdict(list)
    for row in summary_rows:
        by_slug[row["slug"]].append(row)
    impact_rows: list[dict[str, Any]] = []
    for slug, group in sorted(by_slug.items()):
        no_curve = next((row for row in group if row["variant"] == "no_curve"), None)
        singles = [row for row in group if row["variant"].startswith("single_")]
        multis = [row for row in group if row["variant"].startswith("multi_")]
        if not singles:
            continue
        best_single = min(singles, key=lambda row: float(row["fpp_mean"]))
        for multi in multis:
            impact_rows.append({
                "target": multi["target"],
                "slug": slug,
                "multi_variant": multi["variant"],
                "best_single_variant": best_single["variant"],
                "delta_fpp_vs_best_single": (
                    float(multi["fpp_mean"]) - float(best_single["fpp_mean"])
                ),
                "delta_bound_vs_best_single": (
                    float(multi["bound_mean"]) - float(best_single["bound_mean"])
                ),
                "delta_background_vs_best_single": (
                    float(multi["background_mean"]) - float(best_single["background_mean"])
                ),
                "delta_fpp_vs_no_curve": (
                    float(multi["fpp_mean"]) - float(no_curve["fpp_mean"])
                    if no_curve is not None else ""
                ),
                "monotonic_channel_check": "inspect_seed_pairs",
            })
    _write_csv(
        output_dir / "impact_local_multicc.csv",
        impact_rows,
        [
            "target", "slug", "multi_variant", "best_single_variant",
            "delta_fpp_vs_best_single", "delta_bound_vs_best_single",
            "delta_background_vs_best_single", "delta_fpp_vs_no_curve",
            "monotonic_channel_check",
        ],
    )


def _write_ln_evidence_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        ln_evidence = row.get("ln_evidence")
        if not isinstance(ln_evidence, dict):
            continue
        for scenario, value in ln_evidence.items():
            grouped[(row["slug"], row["variant"], scenario)].append(float(value))

    summary_rows = []
    for (slug, variant, scenario), values in sorted(grouped.items()):
        summary_rows.append({
            "slug": slug,
            "variant": variant,
            "scenario": scenario,
            "n_seeds": len(values),
            "ln_evidence_mean": _summary_mean(values),
            "ln_evidence_std": _summary_stdev(values),
        })
    _write_csv(
        output_dir / "ln_evidence_summary_local_multicc.csv",
        summary_rows,
        [
            "slug", "variant", "scenario", "n_seeds",
            "ln_evidence_mean", "ln_evidence_std",
        ],
    )


def _read_selected_curve_owas(output_dir: Path) -> dict[tuple[str, str], float]:
    selected_path = output_dir / "selected_curves_local_multicc.csv"
    if not selected_path.exists():
        return {}
    with selected_path.open(newline="") as f:
        return {
            (row["slug"], row["band"]): float(row["owa_arcsec"])
            for row in csv.DictReader(f)
        }


def _component_owas_for_row(
    row: dict[str, Any],
    selected_owas: dict[tuple[str, str], float],
) -> list[float]:
    row_owas = row.get("component_owas_arcsec")
    if isinstance(row_owas, list) and len(row_owas) == len(row["bands"]):
        return [float(owa) for owa in row_owas]

    missing = [
        band for band in row["bands"]
        if (row["slug"], band) not in selected_owas
    ]
    if missing:
        raise AssertionError(
            "Missing component OWA metadata for "
            f"{row['slug']}:{row['variant']} bands={missing}"
        )
    return [selected_owas[(row["slug"], band)] for band in row["bands"]]


def _all_blind_margin_bound(
    row: dict[str, Any],
    selected_owas: dict[tuple[str, str], float],
) -> float:
    curve_owas = _component_owas_for_row(row, selected_owas)
    if len(curve_owas) < 2:
        return 0.0
    min_owa = min(curve_owas)
    max_owa = max(curve_owas)
    if min_owa <= 0.0 or max_owa <= 0.0:
        return 0.0
    return 2.0 * math.log(max_owa / min_owa)


def _ln_evidence_margin(multi_value: float, best_single_value: float) -> float:
    if multi_value == best_single_value:
        return 0.0
    if multi_value == -math.inf:
        return -math.inf
    if best_single_value == -math.inf:
        return math.inf
    return multi_value - best_single_value


def _write_ln_evidence_monotonicity_check(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    selected_owas: dict[tuple[str, str], float] | None = None,
    tolerance: float = 1e-9,
) -> None:
    by_key = {
        (row["slug"], row["variant"], int(row["seed"]), int(row["n_mc"])): row
        for row in rows
    }
    selected_owas = selected_owas or _read_selected_curve_owas(output_dir)
    check_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    coverage_errors: list[str] = []
    for row in rows:
        if not row["variant"].startswith("multi_"):
            continue
        slug = row["slug"]
        seed = int(row["seed"])
        n_mc = int(row["n_mc"])
        singles = [
            by_key.get((slug, f"single_{band.lower()}", seed, n_mc))
            for band in row["bands"]
        ]
        if any(single is None for single in singles):
            coverage_errors.append(f"{slug}:{row['variant']}:seed{seed}:missing single")
            continue
        multi_lnz = row.get("ln_evidence") or {}
        if not isinstance(multi_lnz, dict):
            coverage_errors.append(f"{slug}:{row['variant']}:seed{seed}:missing multi lnZ")
            continue
        for scenario in FP_SCENARIOS:
            if scenario not in multi_lnz:
                coverage_errors.append(
                    f"{slug}:{row['variant']}:{scenario}:seed{seed}:missing multi scenario"
                )
                continue
            missing_single_scenarios = [
                single["variant"] for single in singles
                if (
                    not isinstance(single.get("ln_evidence"), dict)
                    or scenario not in single["ln_evidence"]
                )
            ]
            if missing_single_scenarios:
                coverage_errors.append(
                    f"{slug}:{row['variant']}:{scenario}:seed{seed}:"
                    f"missing singles={missing_single_scenarios}"
                )
                continue
            single_values = [
                float(single["ln_evidence"][scenario])
                for single in singles
            ]
            multi_value = float(multi_lnz[scenario])
            best_single_value = min(single_values)
            margin = _ln_evidence_margin(multi_value, best_single_value)
            all_blind_margin_bound = _all_blind_margin_bound(row, selected_owas)
            strict_passed = margin <= tolerance
            passed = margin <= all_blind_margin_bound + tolerance
            check_row = {
                "target": row["target"],
                "slug": slug,
                "seed": seed,
                "n_mc": n_mc,
                "multi_variant": row["variant"],
                "scenario": scenario,
                "lnz_multi": multi_value,
                "lnz_best_single": best_single_value,
                "margin": margin,
                "all_blind_margin_bound": all_blind_margin_bound,
                "tolerance": tolerance,
                "strict_passed": strict_passed,
                "passed": passed,
            }
            check_rows.append(check_row)
            if not passed:
                violations.append(check_row)

    if coverage_errors:
        preview = ", ".join(coverage_errors[:5])
        raise AssertionError(
            "Raw FP lnZ monotonicity coverage errors detected: "
            f"{len(coverage_errors)} total; first errors: {preview}"
        )
    if not check_rows:
        raise AssertionError("Raw FP lnZ monotonicity check produced no rows")

    _write_csv(
        output_dir / "ln_evidence_monotonicity_check_local_multicc.csv",
        check_rows,
        [
            "target", "slug", "seed", "n_mc", "multi_variant", "scenario",
            "lnz_multi", "lnz_best_single", "margin", "all_blind_margin_bound",
            "tolerance", "strict_passed", "passed",
        ],
    )
    if violations:
        preview = ", ".join(
            f"{row['slug']}:{row['multi_variant']}:{row['scenario']}:seed{row['seed']}"
            for row in violations[:5]
        )
        raise AssertionError(
            "Raw FP lnZ monotonicity violations detected: "
            f"{len(violations)} total; first violations: {preview}"
        )


def _write_seed_paired_impact(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_key = {
        (row["slug"], row["variant"], int(row["seed"]), int(row["n_mc"])): row
        for row in rows
    }
    impact_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row["variant"].startswith("multi_"):
            continue
        slug = row["slug"]
        seed = int(row["seed"])
        n_mc = int(row["n_mc"])
        singles = [
            by_key.get((slug, f"single_{band.lower()}", seed, n_mc))
            for band in row["bands"]
        ]
        singles = [single for single in singles if single is not None]
        if not singles:
            continue
        best_single = min(singles, key=lambda single: float(single["fpp"]))
        no_curve = by_key.get((slug, "no_curve", seed, n_mc))
        max_single_bound = max(float(single["bound"]) for single in singles)
        max_single_background = max(float(single["background"]) for single in singles)
        min_single_bound = min(float(single["bound"]) for single in singles)
        min_single_background = min(float(single["background"]) for single in singles)
        impact_rows.append({
            "target": row["target"],
            "slug": slug,
            "seed": seed,
            "n_mc": n_mc,
            "multi_variant": row["variant"],
            "bands": "+".join(row["bands"]),
            "best_single_variant": best_single["variant"],
            "fpp_multi": row["fpp"],
            "fpp_best_single": best_single["fpp"],
            "delta_fpp_vs_best_single": float(row["fpp"]) - float(best_single["fpp"]),
            "delta_bound_vs_best_single": (
                float(row["bound"]) - float(best_single["bound"])
            ),
            "delta_background_vs_best_single": (
                float(row["background"]) - float(best_single["background"])
            ),
            "delta_fpp_vs_no_curve": (
                float(row["fpp"]) - float(no_curve["fpp"])
                if no_curve is not None else ""
            ),
            "bound_le_max_component_single": float(row["bound"]) <= max_single_bound,
            "background_le_max_component_single": (
                float(row["background"]) <= max_single_background
            ),
            "bound_le_min_component_single": float(row["bound"]) <= min_single_bound,
            "background_le_min_component_single": (
                float(row["background"]) <= min_single_background
            ),
            "component_single_deltas": json.dumps({
                single["variant"]: {
                    "delta_fpp": float(row["fpp"]) - float(single["fpp"]),
                    "delta_bound": float(row["bound"]) - float(single["bound"]),
                    "delta_background": (
                        float(row["background"]) - float(single["background"])
                    ),
                }
                for single in singles
            }, sort_keys=True),
        })
    _write_csv(
        output_dir / "seed_paired_impact_local_multicc.csv",
        impact_rows,
        [
            "target", "slug", "seed", "n_mc", "multi_variant", "bands",
            "best_single_variant", "fpp_multi", "fpp_best_single",
            "delta_fpp_vs_best_single", "delta_bound_vs_best_single",
            "delta_background_vs_best_single", "delta_fpp_vs_no_curve",
            "bound_le_max_component_single", "background_le_max_component_single",
            "bound_le_min_component_single", "background_le_min_component_single",
            "component_single_deltas",
        ],
    )


def _curve_selection_rows(rows: list[dict[str, str]], targets: list[str]) -> list[dict[str, Any]]:
    output = []
    for slug in targets:
        best = _best_rows_by_band(_target_rows(rows, slug))
        for band in _preferred_band_order(set(best)):
            row = best[band]
            curve = lib.parse_curve(row["file"], row["band"])
            props = lib.curve_properties(curve)
            output.append({
                "target": row["target_ref"],
                "slug": slug,
                "band": band,
                "basename": row["basename"],
                "telescope": row.get("telescope", ""),
                "obs_filter": row.get("obs_filter", ""),
                "sep_at_dmag_3": props.get("sep_at_dmag_3"),
                "dmag_at_0.5": props.get("dmag_at_0.5"),
                "dmag_max": props.get("dmag_max"),
                "iwa_arcsec": props.get("iwa_arcsec"),
                "owa_arcsec": props.get("owa_arcsec"),
            })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-mc", type=int, default=DEFAULT_N_MC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun requested target/variant/seed rows even if JSONL rows exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "runs_local_multicc.jsonl"

    lib.load_env()
    manifest = HARNESS_ROOT / "data" / "curves" / "manifest.csv"
    rows = _load_manifest(manifest)

    selection_targets = list(args.targets)
    if args.summary_only and jsonl_path.exists():
        run_rows = _read_jsonl(jsonl_path)
        selection_targets = sorted({row["slug"] for row in run_rows}) or selection_targets

    selection_rows = _curve_selection_rows(rows, selection_targets)
    selected_owas = _selection_owa_lookup(selection_rows)
    _write_csv(
        output_dir / "selected_curves_local_multicc.csv",
        selection_rows,
        [
            "target", "slug", "band", "basename", "telescope", "obs_filter",
            "sep_at_dmag_3", "dmag_at_0.5", "dmag_max", "iwa_arcsec", "owa_arcsec",
        ],
    )

    if args.summary_only:
        _write_summaries(jsonl_path, output_dir, selected_owas=selected_owas)
        return

    if args.force:
        rows_by_slug = {slug: _target_rows(rows, slug) for slug in args.targets}
        variants_by_slug = {
            slug: _variants_for_target(target_rows)
            for slug, target_rows in rows_by_slug.items()
        }
        keys_to_remove = {
            (slug, variant["variant"], int(seed), int(args.n_mc))
            for slug, variants in variants_by_slug.items()
            for variant in variants
            for seed in args.seeds
        }
        _rewrite_without_keys(jsonl_path, keys_to_remove)

    completed = _completed_keys(jsonl_path)
    cfg_template = AutoFppComputeConfig(compute_backend="local", bin_count=200)
    runs_started = 0

    for slug in args.targets:
        target_rows = _target_rows(rows, slug)
        target_ref = target_rows[0]["target_ref"]
        artifact = lib.load_artifact(slug)
        variants = _variants_for_target(target_rows)
        for variant in variants:
            curve_input = _curve_from_rows(variant["rows"])
            artifact_variant = replace(artifact, contrast_curve=curve_input)
            for seed in args.seeds:
                key = (slug, variant["variant"], int(seed), int(args.n_mc))
                if key in completed:
                    print(f"skip existing {key}", flush=True)
                    continue
                if args.max_runs is not None and runs_started >= args.max_runs:
                    _write_summaries(jsonl_path, output_dir, selected_owas=selected_owas)
                    return
                cfg = replace(
                    cfg_template,
                    compute=Config(
                        n_mc_samples=args.n_mc,
                        n_best_samples=1000,
                        seed=int(seed),
                        n_workers=0,
                    ),
                )
                print(
                    f"run {slug} {variant['variant']} seed={seed} n_mc={args.n_mc}",
                    flush=True,
                )
                t0 = time.time()
                result, _workspace = compute_prepared_artifact(artifact_variant, cfg)
                seconds = time.time() - t0
                row = _result_row(
                    slug=slug,
                    target_ref=target_ref,
                    variant=variant,
                    seed=int(seed),
                    n_mc=int(args.n_mc),
                    result=result,
                    seconds=seconds,
                )
                _append_jsonl(jsonl_path, row)
                completed.add(key)
                runs_started += 1
                print(
                    f"done {slug} {variant['variant']} seed={seed} "
                    f"seconds={seconds:.2f} fpp={row['fpp']:.8g}",
                    flush=True,
                )
    _write_summaries(jsonl_path, output_dir, selected_owas=selected_owas)


if __name__ == "__main__":
    main()
