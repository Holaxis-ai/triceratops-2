from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 45.0
MAX_RETRIES = 3

_NUMERIC_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "radius": (
        "planet_radius_r_earth",
        "planet_radius_rearth",
        "radius_r_earth",
        "radius_rearth",
        "radius",
        "prad",
    ),
    "teff": ("stellar_eff_temp_k", "teff", "teff_k", "stellar_teff", "stellar_teff_k"),
    "snr": ("planet_snr", "snr", "signal_to_noise"),
    "tmag": ("tmag", "tess_mag", "tess_magnitude"),
    "period": ("period_days", "period", "per"),
    "depth": ("depth_ppm", "depth", "dep_ppm", "dep_ppt"),
    "duration": ("duration_hours", "duration_hr", "duration_hrs", "duration", "dur"),
}
_DISPOSITION_ALIASES: tuple[str, ...] = (
    "tfopwg_disposition",
    "disposition",
    "toi_disposition",
)
_KNOWN_PLANET_CODES = {
    "CP",
    "KP",
    "KNOWN_PLANET",
    "KNOWNPLANET",
    "CONFIRMED_PLANET",
    "CONFIRMEDPLANET",
}
_FALSE_POSITIVE_CODES = {"FP", "FA", "FALSE_POSITIVE", "FALSEPOSITIVE"}


def _default_cache_dir() -> Path:
    root = os.getenv("TRICERATOPS_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".triceratops" / "cache"


def _toi_cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "exofop" / "toi_table.pipe"


def _toi_single_cache_path(cache_dir: Path, toi_query: str) -> Path:
    token = (
        str(toi_query)
        .strip()
        .upper()
        .replace("TOI-", "")
        .replace("TOI", "")
        .replace(" ", "")
        .replace("/", "_")
    )
    return Path(cache_dir) / "exofop" / "toi_single" / f"{token}.pipe"


def _read_disk_cache(path: Path, *, cache_ttl_seconds: int) -> str | None:
    if cache_ttl_seconds <= 0:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    age = time.time() - float(stat.st_mtime)
    if age > float(cache_ttl_seconds):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_disk_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _normalize_header(col: str) -> str:
    col = (col or "").strip().lower()
    col = col.replace("%", "pct")
    for ch in ["(", ")", "[", "]", "{", "}", ",", ":", ";", "/"]:
        col = col.replace(ch, " ")
    col = col.replace("-", " ")
    col = "_".join(x for x in col.split() if x)
    return col


def _parse_pipe_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = (text or "").strip().splitlines()
    if not lines:
        return [], []
    header_raw = [h.strip() for h in lines[0].split("|")]
    header_norm = [_normalize_header(h) for h in header_raw]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 2:
            continue
        row: dict[str, str] = {}
        for k, v in zip(header_norm, cols, strict=False):
            if not k:
                continue
            row[k] = v
        rows.append(row)
    return header_norm, rows


@dataclass(frozen=True)
class ExoFOPToiTable:
    fetched_at_unix: float
    headers: list[str]
    rows: list[dict[str, str]]

    def entries_for_tic(self, tic_id: int) -> list[dict[str, str]]:
        target = str(int(tic_id))
        tic_keys = ["tic_id", "tic", "ticid"]
        out: list[dict[str, str]] = []
        for row in self.rows:
            for k in tic_keys:
                if row.get(k) == target:
                    out.append(row)
                    break
        return out


@dataclass(frozen=True)
class ExoFOPToiQueryStats:
    source_rows: int
    matched_rows_before_limit: int
    returned_rows: int
    skipped_non_numeric_rows: int
    filtered_by_disposition_rows: int


@dataclass(frozen=True)
class ExoFOPToiQueryResult:
    rows: list[dict[str, str]]
    stats: ExoFOPToiQueryStats


def _normalize_disposition_text(value: str | None) -> str:
    text = (value or "").strip().upper()
    for ch in ("-", " ", "/"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _parse_float_value(row: dict[str, str], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        raw = row.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        normalized = text.replace(",", "")
        if normalized.lower() in {"nan", "none", "null", "--", "n/a", "na"}:
            continue
        try:
            return float(normalized)
        except Exception:
            continue
    return None


def _row_disposition(row: dict[str, str]) -> str:
    for key in _DISPOSITION_ALIASES:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _disposition_matches_filter(row_disp_norm: str, wanted_norm: set[str]) -> bool:
    if not wanted_norm:
        return True
    if not row_disp_norm:
        return False
    tokens = {part for part in row_disp_norm.split("_") if part}
    return any(target == row_disp_norm or target in tokens for target in wanted_norm)


def query_exofop_toi_rows(
    table: ExoFOPToiTable,
    *,
    radius_min: float | None = None,
    radius_max: float | None = None,
    teff_min: float | None = None,
    teff_max: float | None = None,
    snr_min: float | None = None,
    snr_max: float | None = None,
    tmag_min: float | None = None,
    tmag_max: float | None = None,
    period_min: float | None = None,
    period_max: float | None = None,
    depth_min: float | None = None,
    depth_max: float | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
    include_dispositions: set[str] | None = None,
    exclude_dispositions: set[str] | None = None,
    exclude_known_planets: bool = False,
    exclude_false_positives: bool = False,
    sort_by: str | None = None,
    sort_descending: bool = False,
    max_results: int | None = None,
) -> ExoFOPToiQueryResult:
    """Filter/sort ExoFOP TOI rows with range and disposition constraints."""
    numeric_filters: dict[str, tuple[float | None, float | None]] = {
        "radius": (radius_min, radius_max),
        "teff": (teff_min, teff_max),
        "snr": (snr_min, snr_max),
        "tmag": (tmag_min, tmag_max),
        "period": (period_min, period_max),
        "depth": (depth_min, depth_max),
        "duration": (duration_min, duration_max),
    }
    include_norm = {_normalize_disposition_text(x) for x in (include_dispositions or set()) if x}
    exclude_norm = {_normalize_disposition_text(x) for x in (exclude_dispositions or set()) if x}

    rows_out: list[dict[str, str]] = []
    skipped_non_numeric_rows = 0
    filtered_by_disposition_rows = 0
    for row in table.rows:
        keep = True
        for field_name, (min_value, max_value) in numeric_filters.items():
            if min_value is None and max_value is None:
                continue
            value = _parse_float_value(row, _NUMERIC_FIELD_ALIASES[field_name])
            if value is None:
                skipped_non_numeric_rows += 1
                keep = False
                break
            if min_value is not None and value < float(min_value):
                keep = False
                break
            if max_value is not None and value > float(max_value):
                keep = False
                break
        if not keep:
            continue

        row_disposition = _row_disposition(row)
        row_disp_norm = _normalize_disposition_text(row_disposition)
        if include_norm and not _disposition_matches_filter(row_disp_norm, include_norm):
            filtered_by_disposition_rows += 1
            continue
        if exclude_norm and _disposition_matches_filter(row_disp_norm, exclude_norm):
            filtered_by_disposition_rows += 1
            continue
        if exclude_known_planets and _disposition_matches_filter(row_disp_norm, _KNOWN_PLANET_CODES):
            filtered_by_disposition_rows += 1
            continue
        if exclude_false_positives and _disposition_matches_filter(
            row_disp_norm, _FALSE_POSITIVE_CODES
        ):
            filtered_by_disposition_rows += 1
            continue

        rows_out.append(dict(row))

    matched_rows_before_limit = len(rows_out)

    if sort_by:
        sort_key_text = str(sort_by).strip().lower()
        aliases = _NUMERIC_FIELD_ALIASES.get(sort_key_text)
        if aliases:
            sortable: list[tuple[float, dict[str, str]]] = []
            unsortable: list[dict[str, str]] = []
            for row in rows_out:
                parsed = _parse_float_value(row, aliases)
                if parsed is None:
                    unsortable.append(row)
                else:
                    sortable.append((parsed, row))
            sortable.sort(key=lambda item: item[0], reverse=bool(sort_descending))
            rows_out = [row for _, row in sortable] + unsortable
        else:
            rows_out.sort(
                key=lambda r: str(r.get(sort_by) or "").lower(),
                reverse=bool(sort_descending),
            )

    if max_results is not None and int(max_results) >= 0:
        rows_out = rows_out[: int(max_results)]

    return ExoFOPToiQueryResult(
        rows=rows_out,
        stats=ExoFOPToiQueryStats(
            source_rows=len(table.rows),
            matched_rows_before_limit=matched_rows_before_limit,
            returned_rows=len(rows_out),
            skipped_non_numeric_rows=skipped_non_numeric_rows,
            filtered_by_disposition_rows=filtered_by_disposition_rows,
        ),
    )


_CACHE: ExoFOPToiTable | None = None


def fetch_exofop_toi_table(
    *,
    cache_ttl_seconds: int = 6 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ExoFOPToiTable:
    """Fetch and parse the ExoFOP TOI table (pipe-delimited).

    Caches the full table:
    - in-process (fast for repeated calls within one server process)
    - on-disk (persists across process restarts)
    """
    global _CACHE
    now = time.time()
    if (
        _CACHE is not None
        and cache_ttl_seconds > 0
        and (now - _CACHE.fetched_at_unix) <= float(cache_ttl_seconds)
    ):
        return _CACHE

    cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else _default_cache_dir()
    cache_path = _toi_cache_path(cache_dir)
    cached_text = _read_disk_cache(cache_path, cache_ttl_seconds=int(cache_ttl_seconds))
    if cached_text:
        headers, rows = _parse_pipe_table(cached_text)
        table = ExoFOPToiTable(
            fetched_at_unix=float(cache_path.stat().st_mtime), headers=headers, rows=rows
        )
        _CACHE = table
        return table

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                EXOFOP_TOI_URL,
                params={"sort": "toi", "output": "pipe"},
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            if attempt >= MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise last_exc or RuntimeError("ExoFOP fetch failed")

    with contextlib.suppress(Exception):
        _write_disk_cache(cache_path, response.text)

    headers, rows = _parse_pipe_table(response.text)
    table = ExoFOPToiTable(fetched_at_unix=now, headers=headers, rows=rows)
    _CACHE = table
    return table


def fetch_exofop_toi_table_for_toi(
    toi_query: str | float,
    *,
    cache_ttl_seconds: int = 6 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ExoFOPToiTable:
    """Fetch and parse a TOI-scoped ExoFOP table response.

    Uses ``download_toi.php?toi=<query>`` so callers can avoid downloading the
    full TOI master table when resolving a single target.
    """
    cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else _default_cache_dir()
    cache_path = _toi_single_cache_path(cache_dir, str(toi_query))
    cached_text = _read_disk_cache(cache_path, cache_ttl_seconds=int(cache_ttl_seconds))
    if cached_text:
        headers, rows = _parse_pipe_table(cached_text)
        return ExoFOPToiTable(
            fetched_at_unix=float(cache_path.stat().st_mtime), headers=headers, rows=rows
        )

    query_token = (
        str(toi_query)
        .strip()
        .upper()
        .replace("TOI-", "")
        .replace("TOI", "")
        .strip()
    )
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                EXOFOP_TOI_URL,
                params={"toi": query_token, "output": "pipe"},
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            if attempt >= MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise last_exc or RuntimeError("ExoFOP TOI-scoped fetch failed")

    text = response.text
    if "Sorry, entered TOI is invalid" in text:
        return ExoFOPToiTable(fetched_at_unix=time.time(), headers=[], rows=[])

    with contextlib.suppress(Exception):
        _write_disk_cache(cache_path, text)

    headers, rows = _parse_pipe_table(text)
    return ExoFOPToiTable(fetched_at_unix=time.time(), headers=headers, rows=rows)


def exofop_entries_for_tic(
    *,
    tic_id: int,
    cache_ttl_seconds: int = 6 * 3600,
) -> list[dict[str, Any]]:
    """Return ExoFOP TOI table entries for a TIC ID."""
    table = fetch_exofop_toi_table(cache_ttl_seconds=int(cache_ttl_seconds))
    return [dict(row) for row in table.entries_for_tic(int(tic_id))]
