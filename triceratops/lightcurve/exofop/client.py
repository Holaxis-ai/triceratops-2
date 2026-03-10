from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Value types (frozen dataclasses, inlined from tess-vetter exofop/types.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExoFopTarget:
    tic_id: int
    toi: str | None = None
    aliases: dict[str, str] | None = None


@dataclass(frozen=True)
class ExoFopFileRow:
    file_id: int
    tic_id: int
    toi: str | None
    filename: str
    type: str
    date_utc: str | None = None
    user: str | None = None
    group: str | None = None
    tag: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ExoFopSelectors:
    types: set[str] | None = None
    filename_regex: str | None = None
    tag_ids: set[int] | None = None
    max_files: int | None = None


@dataclass(frozen=True)
class ExoFopFetchResult:
    target: ExoFopTarget
    cache_root: Path
    manifest_path: Path
    filelist_path: Path
    summary_paths: dict[str, Path]
    files_downloaded: list[Path]
    files_skipped: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ExoFopObsNoteRow:
    tic_id: int
    author: str | None
    date_utc: str | None
    data_tag: str | None
    text: str
    note_id: int | None = None
    group: str | None = None
    raw: dict[str, str] | None = None


@dataclass(frozen=True)
class ExoFopToiRow:
    tic_id: int
    toi: str | None
    tfopwg_disposition: str | None
    planet_disposition: str | None
    comments: str | None
    raw: dict[str, str]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ExoFopClient:
    BASE_URL = "https://exofop.ipac.caltech.edu/tess/"

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        cookie_jar_path: str | Path | None = None,
        user_agent: str = "triceratops-fast exofop-client",
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            env_dir = os.getenv("TRICERATOPS_CACHE_DIR")
            self.cache_dir = Path(env_dir) if env_dir else Path.home() / ".triceratops" / "cache"
        self.cookie_jar_path = Path(cookie_jar_path) if cookie_jar_path else None
        self.user_agent = user_agent
        env_timeout = os.getenv("TRICERATOPS_EXOFOP_TIMEOUT_SECONDS")
        if env_timeout is not None and str(env_timeout).strip() != "":
            with contextlib.suppress(TypeError, ValueError):
                timeout_seconds = float(env_timeout)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._cookie_loaded = False
        if self.cookie_jar_path is not None:
            self._cookie_loaded = self._load_cookie_jar(self.cookie_jar_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def cookie_jar_loaded(self) -> bool:
        return self._cookie_loaded

    def resolve_target(self, target: str | int) -> ExoFopTarget:
        if isinstance(target, int):
            return ExoFopTarget(tic_id=int(target), toi=None, aliases={})
        target_str = str(target).strip()
        if target_str.isdigit():
            return ExoFopTarget(tic_id=int(target_str), toi=None, aliases={})

        url = self._url("gototicid.php")
        params = {"target": target_str, "jsonext": "1"}
        payload = self._get_json(url, params=params)

        summary = payload.get("summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict) and summary.get("status") == "OK":
            tic_raw = summary.get("TIC")
            if tic_raw is None:
                raise ValueError(f"ExoFOP resolved '{target_str}' but did not return a TIC in summary payload")
            tic_id = int(tic_raw)
            aliases: dict[str, str] = {}
            for k in ("GAIADR2", "GAIADR3", "KIC", "EPIC"):
                if summary.get(k):
                    aliases[k.lower()] = str(summary.get(k))
            return ExoFopTarget(tic_id=tic_id, toi=None, aliases=aliases)

        params = {"target": target_str, "json": "1"}
        payload = self._get_json(url, params=params)
        if isinstance(payload, dict) and payload.get("status") == "OK":
            tic_raw = payload.get("TIC")
            if tic_raw is None:
                raise ValueError(f"ExoFOP resolved '{target_str}' but did not return a TIC in fallback payload")
            tic_id = int(tic_raw)
            aliases = {}
            for k in ("GaiaDR2", "GAIADR2", "KIC", "EPIC"):
                if payload.get(k):
                    aliases[k.lower()] = str(payload.get(k))
            return ExoFopTarget(tic_id=tic_id, toi=None, aliases=aliases)

        raise ValueError(
            f"ExoFOP could not resolve target '{target_str}': "
            f"{payload.get('message') if isinstance(payload, dict) else payload}"
        )

    def file_list(self, *, tic_id: int, force_refresh: bool = False) -> list[ExoFopFileRow]:
        target_dir = self._target_dir(tic_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "filelist.csv"
        with contextlib.suppress(Exception):
            payload = self.target_json(tic_id=tic_id, force_refresh=force_refresh)
            rows = self._parse_filelist_target_json(payload, fallback_tic_id=int(tic_id))
            if rows:
                self._write_text_atomic(out_path, self._render_filelist_csv(rows))
                return rows

        if out_path.exists() and not force_refresh:
            return self._parse_filelist_csv(
                out_path.read_text(encoding="utf-8", errors="replace"), fallback_tic_id=int(tic_id)
            )
        text = self._get_text(self._url("download_filelist.php"), params={"id": str(int(tic_id)), "output": "csv"})
        self._write_text_atomic(out_path, text)
        return self._parse_filelist_csv(text, fallback_tic_id=int(tic_id))

    def target_json(self, *, tic_id: int, force_refresh: bool = False) -> dict[str, Any]:
        target_dir = self._target_dir(tic_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "target.json"
        if out_path.exists() and not force_refresh:
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        payload = self._get_json(self._url("target.php"), params={"id": str(int(tic_id)), "json": "1"})
        if not isinstance(payload, dict):
            raise ValueError("ExoFOP target.php response is not a JSON object")
        self._write_text_atomic(out_path, json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def obs_notes(self, *, tic_id: int, force_refresh: bool = False) -> list[ExoFopObsNoteRow]:
        target_dir = self._target_dir(tic_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "obsnotes.csv"
        if out_path.exists() and not force_refresh:
            return self._parse_obsnotes_text(
                out_path.read_text(encoding="utf-8", errors="replace"),
                fallback_tic_id=int(tic_id),
            )

        url = self._url("download_obsnotes.php")
        text = self._get_text(url, params={"tid": str(int(tic_id))})
        self._write_text_atomic(out_path, text)
        return self._parse_obsnotes_text(text, fallback_tic_id=int(tic_id))

    def toi_row(
        self,
        *,
        tic_id: int,
        toi: str | float | None = None,
        force_refresh: bool = False,
    ) -> ExoFopToiRow | None:
        with contextlib.suppress(Exception):
            payload = self.target_json(tic_id=int(tic_id), force_refresh=force_refresh)
            row = self._extract_toi_row_from_target_json(payload=payload, tic_id=int(tic_id), toi=toi)
            if row is not None:
                return row

        target_dir = self._target_dir(tic_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "toi_table.pipe"
        if out_path.exists() and not force_refresh:
            text = out_path.read_text(encoding="utf-8", errors="replace")
            return self._extract_toi_row(text=text, tic_id=int(tic_id), toi=toi)

        endpoint = self._url("download_toi.php")
        best_effort_params = [
            {"tic": str(int(tic_id)), "output": "pipe"},
            {"id": str(int(tic_id)), "output": "pipe"},
            {"tid": str(int(tic_id)), "output": "pipe"},
            {"output": "pipe"},
        ]
        last_text: str | None = None
        for params in best_effort_params:
            try:
                text = self._get_text(endpoint, params=params)
            except Exception:
                continue
            last_text = text
            row = self._extract_toi_row(text=text, tic_id=int(tic_id), toi=toi)
            if row is not None:
                self._write_text_atomic(out_path, text)
                return row

        if last_text is None:
            return None
        self._write_text_atomic(out_path, last_text)
        return self._extract_toi_row(text=last_text, tic_id=int(tic_id), toi=toi)

    def index_target(
        self,
        *,
        target: str | int,
        include_summaries: bool = True,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        exo = self.resolve_target(target)
        target_dir = self._target_dir(exo.tic_id)
        summaries_dir = target_dir / "summaries"
        files_dir = target_dir / "files"
        archives_dir = target_dir / "archives"
        for d in (target_dir, summaries_dir, files_dir, archives_dir):
            d.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        files_downloaded: list[Path] = []
        files_skipped: list[str] = []

        file_rows = self.file_list(tic_id=exo.tic_id, force_refresh=force_refresh)
        filelist_path = target_dir / "filelist.csv"

        summary_paths: dict[str, Path] = {}
        if include_summaries:
            try:
                payload = self.target_json(tic_id=int(exo.tic_id), force_refresh=force_refresh)
                summary_payloads = {
                    "spect": payload.get("spectroscopy"),
                    "imaging": payload.get("imaging"),
                    "tseries": payload.get("time_series"),
                    "uploads": payload.get("files"),
                    "stellar": payload.get("stellar_parameters"),
                    "planet": payload.get("planet_parameters"),
                }
                for key, block in summary_payloads.items():
                    out_path = summaries_dir / f"{key}.json"
                    summary_paths[key] = out_path
                    if out_path.exists() and not force_refresh:
                        continue
                    self._write_text_atomic(out_path, json.dumps(block if block is not None else [], indent=2))
            except Exception as exc:
                warnings.append(f"target.php summary discovery failed: {type(exc).__name__}: {exc}")
                summary_specs = {
                    "spect": ("download_spect.php", {"target": str(exo.tic_id), "output": "csv"}),
                    "imaging": ("download_imaging.php", {"target": str(exo.tic_id), "output": "csv"}),
                    "tseries": ("download_tseries.php", {"target": str(exo.tic_id), "output": "csv"}),
                    "uploads": ("download_uploads.php", {"target": str(exo.tic_id), "output": "csv"}),
                    "stellar": ("download_stellar.php", {"id": str(exo.tic_id), "output": "pipe"}),
                    "planet": ("download_planet.php", {"id": str(exo.tic_id), "output": "pipe"}),
                }
                for key, (endpoint, params) in summary_specs.items():
                    out_path = summaries_dir / f"{key}.{self._ext_for_output(params.get('output'))}"
                    summary_paths[key] = out_path
                    if out_path.exists() and not force_refresh:
                        continue
                    try:
                        text = self._get_text(self._url(endpoint), params=params)
                        self._write_text_atomic(out_path, text)
                    except Exception as endpoint_error:
                        warnings.append(f"{endpoint} failed: {type(endpoint_error).__name__}: {endpoint_error}")

        manifest_path = target_dir / "manifest.json"
        self._write_manifest(
            manifest_path=manifest_path,
            target=exo,
            filelist_path=filelist_path,
            summary_paths=summary_paths,
            files_dir=files_dir,
            file_rows=file_rows,
            downloaded_files=files_downloaded,
            skipped=files_skipped,
            warnings=warnings,
        )

        return ExoFopFetchResult(
            target=exo,
            cache_root=target_dir,
            manifest_path=manifest_path,
            filelist_path=filelist_path,
            summary_paths=summary_paths,
            files_downloaded=files_downloaded,
            files_skipped=files_skipped,
            warnings=warnings,
        )

    def fetch_files(
        self,
        *,
        target: str | int,
        selectors: ExoFopSelectors | None = None,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        result = self.index_target(target=target, include_summaries=True, force_refresh=force_refresh)
        tic_id = result.target.tic_id
        target_dir = result.cache_root
        files_dir = target_dir / "files"
        archives_dir = target_dir / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)

        rows = self.file_list(tic_id=tic_id, force_refresh=force_refresh)
        selected = self._filter_rows(rows, selectors)
        selected_names = [r.filename for r in selected]

        zip_path = archives_dir / "files.zip"
        if (not zip_path.exists()) or force_refresh:
            zip_url = self._url("download_files_zip.php")
            content = self._get_bytes(zip_url, params={"id": str(int(tic_id))})
            self._write_bytes_atomic(zip_path, content)

        extracted = self._extract_selected(zip_path=zip_path, out_dir=files_dir, names=selected_names)

        warnings = list(result.warnings)
        files_skipped: list[str] = []
        for r in selected:
            if r.filename not in extracted:
                files_skipped.append(r.filename)
        self._write_manifest(
            manifest_path=result.manifest_path,
            target=result.target,
            filelist_path=result.filelist_path,
            summary_paths=result.summary_paths,
            files_dir=files_dir,
            file_rows=rows,
            downloaded_files=sorted(extracted.values()),
            extracted_files=extracted,
            skipped=files_skipped,
            warnings=warnings,
        )

        return ExoFopFetchResult(
            target=result.target,
            cache_root=result.cache_root,
            manifest_path=result.manifest_path,
            filelist_path=result.filelist_path,
            summary_paths=result.summary_paths,
            files_downloaded=sorted(extracted.values()),
            files_skipped=files_skipped,
            warnings=warnings,
        )

    def fetch(
        self,
        *,
        target: str | int,
        selectors: ExoFopSelectors | None = None,
        include_summaries: bool = True,
        force_refresh: bool = False,
    ) -> ExoFopFetchResult:
        indexed = self.index_target(
            target=target, include_summaries=include_summaries, force_refresh=force_refresh
        )
        if selectors is None:
            return indexed
        return self.fetch_files(target=target, selectors=selectors, force_refresh=force_refresh)

    # -------------------------------------------------------------------------
    # Networking helpers
    # -------------------------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        return self.BASE_URL + endpoint

    def _request(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"request failed: {last_exc}")

    def _get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        resp = self._request("GET", url, params=params)
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def _get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        resp = self._request("GET", url, params=params)
        return resp.content

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        text = self._get_text(url, params=params)
        return json.loads(text)

    # -------------------------------------------------------------------------
    # Cookie jar support
    # -------------------------------------------------------------------------

    def _load_cookie_jar(self, path: Path) -> bool:
        import http.cookiejar

        jar = http.cookiejar.MozillaCookieJar(str(path))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception:
            return False
        for c in jar:
            self.session.cookies.set_cookie(c)
        return True

    # -------------------------------------------------------------------------
    # Parsing / filtering
    # -------------------------------------------------------------------------

    def _parse_filelist_csv(self, text: str, *, fallback_tic_id: int | None = None) -> list[ExoFopFileRow]:
        buf = io.StringIO(text.lstrip("\ufeff"))
        reader = csv.DictReader(buf)
        rows: list[ExoFopFileRow] = []
        for r in reader:
            if not r:
                continue
            filename = (r.get("File Name") or r.get("FileName") or r.get("file_name") or "").strip()
            if not filename:
                continue
            file_id_raw = (r.get("File ID") or r.get("FileID") or r.get("file_id") or "").strip()
            tic_raw = (r.get("TIC") or r.get("TIC ID") or r.get("tic") or "").strip()
            toi = (r.get("TOI") or r.get("toi") or "").strip() or None
            typ = (r.get("Type") or r.get("type") or "").strip() or "Unknown"
            date = (r.get("Date") or r.get("date") or "").strip() or None
            user = (r.get("User") or r.get("user") or "").strip() or None
            group = (r.get("Group") or r.get("group") or "").strip() or None
            tag = (r.get("Tag") or r.get("tag") or "").strip() or None
            desc = (r.get("Description") or r.get("description") or "").strip() or None

            try:
                file_id = int(float(file_id_raw))
            except Exception:
                continue
            if tic_raw:
                try:
                    tic_id = int(float(tic_raw))
                except Exception:
                    tic_id = int(fallback_tic_id) if fallback_tic_id is not None else 0
            else:
                tic_id = int(fallback_tic_id) if fallback_tic_id is not None else 0

            if tic_id <= 0:
                continue

            rows.append(
                ExoFopFileRow(
                    file_id=file_id,
                    tic_id=tic_id,
                    toi=toi,
                    filename=filename,
                    type=typ,
                    date_utc=date,
                    user=user,
                    group=group,
                    tag=tag,
                    description=desc,
                )
            )
        return rows

    def _parse_filelist_target_json(self, payload: dict[str, Any], *, fallback_tic_id: int) -> list[ExoFopFileRow]:
        files_block = payload.get("files")
        if not isinstance(files_block, list):
            return []
        rows: list[ExoFopFileRow] = []
        for item in files_block:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("fname") or "").strip()
            if not filename:
                continue
            fid_raw = item.get("fid")
            if fid_raw is None:
                continue
            try:
                file_id = int(float(fid_raw))
            except Exception:
                continue
            tic_value = item.get("tic_id") or item.get("tic") or fallback_tic_id
            try:
                tic_id = int(float(tic_value))
            except Exception:
                tic_id = int(fallback_tic_id)
            if tic_id <= 0:
                continue
            rows.append(
                ExoFopFileRow(
                    file_id=file_id,
                    tic_id=tic_id,
                    toi=(str(item.get("ftoi")).strip() or None),
                    filename=filename,
                    type=(str(item.get("ftype") or "Unknown").strip() or "Unknown"),
                    date_utc=(str(item.get("fdate")).strip() or None),
                    user=(str(item.get("fuser")).strip() or None),
                    group=(str(item.get("fgroup")).strip() or None),
                    tag=(str(item.get("ftag")).strip() or None),
                    description=(str(item.get("fdesc")).strip() or None),
                )
            )
        return rows

    def _render_filelist_csv(self, rows: list[ExoFopFileRow]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "Type",
                "File Name",
                "File ID",
                "TIC",
                "TOI",
                "Date",
                "User",
                "Group",
                "Tag",
                "Description",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.type,
                    row.filename,
                    row.file_id,
                    row.tic_id,
                    row.toi or "",
                    row.date_utc or "",
                    row.user or "",
                    row.group or "",
                    row.tag or "",
                    row.description or "",
                ]
            )
        return buf.getvalue()

    def _normalize_table_key(self, key: str) -> str:
        text = str(key or "").strip().lower()
        text = text.replace("%", "pct")
        for ch in ("(", ")", "[", "]", "{", "}", ",", ":", ";", "/", "-", "."):
            text = text.replace(ch, " ")
        return "_".join(part for part in text.split() if part)

    def _iter_structured_rows(self, text: str) -> list[dict[str, str]]:
        body = str(text or "").lstrip("\ufeff").replace("\x00", "")
        stripped = body.lstrip().lower()
        if stripped.startswith("<html") or stripped.startswith("<!doctype html"):
            return []
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if not lines:
            return []

        header_line = lines[0]
        delimiter = ","
        header_scores = {candidate: header_line.count(candidate) for candidate in [",", "|", "\t", ";"]}
        best_header_delim = max(header_scores, key=lambda candidate: header_scores[candidate])
        if header_scores.get(best_header_delim, 0) > 0:
            delimiter = best_header_delim
        else:
            sample = "\n".join(lines[:20])
            delimiters = ",|\t;"
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
                delimiter = dialect.delimiter
            except Exception:
                delimiter = self._best_delimiter(lines, candidates=[",", "|", "\t", ";"])

        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter)
        out: list[dict[str, str]] = []
        has_headers = bool(reader.fieldnames) and len([h for h in (reader.fieldnames or []) if h]) > 1
        if has_headers:
            for row in reader:
                if not row:
                    continue
                cleaned: dict[str, str] = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    nk = self._normalize_table_key(k)
                    if not nk:
                        continue
                    cleaned[nk] = str(v).strip() if v is not None else ""
                if cleaned:
                    out.append(cleaned)
            if out:
                return out

        return self._parse_headered_rows_with_delimiter(lines, delimiter=delimiter)

    def _best_delimiter(self, lines: list[str], *, candidates: list[str]) -> str:
        best = ","
        best_score = -1.0
        for candidate in candidates:
            counts = [len(line.split(candidate)) for line in lines[:8]]
            if not counts:
                continue
            if max(counts) < 2:
                continue
            score = float(sum(counts)) / float(len(counts))
            if score > best_score:
                best_score = score
                best = candidate
        return best

    def _parse_headered_rows_with_delimiter(self, lines: list[str], *, delimiter: str) -> list[dict[str, str]]:
        if not lines:
            return []
        header = [self._normalize_table_key(h) for h in lines[0].split(delimiter)]
        header = [h for h in header if h]
        if not header:
            return []
        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            cols = [c.strip() for c in line.split(delimiter)]
            if not cols:
                continue
            row: dict[str, str] = {}
            for idx, key in enumerate(header):
                row[key] = cols[idx] if idx < len(cols) else ""
            if any(str(v).strip() for v in row.values()):
                rows.append(row)
        return rows

    def _find_first_value(self, row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            value = row.get(alias)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _parse_int_best_effort(self, value: str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            return None

    def _parse_obsnotes_text(self, text: str, *, fallback_tic_id: int | None = None) -> list[ExoFopObsNoteRow]:
        rows = self._iter_structured_rows(text)
        out: list[ExoFopObsNoteRow] = []
        for row in rows:
            text_value = self._find_first_value(
                row,
                aliases=(
                    "text",
                    "notes",
                    "note",
                    "obs_note",
                    "obsnotes",
                    "observation_note",
                    "comments",
                    "comment",
                    "description",
                ),
            )
            if text_value is None:
                continue

            author = self._find_first_value(
                row,
                aliases=("author", "user", "username", "observer", "submitter", "name"),
            )
            date_utc = self._find_first_value(
                row,
                aliases=("date", "date_utc", "created_at", "create_date", "obs_date", "upload_date", "lastmod"),
            )
            data_tag = self._find_first_value(
                row,
                aliases=("data_tag", "tag", "tag_id", "tagid", "datatype", "data_type", "category", "table"),
            )
            group = self._find_first_value(row, aliases=("group", "groupname", "team", "program"))
            note_id = self._parse_int_best_effort(
                self._find_first_value(row, aliases=("note_id", "id", "obsnote_id"))
            )
            tic_id = self._parse_int_best_effort(
                self._find_first_value(row, aliases=("tic", "tic_id", "target_tic", "target"))
            )
            if tic_id is None:
                tic_id = int(fallback_tic_id) if fallback_tic_id is not None else 0
            if tic_id <= 0:
                continue

            out.append(
                ExoFopObsNoteRow(
                    tic_id=tic_id,
                    author=author,
                    date_utc=date_utc,
                    data_tag=data_tag,
                    text=text_value,
                    note_id=note_id,
                    group=group,
                    raw=dict(row),
                )
            )
        return out

    def _parse_toi_table_rows(self, text: str) -> list[dict[str, str]]:
        return self._iter_structured_rows(text)

    def _extract_toi_row(
        self,
        *,
        text: str,
        tic_id: int,
        toi: str | float | None,
    ) -> ExoFopToiRow | None:
        rows = self._parse_toi_table_rows(text)
        tic_target = str(int(tic_id))
        toi_target = self._normalize_toi_value(toi)

        selected: dict[str, str] | None = None
        for row in rows:
            row_tic = self._find_first_value(row, aliases=("tic_id", "tic", "ticid"))
            if row_tic is None:
                continue
            row_tic_int = self._parse_int_best_effort(row_tic)
            if row_tic_int is None or str(int(row_tic_int)) != tic_target:
                continue
            if toi_target is not None:
                row_toi = self._normalize_toi_value(
                    self._find_first_value(row, aliases=("toi", "toi_id", "toi_number"))
                )
                if row_toi != toi_target:
                    continue
            selected = row
            break

        if selected is None:
            return None

        toi_value = self._find_first_value(selected, aliases=("toi", "toi_id", "toi_number"))
        tfopwg = self._find_first_value(
            selected,
            aliases=("tfopwg_disposition", "tfopwg_disp", "tfopwg", "disposition"),
        )
        planet_disp = self._find_first_value(
            selected,
            aliases=("planet_disposition", "disp", "disposition", "disposition_group"),
        )
        comments = self._find_first_value(selected, aliases=("comments", "comment", "notes", "remarks"))
        return ExoFopToiRow(
            tic_id=int(tic_id),
            toi=toi_value,
            tfopwg_disposition=tfopwg,
            planet_disposition=planet_disp,
            comments=comments,
            raw=dict(selected),
        )

    def _extract_toi_row_from_target_json(
        self,
        *,
        payload: dict[str, Any],
        tic_id: int,
        toi: str | float | None,
    ) -> ExoFopToiRow | None:
        rows = payload.get("tois")
        if not isinstance(rows, list):
            return None
        toi_target = self._normalize_toi_value(toi)
        selected: dict[str, Any] | None = None
        for item in rows:
            if not isinstance(item, dict):
                continue
            row_toi = self._normalize_toi_value(item.get("toi"))
            if toi_target is not None and row_toi != toi_target:
                continue
            selected = item
            break
        if selected is None:
            return None
        return ExoFopToiRow(
            tic_id=int(tic_id),
            toi=(str(selected.get("toi")).strip() or None),
            tfopwg_disposition=(str(selected.get("disp_tfop")).strip() or None),
            planet_disposition=(str(selected.get("disp_tess")).strip() or None),
            comments=(str(selected.get("notes")).strip() or None),
            raw={str(k): str(v) for k, v in selected.items()},
        )

    def _normalize_toi_value(self, value: str | float | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        text = text.replace("TOI-", "").replace("TOI", "")
        try:
            numeric = float(text)
            return f"{numeric:.2f}"
        except Exception:
            return text

    def _filter_rows(
        self, rows: list[ExoFopFileRow], selectors: ExoFopSelectors | None
    ) -> list[ExoFopFileRow]:
        if selectors is None:
            return rows
        out = rows
        if selectors.types:
            allowed = {t.lower() for t in selectors.types}
            out = [r for r in out if r.type.lower() in allowed]
        if selectors.tag_ids:
            allowed_tags = {int(t) for t in selectors.tag_ids}

            def _tag_ok(tag: str | None) -> bool:
                if tag is None:
                    return False
                try:
                    return int(float(tag)) in allowed_tags
                except Exception:
                    return False

            out = [r for r in out if _tag_ok(r.tag)]
        if selectors.filename_regex:
            rx = re.compile(selectors.filename_regex)
            out = [r for r in out if rx.search(r.filename)]
        if selectors.max_files is not None and selectors.max_files >= 0:
            out = out[: int(selectors.max_files)]
        return out

    # -------------------------------------------------------------------------
    # Files / extraction / hashing
    # -------------------------------------------------------------------------

    def _extract_selected(self, *, zip_path: Path, out_dir: Path, names: list[str]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not names:
            return out
        with zipfile.ZipFile(zip_path, "r") as zf:
            member_names = list(zf.namelist())
            members_by_name = {name: name for name in member_names}
            members_by_basename: dict[str, list[str]] = {}
            for member_name in member_names:
                members_by_basename.setdefault(Path(member_name).name, []).append(member_name)
            basename_counts: dict[str, int] = {}
            for want in names:
                basename_counts[Path(want).name] = basename_counts.get(Path(want).name, 0) + 1
            for want in names:
                member = members_by_name.get(want)
                if member is None:
                    candidates = members_by_basename.get(Path(want).name, [])
                    if len(candidates) == 1:
                        member = candidates[0]
                    elif len(candidates) > 1:
                        member = sorted(candidates, key=lambda item: (len(item), item))[0]
                if member is None:
                    continue
                basename = Path(want).name
                if basename_counts.get(basename, 0) > 1:
                    stem = Path(member).as_posix().replace("/", "__").replace("\\", "__")
                    dest = out_dir / stem
                else:
                    dest = out_dir / basename
                if dest.exists() and dest.stat().st_size > 0:
                    out[want] = dest
                    continue
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                os.replace(tmp, dest)
                out[want] = dest
        return out

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # -------------------------------------------------------------------------
    # Cache layout / manifest
    # -------------------------------------------------------------------------

    def _target_dir(self, tic_id: int) -> Path:
        return self.cache_dir / "exofop" / f"tic_{int(tic_id)}"

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_bytes_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _ext_for_output(self, output: Any) -> str:
        out = str(output or "").lower()
        if out == "csv":
            return "csv"
        if out == "pipe":
            return "pipe"
        return "txt"

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        target: ExoFopTarget,
        filelist_path: Path,
        summary_paths: dict[str, Path],
        files_dir: Path,
        file_rows: list[ExoFopFileRow],
        downloaded_files: Iterable[Path],
        extracted_files: dict[str, Path] | None = None,
        skipped: list[str],
        warnings: list[str],
    ) -> None:
        files_meta: list[dict[str, Any]] = []
        downloaded_set = {Path(p).name for p in downloaded_files}
        for row in file_rows:
            local_path: Path
            if extracted_files is not None and row.filename in extracted_files:
                local_path = extracted_files[row.filename]
            else:
                local_path = files_dir / Path(row.filename).name
            entry: dict[str, Any] = {
                "file_id": row.file_id,
                "filename": row.filename,
                "type": row.type,
                "date_utc": row.date_utc,
                "tag": row.tag,
                "description": row.description,
            }
            if local_path.exists() and local_path.name in downloaded_set:
                try:
                    entry["path"] = str(local_path.relative_to(manifest_path.parent))
                except Exception:
                    entry["path"] = str(local_path)
                entry["bytes"] = int(local_path.stat().st_size)
                entry["sha256"] = self._sha256(local_path)
            files_meta.append(entry)

        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "target": asdict(target),
            "auth": {"cookie_jar_used": bool(self.cookie_jar_path), "cookie_loaded": self.cookie_jar_loaded},
            "sources": {"base_url": self.BASE_URL},
            "filelist": {"path": str(filelist_path.relative_to(manifest_path.parent)), "n_rows": len(file_rows)},
            "summaries": {k: str(v.relative_to(manifest_path.parent)) for k, v in summary_paths.items()},
            "files": files_meta,
            "skipped": skipped,
            "warnings": warnings,
            "errors": [],
        }
        self._write_text_atomic(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
