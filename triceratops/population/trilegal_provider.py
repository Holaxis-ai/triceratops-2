"""TRILEGAL web service provider.

Ports funcs.query_TRILEGAL() and funcs.save_trilegal() from funcs.py:296-380.
Fixes BUG-07: raises TRILEGALQueryError on failure instead of returning 0.0.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from time import sleep

import pandas as pd
from mechanicalsoup import StatefulBrowser
from pandas.errors import EmptyDataError

from triceratops.population.protocols import TRILEGALResult
from triceratops.population.trilegal_parser import parse_trilegal_csv


class TRILEGALQueryError(RuntimeError):
    """Raised when a TRILEGAL web query fails for any reason.

    Replaces the silent ``return 0.0`` in funcs.py:368 (BUG-07).
    """


# TRILEGAL form URLs -- try v1.6 first, fall back to v1.5
_TRILEGAL_URLS = [
    "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6",
    "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.5",
]
_TRILEGAL_BASE = "https://stev.oapd.inaf.it/"
_PHOTSYS = "tab_mag_odfnew/tab_mag_TESS_2mass_kepler.dat"


def _default_cache_dir() -> Path:
    """Return a platform-appropriate cache directory for TRILEGAL results.

    Respects ``$XDG_CACHE_HOME`` when set, otherwise falls back to
    ``~/.cache/triceratops/trilegal/``.  The directory is created if it does
    not already exist.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = cache / "triceratops" / "trilegal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _submit_trilegal_form(ra_deg: float, dec_deg: float) -> str | None:
    """Submit the TRILEGAL web form and return the output URL, or None on failure.

    Ports funcs.query_TRILEGAL() (funcs.py:296-351).
    """
    for url in _TRILEGAL_URLS:
        browser = StatefulBrowser()
        browser.open(url)
        browser.select_form(nr=0)
        browser["gal_coord"] = "2"
        browser["eq_alpha"] = str(ra_deg)
        browser["eq_delta"] = str(dec_deg)
        browser["field"] = "0.1"
        browser["photsys_file"] = _PHOTSYS
        browser["icm_lim"] = "1"
        browser["mag_lim"] = "21"
        browser["binary_kind"] = "0"
        browser.submit_selected()
        sleep(5)

        page = browser.get_current_page()
        if page is None:
            continue
        links = page.select("a")
        if links:
            data_link = links[0].get("href")
            return _TRILEGAL_BASE + data_link[3:]

    return None


def _poll_until_complete(
    url: str,
    timeout_s: float = 600,
    interval_s: float = 10,
) -> None:
    """Poll *url* until TRILEGAL signals normal termination.

    Parameters
    ----------
    url:
        The TRILEGAL output URL to poll.
    timeout_s:
        Maximum seconds to wait before raising :class:`TimeoutError`.
        Default is 600 (10 minutes).
    interval_s:
        Seconds to sleep between polls.  Default is 10.

    Raises
    ------
    TimeoutError
        If the query does not complete within *timeout_s* seconds.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            last = pd.read_csv(url, header=None).iloc[-1:]
        except EmptyDataError:
            time.sleep(interval_s)
            continue
        if last.values[0, 0] == "#TRILEGAL normally terminated":
            return
        time.sleep(interval_s)
    raise TimeoutError(
        f"TRILEGAL query did not complete within {timeout_s}s. "
        f"Check {url} manually."
    )


def _download_and_save(
    output_url: str,
    cache_path: Path,
    poll_timeout_seconds: float = 600,
    poll_interval_seconds: float = 10,
) -> Path:
    """Poll the TRILEGAL results URL until complete, then save CSV.

    Ports funcs.save_trilegal() (funcs.py:354-380).
    """
    _poll_until_complete(
        output_url,
        timeout_s=poll_timeout_seconds,
        interval_s=poll_interval_seconds,
    )

    df = pd.read_csv(output_url, sep=r"\s+")
    df.to_csv(cache_path, index=False)
    return cache_path


class TRILEGALProvider:
    """Queries the TRILEGAL web service for background star populations.

    Ports funcs.save_trilegal() + funcs.query_TRILEGAL().
    Fixes BUG-07: raises TRILEGALQueryError on failure.
    """

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        target_tmag: float,
        cache_path: Path | None = None,
        poll_timeout_seconds: float = 600,
        poll_interval_seconds: float = 10,
    ) -> TRILEGALResult:
        """Submit a TRILEGAL web query and return parsed results.

        If cache_path exists, loads from disk without a web request.
        If cache_path is given but does not exist, saves results there.

        Parameters
        ----------
        ra_deg, dec_deg:
            Sky coordinates of the target field.
        target_tmag:
            TESS magnitude of the target star (used for filtering).
        cache_path:
            Optional explicit path for the cached CSV.  When *None* a
            platform-appropriate default is chosen automatically so that
            results are never written to the current working directory.
        poll_timeout_seconds:
            Maximum time (seconds) to wait for TRILEGAL to finish.
            Default: 600 (10 minutes).
        poll_interval_seconds:
            Polling cadence in seconds.  Default: 10.

        Raises
        ------
        TRILEGALQueryError:
            On any web or parsing failure (including timeout).
        """
        if cache_path is not None and cache_path.exists():
            return parse_trilegal_csv(cache_path, target_tmag)

        if cache_path is None:
            cache_dir = _default_cache_dir()
            cache_path = cache_dir / f"trilegal_{ra_deg:.4f}_{dec_deg:.4f}.csv"

        try:
            output_url = _submit_trilegal_form(ra_deg, dec_deg)
            if output_url is None:
                raise TRILEGALQueryError(
                    f"TRILEGAL service unavailable for ra={ra_deg}, dec={dec_deg}"
                )
            save_to = cache_path
            _download_and_save(
                output_url,
                save_to,
                poll_timeout_seconds=poll_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            return parse_trilegal_csv(cache_path, target_tmag)
        except TRILEGALQueryError:
            raise
        except TimeoutError as exc:
            raise TRILEGALQueryError(
                f"TRILEGAL query timed out for ra={ra_deg}, dec={dec_deg}: {exc}"
            ) from exc
        except Exception as exc:
            raise TRILEGALQueryError(
                f"TRILEGAL query failed for ra={ra_deg}, dec={dec_deg}: {exc}"
            ) from exc
