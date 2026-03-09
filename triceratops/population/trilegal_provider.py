"""TRILEGAL web service provider.

Ports funcs.query_TRILEGAL() and funcs.save_trilegal() from funcs.py:296-380.
Fixes BUG-07: raises TRILEGALQueryError on failure instead of returning 0.0.
"""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
from mechanicalsoup import StatefulBrowser

from triceratops.population.protocols import TRILEGALResult
from triceratops.population.trilegal_parser import parse_trilegal_csv


class TRILEGALQueryError(RuntimeError):
    """Raised when a TRILEGAL web query fails for any reason.

    Replaces the silent ``return 0.0`` in funcs.py:368 (BUG-07).
    """


# TRILEGAL form URLs -- try v1.6 first, fall back to v1.5
_TRILEGAL_URLS = [
    "http://stev.oapd.inaf.it/cgi-bin/trilegal_1.6",
    "http://stev.oapd.inaf.it/cgi-bin/trilegal_1.5",
]
_TRILEGAL_BASE = "http://stev.oapd.inaf.it/"
_PHOTSYS = "tab_mag_odfnew/tab_mag_TESS_2mass_kepler.dat"


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

        links = browser.get_current_page().select("a")
        if links:
            data_link = links[0].get("href")
            return _TRILEGAL_BASE + data_link[3:]

    return None


def _download_and_save(output_url: str, cache_path: Path) -> Path:
    """Poll the TRILEGAL results URL until complete, then save CSV.

    Ports funcs.save_trilegal() (funcs.py:354-380).
    """
    for _ in range(1000):
        last = pd.read_csv(output_url, header=None).iloc[-1:]
        if last.values[0, 0] == "#TRILEGAL normally terminated":
            break
        sleep(10)
    else:
        raise TRILEGALQueryError(
            f"TRILEGAL query did not complete after polling: {output_url}"
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
    ) -> TRILEGALResult:
        """Submit a TRILEGAL web query and return parsed results.

        If cache_path exists, loads from disk without a web request.
        If cache_path is given but does not exist, saves results there.

        Raises:
            TRILEGALQueryError: On any web or parsing failure.
        """
        if cache_path is not None and cache_path.exists():
            return parse_trilegal_csv(cache_path, target_tmag)

        try:
            output_url = _submit_trilegal_form(ra_deg, dec_deg)
            if output_url is None:
                raise TRILEGALQueryError(
                    f"TRILEGAL service unavailable for ra={ra_deg}, dec={dec_deg}"
                )
            save_to = cache_path or Path(f"trilegal_{ra_deg:.4f}_{dec_deg:.4f}.csv")
            _download_and_save(output_url, save_to)
            return parse_trilegal_csv(save_to, target_tmag)
        except TRILEGALQueryError:
            raise
        except Exception as exc:
            raise TRILEGALQueryError(
                f"TRILEGAL query failed for ra={ra_deg}, dec={dec_deg}: {exc}"
            ) from exc
