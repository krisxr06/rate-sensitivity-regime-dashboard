"""
Data loading module for FRED rate series.

Fetch strategy (in priority order):
1. Local parquet cache (if fresh, < CACHE_MAX_AGE_HOURS old)
2. fredapi library  (if FRED_API_KEY env var is set and fredapi installed)
3. FRED public CSV  (always-available fallback)

Note on DGS30: the 30-year constant maturity series was discontinued Feb 2002
and restarted Feb 2006. The resulting gap is preserved as NaN — it is not
interpolated — and analyses that include 30Y automatically exclude those months.
"""

import os
import logging
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR           = Path(__file__).parent.parent / "data" / "processed"
FRED_CSV_URL       = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CACHE_MAX_AGE_HOURS = 24

SERIES_IDS = ["DGS2", "DGS10", "DGS30", "FEDFUNDS"]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(series_id: str) -> Path:
    return DATA_DIR / f"{series_id.lower()}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=CACHE_MAX_AGE_HOURS)


# ── Fetch helpers ──────────────────────────────────────────────────────────────

def _fetch_via_csv(series_id: str) -> pd.Series:
    url = f"{FRED_CSV_URL}?id={series_id}"
    logger.info("Fetching %s from FRED public CSV", series_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.index.name = "DATE"
    df.columns = [series_id]
    df[series_id] = df[series_id].replace(".", pd.NA)
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df[series_id]


def _fetch_via_api(series_id: str, api_key: str) -> pd.Series:
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        logger.info("Fetching %s via FRED API", series_id)
        s = fred.get_series(series_id)
        s.name = series_id
        return s
    except ImportError:
        logger.warning("fredapi not installed — falling back to CSV")
        return _fetch_via_csv(series_id)
    except Exception as exc:
        logger.warning("FRED API failed for %s (%s) — falling back to CSV", series_id, exc)
        return _fetch_via_csv(series_id)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_series(series_id: str, force_refresh: bool = False) -> pd.Series:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(series_id)

    if not force_refresh and not _is_stale(cache):
        logger.info("Loading %s from cache", series_id)
        return pd.read_parquet(cache)[series_id]

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    series  = _fetch_via_api(series_id, api_key) if api_key else _fetch_via_csv(series_id)
    series.name = series_id
    pd.DataFrame(series).to_parquet(cache)
    return series


def load_all(force_refresh: bool = False) -> pd.DataFrame:
    """Load all four FRED series into a single wide DataFrame."""
    frames = [load_series(sid, force_refresh=force_refresh).rename(sid) for sid in SERIES_IDS]
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
