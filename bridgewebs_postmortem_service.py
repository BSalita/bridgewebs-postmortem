"""Headless access to cached BridgeWebs postmortem dataframes.

The Streamlit app (bridgewebs_postmortem_streamlit.py) persists each fully
augmented UNFILTERED board-results dataframe to
cache/df-{club}-{session}.parquet right after augmentation (see
save_augmented_df_to_cache). This module is the shared, Streamlit-free core
used by bridgewebs_postmortem_mcp_server.py: it enumerates those parquets,
re-derives the per-player flag columns by player NAME (BridgeWebs has no
player ids; same logic as filter_dataframe in the app, driven by the
Player_Name_[NESW] and PairId_NS/EW columns), and runs DuckDB SQL against the
dataframe registered as 'self', mirroring how the app's SQL favorites work.

Env:
  BRIDGEWEBS_POSTMORTEM_CACHE_DIR  cache directory (default ./cache next to this file)
"""

import os
import pathlib
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("BRIDGEWEBS_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache")))

CON_REGISTER_NAME = "self"
DEFAULT_SQL_ROW_LIMIT = 500
MAX_SQL_ROW_LIMIT = 2000
MAX_SCHEMA_COLUMNS = 1000

# df-{club}-{session}.parquet; both tokens are sanitized dash-free by the app.
_CACHE_FILE_RE = re.compile(r"^df-(?P<club>[^-]+)-(?P<session_id>[^-]+)\.parquet$")

# (player_direction, partner_direction, pair_direction, opponent_pair_direction)
_SEAT_TUPLES = (
    ("N", "S", "NS", "EW"),
    ("S", "N", "NS", "EW"),
    ("E", "W", "EW", "NS"),
    ("W", "E", "EW", "NS"),
)

# Default column set for the per-board summary tool; intersected with the
# actual dataframe columns.
BOARD_SUMMARY_COLUMNS = [
    "Board", "Contract", "Declarer_Direction", "Declarer_Name",
    "Result", "Tricks", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
    "MP_NS", "MP_EW", "Par_NS", "ParContract",
    "PairId_NS", "PairId_EW", "PBN",
]


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _fuzzy_score(candidate: object, query: object) -> float:
    haystack = _normalize_text(candidate)
    needle = _normalize_text(query)
    if not haystack or not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    return SequenceMatcher(None, needle, haystack).ratio()


def _fuzzy_matches(candidate: object, query: object, threshold: float = 0.72) -> bool:
    return _fuzzy_score(candidate, query) >= threshold


def _parse_cache_filename(name: str) -> Optional[Dict[str, str]]:
    m = _CACHE_FILE_RE.match(name)
    if m is None:
        return None
    return {"club": m.group("club"), "session_id": m.group("session_id")}


def list_cached_postmortems(club: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cached postmortems (newest file first), optionally for one club."""
    out: List[Dict[str, Any]] = []
    if not CACHE_DIR.is_dir():
        return out
    for f in CACHE_DIR.glob("df-*.parquet"):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        if club is not None and not _fuzzy_matches(parsed["club"], club):
            continue
        stat = f.stat()
        out.append(
            {
                "club": parsed["club"],
                "session_id": parsed["session_id"],
                "file": f.name,
                "size_bytes": stat.st_size,
                "cached_at": stat.st_mtime,
            }
        )
    out.sort(key=lambda d: d["cached_at"], reverse=True)
    return out


def dataset_info() -> Dict[str, Any]:
    cached = list_cached_postmortems()
    return {
        "cache_dir": str(CACHE_DIR),
        "cached_postmortems": len(cached),
        "clubs": sorted({c["club"] for c in cached}),
        "note": (
            "Postmortems are produced on demand by the Streamlit app "
            "(https://bridgewebs.postmortem.chat: enter a BridgeWebs results "
            "URL and select a player); this service reads its parquet cache."
        ),
    }


def _resolve_cache_file(club: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[pathlib.Path, Dict[str, Any]]:
    cached = list_cached_postmortems(club)
    if not cached:
        raise FileNotFoundError(
            f"No cached BridgeWebs postmortem{f' for club {club}' if club else ''}. "
            f"Generate one first by loading https://bridgewebs.postmortem.chat, "
            f"entering the BridgeWebs results URL and selecting a player."
        )
    if session_id is None:
        entry = cached[0]  # newest cache file
    else:
        entry = next((c for c in cached if c["session_id"] == str(session_id)), None)
        if entry is None:
            raise FileNotFoundError(
                f"No cached BridgeWebs postmortem for session {session_id}"
                f"{f' (club {club})' if club else ''}. "
                f"Cached sessions: {[(c['club'], c['session_id']) for c in cached]}"
            )
    return CACHE_DIR / entry["file"], entry


# Small in-process cache keyed by (path, mtime); a few frames resident at most.
_df_cache: Dict[Tuple[str, float], pl.DataFrame] = {}
_df_cache_lock = threading.Lock()
_DF_CACHE_MAX = 4


def _read_parquet_cached(path: pathlib.Path) -> pl.DataFrame:
    key = (str(path), path.stat().st_mtime)
    with _df_cache_lock:
        if key in _df_cache:
            return _df_cache[key]
    df = pl.read_parquet(path)
    with _df_cache_lock:
        if len(_df_cache) >= _DF_CACHE_MAX:
            _df_cache.pop(next(iter(_df_cache)))
        _df_cache[key] = df
    return df


def player_names(df: pl.DataFrame) -> List[str]:
    """Unique player names appearing in the Player_Name_[NESW] columns."""
    names: set = set()
    for seat in "NESW":
        col = f"Player_Name_{seat}"
        if col in df.columns:
            names.update(
                n.strip()
                for n in df[col].drop_nulls().unique().to_list()
                if isinstance(n, str) and n.strip()
            )
    return sorted(names)


def personalize(df: pl.DataFrame, player_name: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Add the player-centric flag columns exactly as filter_dataframe in the
    app does, resolving the player name with typo-tolerant matching."""
    requested_name = str(player_name).strip()
    names = player_names(df)
    ranked = sorted(
        ((_fuzzy_score(name, requested_name), name) for name in names),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.72:
        raise ValueError(
            f"No player name fuzzy match for {requested_name!r}; available: {names}"
        )
    name = ranked[0][1]
    for player_direction, partner_direction, pair_direction, opponent_pair_direction in _SEAT_TUPLES:
        col = f"Player_Name_{player_direction}"
        if col not in df.columns:
            continue
        rows = df.filter(pl.col(col).str.strip_chars().str.to_lowercase() == name.lower())
        if rows.height == 0:
            continue
        pair_number = rows[f"PairId_{pair_direction}"][0]
        partner_name = rows[f"Player_Name_{partner_direction}"][0]
        df = df.with_columns(
            pl.col(f"PairId_{pair_direction}").eq(str(pair_number)).alias("Boards_I_Played"),
        )
        df = df.with_columns(
            pl.col("Boards_I_Played").and_(pl.col("Declarer_Direction").eq(player_direction)).alias("Boards_I_Declared"),
            pl.col("Boards_I_Played").and_(pl.col("Declarer_Direction").eq(partner_direction)).alias("Boards_Partner_Declared"),
        )
        df = df.with_columns(
            pl.col("Boards_I_Played").alias("Boards_We_Played"),
            pl.col("Boards_I_Played").alias("Our_Boards"),
            (pl.col("Boards_I_Declared") | pl.col("Boards_Partner_Declared")).alias("Boards_We_Declared"),
        )
        df = df.with_columns(
            (pl.col("Boards_I_Played") & ~pl.col("Boards_We_Declared") & pl.col("Contract").ne("PASS")).alias("Boards_Opponent_Declared"),
        )
        meta = {
            "requested_player_name": requested_name,
            "player_name": rows[col][0],
            "player_direction": player_direction,
            "partner_name": partner_name,
            "partner_direction": partner_direction,
            "pair_direction": pair_direction,
            "opponent_pair_direction": opponent_pair_direction,
            "pair_number": pair_number,
            "game_date": str(df["Date"].first()) if "Date" in df.columns else None,
        }
        score_col = f"ScorePercent_{pair_direction}"
        if score_col in rows.columns:
            meta["score_percent"] = rows[score_col][0]
        return df, meta
    raise ValueError(
        f"Player {name!r} not found in any Player_Name_[NESW] column of the "
        f"cached postmortem. Discover names with the players tool."
    )


def load_postmortem(
    player_name: Optional[str] = None,
    club: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a cached postmortem (newest when club/session_id are None) and,
    when player_name is given, personalize it. Returns (df, meta)."""
    path, entry = _resolve_cache_file(club, session_id)
    df = _read_parquet_cached(path)
    if player_name is not None:
        df, meta = personalize(df, player_name)
    else:
        meta = {"game_date": str(df["Date"].first()) if "Date" in df.columns else None}
    meta["club"] = entry["club"]
    meta["session_id"] = entry["session_id"]
    meta["cache_file"] = entry["file"]
    return df, meta


def process_sql_macros(sql: str, meta: Dict[str, Any]) -> str:
    """Same substitutions as PostmortemBase.process_prompt_macros."""
    for macro, key in (
        ("{Player_Direction}", "player_direction"),
        ("{Partner_Direction}", "partner_direction"),
        ("{Pair_Direction}", "pair_direction"),
        ("{Opponent_Pair_Direction}", "opponent_pair_direction"),
    ):
        value = meta.get(key)
        if value is not None:
            sql = sql.replace(macro, str(value))
    return sql


def run_sql(df: pl.DataFrame, sql: str, meta: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    """Run DuckDB SQL against the postmortem dataframe registered as 'self'."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    sql = process_sql_macros(sql.strip().rstrip(";"), meta)
    # Same convenience as the app's ShowDataFrameTable: allow DuckDB's
    # FROM-first syntax by prepending the table when it is not referenced.
    if f"from {CON_REGISTER_NAME}" not in sql.lower():
        sql = f"FROM {CON_REGISTER_NAME} " + sql
    # external access off: only the registered dataframe is queryable.
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        con.register(CON_REGISTER_NAME, df)
        result = con.execute(sql).pl()
    finally:
        con.close()
    truncated = result.height > limit
    result = result.head(limit)
    return {
        "sql": sql,
        "columns": result.columns,
        "rows": result.to_dicts(),
        "row_count": result.height,
        "truncated": truncated,
    }


def board_results(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    only_my_boards: bool = True,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-board rows, defaulting to the boards the player actually played."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    if only_my_boards and "Boards_I_Played" in df.columns:
        df = df.filter(pl.col("Boards_I_Played"))
    wanted = columns or BOARD_SUMMARY_COLUMNS
    missing = [c for c in wanted if c not in df.columns]
    selected = [c for c in wanted if c in df.columns]
    if not selected:
        raise ValueError(f"None of the requested columns exist. Missing: {missing}")
    if "Board" in df.columns:
        df = df.sort("Board")
    df = df.select(selected).head(limit)
    return {
        "meta": meta,
        "columns": selected,
        "missing_columns": missing,
        "rows": df.to_dicts(),
        "row_count": df.height,
    }


def schema_columns(df: pl.DataFrame, pattern: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Column names (with dtypes) of the augmented postmortem dataframe,
    optionally filtered by a case-insensitive regex. The frame has thousands
    of columns, hence the cap."""
    limit = max(1, min(limit or MAX_SCHEMA_COLUMNS, MAX_SCHEMA_COLUMNS))
    names = sorted(df.columns)
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        names = [c for c in names if rx.search(c)]
    truncated = len(names) > limit
    names = names[:limit]
    dtypes = dict(zip(df.columns, (str(t) for t in df.dtypes)))
    return {
        "total_columns": df.width,
        "matched_columns": len(names),
        "truncated": truncated,
        "columns": {c: dtypes[c] for c in names},
    }
