"""MCP server exposing BridgeWebs bridge game postmortem data.

Transport: streamable HTTP (endpoint /mcp) on BRIDGEWEBS_POSTMORTEM_MCP_PORT
(default 8514), stateless with JSON responses so plain HTTP clients and
cloudflared work without session affinity. Same pattern as
Bridge_Game_Postmortem_Chatbot/acbl_postmortem_mcp_server.py.

Every tool calls the first-party BridgeWebs REST API. This MCP process does
not import report libraries, read parquet, scrape, or call third-party APIs.

Deployment: bridgewebs-mcp container, started by
../7nt/postmortem_start.ps1. GET /health is used by the wslc watchdog and
deploy health checks.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

import requests

BRIDGEWEBS_POSTMORTEM_MCP_PORT = int(os.environ.get("BRIDGEWEBS_POSTMORTEM_MCP_PORT", "8514"))
BRIDGEWEBS_POSTMORTEM_API_BASE_URL = os.environ.get(
    "BRIDGEWEBS_POSTMORTEM_API_BASE_URL", "http://127.0.0.1:8521"
).rstrip("/")
_TIMEOUT_S = 300

mcp = MCPServer("bridgewebs-postmortem")


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(
        f"{BRIDGEWEBS_POSTMORTEM_API_BASE_URL}{path}",
        params={key: value for key, value in (params or {}).items() if value is not None},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{BRIDGEWEBS_POSTMORTEM_API_BASE_URL}{path}",
        json=payload,
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for the wslc watchdog / deploy health check."""
    return JSONResponse({"service": "bridgewebs-mcp", "api": _get("/health")})


@mcp.tool()
def bridgewebs_postmortem_dataset_info() -> Dict[str, Any]:
    """Summary of the BridgeWebs postmortem cache: how many postmortems are
    cached, for which clubs, and how new ones get generated."""
    return _get("/bridgewebs/dataset-info")


@mcp.tool()
def bridgewebs_postmortem_sessions(club: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """List cached BridgeWebs postmortem sessions (newest first), optionally
    for one club. Each entry has club, session_id (BridgeWebs event id), and
    cache timestamps. A session must appear here before the
    players/boards/sql/schema tools can query it."""
    return _get(
        "/bridgewebs/sessions",
        {"club": club, "limit": max(1, min(limit, 500))},
    )


@mcp.tool()
def bridgewebs_postmortem_players(
    club: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Player names appearing in one cached BridgeWebs postmortem (newest
    session when club/session_id are omitted). BridgeWebs identifies players
    by NAME, so use these values as player_name in the boards/sql tools."""
    return _get(
        "/bridgewebs/players",
        {"club": club, "session_id": session_id},
    )


@mcp.tool()
def bridgewebs_postmortem_boards(
    player_name: str,
    club: Optional[str] = None,
    session_id: Optional[str] = None,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Per-board results for one cached BridgeWebs postmortem, personalized
    for a player (by name; discover names with bridgewebs_postmortem_players):
    contract, declarer, result, tricks, scores, matchpoint percentages, and
    the deal (PBN).

    club/session_id: omit for the most recently cached game.
    only_my_boards: True (default) limits rows to boards the player's pair
    actually played; False returns every board in the game (all pairs).
    columns: optional comma-separated column names to override the default
    summary set (discover names with bridgewebs_postmortem_schema).
    """
    return _get(
        "/bridgewebs/boards",
        {
            "player_name": player_name,
            "club": club,
            "session_id": session_id,
            "only_my_boards": only_my_boards,
            "columns": columns,
            "limit": limit,
        },
    )


@mcp.tool()
def bridgewebs_postmortem_sql(
    player_name: str,
    sql: str,
    club: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Run a DuckDB SQL query against one cached BridgeWebs postmortem,
    registered as table 'self' (one row per board result, thousands of
    augmented columns: double-dummy, par, single-dummy expected values, HCP,
    ...), personalized for a player (by name).

    'FROM self' is prepended when the query does not reference it, so both
    'SELECT ... FROM self ...' and DuckDB's 'SELECT ...' shorthand work.
    Personalization macros are substituted before execution:
    {Player_Direction}, {Partner_Direction}, {Pair_Direction},
    {Opponent_Pair_Direction}. Boolean helper columns include Boards_I_Played,
    Boards_I_Declared, Boards_We_Declared, Boards_Opponent_Declared.
    club/session_id: omit for the most recently cached game.
    """
    return _post(
        "/bridgewebs/sql",
        {
            "player_name": player_name,
            "sql": sql,
            "club": club,
            "session_id": session_id,
            "limit": limit,
        },
    )


@mcp.tool()
def bridgewebs_postmortem_schema(
    club: Optional[str] = None,
    session_id: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Column names and dtypes of one cached BridgeWebs postmortem dataframe.
    The frame has thousands of augmented columns, so pass a case-insensitive
    regex pattern (e.g. 'Pct|Score', '^DD_', 'PairId') to search for relevant
    ones before writing bridgewebs_postmortem_sql queries."""
    return _get(
        "/bridgewebs/schema",
        {
            "club": club,
            "session_id": session_id,
            "pattern": pattern,
            "limit": limit,
        },
    )


if __name__ == "__main__":
    print(
        f"[bridgewebs-mcp] start {datetime.now(timezone.utc).isoformat()} "
        f"on :{BRIDGEWEBS_POSTMORTEM_MCP_PORT}; "
        f"api -> {BRIDGEWEBS_POSTMORTEM_API_BASE_URL}",
        flush=True,
    )
    # Stateless + JSON responses: plain request/response tools, no session
    # affinity needed behind cloudflared, and curl-testable.
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=BRIDGEWEBS_POSTMORTEM_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
