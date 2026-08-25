"""First-party REST API for cached BridgeWebs postmortems."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import bridgewebs_postmortem_service as service


app = FastAPI(title="BridgeWebs Postmortem API", version="1.0.0")


class SqlRequest(BaseModel):
    player_name: str
    sql: str
    club: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 500


def _run(callable_, /, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    info = _run(service.dataset_info)
    return {"status": "ok", "service": "bridgewebs-postmortem-api", **info}


@app.get("/bridgewebs/dataset-info")
def dataset_info() -> dict:
    return _run(service.dataset_info)


@app.get("/bridgewebs/sessions")
def sessions(
    club: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    rows = _run(service.list_cached_postmortems, club)[:limit]
    return {"sessions": rows, "count": len(rows)}


@app.get("/bridgewebs/players")
def players(
    club: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
) -> dict:
    frame, meta = _run(service.load_postmortem, None, club, session_id)
    names = _run(service.player_names, frame)
    return {"meta": meta, "players": names, "count": len(names)}


@app.get("/bridgewebs/boards")
def boards(
    player_name: str,
    club: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    only_my_boards: bool = Query(True),
    columns: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=5000),
) -> dict:
    frame, meta = _run(service.load_postmortem, player_name, club, session_id)
    selected = [column.strip() for column in columns.split(",")] if columns else None
    return _run(
        service.board_results,
        frame,
        meta,
        only_my_boards=only_my_boards,
        columns=selected,
        limit=limit,
    )


@app.post("/bridgewebs/sql")
def sql(request: SqlRequest) -> dict:
    frame, meta = _run(
        service.load_postmortem,
        request.player_name,
        request.club,
        request.session_id,
    )
    result = _run(service.run_sql, frame, request.sql, meta, limit=request.limit)
    result["meta"] = meta
    return result


@app.get("/bridgewebs/schema")
def schema(
    club: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    pattern: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
) -> dict:
    frame, _meta = _run(service.load_postmortem, None, club, session_id)
    return _run(service.schema_columns, frame, pattern=pattern, limit=limit)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BRIDGEWEBS_POSTMORTEM_API_PORT", "8521"))
    print(
        f"[bridgewebs-postmortem-api] start "
        f"{datetime.now(timezone.utc).isoformat()} port={port}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
