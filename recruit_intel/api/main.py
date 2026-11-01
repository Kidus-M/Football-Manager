"""HTTP API for the dashboard: `uvicorn recruit_intel.api.main:app`.

Read-only over the DuckDB file; every route is a thin wrapper over
api/service.py, which reuses the same analysis code as the CLI.
"""

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from recruit_intel.api import service

app = FastAPI(title="Recruitment intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _models_needed(exc: Exception) -> HTTPException:
    return HTTPException(503, f"{exc}. Install the ml extra and run `recruit train`.")


@app.get("/api/club")
def club():
    return service.club_info()


@app.get("/api/formations")
def formations():
    return service.formations()


@app.get("/api/lineup")
def lineup(formation: str | None = None):
    try:
        return service.lineup_view(formation)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/squad")
def squad():
    return service.squad()


@app.get("/api/players")
def find_players(
    q: Annotated[str, Query(max_length=60)] = "", limit: Annotated[int, Query(le=50)] = 12
):
    return service.find(q, limit)


@app.get("/api/players/{player_id}")
def player(player_id: str):
    found = service.player(player_id)
    if found is None:
        raise HTTPException(404, f"no rated player {player_id}")
    return found


@app.get("/api/search")
def search(
    slot: str,
    formation: str | None = None,
    max_age: Annotated[int | None, Query(ge=16, le=40)] = None,
    max_value_m: Annotated[float | None, Query(gt=0)] = None,
    contract_until: int | None = None,
    leagues: Annotated[list[str] | None, Query()] = None,
    min_minutes: Annotated[int | None, Query(ge=0)] = None,
    upgrades: bool = False,
    limit: Annotated[int, Query(le=100)] = 20,
):
    try:
        return service.search(slot, formation, max_age, max_value_m, contract_until, leagues,
                              min_minutes, upgrades, limit)  # fmt: skip
    except KeyError as exc:
        raise HTTPException(404, f"no slot {slot} in the formation") from exc


@app.get("/api/tactics")
def tactics(press: str | None = None, buildup: str | None = None, width: str | None = None):
    try:
        return service.tactics_view(press, buildup, width)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/map")
def similarity_map():
    try:
        return service.similarity_map()
    except (ImportError, FileNotFoundError) as exc:
        raise _models_needed(exc) from exc


@app.get("/api/similar/{player_id}")
def similar(player_id: str, younger: bool = False, cheaper: bool = False,
            same_position: bool = False, limit: Annotated[int, Query(le=50)] = 12):  # fmt: skip
    try:
        return service.similar(player_id, younger, cheaper, same_position, limit)
    except (ImportError, FileNotFoundError) as exc:
        raise _models_needed(exc) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
