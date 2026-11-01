# Football Recruitment Intelligence Platform

A club-agnostic recruitment and squad-analysis platform: point it at any club, and it profiles the current squad, finds the gaps against the club's tactical system, and searches the market for players who fill them. The first configuration is Manchester United.

Runs locally on free data. A personal research project, not a commercial product.

## Quick start

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/recruit ingest          # FPL, Understat (club's pool), ClubElo, club's Transfermarkt squad
.venv/Scripts/recruit ingest --market-pool ENG   # + every Transfermarkt squad in those leagues (slow first run)
.venv/Scripts/recruit squad           # current squad: value, contract, per-90 stats
.venv/Scripts/pytest
```

Club is chosen with `--club <file stem>` (default `man_utd`, from `config/clubs/`).

## Layout

| Path | Purpose |
| --- | --- |
| `config/clubs/` | one YAML per club: formation, roles, recruitment pool, source names, theme |
| `config/roles.yaml` | shared role library (weighted per-90 stat profiles) |
| `recruit_intel/ingest/` | one adapter per source, sharing a cached, rate-limited HTTP client |
| `recruit_intel/identity/` | cross-source player matching |
| `recruit_intel/features/` | metric catalogue, per-90, percentiles, league adjustment |
| `recruit_intel/models/` | embeddings, roles, projection, value, fee range |
| `recruit_intel/analysis/` | squad profile, gap analysis, tactics |
| `recruit_intel/search/` | filters and ranking |
| `recruit_intel/api/` | FastAPI backend |
| `data/` | DuckDB file and raw cache (gitignored) |
