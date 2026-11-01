# Football Recruitment Intelligence Platform

A club-agnostic recruitment and squad-analysis platform: point it at any club, and it profiles the current squad, finds the gaps against the club's tactical system, and searches the market for players who fill them. The first configuration is Manchester United.

Runs locally on free data. A personal research project, not a commercial product.

## Quick start

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/recruit ingest          # FPL, Understat (club's pool), ClubElo, club's Transfermarkt squad
.venv/Scripts/recruit ingest --market-pool ENG   # + every Transfermarkt squad in those leagues (slow first run)
.venv/Scripts/recruit match           # rebuild canonical player ids (ingest also runs this)
.venv/Scripts/recruit squad           # current squad: value, contract, per-90 stats
.venv/Scripts/recruit profile         # squad strengths, weaknesses and best role (percentiles)
.venv/Scripts/recruit profile Mainoo Tielemans --radar data/radars/cm.svg   # compare + SVG radar
.venv/Scripts/recruit gaps --depth    # best XI for the club's formation, cover, recruitment priorities
.venv/Scripts/recruit gaps --formation 3-4-2-1   # same squad, another shape
.venv/Scripts/recruit search          # ranked market candidates for the top 3 gaps
.venv/Scripts/recruit search --slot ST --max-value 50 --max-age 25 --upgrades
.venv/Scripts/recruit tactics         # how the XI should play: levers, per-position instructions
.venv/Scripts/recruit tactics --press high --buildup direct   # what-if: who fits, which gaps open
.venv/Scripts/pip install -e ".[ml]"   # models (see pyproject for CPU-only torch)
.venv/Scripts/recruit ingest --seasons 11   # history for training (Understat 2016-)
.venv/Scripts/recruit train           # embeddings + league-translation model, with validation
.venv/Scripts/recruit similar Mbeumo --younger --cheaper --not-ours
.venv/Scripts/recruit roles           # playing roles discovered by clustering
.venv/Scripts/recruit bargains --group CB --max-age 25
.venv/Scripts/recruit ingest --seasons 11 --managers   # manager histories (slow first run)
.venv/Scripts/recruit style "Manchester United"   # strength-adjusted style by season, teams alike
.venv/Scripts/recruit manager "De Zerbi"          # a manager's style across clubs
.venv/Scripts/recruit fit Diakite "Wesley Fofana" # how a club's style suits players
.venv/Scripts/pytest
```

Club is chosen with `--club <file stem>` (default `man_utd`, from `config/clubs/`).

## Dashboard

A Next.js app in `app/` over a read-only FastAPI backend (`recruit_intel/api/`).

```bash
.venv/Scripts/pip install -e ".[api,ml]"
.venv/Scripts/uvicorn recruit_intel.api.main:app --port 8000   # API (warms its caches on start)
npm --prefix app install
npm --prefix app run dev                                      # http://localhost:3000
```

Views: **Lineup** (best XI on the pitch, gaps glowing in club colour, formation
switch), **Recruit** (ranked candidates with live filters), **Compare** (radars),
**Similar** (style map of every player), **Tactics** (pressing/build-up/width
what-ifs re-run the XI). Ctrl/Cmd+K jumps to any player or view. The club's
colours come from its config, so another club config re-skins the app. Design
rationale and references: `docs/design/`.

## Player identity

Each source names players differently ("Bruno Borges Fernandes", "Bruno Fernandes"),
so every source row is linked to one canonical `player_id` in `player_source_ids`.
Transfermarkt anchors the ids (`tm-<id>`); FPL links by birth date, Understat by
club, then league, then globally, each step stricter. Names must agree on surname,
and clubs that fuzzy matching can't map go in `config/team_aliases.yaml`.
`recruit match` prints the match rate; the target is 95%+ of this season's top-5
league players.

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
