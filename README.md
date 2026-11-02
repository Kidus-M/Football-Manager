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

## Defensive stats (API-Football) and data checks

Tackles, interceptions, blocks, duels and dribbled-past for every league come
from [API-Football](https://www.api-football.com). Put a free key in `.env`
(see `.env.example`), then run once a day until it reports nothing left:

```bash
.venv/Scripts/recruit ingest --apifootball ENG ESP ITA GER FRA --apifootball-history 2
.venv/Scripts/recruit train --defence-only   # retrain the defensive estimator after a load
```

The free plan allows 100 requests a day, only seasons 2022-23 to 2024-25, and
only three pages per listing, so players are fetched club by club (~100
requests per league-season: 2024-25 for all five leagues first, then the two
seasons before it, about a week in all). Records where API-Football attaches
one stat line to two clubs are dropped. It never records dribbled-past, so no
role uses that metric. A paid plan (Pro, about $19/month) covers the current
season with the same code.

**How the window gets defensive numbers.** Each metric in each view is one of:

| Kind | Meaning |
| --- | --- |
| measured | this window's own data (FPL tackles and recoveries in the Premier League) |
| older | the player's latest API-Football season, as-is |
| carried | estimated from that season plus this season's profile and team |
| estimated | no data of his own: estimated from playing style, team, league and age |

The estimator (`recruit_intel/models/defence.py`) is a set of quantile
gradient-boosting models trained on every measured API-Football season, and
validated on held-out players: rank correlation within position group, and
how often the measured value falls inside the 10-90% range (ranges are
widened until it's 80%). Until two or more seasons are loaded it can't learn
from a player's older season, so players with one keep it as-is ("older").
With 2024-25 for three leagues, estimates without data rank players at
0.13-0.40 correlation: a prompt to go and watch someone, not a verdict.

Defensive volume (tackles, interceptions, blocks, recoveries) is adjusted to
50% possession, with possession estimated from Understat's PPDA pass counts,
so defenders at dominant teams aren't marked down for having less to do.
Estimates and older seasons are ranked against measured players only.

Role fits carry a likely range and a **confidence**: the share of the role's
weight backed by data, each metric counted by how reliably its kind ranked
players in validation. Search discounts fits by confidence, shows the range,
and `recruit search --own-data-only` (or the dashboard toggle) leaves out
players whose fit leans on pure estimates. Estimates are never listed as
strengths or weaknesses.

```bash
.venv/Scripts/recruit audit          # data-quality report
.venv/Scripts/recruit audit --fix    # repair the safely fixable problems
```

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
