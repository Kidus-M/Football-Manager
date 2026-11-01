# Moodboard

Collected on 2026-11-01 before building the front end, per the spec: Awwwards
sports Sites of the Day and Dribbble sports-analytics shots. Links only; notes
are what each reference teaches this project, not descriptions to copy.

## Awwwards (sports)

| Reference | Take from it |
| --- | --- |
| [Field Rank](https://www.awwwards.com/sites/field-rank) | Rankings as the product: results turned into season tables and charts. Closest in purpose; data first, decoration second. |
| [World Cup — The Immortals](https://www.awwwards.com/sites/world-cup-the-immortals) | Football history told through moments; timeline as navigation (season timeline view). |
| [The Last Tango](https://www.awwwards.com/sites/the-last-tango) | Goals as point clouds: data as material. Supports drawing the similarity map as a field of points, not a scatter chart. |
| [Sixes Social Cricket](https://www.awwwards.com/sites/sixes-social-cricket) | Bold sport typography carrying the identity; condensed display type works for sport. |
| [Boom Ball](https://www.awwwards.com/sites/boom-ball) | High-energy motion; a reminder of what *not* to do in an analysis tool: motion must explain, not perform. |
| [The Ride to Rio](https://www.awwwards.com/sites/the-ride-to-rio) | A 3-D map you move through; supports the tilted "broadcast camera" pitch. |
| [Justin Jefferson](https://www.awwwards.com/sites/justin-jefferson) | A player's record as a portrait: the player page should read like a profile, not a stats dump. |
| [HelmetVerse](https://www.awwwards.com/sites/helmetverse) | Kit/equipment as a design object; kit numbers and plates as UI elements. |
| [Reel Motion](https://www.awwwards.com/sites/reel-motion) | Built dark-first because its users are; same reasoning as ours (match-night, late analysis). |
| [Palaraiffeisen](https://www.awwwards.com/sites/palaraiffeisen) | Venue signage and wayfinding: stadium signage is a usable type reference. |

## Dribbble (football analytics dashboards)

| Reference | Take from it |
| --- | --- |
| [Topnotch Studio — Football Analytics Dashboard](https://dribbble.com/shots/27547707-Football-Analytics-Dashboard) | Light card grid; legible but generic. |
| [Nexux Lab — Football Analytics Dashboard](https://dribbble.com/shots/27119431-Football-Analytics-Dashboard) | Dark UI with player photography; photography dominates the data. |
| [Emura — Supermob Player Analytics](https://dribbble.com/shots/27388369-Supermob-Player-Analytics-Dashboard) | Player profile layout: identity block, then percentile bars. |
| [Orbix — Clutch Lens live match](https://dribbble.com/shots/27558162-Clutch-Lens-FIFA-World-Cup-2026-Live-Match-Dashboard) | Dark live-match board; dense but scannable. |
| [md sabuj mia — FIFA 2026 Live Intelligence](https://dribbble.com/shots/27533835-FIFA-2026-Live-Intelligence-AI-Powered-Analytics-Dashboard) | Typical "AI dashboard" look: glass cards, glows. |
| [Blend wrk — AI-Powered Match Zone](https://dribbble.com/shots/27043360-AI-Powered-Football-Match-Zone) | Same cluster: dark glass, neon. |
| [Lilian Labbe — Ball Knowledge](https://dribbble.com/shots/27736762-Ball-Knowledge-Saas-Football-Data-Analytics-Dashboard-UI) | SaaS framing of football data. |
| [Rifat Zulfan — Football Scouting Analytics](https://dribbble.com/shots/27180578-Football-Scouting-Analytics-Dashboard) | Scouting focus: shortlist + profile side by side. Matches our recruitment funnel. |
| [Atish Chakma — Match Analytics + chat](https://dribbble.com/shots/26517374-Football-Match-Analytics-Dashboard-Live-Stats-Chat) | Live stats, not our use case. |
| [Mohamed Yasser — Sports Analytics](https://dribbble.com/shots/27706955-Sports-Analytics-Football-Dashboard) | World Cup dashboard; generic card grid. |

## What the board says

The Dribbble football dashboards converge on one look: dark glass cards, a neon
accent, cut-out player photos, a KPI row on top. That is the template to avoid.
The Awwwards references that work treat the sport's own artefacts (the pitch,
kit numbers, signage, the timeline of a season) as the interface.

**Decisions**

- **Type pairing:** Big Shoulders Display (stadium signage, kit numbers) for
  names and numbers, Instrument Sans for reading, Martian Mono for data labels.
- **Motion language:** "broadcast camera". Motion only moves things the way a
  tactics camera does: positions glide when the formation changes, a player
  plate lifts into his profile, nothing bounces or loops. Reduced motion turns
  every transition into a cut.
