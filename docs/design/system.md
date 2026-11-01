# Design system

The front end's single job: help a recruitment team decide who to sign and
why. Its thesis view is the team sheet on the pitch, not a KPI grid.

## Colour

Dark-first (spec). The dark is a floodlit pitch at night, not neutral black,
and the accent comes from the club config, so switching club re-skins the app.

| Token | Hex | Use |
| --- | --- | --- |
| `--night` | `#0B1210` | page background: the pitch at night |
| `--turf` | `#13201B` | raised surfaces, the pitch fill |
| `--line` | `#26352E` | dividers, card edges, pitch markings at rest |
| `--chalk` | `#E9EDE4` | primary text, pitch lines in focus |
| `--fog` | `#8E9A91` | secondary text, axis labels |
| `--club` | config `theme.accent` (`#DA291C` for Man United) | gaps, the selected thing, primary actions |
| `--club-2` | config `theme.secondary` (`#FBE122`) | rare highlight: an upgrade over the starter |

Fit is shown on a single ramp from `--line` (poor) to `--chalk` (strong), with
`--club` reserved for problems, so red always means "needs attention".

## Type

| Role | Face | Notes |
| --- | --- | --- |
| Display | Big Shoulders Display 700-900 | names, kit-number-style figures; uppercase sparingly |
| Body | Instrument Sans 400/500/600 | all reading text |
| Data | Martian Mono 400 | percentiles, values, labels on charts; tabular |

Scale (rem): 0.75 data / 0.875 small / 1 body / 1.25 / 1.75 / 2.75 / 4.5 display.

## Layout

```
+------+-----------------------------------------------+
| rail |  view title                       club · season|
|      +-------------------------------+---------------+
| Sq   |                               |  selected     |
| Lu   |   PITCH (tilted, broadcast)   |  slot / player|
| Re   |                               |---------------|
| Co   |                               |  priorities   |
| Si   |                               |               |
| Ta   +-------------------------------+---------------+
+------+   cmd+K: jump to any player, team or view       |
```

A left rail of views; each view has one hero element and supporting panels.
Below 900px the rail becomes a bottom bar and panels stack under the hero.

## Signature

The tilted pitch: chalk markings in perspective like a broadcast tactical
camera, player plates standing on it with kit-number figures for fit, and
gaps glowing in the club colour. A "flat view" toggle and reduced-motion
setting remove the tilt and transitions.
