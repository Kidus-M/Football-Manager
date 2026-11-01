"""Percentile radar charts as standalone SVG.

Plain SVG with no plotting dependency, so the same output can be served by
the API later or dropped into the front end. Dark-first, with the club's
theme colours for the first player and neutral tones for comparisons.
"""

import math
from html import escape

from recruit_intel.features.metrics import METRICS

SIZE = 560
CENTER = SIZE / 2
RADIUS = 190
RINGS = (25, 50, 75, 100)
BG, GRID, TEXT, MUTED = "#0d0e10", "#2a2d33", "#f2f2f0", "#8b8f98"
COMPARE_COLOURS = ("#4fc3f7", "#c5e1a5")


def _point(axis: int, n_axes: int, pct: float) -> tuple[float, float]:
    angle = -math.pi / 2 + 2 * math.pi * axis / n_axes
    r = RADIUS * max(0.0, min(pct, 100)) / 100
    return CENTER + r * math.cos(angle), CENTER + r * math.sin(angle)


def _polygon(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render(
    players: list[tuple[str, dict[str, float]]],
    metrics: list[str],
    title: str,
    subtitle: str = "",
    accent: str = "#DA291C",
) -> str:
    """players: [(name, {metric: percentile})]; missing metrics plot at 0 and
    are labelled 'n/a'."""
    if len(metrics) < 3:
        raise ValueError("a radar needs at least 3 metrics")
    n = len(metrics)
    colours = (accent, *COMPARE_COLOURS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE + 70}" '
        f'viewBox="0 0 {SIZE} {SIZE + 70}" '
        f'font-family="Inter, Segoe UI, sans-serif">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<text x="24" y="38" fill="{TEXT}" font-size="22" font-weight="700">'
        f"{escape(title)}</text>",
        f'<text x="24" y="60" fill="{MUTED}" font-size="13">{escape(subtitle)}</text>',
        '<g transform="translate(0,40)">',
    ]

    for ring in RINGS:
        ring_points = [_point(i, n, ring) for i in range(n)]
        parts.append(
            f'<polygon points="{_polygon(ring_points)}" fill="none" stroke="{GRID}" '
            f'stroke-width="{1.5 if ring == 50 else 1}"/>'
        )
    for i, metric in enumerate(metrics):
        x, y = _point(i, n, 100)
        parts.append(
            f'<line x1="{CENTER}" y1="{CENTER}" x2="{x:.1f}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        lx, ly = _point(i, n, 118)
        anchor = "middle" if abs(lx - CENTER) < 20 else ("start" if lx > CENTER else "end")
        values = " / ".join(
            "n/a" if pcts.get(metric) is None else f"{pcts[metric]:.0f}" for _, pcts in players
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{TEXT}" font-size="12" '
            f'text-anchor="{anchor}">{escape(METRICS[metric].label)}</text>'
            f'<text x="{lx:.1f}" y="{ly + 15:.1f}" fill="{MUTED}" font-size="11" '
            f'text-anchor="{anchor}">{values}</text>'
        )

    for (_name, pcts), colour in zip(players, colours, strict=False):
        pts = [_point(i, n, pcts.get(m) or 0) for i, m in enumerate(metrics)]
        parts.append(
            f'<polygon points="{_polygon(pts)}" fill="{colour}" fill-opacity="0.22" '
            f'stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>'
        )
        parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colour}"/>' for x, y in pts)
    parts.append("</g>")

    for i, ((name, _), colour) in enumerate(zip(players, colours, strict=False)):
        y = SIZE + 48
        x = 24 + i * 180
        parts.append(
            f'<rect x="{x}" y="{y - 11}" width="12" height="12" rx="2" fill="{colour}"/>'
            f'<text x="{x + 18}" y="{y}" fill="{TEXT}" font-size="13">{escape(name)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
