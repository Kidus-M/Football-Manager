import xml.etree.ElementTree as ET

import pytest

from recruit_intel.analysis.radar import render

METRICS = ["npxg", "xa", "shots", "key_passes"]


def test_renders_valid_svg_with_one_polygon_per_player():
    svg = render(
        [("A & B", {"npxg": 90, "xa": 10, "shots": 50, "key_passes": 70}),
         ("C", {"npxg": 40, "xa": None, "shots": 60, "key_passes": 20})],
        METRICS, "A & B vs C",
    )  # fmt: skip
    root = ET.fromstring(svg)  # parses, so names were escaped
    ns = "{http://www.w3.org/2000/svg}"
    filled = [p for p in root.iter(f"{ns}polygon") if p.get("fill") != "none"]
    assert len(filled) == 2
    assert "n/a" in svg  # missing metric is labelled, not hidden


def test_needs_three_metrics():
    with pytest.raises(ValueError):
        render([("A", {})], ["npxg", "xa"], "A")
