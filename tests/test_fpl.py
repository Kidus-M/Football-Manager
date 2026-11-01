from recruit_intel.ingest import fpl

BOOTSTRAP = {
    "events": [{"deadline_time": "2026-08-21T17:30:00Z"}],
    "teams": [{"id": 16, "name": "Man Utd"}],
    "elements": [
        {
            "id": 7,
            "first_name": "Test",
            "second_name": "Player",
            "team": 16,
            "element_type": 3,
            "birth_date": "2001-05-04",
            "minutes": 900,
            "removed": False,
            **{field: "1.5" for field in fpl.METRIC_FIELDS},
        },
        {"id": 8, "removed": True},
    ],
}


def test_parse_maps_positions_metrics_and_season():
    out = fpl.parse(BOOTSTRAP)
    players = out["source_players"]
    assert players.to_dict("records")[0]["position"] == "MID"
    assert len(players) == 1  # removed players dropped
    stats = out["player_season_stats"]
    assert set(stats["metric"]) == set(fpl.METRIC_FIELDS.values())
    assert (stats["season"] == 2026).all()
    assert stats["value"].iloc[0] == 1.5
