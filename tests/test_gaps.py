import pandas as pd

from recruit_intel.analysis.gaps import priorities, slot_issues
from recruit_intel.analysis.lineup import Candidate, SlotDepth
from recruit_intel.config import Slot

SLOT = Slot(key="ST", groups=["ST"], role="complete_forward", x=50, y=85)


def _depth(starter_fit, *backup_fits):
    starter = Candidate("s", "Starter", starter_fit, starter_fit, 900) if starter_fit else None
    backups = [Candidate(f"b{i}", f"B{i}", f, f, 900) for i, f in enumerate(backup_fits)]
    return SlotDepth(SLOT, starter, backups)


def severities(issues):
    return [i.severity for i in issues]


def test_strong_starter_with_cover_has_no_issues():
    assert slot_issues(_depth(80, 60), age=25, contract_year=2030, season_start=2026) == []


def test_empty_slot_is_critical():
    assert severities(slot_issues(_depth(None), None, None, 2026)) == ["critical"]


def test_fit_bands():
    assert severities(slot_issues(_depth(40, 60), 25, 2030, 2026)) == ["critical"]
    assert severities(slot_issues(_depth(50, 60), 25, 2030, 2026)) == ["high"]
    assert severities(slot_issues(_depth(60, 60), 25, 2030, 2026)) == ["medium"]


def test_cover_contract_and_age_flags():
    issues = slot_issues(_depth(80, 45), age=33, contract_year=2027, season_start=2026)
    texts = " | ".join(i.text for i in issues)
    assert "no adequate cover" in texts
    assert "contract ends 2027" in texts
    assert "is 33" in texts


def test_unrated_cover_is_unproven_not_missing():
    depth = _depth(80)
    depth.unrated = ["Tom Heaton"]
    issues = slot_issues(depth, 25, 2030, 2026)
    assert [(i.severity, i.text) for i in issues] == [("low", "cover unproven: Tom Heaton unrated")]


def test_slot_with_only_unrated_players_is_high_not_critical():
    depth = _depth(None)
    depth.unrated = ["Matthijs de Ligt"]
    assert severities(slot_issues(depth, None, None, 2026)) == ["high"]


def test_priorities_orders_by_severity_then_fit():
    report = pd.DataFrame(
        [
            {"slot": "LB", "severity": "medium", "fit": 60},
            {"slot": "ST", "severity": "critical", "fit": 40},
            {"slot": "GK", "severity": None, "fit": 80},
            {"slot": "RB", "severity": "medium", "fit": 55},
        ]
    )
    assert list(priorities(report).slot) == ["ST", "RB", "LB"]
