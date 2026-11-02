"""Export the showcase picks (search results + player profiles) to showcase/out/picks.json."""

import json
from pathlib import Path

from recruit_intel.api import service

OUT = Path(__file__).parent / "out"
MAX_AGE = 25
PICKS = [  # (slot, name): the best under-25 option the search ranks for each priority
    ("GK", "Bart Verbruggen"),
    ("RB", "Javi Rueda"),
    ("LCB", "Enzo Boyomo"),
    ("LB", "Maxim De Cuyper"),
    ("AM", "Nicola Zalewski"),
    ("ST", "Karl Etta Eyong"),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lineup = service.lineup_view()
    picks = []
    for slot, name in PICKS:
        result = service.search(slot, max_age=MAX_AGE, limit=20)
        cand = next(c for c in result["candidates"] if c["name"] == name)
        starter = next(s for s in lineup["slots"] if s["key"] == slot)["starter"]
        starter_profile = service.player(starter["player_id"]) if starter else None
        picks.append({"slot": slot, "role": result["role"], "starter": result["starter"],
                      "candidate": cand, "profile": service.player(cand["player_id"]),
                      "starter_profile": starter_profile})  # fmt: skip
    data = {"club": service.club_info(), "lineup": lineup, "picks": picks}
    (OUT / "picks.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in picks:
        c = p["candidate"]
        print(p["slot"], c["name"], c["age"], c["value_eur"], round(c["fit"]), p["starter"])


if __name__ == "__main__":
    main()
