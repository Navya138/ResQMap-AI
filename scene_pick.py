"""Choose good demo scenes: plenty of buildings AND visible damage, spread across disaster events."""
import json
from collections import defaultdict
from config import CLS2ID


def count_buildings(label_path):
    total = damaged = 0
    for f in json.load(open(label_path))["features"]["xy"]:
        p = f["properties"]
        if p.get("feature_type") == "building" and p.get("subtype") in CLS2ID:
            total += 1
            damaged += p["subtype"] != "no-damage"
    return total, damaged


def pick_scenes(recs, n, min_total=15, min_damaged=5):
    by = defaultdict(list)
    for r in recs:
        total, damaged = count_buildings(r["label"])
        if total >= min_total and damaged >= min_damaged:
            by[r["event"]].append((damaged, r))
    for ev in by:
        by[ev].sort(key=lambda t: -t[0])
    picked, i = [], 0
    while len(picked) < n and any(i < len(v) for v in by.values()):
        for ev in sorted(by):
            if i < len(by[ev]) and len(picked) < n:
                picked.append(by[ev][i][1])
        i += 1
    return picked
