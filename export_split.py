"""Physically separate the 80:20 split into two folders, or check which side one picture is on.

  python export_split.py                       # copy files into split_dataset/train and split_dataset/test
  python export_split.py --link                # same, but hard links (no extra disk space, same drive only)
  python export_split.py --check hurricane-michael_00000123
"""
import argparse, csv, json, os, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="split_dataset")
ap.add_argument("--link", action="store_true")
ap.add_argument("--check", metavar="NAME", help="scene or file name (part of the name is enough)")
a = ap.parse_args()

sp = json.load(open(Path("data") / "splits.json"))

if a.check:
    hits = [(side, r["id"]) for side in ("train", "test") for r in sp[side] if a.check in r["id"] or a.check in r["pre"] or a.check in r["post"]]
    print("\n".join(f"{i}  ->  {side.upper()}" for side, i in hits[:20]) or "not found in either split")
    raise SystemExit

out = Path(a.out)
for side in ("train", "test"):
    (out / side / "images").mkdir(parents=True, exist_ok=True)
    (out / side / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sp[side]:
        for key, sub in (("pre", "images"), ("post", "images"), ("label", "labels")):
            src = Path(r[key]); dst = out / side / sub / src.name
            if dst.exists():
                continue
            if a.link:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        rows.append((r["id"], r["event"]))
    with open(out / f"{side}_files.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene_id", "event"]); w.writerows(rows)
    print(f"{side}: {len(rows)} scenes -> {out / side}")
print("Done. Lists are in", out / "train_files.csv", "and", out / "test_files.csv")
