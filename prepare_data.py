"""Scan xBD, make the 80:20 train/test split, and cut building-pair crops for the damage classifier.

xBD layout expected:  <root>/<tier>/images/*.png  and  <root>/<tier>/labels/*.json
(tiers such as train, tier3, test, hold - all labelled tiers are pooled, then split 80:20 per disaster event)
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from shapely import wkt
from config import *
from utils import crop_pair


def find_records(root):
    recs = []
    for lab in Path(root).glob("*/labels/*_post_disaster.json"):
        stem = lab.stem
        pre_name = stem.replace("_post_disaster", "_pre_disaster")
        imgs = lab.parent.parent / "images"
        pre, post = imgs / f"{pre_name}.png", imgs / f"{stem}.png"
        if pre.exists() and post.exists():
            rid = stem.replace("_post_disaster", "")
            recs.append(dict(id=rid, event=rid.rsplit("_", 1)[0],
                             pre=str(pre), post=str(post), label=str(lab)))
    return recs


def split_80_20(recs, ratio, seed):
    by = defaultdict(list)
    for r in recs:
        by[r["event"]].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for ev in sorted(by):
        items = sorted(by[ev], key=lambda r: r["id"])
        rng.shuffle(items)
        k = max(1, int(round(len(items) * ratio)))
        test += items[:k]
        train += items[k:]
    return train, test


def collect(recs, cap, rng):
    ents = []
    for r in recs:
        feats = json.load(open(r["label"]))["features"]["xy"]
        for f in feats:
            p = f["properties"]
            sub = p.get("subtype")
            if p.get("feature_type") == "building" and sub in CLS2ID:
                ents.append((r["id"], f["wkt"], CLS2ID[sub], p.get("uid", "")))
    if cap:
        by = defaultdict(list)
        for e in ents:
            by[e[2]].append(e)
        ents = []
        for c, lst in by.items():
            rng.shuffle(lst)
            ents += lst[:cap]
    return ents


def save_crops(ents, recs, split, out):
    rec = {r["id"]: r for r in recs}
    by = defaultdict(list)
    for e in ents:
        by[e[0]].append(e)
    d = out / "crops" / split
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid, lst in tqdm(by.items(), desc=f"crops/{split}"):
        pre = Image.open(rec[rid]["pre"]).convert("RGB")
        post = Image.open(rec[rid]["post"]).convert("RGB")
        for _, w, c, uid in lst:
            g = wkt.loads(w)
            path = d / f"{rid}_{uid}.png"
            crop_pair(pre, post, g.bounds).save(path)
            rows.append((str(path), c, rec[rid]["event"]))
    with open(out / f"crops_{split}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "label", "event"])
        wr.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="xBD root folder")
    ap.add_argument("--max_per_class", type=int, default=8000,
                    help="cap crops per class in TRAIN (0 = no cap); test cap is 25%% of this")
    a = ap.parse_args()

    recs = find_records(a.root)
    assert recs, "No xBD image pairs found - check --root"
    train, test = split_80_20(recs, TEST_RATIO, SEED)
    json.dump({"train": train, "test": test}, open(DATA / "splits.json", "w"), indent=1)
    print(f"{len(recs)} image pairs -> train {len(train)} | test {len(test)}  (80:20 per disaster event)")

    rng = random.Random(SEED)
    cap_te = a.max_per_class // 4 if a.max_per_class else 0
    for name, rs, cap in (("train", train, a.max_per_class), ("test", test, cap_te)):
        rows = save_crops(collect(rs, cap, rng), rs, name, DATA)
        print(name, "crops:", len(rows))
