"""Generate a SYNTHETIC xBD-style dataset (fake satellite scenes + building polygons + damage labels).
Same folder/JSON layout as real xBD, so every other script works unchanged.
For demonstrating the pipeline when the real dataset isn't available - NOT real disaster imagery.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
EVENTS = {  # name: (background rgb, lon, lat)
    "synth-hurricane": ((92, 118, 78), 78.40, 17.40),
    "synth-flood": ((120, 112, 88), 78.55, 17.45),
    "synth-quake": ((135, 118, 98), 78.30, 17.55),
}
ROOFS = [(150, 150, 155), (172, 108, 92), (205, 192, 170), (118, 124, 132), (182, 152, 110)]


def rect(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + x * c - y * s, cy + x * s + y * c)
            for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def wkt_poly(pts, f=lambda p: p):
    q = [f(p) for p in pts]; q.append(q[0])
    return "POLYGON ((" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in q) + "))"


def background(rng, size, base):
    low = rng.normal(0, 14, (size // 40, size // 40, 3)).astype(np.float32)
    low = np.asarray(Image.fromarray(np.clip(low + 128, 0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), np.float32) - 128
    img = np.array(base, np.float32) + low + rng.normal(0, 3, (size, size, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_scene(rng, name, idx, size, cell, base, lon_b, lat_b):
    bg = background(rng, size, base)
    pre, post = bg.copy(), bg.copy()
    pre_d, post_d = ImageDraw.Draw(pre), ImageDraw.Draw(post)
    for k in range(0, size + 1, cell):                      # roads
        for d in (pre_d, post_d):
            d.line([(k, 0), (k, size)], fill=(88, 88, 92), width=5)
            d.line([(0, k), (size, k)], fill=(88, 88, 92), width=5)

    n_ep = int(rng.integers(1, 4))
    eps = rng.uniform(size * .15, size * .85, (n_ep, 2)); rad = rng.uniform(size * .15, size * .3, n_ep)
    blds = []
    for gy in range(size // cell):
        for gx in range(size // cell):
            if rng.random() > .62:
                continue
            cx, cy = (gx + .5) * cell + rng.uniform(-4, 4), (gy + .5) * cell + rng.uniform(-4, 4)
            w, h = rng.uniform(14, 24, 2)
            pts = rect(cx, cy, w, h, rng.uniform(-.15, .15))
            s = max(np.exp(-(np.hypot(cx - eps[:, 0], cy - eps[:, 1]) / rad) ** 2)) + rng.normal(0, .12)
            cls = 3 if s > .72 else 2 if s > .5 else 1 if s > .3 else 0
            roof = tuple(int(np.clip(c + rng.normal(0, 8), 0, 255)) for c in ROOFS[rng.integers(len(ROOFS))])
            blds.append((pts, cls, roof))

    rubble_mask = Image.new("L", (size, size), 0); rm = ImageDraw.Draw(rubble_mask)
    for pts, cls, roof in blds:
        shadow = [(x + 3, y + 3) for x, y in pts]
        for d in (pre_d, post_d):
            d.polygon(shadow, fill=(55, 55, 55))
        pre_d.polygon(pts, fill=roof)
        if cls == 0:
            post_d.polygon(pts, fill=tuple(int(np.clip(c * rng.uniform(.9, 1.0) + rng.normal(0, 3), 0, 255)) for c in roof))
            if rng.random() < .10:                           # harmless look-alike (e.g. a parked tarp)
                cx, cy = np.mean(pts, 0); post_d.rectangle([cx - 3, cy - 2, cx + 2, cy + 2], fill=(60, 95, 160))
        elif cls == 1:                                       # tarp patch + dirt
            post_d.polygon(pts, fill=tuple(int(c * rng.uniform(.82, .95)) for c in roof))
            cx, cy = np.mean(pts, 0)
            if rng.random() < .65:
                post_d.rectangle([cx - 4, cy - 3, cx + 3, cy + 3], fill=(60, 95, 160))
        elif cls == 2:                                       # half the roof gone
            post_d.polygon(pts, fill=roof)
            t = rng.uniform(.25, .65)
            half = [pts[0], pts[1], (pts[1][0] + t * (pts[2][0] - pts[1][0]), pts[1][1] + t * (pts[2][1] - pts[1][1])),
                    (pts[0][0] + t * (pts[3][0] - pts[0][0]), pts[0][1] + t * (pts[3][1] - pts[0][1]))]
            post_d.polygon(half, fill=(88, 68, 52))
            for _ in range(6):
                dx, dy = rng.normal(0, 9, 2); cx, cy = np.mean(pts, 0)
                post_d.ellipse([cx + dx - 1, cy + dy - 1, cx + dx + 1, cy + dy + 1], fill=(100, 85, 70))
        else:                                                # destroyed -> rubble
            rm.polygon([(x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for x, y in pts], fill=255)
            cx, cy = np.mean(pts, 0)
            for _ in range(10):
                dx, dy = rng.normal(0, 12, 2)
                post_d.ellipse([cx + dx - 1.5, cy + dy - 1.5, cx + dx + 1.5, cy + dy + 1.5], fill=(105, 90, 75))
    rub = np.clip(rng.normal(105, 28, (size, size, 3)) * np.array([1.0, .88, .75]), 0, 255).astype(np.uint8)
    post = Image.composite(Image.fromarray(rub), post, rubble_mask.filter(ImageFilter.GaussianBlur(.8)))
    post = Image.fromarray(np.clip(np.asarray(post, np.float32) * rng.uniform(.95, 1.05) + rng.normal(0, 3, (size, size, 3)), 0, 255).astype(np.uint8))

    lon0, lat0 = lon_b + (idx % 10) * .01, lat_b + (idx // 10) * .01
    dlon, dlat = .5 / (111320 * math.cos(math.radians(lat0))), -.5 / 110540
    geo = lambda p: (lon0 + p[0] * dlon, lat0 + p[1] * dlat)
    xy, ll = [], []
    for i, (pts, cls, _) in enumerate(blds):
        pr = {"feature_type": "building", "subtype": CLASSES[cls], "uid": f"{name}-{i}"}
        xy.append({"properties": pr, "wkt": wkt_poly(pts)})
        ll.append({"properties": pr, "wkt": wkt_poly(pts, geo)})
    return pre, post, {"features": {"xy": xy, "lng_lat": ll}, "metadata": {"synthetic": True}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_xbd")
    ap.add_argument("--scenes_per_event", type=int, default=20)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--cell", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out) / "train"
    (root / "images").mkdir(parents=True, exist_ok=True); (root / "labels").mkdir(exist_ok=True)
    total = 0
    for ev, (base, lon, lat) in EVENTS.items():
        for i in range(a.scenes_per_event):
            sid = f"{ev}_{i:08d}"
            pre, post, lab = make_scene(rng, sid, i, a.size, a.cell, base, lon, lat)
            pre.save(root / "images" / f"{sid}_pre_disaster.png"); post.save(root / "images" / f"{sid}_post_disaster.png")
            json.dump(lab, open(root / "labels" / f"{sid}_post_disaster.json", "w"))
            total += len(lab["features"]["xy"])
    print(f"Synthetic dataset written to {root.parent}: {len(EVENTS) * a.scenes_per_event} scenes, {total} buildings")
