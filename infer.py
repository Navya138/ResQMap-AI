"""Run the full pipeline on test scenes and write demo/<scene>/ (cached so the live demo never waits on a GPU).
 
  python infer.py --n 4                       # 4 scenes from the held-out 20% split
  python infer.py --scene hurricane-michael_00000123
  python infer.py --n 4 --use_gt              # safety net: ground-truth footprints, predicted damage
 
The same pipeline is also used by app.py (upload mode) through load_models() and analyze().
"""
import argparse, json
import numpy as np, torch, cv2
from PIL import Image
from config import *
from scene_pick import pick_scenes
from model_defs import make_damage_model
from utils import (crop_pair, pair_to_array, fit_affine, fallback_affine, pix2geo,
                   load_features, polys, MEAN3, STD3)
 
 
@torch.no_grad()
def segment(model, pre, dev):
    x = ((np.asarray(pre, dtype=np.float32) / 255 - MEAN3) / STD3).transpose(2, 0, 1)
    p = torch.sigmoid(model(torch.from_numpy(x)[None].to(dev)))[0, 0].cpu().numpy()
    return (p > .5).astype(np.uint8)
 
 
def mask_to_polys(mask, min_area=40):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if len(c) < 3 or cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
        if len(c) >= 3:
            out.append(c.astype(float))
    return out
 
 
def gt_polys(label):
    xy, _ = load_features(label)
    return [np.array(p.exterior.coords)[:, :2] for f in xy
            if f["properties"].get("feature_type") == "building" for p in polys(f["wkt"])]
 
 
@torch.no_grad()
def classify(model, pre, post, pl, dev, bs=128):
    xs = [pair_to_array(crop_pair(pre, post, (*p.min(0), *p.max(0)))) for p in pl]
    ids, conf = [], []
    for i in range(0, len(xs), bs):
        pr = torch.softmax(model(torch.from_numpy(np.stack(xs[i:i + bs])).to(dev)), 1).cpu()
        c, k = pr.max(1); ids += k.tolist(); conf += c.tolist()
    return ids, conf
 
 
def load_models(dev, need_seg):
    """Load the damage classifier, and the building segmenter only if footprints are not given."""
    dmg = make_damage_model(pretrained=False).to(dev)
    dmg.load_state_dict(torch.load(MODELS / "damage.pt", map_location=dev)); dmg.eval()
    seg = None
    if need_seg:
        import segmentation_models_pytorch as smp
        seg = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
        seg.load_state_dict(torch.load(MODELS / "seg.pt", map_location=dev)); seg.eval()
    return dmg, seg
 
 
def analyze(dmg, seg, dev, pre, post, label, scene_id, event, use_gt, center=(0.0, 0.0)):
    """Analyse one before/after pair and write demo/<scene_id>/{pre.jpg,post.jpg,buildings.geojson,meta.json}.
 
    pre, post : PIL RGB images of the same size
    label     : path to an xBD label .json, or None (then use_gt must be False and `center` is used for the map)
    Returns the output folder, or None if no buildings were found.
    """
    pl = gt_polys(label) if use_gt else mask_to_polys(segment(seg, pre, dev))
    if not pl:
        return None
    ids, conf = classify(dmg, pre, post, pl, dev)
    M = fit_affine(label) if label else None
    if M is None:
        M = fallback_affine(*center, size=pre.size[0])
    feats = []
    for k, (p, c, s) in enumerate(zip(pl, ids, conf)):
        ring = [list(pix2geo(M, x, y)) for x, y in p]
        ring.append(ring[0])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"id": k, "damage": CLASSES[c], "conf": round(float(s), 3)}})
    W, H = pre.size
    corners = [pix2geo(M, x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H))]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    out = DEMO / scene_id; out.mkdir(parents=True, exist_ok=True)
    pre.save(out / "pre.jpg", quality=85); post.save(out / "post.jpg", quality=85)
    json.dump({"type": "FeatureCollection", "features": feats}, open(out / "buildings.geojson", "w"))
    json.dump({"id": scene_id, "event": event,
               "mode": "ground-truth footprints" if use_gt else "predicted footprints",
               "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]]}, open(out / "meta.json", "w"))
    return out
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--use_gt", action="store_true")
    ap.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], metavar=("LON", "LAT"),
                    help="only used if a scene has no geo-referenced label")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
 
    sp = json.load(open(DATA / "splits.json"))["test"]
    if a.scene:
        recs = [r for r in sp if r["id"] == a.scene]
    else:
        recs = pick_scenes(sp, a.n)
    assert recs, "no suitable scene found in the test split"
 
    dmg, seg = load_models(dev, need_seg=not a.use_gt)
 
    for r in recs:
        pre, post = Image.open(r["pre"]).convert("RGB"), Image.open(r["post"]).convert("RGB")
        out = analyze(dmg, seg, dev, pre, post, r["label"], r["id"], r["event"], a.use_gt, tuple(a.center))
        if out is None:
            print(f"{r['id']}: no buildings found - skipped"); continue
        n = len(json.load(open(out / "buildings.geojson"))["features"])
        print(f"{r['id']}: {n} buildings -> {out}")
 