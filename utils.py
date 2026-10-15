import json
import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import MultiPolygon
from config import CROP, PAD, IMAGENET_MEAN, IMAGENET_STD

MEAN3 = np.array(IMAGENET_MEAN, dtype=np.float32)
STD3 = np.array(IMAGENET_STD, dtype=np.float32)


def polys(wkt_str):
    g = wkt.loads(wkt_str)
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g]


def load_features(label_path):
    d = json.load(open(label_path))
    return d["features"]["xy"], d["features"].get("lng_lat", [])


def mask_from_label(label_path, size=1024):
    """Rasterise building footprints from an xBD label json."""
    xy, _ = load_features(label_path)
    m = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(m)
    for f in xy:
        if f["properties"].get("feature_type") != "building":
            continue
        for p in polys(f["wkt"]):
            dr.polygon([tuple(c[:2]) for c in p.exterior.coords], fill=1)
    return np.array(m, dtype=np.uint8)


def crop_pair(pre, post, bbox):
    """Square crop around a building from pre and post images, pasted side by side."""
    minx, miny, maxx, maxy = bbox
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny, 16) / 2 + PAD
    box = tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    a = pre.crop(box).resize((CROP, CROP), Image.BILINEAR)
    b = post.crop(box).resize((CROP, CROP), Image.BILINEAR)
    out = Image.new("RGB", (2 * CROP, CROP))
    out.paste(a, (0, 0))
    out.paste(b, (CROP, 0))
    return out


def pair_to_array(img):
    """Side-by-side PIL image -> normalised float32 array of shape 6 x CROP x CROP."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    pre, post = a[:, :CROP], a[:, CROP:]
    pre, post = (pre - MEAN3) / STD3, (post - MEAN3) / STD3
    return np.concatenate([pre, post], axis=2).transpose(2, 0, 1).astype(np.float32)


def fit_affine(label_path):
    """Pixel -> (lon, lat) affine fitted from the xBD label's paired xy / lng_lat polygons."""
    xy, ll = load_features(label_path)
    geo = {f["properties"].get("uid"): wkt.loads(f["wkt"]).centroid for f in ll}
    P, G = [], []
    for f in xy:
        u = f["properties"].get("uid")
        if u in geo:
            c = wkt.loads(f["wkt"]).centroid
            P.append([c.x, c.y, 1.0])
            G.append([geo[u].x, geo[u].y])
    if len(P) < 3:
        return None
    X, *_ = np.linalg.lstsq(np.array(P), np.array(G), rcond=None)
    return X.T  # 2 x 3


def fallback_affine(lon, lat, mpp=0.5, size=1024):
    dlon = mpp / (111320 * np.cos(np.radians(lat)))
    dlat = -mpp / 110540
    return np.array([[dlon, 0, lon - size / 2 * dlon], [0, dlat, lat - size / 2 * dlat]])


def pix2geo(M, x, y):
    lon, lat = M @ np.array([x, y, 1.0])
    return float(lon), float(lat)
