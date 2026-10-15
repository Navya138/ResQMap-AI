"""Stage 3 - turn per-building damage predictions into a ranked inspection queue of zones."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from config import SEV_WEIGHT

COLS = ["rank", "zone_id", "score", "n_damaged", "destroyed", "major", "minor",
        "needs_verification", "reason", "lon", "lat"]


def to_xy(lon, lat, lat0):
    return np.c_[np.asarray(lon) * 111320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110540]


def _norm(s):
    s = np.asarray(s, dtype=float)
    return s / s.max() if s.max() > 0 else s * 0


def build_queue(df, eps_m=40, min_samples=3, w_sev=.6, w_den=.25, w_fac=.15,
                facilities=None, low_conf=.6):
    """df columns: lon, lat, damage, conf.  facilities (optional): lon, lat, name."""
    if df is None or len(df) == 0 or "damage" not in df:
        return pd.DataFrame(columns=COLS)
    d = df[df.damage != "no-damage"].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=COLS)
    lat0 = d.lat.mean()
    xy = to_xy(d.lon, d.lat, lat0)
    lab = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xy)
    nxt = lab.max() + 1
    for i in np.where(lab == -1)[0]:          # isolated damaged buildings become their own zone
        lab[i] = nxt; nxt += 1
    d["zone"] = lab

    rows = []
    for z, g in d.groupby("zone"):
        gx = xy[g.index]
        ext = (gx.max(0) - gx.min(0)) + 30
        area_ha = max(ext[0] * ext[1], 600) / 1e4
        c = g.damage.value_counts()
        rows.append(dict(zone_id=int(z), n_damaged=len(g),
                         destroyed=int(c.get("destroyed", 0)), major=int(c.get("major-damage", 0)),
                         minor=int(c.get("minor-damage", 0)),
                         sev=float((g.damage.map(SEV_WEIGHT) * g.conf).sum()),
                         density=len(g) / area_ha, lon=g.lon.mean(), lat=g.lat.mean(),
                         uncertain=float((g.conf < low_conf).mean())))
    q = pd.DataFrame(rows)

    use_fac = facilities is not None and len(facilities) > 0
    fac = np.zeros(len(q))
    near = [""] * len(q)
    if use_fac:
        fxy = to_xy(facilities.lon, facilities.lat, lat0)
        zxy = to_xy(q.lon, q.lat, lat0)
        dist = np.linalg.norm(zxy[:, None, :] - fxy[None, :, :], axis=2)
        fac = np.clip(1 - dist.min(1) / 300, 0, 1)
        names = facilities["name"].tolist() if "name" in facilities else ["facility"] * len(facilities)
        near = [names[j] if f > 0 else "" for j, f in zip(dist.argmin(1), fac)]

    wsum = w_sev + w_den + (w_fac if use_fac else 0)
    q["score"] = (100 * (w_sev * _norm(q.sev) + w_den * _norm(q.density) + w_fac * fac) / max(wsum, 1e-9)).round(1)
    q["needs_verification"] = q.uncertain > .4

    def why(i, r):
        s = f"{r.destroyed} destroyed, {r.major} major, {r.minor} minor"
        if r.density >= q.density.quantile(.75): s += "; dense cluster"
        if near[i]: s += f"; near {near[i]}"
        if r.needs_verification: s += "; low model confidence"
        return s
    q["reason"] = [why(i, r) for i, r in q.iterrows()]
    q = q.sort_values("score", ascending=False).reset_index(drop=True)
    q.insert(0, "rank", q.index + 1)
    return q[COLS]
