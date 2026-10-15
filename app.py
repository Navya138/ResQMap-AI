"""ResQMap AI - Streamlit demo.   streamlit run app.py"""
import hashlib, json
from pathlib import Path
import pandas as pd, streamlit as st, folium
from streamlit_folium import st_folium
from config import DEMO, CLASSES, MODELS
from priority import build_queue
 
COL = {"no-damage": "#2ecc71", "minor-damage": "#f1c40f", "major-damage": "#e67e22", "destroyed": "#e74c3c"}
st.set_page_config("ResQMap AI", layout="wide")
st.title("ResQMap AI")
st.caption("From disaster images to response priorities - AI suggests, humans decide.")
 
 
@st.cache_data
def load(p):
    gj = json.load(open(Path(p) / "buildings.geojson")); meta = json.load(open(Path(p) / "meta.json"))
    rows = []
    for f in gj["features"]:
        ring = f["geometry"]["coordinates"][0]
        rows.append(dict(lon=sum(c[0] for c in ring) / len(ring), lat=sum(c[1] for c in ring) / len(ring),
                         damage=f["properties"]["damage"], conf=f["properties"]["conf"]))
    return gj, meta, pd.DataFrame(rows, columns=["lon", "lat", "damage", "conf"])
 
 
@st.cache_resource
def get_models(need_seg):
    import torch
    from infer import load_models
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dmg, seg = load_models(dev, need_seg)
    return dmg, seg, dev
 
 
def upload_panel(sb):
    """Sidebar controls for analysing the user's own before/after pictures. Returns the scene folder or None."""
    sb.markdown("### Upload a scene")
    pre_f = sb.file_uploader("Before image (pre-disaster)", type=["png", "jpg", "jpeg"], key="up_pre")
    post_f = sb.file_uploader("After image (post-disaster)", type=["png", "jpg", "jpeg"], key="up_post")
    lab_f = sb.file_uploader("xBD label file (.json) - optional", type="json", key="up_lab")
    center = (0.0, 0.0)
    if lab_f is None:
        sb.caption("No label file: buildings are found by the segmentation model (needs models/seg.pt). "
                   "Type where the image is on the map.")
        lon = sb.number_input("Longitude", value=0.0, format="%.6f")
        lat = sb.number_input("Latitude", value=0.0, format="%.6f")
        center = (lon, lat)
    run = sb.button("Run analysis", type="primary", disabled=not (pre_f and post_f))
    if run:
        from PIL import Image
        pre, post = Image.open(pre_f).convert("RGB"), Image.open(post_f).convert("RGB")
        use_gt = lab_f is not None
        if pre.size != post.size:
            sb.error(f"The two images must be the same size (before {pre.size}, after {post.size}).")
        elif not use_gt and not (MODELS / "seg.pt").exists():
            sb.error("No label file and no models/seg.pt, so I can't find buildings. Upload the xBD label .json.")
        elif not use_gt and (pre.size[0] % 32 or pre.size[1] % 32):
            sb.error("Without a label file the image width and height must be multiples of 32 (xBD is 1024x1024).")
        else:
            if pre.size != (1024, 1024):
                sb.warning("The model was trained on 1024x1024 tiles, so other sizes may be less accurate.")
            raw = pre_f.getvalue() + post_f.getvalue() + (lab_f.getvalue() if lab_f else b"")
            sid = "upload_" + hashlib.md5(raw).hexdigest()[:8]
            label = None
            if lab_f:
                (DEMO / sid).mkdir(parents=True, exist_ok=True)
                label = DEMO / sid / "label.json"
                label.write_bytes(lab_f.getvalue())
            try:
                from infer import analyze
                with st.spinner("Running the damage model... (this can take a minute on CPU)"):
                    dmg, seg, dev = get_models(need_seg=not use_gt)
                    out = analyze(dmg, seg, dev, pre, post, label, sid, "uploaded scene", use_gt, center)
                if out is None:
                    sb.warning("No buildings were found in this image.")
                else:
                    load.clear()
                    st.session_state["uploaded_scene"] = str(out)
            except Exception as e:
                sb.error(f"Analysis failed: {e}")
    s = st.session_state.get("uploaded_scene")
    return Path(s) if s else None
 
 
sb = st.sidebar
mode = sb.radio("Input", ["Dataset scene", "Upload your own"], horizontal=True)
if mode == "Dataset scene":
    scenes = [p for p in sorted(DEMO.glob("*")) if (p / "buildings.geojson").exists()]
    if not scenes:
        st.warning("No scenes yet. Run `python infer.py --n 4` after training (see README), "
                   "or switch to 'Upload your own' in the sidebar."); st.stop()
    scene = sb.selectbox("Disaster scene", scenes, format_func=lambda p: p.name)
else:
    scene = upload_panel(sb)
    if scene is None:
        st.info("Upload a before and an after image in the sidebar, then press **Run analysis**."); st.stop()
    sb.success(f"Showing: {scene.name}")
 
gj, meta, df = load(str(scene))
if df.empty:
    st.warning("This scene has no detected buildings. Pick another scene in the sidebar, or re-run infer.py.")
    st.stop()
sb.markdown("### Priority weights")
w_sev = sb.slider("Damage severity", 0.0, 1.0, .6, .05)
w_den = sb.slider("Cluster density", 0.0, 1.0, .25, .05)
w_fac = sb.slider("Critical facility nearby", 0.0, 1.0, .15, .05)
eps = sb.slider("Zone radius (m)", 15, 100, 40, 5)
show_all = sb.checkbox("Show undamaged buildings", False)
fac_file = sb.file_uploader("Critical facilities CSV (lon,lat,name)", type="csv")
fac = pd.read_csv(fac_file) if fac_file else None
 
q = build_queue(df, eps_m=eps, w_sev=w_sev, w_den=w_den, w_fac=w_fac, facilities=fac)
 
cols = st.columns(5)
cols[0].metric("Buildings detected", len(df))
for c, k in zip(cols[1:], CLASSES):
    c.metric(k.replace("-damage", ""), int((df.damage == k).sum()))
st.caption(f"Mode: {meta['mode']} - event: {meta['event']}")
 
left, right = st.columns([3, 2])
with right:
    st.subheader("Assessment queue")
    zoom_to = st.selectbox("Zoom to zone", ["(whole scene)"] + q.zone_id.tolist())
    st.dataframe(q.drop(columns=["lon", "lat"]), hide_index=True, height=480, use_container_width=True)
    st.download_button("Download queue (CSV)", q.to_csv(index=False), f"{meta['id']}_queue.csv", "text/csv")
 
with left:
    (s, w), (n, e) = meta["bounds"]
    center, zoom = [(s + n) / 2, (w + e) / 2], 17
    if zoom_to != "(whole scene)":
        r = q[q.zone_id == zoom_to].iloc[0]; center, zoom = [r.lat, r.lon], 20
    m = folium.Map(center, zoom_start=zoom, max_zoom=21, tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Satellite basemap", max_zoom=19).add_to(m)
    for nm in ("pre", "post"):
        folium.raster_layers.ImageOverlay(str(scene / f"{nm}.jpg"), meta["bounds"], name=f"{nm}-disaster image",
                                          show=(nm == "post")).add_to(m)
    feats = [f for f in gj["features"] if show_all or f["properties"]["damage"] != "no-damage"]
    folium.GeoJson({"type": "FeatureCollection", "features": feats}, name="Buildings",
                   style_function=lambda f: {"color": COL[f["properties"]["damage"]], "weight": 1,
                                             "fillColor": COL[f["properties"]["damage"]], "fillOpacity": .55},
                   tooltip=folium.GeoJsonTooltip(["damage", "conf"])).add_to(m)
    for _, r in q.head(15).iterrows():
        folium.Marker([r.lat, r.lon], tooltip=f"#{r['rank']}: {r.reason}",
                      icon=folium.DivIcon(html=f'<div style="background:#c0392b;color:#fff;border-radius:50%;width:24px;height:24px;'
                                               f'text-align:center;line-height:24px;font-weight:700;border:2px solid #fff">{r["rank"]}</div>')).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, height=560, use_container_width=True, returned_objects=[])
 
st.subheader("Before / after")
a, b = st.columns(2)
a.image(str(scene / "pre.jpg"), caption="Pre-disaster", use_container_width=True)
b.image(str(scene / "post.jpg"), caption="Post-disaster", use_container_width=True)
 
