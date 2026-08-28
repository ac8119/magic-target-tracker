"""
MAGIC target-selection explorer — interactive slider cuts over a full MAGIC
catalog with linked on-sky / distance-modulus / [Fe/H]-precision panels.

Data flow
  1. A catalog FITS file is chosen from MAGIC_CATALOG_GLOBS (the filename acts
     as the catalog version label).
  2. First use converts it to a column-pruned float32 Parquet cache in
     data/explorer_cache/ (keyed by filename + mtime + ledger mtime + schema),
     precomputing Galactic l/b, angular separations from the LMC/SMC centers,
     and an `observed` flag = 2" cross-match against data/master_exclusion.csv
     (the tracker's ledger, same tolerance as the app's checker). Build it from
     the command line with:  python3 explorer.py <catalog.fits>
  3. Every widget change re-filters the FULL cached table with numpy boolean
     masks; the headline metrics always come from the full filtered set.
     Only the display decimates: scatter layers switch to full-set 2D
     histograms above SCATTER_MAX rows.

Panels live in the PANELS registry at the bottom — to add a linked panel,
write one function taking (df, mask, ui) and register it.

Only ra/dec are required of a catalog; every cut whose columns are missing is
simply not offered. Column names follow the MAGIC merged catalogs (see CANDS).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUSION_CSV = os.path.join(APP_DIR, "data", "master_exclusion.csv")
CACHE_DIR = os.path.join(APP_DIR, "data", "explorer_cache")

CATALOG_GLOBS = os.environ.get("MAGIC_CATALOG_GLOBS", ":".join([
    "~/Documents/Research/magic-validation/new_distances/*.fits",
    "~/Dropbox (MIT)/my_papers/magic_overview/raw_catalog_to_usable/*.fits",
    "~/Documents/Research/magic-scratch/cats/*.fits",
])).split(":")
DEFAULT_CATALOG = "2025B_magic_noSMC_g195_ebv02_classified.fits"

# local-volume-database copies searched in order (Pace's LVDB)
LVDB_DIRS = [os.environ.get(
                 "MAGIC_LVDB_DIR",
                 "~/Documents/Research/magic-dwarf-outskirts/local_volume_database"),
             os.path.join(APP_DIR, "data", "lvdb")]
LVDB_DWARF_FILES = ["dwarf_mw.csv"]
LVDB_CLUSTER_FILES = ["gc_harris.csv", "gc_mw_new.csv", "gc_dwarf_hosted.csv"]

MATCH_RADIUS_ARCSEC = 2.0        # ledger cross-match, same as app default
LMC = (80.89, -69.76, 5.0)       # ra, dec, default excision radius (deg)
SMC = (13.19, -72.83, 3.0)
SCATTER_MAX = 150_000            # above this, scatter layers become 2D histograms
CHUNK = 2_000_000                # FITS -> Parquet conversion chunk (rows)
SCHEMA_VERSION = 1

# slider bounds = catalog percentiles clipped to these physical windows,
# so a handful of junk-photometry rows can't stretch a slider to uselessness
HARD_BOUNDS = {"pmra": (-30, 30), "pmdec": (-30, 30), "ebv": (0, 1),
               "gi0": (-2, 5), "feh": (-5, 2), "e_feh": (0, 5),
               "dmod": (0, 25), "magerr_g": (0, 2), "magerr_cahk": (0, 2)}

# canonical column -> catalog column candidates (first match wins)
CANDS = {
    "ra": ["ra"], "dec": ["dec"],
    "pmra": ["pmra"], "pmdec": ["pmdec"],
    "ebv": ["ebv_sfd98", "ebv"],
    "feh": ["feh"], "e_feh": ["e_feh"], "dmod": ["dmod"],
    "magerr_g": ["magerr_psf_g"], "magerr_cahk": ["magerr_psf_cahk"],
}
# (g-i)_0: dereddened if available, else instrumental
GI0_CANDS = [("g_dered", "i_dered"), ("mag_psf_g", "mag_psf_i")]


# ──────────────────────── catalog discovery ────────────────────────
def find_catalogs():
    """{basename: path} of every FITS catalog on the configured search paths."""
    out = {}
    for pat in CATALOG_GLOBS:
        for p in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.getsize(p) > 1e6:          # skip tiny helper tables
                out.setdefault(os.path.basename(p), p)
    return out


def cache_paths(cat_path):
    key = "v{}_{}_{}".format(
        SCHEMA_VERSION, int(os.path.getmtime(cat_path)),
        int(os.path.getmtime(EXCLUSION_CSV)) if os.path.exists(EXCLUSION_CSV) else 0)
    stem = os.path.splitext(os.path.basename(cat_path))[0]
    base = os.path.join(CACHE_DIR, f"{stem}__{key}")
    return base + ".parquet", base + ".json"


# ──────────────────────── FITS -> Parquet cache ────────────────────────
def _angsep_deg(ra1, dec1, ra2, dec2):
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    a = (np.sin((dec2 - dec1) / 2) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(np.asarray(ra_deg, float)), np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def build_cache(cat_path, progress=None):
    """Convert one catalog to the pruned Parquet cache + metadata sidecar."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy import units as u
    from scipy.spatial import cKDTree

    tree = None
    if os.path.exists(EXCLUSION_CSV):
        led = pd.read_csv(EXCLUSION_CSV)
        tree = cKDTree(_unit_vectors(led["ra"].values, led["dec"].values))
    chord = 2 * np.sin(np.radians(MATCH_RADIUS_ARCSEC / 3600.0) / 2)

    frames, missing = [], []
    with fits.open(cat_path, memmap=True) as hdul:
        hdu = hdul[1]
        n = hdu.header["NAXIS2"]
        names = set(hdu.columns.names)
        for start in range(0, n, CHUNK):
            rec = hdu.data[start:start + CHUNK]
            cols = {}
            for canon, cands in CANDS.items():
                src = next((c for c in cands if c in names), None)
                if src is None:
                    if start == 0:
                        missing.append(canon)
                    cols[canon] = np.full(len(rec), np.nan, dtype=np.float32)
                else:
                    cols[canon] = np.asarray(rec[src], dtype=np.float32)
            gi = next(((g, i) for g, i in GI0_CANDS if g in names and i in names), None)
            if gi:
                cols["gi0"] = (np.asarray(rec[gi[0]], np.float32)
                               - np.asarray(rec[gi[1]], np.float32))
            else:
                if start == 0:
                    missing.append("gi0")
                cols["gi0"] = np.full(len(rec), np.nan, dtype=np.float32)
            if "star_class" in names:
                sc = np.char.strip(rec["star_class"].astype(str))
            elif "is_rgb" in names:
                sc = np.where(rec["is_rgb"], "RGB", "MS")
            else:
                sc = np.full(len(rec), "unknown")
            cols["star_class"] = sc

            ra64 = np.asarray(rec["ra"], float)
            dec64 = np.asarray(rec["dec"], float)
            gal = SkyCoord(ra=ra64 * u.deg, dec=dec64 * u.deg).galactic
            cols["l"] = gal.l.deg.astype(np.float32)
            cols["b"] = gal.b.deg.astype(np.float32)
            cols["sep_lmc"] = _angsep_deg(LMC[0], LMC[1], ra64, dec64).astype(np.float32)
            cols["sep_smc"] = _angsep_deg(SMC[0], SMC[1], ra64, dec64).astype(np.float32)
            if tree is not None:
                d, _ = tree.query(_unit_vectors(ra64, dec64), k=1,
                                  distance_upper_bound=chord)
                cols["observed"] = d <= chord
            else:
                cols["observed"] = np.zeros(len(rec), bool)
            frames.append(pd.DataFrame(cols))
            if progress:
                progress(min(1.0, (start + len(rec)) / n))

    df = pd.concat(frames, ignore_index=True)
    df["star_class"] = df["star_class"].astype("category")

    def rng(col, lo_q=0.5, hi_q=99.5):
        v = df[col].values
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        return [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))]

    meta = {
        "catalog": cat_path, "n_rows": len(df),
        "n_observed": int(df["observed"].sum()),
        "missing": missing,
        "star_classes": df["star_class"].value_counts().to_dict(),
        "ranges": {c: rng(c) for c in
                   ("pmra", "pmdec", "ebv", "gi0", "feh", "e_feh", "dmod",
                    "magerr_g", "magerr_cahk")},
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    pq, js = cache_paths(cat_path)
    df.to_parquet(pq, index=False)
    with open(js, "w") as f:
        json.dump(meta, f, indent=1)
    return pq, js


# ──────────────────────── filtering (pure, tested) ────────────────────────
def apply_cuts(df, cuts):
    """AND together enabled cuts; NaN never passes an enabled numeric cut.

    Each cut: {"col", "kind": range|max|min|isin, "value", "enabled": bool}.
    Returns a boolean numpy mask over the full table (never decimated).
    """
    mask = np.ones(len(df), dtype=bool)
    for cut in cuts:
        if not cut.get("enabled", True):
            continue
        v = (df[cut["col"]].to_numpy() if hasattr(df[cut["col"]], "to_numpy")
             else np.asarray(df[cut["col"]]))
        kind, val = cut["kind"], cut["value"]
        if kind == "range":
            mask &= (v >= val[0]) & (v <= val[1])
        elif kind == "max":
            mask &= v <= val
        elif kind == "min":
            mask &= v >= val
        elif kind == "isin":
            mask &= np.isin(np.asarray(v, dtype=object), list(val))
        else:
            raise ValueError(f"unknown cut kind {kind}")
    return mask


def metrics(df, mask):
    """Headline numbers, always from the FULL filtered set."""
    n = int(mask.sum())
    n_obs = int((mask & df["observed"].to_numpy()).sum())
    return {"selected": n, "observed": n_obs, "remaining": n - n_obs}


# ──────────────────────── LVDB overlays ────────────────────────
def load_lvdb():
    """(dwarfs, clusters) DataFrames [name, ra, dec] from the first hit per file."""
    def gather(fnames):
        frames = []
        for fn in fnames:
            for d in LVDB_DIRS:
                p = os.path.join(os.path.expanduser(d), fn)
                if os.path.exists(p):
                    t = pd.read_csv(p)
                    frames.append(pd.DataFrame({
                        "name": t.get("name", t.get("key", "")),
                        "ra": pd.to_numeric(t["ra"], errors="coerce"),
                        "dec": pd.to_numeric(t["dec"], errors="coerce")}).dropna())
                    break
        return (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame(columns=["name", "ra", "dec"]))
    return gather(LVDB_DWARF_FILES), gather(LVDB_CLUSTER_FILES)


# ═══════════════════════════ Streamlit page ═══════════════════════════
def _st():
    import streamlit as st
    return st


def _load_cached(pq, js):
    st = _st()

    @st.cache_resource(show_spinner="Loading catalog cache ...")
    def _load(pq_path, js_path):
        df = pd.read_parquet(pq_path)
        with open(js_path) as f:
            meta = json.load(f)
        return df, meta
    return _load(pq, js)


def _range_slider(st, label, bounds, key, fmt="%.2f", pad=0.05):
    lo, hi = bounds
    span = (hi - lo) or 1.0
    lo, hi = lo - pad * span, hi + pad * span
    on = st.checkbox(label, key=key + ":on")
    val = st.slider(label, lo, hi, (lo, hi), (hi - lo) / 200, format=fmt,
                    key=key, label_visibility="collapsed", disabled=not on)
    return on, val


def _max_slider(st, label, bounds, key, fmt="%.3f"):
    lo, hi = 0.0, bounds[1]
    on = st.checkbox(label, key=key + ":on")
    val = st.slider(label, lo, hi, hi, (hi - lo) / 200, format=fmt,
                    key=key, label_visibility="collapsed", disabled=not on)
    return on, val


def render():
    """The 'Target explorer' page."""
    st = _st()
    st.header("Target-selection explorer")

    cats = find_catalogs()
    if not cats:
        st.info("No MAGIC catalogs found on this machine "
                "(searched: {}).".format(", ".join(CATALOG_GLOBS)))
        return
    labels = {f"{n}  ({os.path.getsize(p) / 1e9:.1f} GB)": n for n, p in cats.items()}
    default = next((i for i, n in enumerate(labels.values()) if n == DEFAULT_CATALOG), 0)
    choice = st.selectbox("Catalog (version)", list(labels), index=default)
    cat_path = cats[labels[choice]]

    pq, js = cache_paths(cat_path)
    if not (os.path.exists(pq) and os.path.exists(js)):
        st.warning("No Parquet cache yet for this catalog (first use). Building it "
                   "reads the whole FITS file once — a few minutes for multi-GB files. "
                   f"You can also prebuild from a terminal:\n\n"
                   f"`python3 explorer.py '{cat_path}'`")
        if st.button("Build cache now"):
            bar = st.progress(0.0)
            build_cache(cat_path, progress=bar.progress)
            bar.empty()
            st.rerun()
        return
    df, meta = _load_cached(pq, js)
    rng = {}
    for col, r in meta["ranges"].items():
        if r:
            h = HARD_BOUNDS.get(col, (-np.inf, np.inf))
            rng[col] = [max(r[0], h[0]), min(r[1], h[1])]
    key = os.path.basename(pq)  # widget namespace per catalog version

    ctrl, view = st.columns([1, 3], gap="medium")

    # ── cuts (each with an enable toggle; disabled cuts filter nothing) ──
    cuts = []
    with ctrl:
        st.subheader("Cuts")
        pops = {"both": ["RGB", "MS"], "RGB": ["RGB"], "MS": ["MS"],
                "ambiguous": ["ambiguous"], "all": sorted(meta["star_classes"])}
        pop = st.radio("Population ([Fe/H], dmod are per-class values)",
                       list(pops), horizontal=True, key=key + ":pop")
        cuts.append({"col": "star_class", "kind": "isin", "value": pops[pop],
                     "enabled": pop != "all"})

        def add(cut_on, col, kind, value):
            cuts.append({"col": col, "kind": kind, "value": value, "enabled": cut_on})

        for col, label, fmt in (("pmra", "pmra (mas/yr)", "%.1f"),
                                ("pmdec", "pmdec (mas/yr)", "%.1f"),
                                ("gi0", "(g-i)₀ color", "%.2f"),
                                ("feh", "[Fe/H]", "%.2f"),
                                ("dmod", "distance modulus", "%.2f")):
            if rng.get(col):
                on, val = _range_slider(st, label, rng[col], f"{key}:{col}", fmt)
                add(on, col, "range", val)
        for col, label in (("ebv", "E(B-V) max"),
                           ("e_feh", "[Fe/H] error max"),
                           ("magerr_g", "depth: σ(g) max (mag)"),
                           ("magerr_cahk", "depth: σ(CaHK) max (mag)")):
            if rng.get(col):
                on, val = _max_slider(st, label, rng[col], f"{key}:{col}")
                add(on, col, "max", val)

        classes = sorted(meta["star_classes"])
        on = st.checkbox("star_class", key=key + ":cls:on")
        sel = st.multiselect("star_class", classes, default=classes,
                             key=key + ":cls", label_visibility="collapsed",
                             disabled=not on)
        add(on, "star_class", "isin", sel)

        st.subheader("Excise Clouds")
        for name, (cra, cdec, rdef), col in (("LMC", LMC, "sep_lmc"),
                                             ("SMC", SMC, "sep_smc")):
            on = st.checkbox(f"cut {name} center ({cra}, {cdec})", key=f"{key}:{col}:on")
            r = st.slider(f"{name} radius (deg)", 0.5, 15.0, rdef, 0.5,
                          key=f"{key}:{col}", disabled=not on)
            add(on, col, "min", r)

    mask = apply_cuts(df, cuts)
    m = metrics(df, mask)

    with view:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("In catalog", f"{len(df):,}")
        c2.metric("Passing cuts", f"{m['selected']:,}")
        c3.metric("Already observed", f"{m['observed']:,}")
        c4.metric("Remaining to observe", f"{m['remaining']:,}")
        ui = {"st": st, "key": key}
        for title, fn in PANELS:
            st.subheader(title)
            fn(df, mask, ui)


# ──────────────────────── linked panels ────────────────────────
# Each panel: fn(df, mask, ui) — df is the FULL table, mask the current cuts.
# Register new panels in PANELS below; they automatically share the same cuts.

def _density_or_scatter(fig_go, x, y, name, nbins=(360, 200)):
    """Full-set 2D histogram above SCATTER_MAX, WebGL scatter below."""
    if len(x) > SCATTER_MAX:
        H, xe, ye = np.histogram2d(x, y, bins=nbins)
        return fig_go.Heatmap(
            x=0.5 * (xe[:-1] + xe[1:]), y=0.5 * (ye[:-1] + ye[1:]),
            z=np.where(H.T > 0, np.log10(H.T, where=H.T > 0), np.nan),
            colorscale="Viridis", colorbar=dict(title="log₁₀ N"), name=name)
    return fig_go.Scattergl(x=x, y=y, mode="markers", name=name,
                            marker=dict(size=2, color="#4c78a8", opacity=0.5))


def panel_sky(df, mask, ui):
    import plotly.graph_objects as go
    st = ui["st"]
    c1, c2, c3 = st.columns(3)
    frame = c1.radio("Frame", ["Equatorial (RA/Dec)", "Galactic (l/b)"],
                     horizontal=True, key=ui["key"] + ":frame")
    show_dw = c2.checkbox("dwarf galaxies (LVDB)", True, key=ui["key"] + ":dw")
    show_gc = c3.checkbox("MW star clusters (LVDB)", True, key=ui["key"] + ":gc")
    gal = frame.startswith("Galactic")
    xc, yc = ("l", "b") if gal else ("ra", "dec")

    x, y = df[xc].to_numpy()[mask], df[yc].to_numpy()[mask]
    fig = go.Figure()
    if len(x):
        fig.add_trace(_density_or_scatter(go, x, y, "targets"))
    obs = mask & df["observed"].to_numpy()
    if obs.any():
        fig.add_trace(go.Scattergl(
            x=df[xc].to_numpy()[obs], y=df[yc].to_numpy()[obs], mode="markers",
            name="already observed",
            marker=dict(symbol="x", size=7, color="#e45756")))

    if len(x):
        xr = [float(np.nanmin(x)), float(np.nanmax(x))]
        yr = [float(np.nanmin(y)), float(np.nanmax(y))]
    else:
        xr, yr = [0, 360], [-90, 90]
    dwarfs, clusters = load_lvdb()
    for show, cat, sym, color, label in (
            (show_dw, dwarfs, "star", "#f2b701", "dwarf galaxies"),
            (show_gc, clusters, "triangle-up", "#00b8d9", "star clusters")):
        if not show or not len(cat):
            continue
        if gal:
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            g = SkyCoord(ra=cat["ra"].values * u.deg,
                         dec=cat["dec"].values * u.deg).galactic
            cx, cy = g.l.deg, g.b.deg
        else:
            cx, cy = cat["ra"].values, cat["dec"].values
        inview = ((cx >= xr[0]) & (cx <= xr[1]) & (cy >= yr[0]) & (cy <= yr[1]))
        fig.add_trace(go.Scatter(
            x=cx[inview], y=cy[inview], mode="markers+text", name=label,
            text=cat["name"].values[inview], textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(symbol=sym, size=9, color=color,
                        line=dict(width=1, color="black"))))

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=xc, yaxis_title=yc,
                      legend=dict(orientation="h", y=1.06))
    if not gal:
        fig.update_xaxes(autorange="reversed")  # RA increases leftward
    st.plotly_chart(fig, use_container_width=True)


def panel_dmod(df, mask, ui):
    import plotly.graph_objects as go
    st = ui["st"]
    v = df["dmod"].to_numpy()[mask]
    v = v[np.isfinite(v)]
    fig = go.Figure()
    if len(v):
        cnt, edges = np.histogram(v, bins=120)
        fig.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=cnt,
                             name="targets", marker_color="#4c78a8"))
        o = df["dmod"].to_numpy()[mask & df["observed"].to_numpy()]
        o = o[np.isfinite(o)]
        if len(o):
            cnt2, _ = np.histogram(o, bins=edges)
            fig.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=cnt2,
                                 name="already observed", marker_color="#e45756"))
        fig.update_layout(barmode="overlay", height=340,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="distance modulus (per-class)",
                          yaxis_title="N", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No finite dmod values in the current selection "
                   "(ambiguous/invalid stars have none).")


def panel_feh(df, mask, ui):
    import plotly.graph_objects as go
    st = ui["st"]
    x = df["feh"].to_numpy()[mask]
    y = df["e_feh"].to_numpy()[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    fig = go.Figure()
    if ok.any():
        fig.add_trace(_density_or_scatter(go, x[ok], y[ok], "targets",
                                          nbins=(240, 160)))
        obs = mask & df["observed"].to_numpy()
        xo, yo = df["feh"].to_numpy()[obs], df["e_feh"].to_numpy()[obs]
        oko = np.isfinite(xo) & np.isfinite(yo)
        if oko.any():
            fig.add_trace(go.Scattergl(
                x=xo[oko], y=yo[oko], mode="markers", name="already observed",
                marker=dict(symbol="x", size=7, color="#e45756")))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="[Fe/H] (per-class)",
                          yaxis_title="σ([Fe/H])")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No finite [Fe/H] values in the current selection.")


PANELS = [
    ("On-sky", panel_sky),
    ("Distance modulus", panel_dmod),
    ("[Fe/H] vs its uncertainty", panel_feh),
]


# ──────────────────────── CLI cache prebuild ────────────────────────
if __name__ == "__main__":
    import sys
    import time
    for path in sys.argv[1:]:
        t0 = time.time()
        print(f"Building cache for {path} ...")
        pq, _ = build_cache(path, progress=lambda f: print(f"  {f:5.0%}", end="\r"))
        print(f"\n  -> {pq}  ({time.time() - t0:.0f} s)")
