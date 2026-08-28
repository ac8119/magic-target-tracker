"""
MAGIC target-selection explorer — interactive slider cuts over a full MAGIC
catalog with linked on-sky / distance-modulus / [Fe/H]-precision panels.

Data flow
  1. A catalog FITS file is chosen from MAGIC_CATALOG_GLOBS (the filename acts
     as the catalog version label).
  2. First use converts it to a column-pruned float32 Parquet cache in
     data/explorer_cache/ (keyed by filename + mtime + ledger mtime + schema),
     precomputing Galactic l/b, angular separations from the LMC/SMC centers,
     and a category-aware ledger cross-match (1", the MAGIC pipeline
     convention) against data/master_exclusion.csv: obs_cat = observed-by-us
     category (MAGIC/nonMAGIC Magellan, GMOS; beats literature) and lit_known
     = Literature match. Build from the command line with:
     python3 explorer.py <catalog.fits>
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

# FALLBACK ONLY: Ani's workstation paths, used when neither
# st.secrets["catalogs"] nor MAGIC_CATALOG_GLOBS is set (keeps a bare local
# checkout working). Deployments configure paths in their own secrets.
LOCAL_FALLBACK_GLOBS = [
    "~/Documents/Research/magic-validation/new_distances/*.fits",
    "~/Dropbox (MIT)/my_papers/magic_overview/raw_catalog_to_usable/*.fits",
    "~/Documents/Research/magic-scratch/cats/*.fits",
]
DEFAULT_CATALOG = "2025B_magic_noSMC_g195_ebv02_classified.fits"

# local-volume-database copies searched in order (Pace's LVDB)
LVDB_DIRS = [os.environ.get(
                 "MAGIC_LVDB_DIR",
                 "~/Documents/Research/magic-dwarf-outskirts/local_volume_database"),
             os.path.join(APP_DIR, "data", "lvdb")]
LVDB_DWARF_FILES = ["dwarf_mw.csv"]
LVDB_CLUSTER_FILES = ["gc_harris.csv", "gc_mw_new.csv", "gc_dwarf_hosted.csv"]
LVDB_MAX_DIST_KPC = 300.0     # drop local-volume systems beyond the MW halo

MATCH_RADIUS_ARCSEC = 1.0        # ledger cross-match — the 1" MAGIC pipeline
                                 # convention (make_targets.py, SIMBAD query);
                                 # the app's interactive checker stays at 2"
OBSERVED_CATEGORIES = ("MAGIC_Magellan", "nonMAGIC_Magellan", "GMOS")
SIMBAD_RADIUS_ARCSEC = 1.0       # CDS X-Match radius (make_targets.py convention)
SIMBAD_MAX_ROWS = 50_000         # refuse to upload more rows than this to CDS
SIMBAD_CACHE_CSV = os.path.join(APP_DIR, "data", "simbad_cache.csv")
LMC = (80.89, -69.76, 5.0)       # ra, dec, default excision radius (deg)
SMC = (13.19, -72.83, 3.0)
SCATTER_MAX = 150_000            # above this, scatter layers become 2D histograms
CHUNK = 2_000_000                # FITS -> Parquet conversion chunk (rows)
SCHEMA_VERSION = 4

# slider bounds = catalog percentiles clipped to these physical windows,
# so a handful of junk-photometry rows can't stretch a slider to uselessness
HARD_BOUNDS = {"pmra": (-30, 30), "pmdec": (-30, 30), "ebv": (0, 1),
               "gi0": (-2, 5), "feh": (-5, 2), "e_feh": (0, 5),
               "dmod": (0, 25), "mag_g": (10, 25)}

# ── Fiducial preset ─────────────────────────────────────────────────────
# One-click "standard MAGIC low-metallicity giant" selection, applied by the
# [Fiducial cuts] button. EDIT ME as the survey conventions evolve.
# Provenance: scripts/make_targets.py in magic-low-metallicity-followup —
#   giants:  |pmra| < 3.5 and |pmdec| < 3.5 mas/yr (its hardcoded giant cut);
#            its loggs < 4.0 and parallax < 0.4 cuts have no cached column
#            here, so population = "RGB" (the catalog classification) stands
#            in for them;
#   [Fe/H] < -3.0: the "fehm-30" threshold used by nearly every dated run;
#   mag_g <= 18.5: gmax of the most recent MIKE selection (155_185 runs;
#            the 251018 MagE backup used 19.0) — the pipeline's gmin bright
#            limit has no counterpart since the depth cut is max-only;
#   LMC/SMC excision radii per the *_noSMC selection variants.
FIDUCIAL = {
    "population": "RGB",
    "pmra": (-3.5, 3.5),      # mas/yr
    "pmdec": (-3.5, 3.5),     # mas/yr
    "feh": (-5.0, -3.0),      # dex
    "gi0": (0.2, 1.5),        # overview paper quality cut (magic.tex ~l.447):
                              # "0.2 < (g-i)_0 < 1.5 to exclude the coolest
                              # stars and regions significantly bluer than
                              # the main-sequence turnoff"
    "mag_g": 18.5,            # keep stars brighter than this
    "dmod": (17.39, 25.0),    # dmod > 17.39 <=> d > 30 kpc (5*log10(30000/10));
                              # the upper bound is the slider's hard limit,
                              # i.e. effectively no far cut
    "ebv": 0.05,              # stricter than the catalog's baked-in E(B-V)<0.2
    "sep_lmc": 5.0,           # excision radius around the LMC (deg)
    "sep_smc": 3.0,           # excision radius around the SMC (deg)
}
# every cut column a Clear-all must switch off
CUT_COLS = ("pmra", "pmdec", "gi0", "feh", "dmod", "ebv", "e_feh", "mag_g",
            "sep_lmc", "sep_smc")

# every column a built cache / cloud subset carries
SCHEMA_COLUMNS = ["ra", "dec", "pmra", "pmdec", "ebv", "feh", "e_feh", "dmod",
                  "mag_g", "gi0", "star_class", "l", "b", "sep_lmc", "sep_smc",
                  "obs_cat", "lit_known"]
RANGE_COLS = ("pmra", "pmdec", "ebv", "gi0", "feh", "e_feh", "dmod", "mag_g")

# canonical column -> catalog column candidates (first match wins)
CANDS = {
    "ra": ["ra"], "dec": ["dec"],
    "pmra": ["pmra"], "pmdec": ["pmdec"],
    "ebv": ["ebv_sfd98", "ebv"],
    "feh": ["feh"], "e_feh": ["e_feh"], "dmod": ["dmod"],
    "mag_g": ["mag_psf_g"],
}
# (g-i)_0: dereddened if available, else instrumental
GI0_CANDS = [("g_dered", "i_dered"), ("mag_psf_g", "mag_psf_i")]


# ──────────────────────── catalog discovery ────────────────────────
def resolve_globs(secrets, user=None, env_value=None):
    """Catalog search globs, in precedence order:
    1. secrets["catalogs"]: optional per-user globs under [catalogs.users]
       (keyed by the password-gate login), merged user-first with the
       deployment-wide "globs" list — the primary mechanism;
    2. the MAGIC_CATALOG_GLOBS env var (colon-separated) for CLI/dev use;
    3. LOCAL_FALLBACK_GLOBS.
    `secrets` is any mapping shaped like st.secrets (testable directly)."""
    try:
        cats = secrets["catalogs"]
    except (KeyError, FileNotFoundError, TypeError):
        cats = {}
    globs = []
    users = cats.get("users", {})
    if user and user in users:
        globs += [str(g) for g in users[user]]
    globs += [str(g) for g in cats.get("globs", [])]
    if globs:
        return globs
    if env_value:
        return [g for g in env_value.split(":") if g]
    return list(LOCAL_FALLBACK_GLOBS)


def path_allowed(path, allowed_roots):
    """True if a user-entered path/glob stays inside an allowlisted root.
    The non-wildcard prefix is resolved (symlinks, '..') before checking, so
    web users cannot browse outside the roots a deployment explicitly opens."""
    base = os.path.realpath(os.path.expanduser(str(path).split("*")[0]))
    for root in allowed_roots:
        r = os.path.realpath(os.path.expanduser(str(root))).rstrip(os.sep)
        if base == r or base.startswith(r + os.sep):
            return True
    return False


def _secrets():
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return {}


def find_catalogs(user=None, extra_globs=()):
    """{basename: path} of every FITS catalog on the resolved search paths.
    extra_globs: session-scoped, allowlist-validated paths added in the UI."""
    pats = list(extra_globs) + resolve_globs(
        _secrets(), user, os.environ.get("MAGIC_CATALOG_GLOBS"))
    out = {}
    for pat in pats:
        p_exp = os.path.expanduser(pat)
        for p in sorted(glob.glob(p_exp) if any(c in p_exp for c in "*?[")
                        else glob.glob(os.path.join(p_exp, "*.fits"))):
            if p.endswith(".fits") and os.path.getsize(p) > 1e6:
                out.setdefault(os.path.basename(p), p)
    return out


def release_spec(secrets):
    """Parsed [catalogs.release] secrets: repo/tag/token plus asset name list
    (a plain string `asset` becomes a one-element list); None if unset or
    incomplete. The token never leaves this dict."""
    try:
        rel = secrets["catalogs"]["release"]
    except (KeyError, FileNotFoundError, TypeError):
        return None
    assets = rel.get("asset", [])
    if isinstance(assets, str):
        assets = [assets]
    if not (rel.get("repo") and rel.get("tag") and assets):
        return None
    return {"repo": str(rel["repo"]), "tag": str(rel["tag"]),
            "assets": [str(a) for a in assets],
            "token": str(rel.get("token", ""))}


def fetch_release_asset(repo, tag, asset, token, dest, session=None,
                        progress=None):
    """Stream one GitHub release asset to dest via the API: look up the asset
    id under the tag, then GET the asset with Accept: octet-stream following
    the redirect. Errors carry HTTP statuses only — never the token."""
    import requests
    s = session or requests.Session()
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    r = s.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
              headers={**auth, "Accept": "application/vnd.github+json"},
              timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"release lookup failed (HTTP {r.status_code})")
    hit = next((a for a in r.json().get("assets", [])
                if a.get("name") == asset), None)
    if hit is None:
        raise RuntimeError(f"asset '{asset}' not found in release {tag}")
    r2 = s.get(f"https://api.github.com/repos/{repo}/releases/assets/{hit['id']}",
               headers={**auth, "Accept": "application/octet-stream"},
               stream=True, allow_redirects=True, timeout=300)
    if r2.status_code != 200:
        raise RuntimeError(f"asset download failed (HTTP {r2.status_code})")
    total = int(hit.get("size") or 0)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp, done = dest + ".part", 0
    with open(tmp, "wb") as f:
        for chunk in r2.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(min(1.0, done / total))
    os.replace(tmp, dest)
    return dest


def release_paths(rel, asset):
    """Cache-dir paths for a downloaded release asset (already schema
    Parquet — the download IS the cache, so one fetch per container)."""
    stem = os.path.splitext(asset)[0]
    base = os.path.join(CACHE_DIR, f"release_{rel['tag']}_{stem}")
    return base + ".parquet", base + ".json"


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


def _make_meta(df, catalog, missing=()):
    """Metadata sidecar (slider ranges, class counts) for a schema DataFrame —
    used for built caches and for downloaded release-asset subsets alike."""
    def rng(col, lo_q=0.5, hi_q=99.5):
        if col not in df.columns:
            return None
        v = np.asarray(df[col].values, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        return [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))]

    return {
        "catalog": str(catalog), "n_rows": int(len(df)),
        "n_observed_us": (int((df["obs_cat"].astype(str) != "").sum())
                          if "obs_cat" in df.columns else 0),
        "n_lit_known": (int(df["lit_known"].sum())
                        if "lit_known" in df.columns else 0),
        "missing": list(missing),
        "star_classes": {str(k): int(n) for k, n
                         in df["star_class"].value_counts().items()},
        "ranges": {c: rng(c) for c in RANGE_COLS},
    }


def classify_against_ledger(ra, dec, ledger, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """Category-aware ledger cross-match at radius_arcsec.

    Returns (obs_cat, lit_known): obs_cat is the OBSERVED_CATEGORIES category
    of the nearest observation within the radius ("" if none — an observation
    always beats a literature entry), lit_known flags a Literature entry
    within the radius.
    """
    from scipy.spatial import cKDTree
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
    xyz = _unit_vectors(ra, dec)
    obs_cat = np.full(len(xyz), "", dtype=object)
    lit_known = np.zeros(len(xyz), dtype=bool)
    obs = ledger[ledger["category"].isin(OBSERVED_CATEGORIES)]
    if len(obs):
        d, i = cKDTree(_unit_vectors(obs["ra"].values, obs["dec"].values)).query(
            xyz, k=1, distance_upper_bound=chord)
        hit = d <= chord
        obs_cat[hit] = obs["category"].values[np.clip(i[hit], 0, len(obs) - 1)]
    lit = ledger[ledger["category"] == "Literature"]
    if len(lit):
        d, _ = cKDTree(_unit_vectors(lit["ra"].values, lit["dec"].values)).query(
            xyz, k=1, distance_upper_bound=chord)
        lit_known = d <= chord
    return obs_cat, lit_known


def build_cache(cat_path, progress=None):
    """Convert one catalog to the pruned Parquet cache + metadata sidecar."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy import units as u

    ledger = (pd.read_csv(EXCLUSION_CSV) if os.path.exists(EXCLUSION_CSV)
              else pd.DataFrame(columns=["ra", "dec", "category"]))

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
            obs_cat, lit_known = classify_against_ledger(ra64, dec64, ledger)
            cols["obs_cat"] = obs_cat
            cols["lit_known"] = lit_known
            frames.append(pd.DataFrame(cols))
            if progress:
                progress(min(1.0, (start + len(rec)) / n))

    df = pd.concat(frames, ignore_index=True)
    df["star_class"] = df["star_class"].astype("category")
    df["obs_cat"] = df["obs_cat"].astype("category")

    meta = _make_meta(df, cat_path, missing)
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


def metrics(df, mask, lit_is_observed=True):
    """Headline numbers, always from the FULL filtered set.

    observed_us = the OBSERVED_CATEGORIES ledger entries (our observations);
    literature  = Literature-only matches (an observation beats literature);
    remaining excludes literature-known stars only if lit_is_observed.
    """
    n = int(mask.sum())
    obs_us = mask & (np.asarray(df["obs_cat"]) != "")
    lit = mask & np.asarray(df["lit_known"]) & ~obs_us
    n_obs, n_lit = int(obs_us.sum()), int(lit.sum())
    return {"selected": n, "observed_us": n_obs, "literature": n_lit,
            "remaining": n - n_obs - (n_lit if lit_is_observed else 0)}


# ──────────────────────── LVDB overlays ────────────────────────
OCC_BIN_DEG = 2.0  # coarse sky pixel for the marker occupancy test


def occupancy_grid(star_ra, star_dec, bin_deg=OCC_BIN_DEG):
    """~bin_deg RA/Dec sky pixels holding >=1 star, dilated to the 8 adjacent
    pixels (RA wraparound handled; Dec clipped at the poles). Build it once
    per interaction, then test each marker set against it."""
    nx, ny = int(round(360 / bin_deg)), int(round(180 / bin_deg))
    if not len(star_ra):
        return np.zeros((nx, ny), dtype=bool)
    occ = np.histogram2d(np.mod(star_ra, 360.0), star_dec, bins=[nx, ny],
                         range=[[0, 360], [-90, 90]])[0] > 0
    dil = np.zeros_like(occ)
    for di in (-1, 0, 1):
        r = np.roll(occ, di, axis=0)          # RA wraps around
        dil |= r
        dil[:, 1:] |= r[:, :-1]               # Dec neighbors, clipped at poles
        dil[:, :-1] |= r[:, 1:]
    return dil


def grid_lookup(grid, m_ra, m_dec, bin_deg=OCC_BIN_DEG):
    """Which markers land on an occupied (dilated) pixel of the grid."""
    m_ra, m_dec = np.asarray(m_ra, float), np.asarray(m_dec, float)
    nx, ny = grid.shape
    ix = np.clip((np.mod(m_ra, 360.0) / bin_deg).astype(int), 0, nx - 1)
    iy = np.clip(((m_dec + 90.0) / bin_deg).astype(int), 0, ny - 1)
    return grid[ix, iy]


def occupied(star_ra, star_dec, m_ra, m_dec, bin_deg=OCC_BIN_DEG):
    """Convenience wrapper: occupancy_grid + grid_lookup in one call."""
    return grid_lookup(occupancy_grid(star_ra, star_dec, bin_deg),
                       m_ra, m_dec, bin_deg)

def load_lvdb():
    """(dwarfs, clusters) DataFrames [name, ra, dec] from the first hit per
    file, keeping only systems within LVDB_MAX_DIST_KPC (heliocentric)."""
    def gather(fnames):
        frames = []
        for fn in fnames:
            for d in LVDB_DIRS:
                p = os.path.join(os.path.expanduser(d), fn)
                if not os.path.exists(p):
                    continue
                t = pd.read_csv(p)
                if "distance" in t.columns:          # LVDB heliocentric, kpc
                    dist = pd.to_numeric(t["distance"], errors="coerce")
                elif "distance_modulus" in t.columns:
                    dm = pd.to_numeric(t["distance_modulus"], errors="coerce")
                    dist = 10 ** (dm / 5 - 2)
                else:
                    dist = pd.Series(0.0, index=t.index)
                t = t[dist.fillna(np.inf) < LVDB_MAX_DIST_KPC]
                frames.append(pd.DataFrame({
                    "name": t.get("name", t.get("key", "")),
                    "ra": pd.to_numeric(t["ra"], errors="coerce"),
                    "dec": pd.to_numeric(t["dec"], errors="coerce")}).dropna())
                break
        return (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame(columns=["name", "ra", "dec"]))
    return gather(LVDB_DWARF_FILES), gather(LVDB_CLUSTER_FILES)


# ──────────────────────── SIMBAD cross-match ────────────────────────
# On-demand only (button press): the filtered stars are uploaded to the CDS
# X-Match service and matched against SIMBAD at SIMBAD_RADIUS_ARCSEC — the
# same service + radius scripts/make_targets.py uses via stilts cdsskymatch.
# Implementation uses astroquery.xmatch (already installed; pure Python).

SIMBAD_COLS = ["simbad_main_id", "simbad_main_type", "simbad_sep_arcsec"]
SIMBAD_COLOR = "#2ca02c"   # green used for every "In SIMBAD" overlay
                           # (violet was hard to read on the dmod histogram;
                           # green kept everywhere for consistency)


def run_simbad_xmatch(ra, dec, radius_arcsec=SIMBAD_RADIUS_ARCSEC):
    """Query CDS X-Match against SIMBAD. Returns a DataFrame with columns
    idx (position into the input arrays) + SIMBAD_COLS, best match per star."""
    from astropy import units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    t = Table({"idx": np.arange(len(ra)),
               "ra": np.asarray(ra, float), "dec": np.asarray(dec, float)})
    res = XMatch.query(cat1=t, cat2="simbad",
                       max_distance=radius_arcsec * u.arcsec,
                       colRA1="ra", colDec1="dec")
    if len(res) == 0:
        return pd.DataFrame(columns=["idx"] + SIMBAD_COLS)
    r = res.to_pandas().sort_values("angDist").drop_duplicates("idx")
    return pd.DataFrame({
        "idx": r["idx"].astype(int).values,
        "simbad_main_id": r["main_id"].astype(str).values,
        "simbad_main_type": r["main_type"].astype(str).values,
        "simbad_sep_arcsec": r["angDist"].round(2).values})


def merge_simbad(df, mask, xmatch_fn=run_simbad_xmatch):
    """Cross-match the filtered rows of df against SIMBAD.

    Returns a DataFrame with SIMBAD_COLS indexed by df row position (matches
    only). xmatch_fn is injectable so tests never touch the network.
    """
    idx = np.flatnonzero(mask)
    res = xmatch_fn(df["ra"].to_numpy()[idx].astype(float),
                    df["dec"].to_numpy()[idx].astype(float))
    out = res[SIMBAD_COLS].copy()
    out.index = idx[res["idx"].to_numpy()]
    return out


def _coord_key(ra, dec):
    """Integer key from coordinates rounded to 1e-5 deg (0.036 arcsec)."""
    r = np.round(np.mod(np.asarray(ra, float), 360.0) * 1e5).astype(np.int64)
    d = np.round((np.asarray(dec, float) + 90.0) * 1e5).astype(np.int64)
    return r * 100_000_000 + d


def load_simbad_cache(df):
    """Positive SIMBAD matches persisted from earlier sessions, joined back to
    df rows by rounded coordinates. Returns same shape as merge_simbad."""
    if not os.path.exists(SIMBAD_CACHE_CSV):
        return pd.DataFrame(columns=SIMBAD_COLS)
    cache = (pd.read_csv(SIMBAD_CACHE_CSV)
             .drop_duplicates("coord_key").set_index("coord_key"))
    keys = _coord_key(df["ra"].to_numpy(), df["dec"].to_numpy())
    pos = np.flatnonzero(np.isin(keys, cache.index.to_numpy()))
    out = cache.loc[keys[pos], SIMBAD_COLS].copy()
    out.index = pos
    return out


def append_simbad_cache(df, matches):
    """Persist positive matches (keyed by rounded coordinates, git-ignored)."""
    if not len(matches):
        return
    new = matches.copy()
    new.insert(0, "coord_key", _coord_key(df["ra"].to_numpy()[matches.index],
                                          df["dec"].to_numpy()[matches.index]))
    if os.path.exists(SIMBAD_CACHE_CSV):
        old = pd.read_csv(SIMBAD_CACHE_CSV)
        new = pd.concat([old, new[~new["coord_key"].isin(old["coord_key"])]],
                        ignore_index=True)
    new.to_csv(SIMBAD_CACHE_CSV, index=False)


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


# Each cut control keeps its canonical value in session_state["<key>:val"];
# the slider and the typed number boxes are synced to it two-way via
# callbacks. Typed values may exceed the slider's percentile-derived bounds —
# the typed value always wins (the slider only clips what it displays).
# The [Fiducial cuts] button stages values under "<key>:pending", which the
# control consumes (and enables itself) on the next rerun.

def _consume_pending(st, key, vkey, onkey, cast):
    pend = st.session_state.pop(key + ":pending", None)
    if pend is not None:
        st.session_state[vkey] = cast(pend)
        st.session_state[onkey] = True
    return pend is not None


def _range_cut(st, label, bounds, key, fmt="%.2f", pad=0.05):
    lo, hi = float(bounds[0]), float(bounds[1])
    span = (hi - lo) or 1.0
    slo, shi = lo - pad * span, hi + pad * span
    step = (shi - slo) / 200
    vkey, skey = key + ":val", key + ":sl"
    lokey, hikey, onkey = key + ":lo", key + ":hi", key + ":on"

    def clip(x):
        return float(min(max(float(x), slo), shi))

    pended = _consume_pending(st, key, vkey, onkey,
                              lambda v: (float(v[0]), float(v[1])))
    st.session_state.setdefault(vkey, (slo, shi))

    def from_slider():
        v = st.session_state[skey]
        st.session_state[vkey] = (float(v[0]), float(v[1]))
        st.session_state[lokey], st.session_state[hikey] = st.session_state[vkey]

    def from_boxes():
        v = sorted((float(st.session_state[lokey]), float(st.session_state[hikey])))
        st.session_state[vkey] = tuple(v)

    cur = st.session_state[vkey]
    st.session_state[skey] = (clip(cur[0]), clip(cur[1]))
    st.session_state.setdefault(lokey, cur[0])
    st.session_state.setdefault(hikey, cur[1])
    if pended:
        st.session_state[lokey], st.session_state[hikey] = cur

    on = st.checkbox(label, key=onkey)
    st.slider(label, slo, shi, step=step, format=fmt, key=skey,
              label_visibility="collapsed", disabled=not on,
              on_change=from_slider)
    c1, c2 = st.columns(2)
    c1.number_input(label + " min", step=step, format=fmt, key=lokey,
                    disabled=not on, on_change=from_boxes,
                    label_visibility="collapsed")
    c2.number_input(label + " max", step=step, format=fmt, key=hikey,
                    disabled=not on, on_change=from_boxes,
                    label_visibility="collapsed")
    return on, st.session_state[vkey]


def _thresh_cut(st, label, bounds, key, fmt="%.2f", default=None):
    lo, hi = float(bounds[0]), float(bounds[1])
    step = ((hi - lo) or 1.0) / 200
    vkey, skey, bkey, onkey = key + ":val", key + ":sl", key + ":box", key + ":on"

    pended = _consume_pending(st, key, vkey, onkey, float)
    st.session_state.setdefault(vkey, float(default if default is not None else hi))

    def from_slider():
        st.session_state[vkey] = float(st.session_state[skey])
        st.session_state[bkey] = st.session_state[vkey]

    def from_box():
        st.session_state[vkey] = float(st.session_state[bkey])

    st.session_state[skey] = float(min(max(st.session_state[vkey], lo), hi))
    st.session_state.setdefault(bkey, st.session_state[vkey])
    if pended:
        st.session_state[bkey] = st.session_state[vkey]

    on = st.checkbox(label, key=onkey)
    st.slider(label, lo, hi, step=step, format=fmt, key=skey,
              label_visibility="collapsed", disabled=not on,
              on_change=from_slider)
    st.number_input(label + " value", step=step, format=fmt, key=bkey,
                    disabled=not on, on_change=from_box,
                    label_visibility="collapsed")
    return on, st.session_state[vkey]


def _queue_fiducial(st, key, rng):
    """Stage the FIDUCIAL preset (button callback path — widgets pick the
    values up when they are instantiated later in the same rerun)."""
    for col, val in FIDUCIAL.items():
        if col == "population":
            st.session_state[key + ":pop"] = val
        elif col in ("sep_lmc", "sep_smc") or rng.get(col):
            st.session_state[f"{key}:{col}:pending"] = val


def _clear_cuts(st, key):
    st.session_state[key + ":pop"] = "both"
    for col in CUT_COLS:
        st.session_state[f"{key}:{col}:on"] = False
        st.session_state.pop(f"{key}:{col}:pending", None)


def render():
    """The 'Target explorer' page."""
    st = _st()
    st.header("Target-selection explorer")

    user = st.session_state.get("user")
    extra_key = "explorer:extra_globs"
    extras = st.session_state.get(extra_key, [])

    # optional runtime path entry — shown ONLY when the deployment
    # allowlists roots in secrets (default-closed, like the feature flag)
    try:
        allowed_roots = list(_secrets()["catalogs"]["allowed_roots"])
    except (KeyError, FileNotFoundError, TypeError):
        allowed_roots = []
    if allowed_roots:
        with st.sidebar.expander("Add catalog path (this session)"):
            newp = st.text_input("Directory or glob under an allowed root",
                                 key="explorer:addpath")
            if st.button("Add path", key="explorer:addpath_btn") and newp:
                if path_allowed(newp, allowed_roots):
                    if newp not in extras:
                        extras = extras + [newp]
                        st.session_state[extra_key] = extras
                    st.success("Added for this session.")
                else:
                    st.warning("Rejected: path is outside the allowed "
                               "catalog roots for this deployment.")

    cats = find_catalogs(user=user, extra_globs=extras)
    rel = release_spec(_secrets())
    options = {}   # display label -> ("local", path) | ("release", asset)
    for n, p in cats.items():
        options[f"{n}  ({os.path.getsize(p) / 1e9:.1f} GB)"] = ("local", p)
    for a in (rel["assets"] if rel else []):
        options[f"{a}@{rel['tag']}"] = ("release", a)
    if not options:
        st.info("No MAGIC catalogs found (searched: {}). Configure paths or a "
                "release asset in st.secrets['catalogs'] — see "
                "secrets.toml.example.".format(", ".join(resolve_globs(
                    _secrets(), user, os.environ.get("MAGIC_CATALOG_GLOBS")))))
        return
    default = next((i for i, (kind, ref) in enumerate(options.values())
                    if kind == "local"
                    and os.path.basename(ref) == DEFAULT_CATALOG), 0)
    choice = st.selectbox("Catalog (version)", list(options), index=default)
    kind, ref = options[choice]

    if kind == "release":
        # already explorer-schema Parquet: download once per container into
        # the cache dir (the file on disk is the cache), then load normally
        pq, js = release_paths(rel, ref)
        if not os.path.exists(pq):
            try:
                with st.spinner(f"Downloading {ref} from the release ..."):
                    bar = st.progress(0.0)
                    fetch_release_asset(rel["repo"], rel["tag"], ref,
                                        rel["token"], pq,
                                        progress=bar.progress)
                    bar.empty()
            except Exception as e:
                st.warning(f"Release catalog unavailable: {e} — "
                           "the rest of the app keeps working.")
                return
        if not os.path.exists(js):
            dfr = pd.read_parquet(pq)
            need = {"ra", "dec", "feh", "star_class"}
            if not need <= set(dfr.columns):
                st.warning(f"{ref} is not an explorer-schema Parquet "
                           f"(missing columns: {sorted(need - set(dfr.columns))}) "
                           "— rebuild it with build_cloud_subset.py.")
                return
            with open(js, "w") as f:
                json.dump(_make_meta(dfr, f"{ref}@{rel['tag']}"), f, indent=1)
    else:
        cat_path = ref
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
        b1, b2 = st.columns(2)
        if b1.button("Fiducial cuts", use_container_width=True,
                     help="Standard MAGIC low-metallicity giant selection — "
                          "edit the FIDUCIAL dict at the top of explorer.py"):
            _queue_fiducial(st, key, rng)
        if b2.button("Clear all", use_container_width=True):
            _clear_cuts(st, key)

        pops = {"RGB": ["RGB"], "MS": ["MS"], "both": ["RGB", "MS"],
                "include ambiguous": ["RGB", "MS", "ambiguous"]}
        st.session_state.setdefault(key + ":pop", "both")
        pop = st.radio("Population ([Fe/H], dmod are per-class values)",
                       list(pops), horizontal=True, key=key + ":pop")
        cuts.append({"col": "star_class", "kind": "isin", "value": pops[pop],
                     "enabled": True})

        def add(cut_on, col, kind, value):
            cuts.append({"col": col, "kind": kind, "value": value, "enabled": cut_on})

        for col, label, fmt in (("pmra", "pmra (mas/yr)", "%.1f"),
                                ("pmdec", "pmdec (mas/yr)", "%.1f"),
                                ("gi0", "(g-i)₀ color", "%.2f"),
                                ("feh", "[Fe/H]", "%.2f"),
                                ("dmod", "distance modulus", "%.2f")):
            if rng.get(col):
                on, val = _range_cut(st, label, rng[col], f"{key}:{col}", fmt)
                add(on, col, "range", val)
        for col, label, fmt, lo in (("ebv", "E(B-V) max", "%.3f", 0.0),
                                    ("e_feh", "[Fe/H] error max", "%.2f", 0.0),
                                    ("mag_g", "depth: g max (mag_psf_g)",
                                     "%.2f", None)):
            if rng.get(col):
                bounds = (rng[col][0] if lo is None else lo, rng[col][1])
                on, val = _thresh_cut(st, label, bounds, f"{key}:{col}", fmt)
                add(on, col, "max", val)

        st.subheader("Excise Clouds")
        for name, (cra, cdec, rdef), col in (("LMC", LMC, "sep_lmc"),
                                             ("SMC", SMC, "sep_smc")):
            on, r = _thresh_cut(st, f"cut {name} ({cra}, {cdec}) — radius (deg)",
                                (0.5, 15.0), f"{key}:{col}", "%.1f", default=rdef)
            add(on, col, "min", r)

    mask = apply_cuts(df, cuts)

    skey = key + ":simbad"
    if skey not in st.session_state:
        st.session_state[skey] = load_simbad_cache(df)
        st.session_state[skey + ":queried"] = set()

    with view:
        mrow = st.container()   # metrics render last (fresh after SIMBAD click)
        lit_obs = st.checkbox("count literature-known stars as already observed",
                              value=True, key=key + ":litobs")
        m = metrics(df, mask, lit_is_observed=lit_obs)

        # on-demand SIMBAD cross-match — only ever queries on button press
        n_sel = m["selected"]
        over = n_sel > SIMBAD_MAX_ROWS
        if st.button(f"Check SIMBAD ({n_sel:,} filtered stars, "
                     f"{SIMBAD_RADIUS_ARCSEC:.0f} arcsec CDS X-Match)",
                     disabled=(n_sel == 0 or over)):
            import hashlib
            h = hashlib.md5(np.flatnonzero(mask).tobytes()).hexdigest()
            if h in st.session_state[skey + ":queried"]:
                st.info("This exact selection was already checked this session.")
            else:
                try:
                    with st.spinner("Querying CDS X-Match against SIMBAD ..."):
                        res = merge_simbad(df, mask)
                    append_simbad_cache(df, res)
                    sim = st.session_state[skey]
                    st.session_state[skey] = pd.concat(
                        [sim[~sim.index.isin(res.index)], res]).sort_index()
                    st.session_state[skey + ":queried"].add(h)
                    st.success(f"{len(res):,} of {n_sel:,} filtered stars "
                               f"are in SIMBAD.")
                except Exception as e:
                    st.warning(f"SIMBAD X-Match failed (offline or CDS "
                               f"service error) — the app keeps working: {e}")
        if over:
            st.caption(f"SIMBAD check disabled: {n_sel:,} rows exceed the "
                       f"{SIMBAD_MAX_ROWS:,} upload cap — tighten the cuts.")

        sim = st.session_state[skey]
        sim_mask = np.zeros(len(df), dtype=bool)
        if len(sim.index):
            sim_mask[sim.index.to_numpy()] = True
        sim_mask &= mask
        n_simbad = int(sim_mask.sum())
        with mrow:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("In catalog", f"{len(df):,}")
            c2.metric("Passing cuts", f"{m['selected']:,}")
            c3.metric("Observed by us", f"{m['observed_us']:,}")
            c4.metric("Literature-known", f"{m['literature']:,}")
            c5.metric("Remaining to observe", f"{m['remaining']:,}")
            c6.metric("In SIMBAD", f"{n_simbad:,}",
                      help="From this session's checks plus the persistent "
                           "cache; run 'Check SIMBAD' to update.")

        import inspect
        ui = {"st": st, "key": key, "sim": sim, "sim_mask": sim_mask,
              "cuts": cuts,
              # click-to-highlight: selected df row positions live under
              # sel_key; the e_feh panel writes it (plotly selection events,
              # Streamlit >= 1.35 only), the table panel consumes it
              "sel_key": key + ":sel_rows",
              "plotly_select": "on_select"
                               in inspect.signature(st.plotly_chart).parameters}
        for title, fn in PANELS:
            st.subheader(title)
            fn(df, mask, ui)


# ──────────────────── click-to-highlight helpers ────────────────────
def move_selected_first(tab, sel_rows):
    """Reorder a table view so selected df row positions come first.
    Returns (reordered_tab, selected_rows_present_in_tab)."""
    present = set(tab.index)
    sel = [r for r in sel_rows if r in present]
    if not sel:
        return tab, []
    return pd.concat([tab.loc[sel], tab.drop(index=sel)]), sel


def star_detail(df, row, sim):
    """One-line summary of a single star (df row position) for the table."""
    r = df.iloc[row]
    status = (str(r["obs_cat"]) if str(r["obs_cat"])
              else ("literature-known" if bool(r["lit_known"]) else "unobserved"))
    line = (f"**({r['ra']:.5f}, {r['dec']:.5f})** · g = {r['mag_g']:.2f} · "
            f"[Fe/H] = {r['feh']:.2f} ± {r['e_feh']:.2f} · "
            f"dmod = {r['dmod']:.2f} · {r['star_class']} · {status}")
    if row in sim.index:
        line += (f" · SIMBAD: {sim.loc[row, 'simbad_main_id']} "
                 f"({sim.loc[row, 'simbad_main_type']})")
    return line


# ──────────────────────── linked panels ────────────────────────
# Each panel: fn(df, mask, ui) — df is the FULL table, mask the current cuts.
# ui carries the shared overlay state, so every registered panel gets it for
# free: ui["sim_mask"] is a full-length boolean array of the SIMBAD-matched
# stars ALREADY intersected with the filter mask (all-False until a SIMBAD
# query has run — panels then show nothing extra), styled SIMBAD_COLOR with
# the legend entry "In SIMBAD"; ui["sim"] is the match table for row detail.
# Register new panels in PANELS below; they automatically share the cuts.

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
    obs_us = mask & (np.asarray(df["obs_cat"]) != "")
    lit = mask & np.asarray(df["lit_known"]) & ~obs_us
    for sel, name, marker in (
            (obs_us, "observed by us",
             dict(symbol="x", size=7, color="#e45756")),
            (lit, "literature-known",
             dict(symbol="circle-open", size=7, color="#f58518")),
            (ui["sim_mask"], "In SIMBAD",
             dict(symbol="diamond-open", size=8, color=SIMBAD_COLOR))):
        if sel.any():
            fig.add_trace(go.Scattergl(
                x=df[xc].to_numpy()[sel], y=df[yc].to_numpy()[sel],
                mode="markers", name=name, marker=marker))

    occ = occupancy_grid(df["ra"].to_numpy()[mask], df["dec"].to_numpy()[mask])
    dwarfs, clusters = load_lvdb()
    for show, cat, sym, color, label in (
            (show_dw, dwarfs, "star", "#f2b701", "dwarf galaxies"),
            (show_gc, clusters, "triangle-up", "#00b8d9", "star clusters")):
        if not show or not len(cat):
            continue
        # occupancy test in RA/Dec: marker (or an 8-neighbor pixel) must
        # hold at least one star surviving the current cuts
        keep = grid_lookup(occ, cat["ra"].values, cat["dec"].values)
        if gal:
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            g = SkyCoord(ra=cat["ra"].values * u.deg,
                         dec=cat["dec"].values * u.deg).galactic
            cx, cy = g.l.deg, g.b.deg
        else:
            cx, cy = cat["ra"].values, cat["dec"].values
        fig.add_trace(go.Scatter(
            x=cx[keep], y=cy[keep], mode="markers+text", name=label,
            text=cat["name"].values[keep], textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(symbol=sym, size=9, color=color,
                        line=dict(width=1, color="black"))))

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=xc, yaxis_title=yc,
                      legend=dict(orientation="h", y=1.06))
    if not gal:
        fig.update_xaxes(range=[360, 0])  # full 0-360 deg, RA increasing leftward
    st.plotly_chart(fig, use_container_width=True)


KPC_TICKS = (10, 20, 30, 50, 100, 150, 200)   # top-axis labels (kpc)


def _dmod_to_kpc(dmod):
    return 10 ** ((np.asarray(dmod, float) - 10.0) / 5.0)


def panel_dmod(df, mask, ui):
    import plotly.graph_objects as go
    st = ui["st"]
    v = df["dmod"].to_numpy()[mask]
    v = v[np.isfinite(v)]
    if not len(v):
        st.caption("No finite dmod values in the current selection "
                   "(ambiguous/invalid stars have none).")
        return

    # adaptive binning: ~2*sqrt(N) bins keeps fiducial-scale selections
    # (a few hundred stars) from being shredded across 120 near-empty bins
    nbins = int(np.clip(2 * np.sqrt(len(v)), 20, 120))
    cnt, edges = np.histogram(v, bins=nbins)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    kpc = _dmod_to_kpc(ctr)
    hover = ("dmod %{x:.2f} · %{customdata:.1f} kpc · N = %{y}"
             "<extra>%{fullData.name}</extra>")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ctr, y=cnt, customdata=kpc, name="targets",
                         marker_color="#4c78a8", hovertemplate=hover))
    o = df["dmod"].to_numpy()[mask & (np.asarray(df["obs_cat"]) != "")]
    o = o[np.isfinite(o)]
    if len(o):
        cnt2, _ = np.histogram(o, bins=edges)
        fig.add_trace(go.Bar(x=ctr, y=cnt2, customdata=kpc,
                             name="observed by us", marker_color="#e45756",
                             hovertemplate=hover))
    s = df["dmod"].to_numpy()[ui["sim_mask"]]
    s = s[np.isfinite(s)]
    if len(s):
        cnt3, _ = np.histogram(s, bins=edges)
        fig.add_trace(go.Scatter(
            x=ctr, y=np.where(cnt3 > 0, cnt3, np.nan),   # gaps, not log(0)
            customdata=kpc, name="In SIMBAD", mode="lines",
            line=dict(color=SIMBAD_COLOR, width=3, shape="hvh"),
            hovertemplate=hover))

    # dashed line where an enabled dmod cut starts (fiducial: d > 30 kpc)
    for cut in ui.get("cuts", []):
        if (cut["col"] == "dmod" and cut["kind"] == "range"
                and cut.get("enabled")):
            lo = float(cut["value"][0])
            if edges[0] <= lo <= edges[-1]:
                fig.add_vline(x=lo, line_dash="dash", line_color="#666",
                              annotation_text=f"d > {_dmod_to_kpc(lo):.0f} kpc",
                              annotation_position="top right")

    # secondary top axis in physical distance: dmod = 5 log10(d / kpc) + 10
    fig.add_trace(go.Scatter(x=[float(ctr[0]), float(ctr[-1])], y=[None, None],
                             xaxis="x2", showlegend=False, hoverinfo="skip"))
    # linear y shows the distribution's shape at fiducial-scale N; log keeps
    # the small observed/SIMBAD overlays visible against millions of targets
    log_y = len(v) > 10_000
    fig.update_layout(
        barmode="overlay", height=360, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="distance modulus (RGB/MS per class)",
        yaxis_title="N", yaxis_type="log" if log_y else "linear",
        xaxis2=dict(matches="x", overlaying="x", side="top",
                    tickvals=[5 * np.log10(k) + 10 for k in KPC_TICKS],
                    ticktext=[str(k) for k in KPC_TICKS],
                    title=dict(text="distance (kpc)", font=dict(size=11)),
                    showgrid=False))
    st.plotly_chart(fig, use_container_width=True)


CLICK_MAX = 20_000   # clickable-scatter cap on top of the density layer


def panel_feh(df, mask, ui):
    import plotly.graph_objects as go
    st = ui["st"]
    feh, e_feh = df["feh"].to_numpy(), df["e_feh"].to_numpy()
    rows = np.flatnonzero(mask)
    ok = np.isfinite(feh[rows]) & np.isfinite(e_feh[rows])
    rows_ok = rows[ok]
    if not len(rows_ok):
        st.caption("No finite [Fe/H] values in the current selection.")
        return

    ui["click_note"] = ""
    fig = go.Figure()
    if len(rows_ok) > SCATTER_MAX:
        H, xe, ye = np.histogram2d(feh[rows_ok], e_feh[rows_ok],
                                   bins=(240, 160))
        fig.add_trace(go.Heatmap(
            x=0.5 * (xe[:-1] + xe[1:]), y=0.5 * (ye[:-1] + ye[1:]),
            z=np.where(H.T > 0, np.log10(H.T, where=H.T > 0), np.nan),
            colorscale="Viridis", colorbar=dict(title="log₁₀ N"),
            name="targets"))
        # clicks need points: decimated, display-only clickable layer
        pick = np.random.default_rng(0).choice(len(rows_ok), CLICK_MAX,
                                               replace=False)
        rows_click = rows_ok[pick]
        ui["click_note"] = (f"clickable layer decimated to {CLICK_MAX:,} of "
                            f"{len(rows_ok):,} points (display-only)")
        fig.add_trace(go.Scattergl(
            x=feh[rows_click], y=e_feh[rows_click], mode="markers",
            customdata=rows_click, showlegend=False, name="targets",
            marker=dict(size=4, color="rgba(76,120,168,0.35)")))
    else:
        fig.add_trace(go.Scattergl(
            x=feh[rows_ok], y=e_feh[rows_ok], mode="markers", name="targets",
            customdata=rows_ok,
            marker=dict(size=5, color="#3a5f8a", opacity=0.75)))

    obs_rows = np.flatnonzero(mask & (np.asarray(df["obs_cat"]) != ""))
    obs_rows = obs_rows[np.isfinite(feh[obs_rows]) & np.isfinite(e_feh[obs_rows])]
    if len(obs_rows):
        fig.add_trace(go.Scattergl(
            x=feh[obs_rows], y=e_feh[obs_rows], mode="markers",
            name="observed by us", customdata=obs_rows,
            marker=dict(symbol="x", size=9, color="#e45756")))
    sim_rows = np.flatnonzero(ui["sim_mask"])
    sim_rows = sim_rows[np.isfinite(feh[sim_rows]) & np.isfinite(e_feh[sim_rows])]
    if len(sim_rows):
        fig.add_trace(go.Scattergl(
            x=feh[sim_rows], y=e_feh[sim_rows], mode="markers",
            name="In SIMBAD", customdata=sim_rows,
            marker=dict(symbol="diamond-open", size=10, color=SIMBAD_COLOR)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="[Fe/H] (per-class)",
                      yaxis_title="σ([Fe/H])")

    # click / box / lasso selection -> highlight in the target table.
    # st.plotly_chart selection events exist from Streamlit 1.35 on; on older
    # versions the plot stays static and we say so rather than emulating it.
    sel_key = ui["sel_key"]
    if ui["plotly_select"]:
        ck = ui["key"] + ":feh_sel"
        if st.session_state.get(sel_key) and st.button(
                "Clear selection", key=ui["key"] + ":sel_clear"):
            st.session_state[sel_key] = []
            st.session_state.pop(ck, None)   # resets the chart's selection
        ev = st.plotly_chart(fig, use_container_width=True, key=ck,
                             on_select="rerun",
                             selection_mode=("points", "box", "lasso"))
        picked = []
        for p in getattr(getattr(ev, "selection", None), "points", []) or []:
            v = p.get("customdata")
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if v is not None:
                picked.append(int(v))
        if picked:   # a rerun without an active chart selection keeps the
            st.session_state[sel_key] = sorted(set(picked))   # current one
        st.caption("Click a star (or box/lasso-select) to highlight it in "
                   "the target table below; use the button above the chart "
                   "to clear a selection.")
    else:
        st.plotly_chart(fig, use_container_width=True)
        import streamlit as _stmod
        st.caption(f"Click-to-highlight needs Streamlit ≥ 1.35 (installed: "
                   f"{_stmod.__version__}) — `pip install -U streamlit` "
                   "to enable it.")


def panel_table(df, mask, ui):
    st = ui["st"]
    n = int(mask.sum())
    if n > SIMBAD_MAX_ROWS:
        st.caption(f"{n:,} rows — tighten the cuts below {SIMBAD_MAX_ROWS:,} "
                   "to browse or download the target table.")
        return
    cols = [c for c in ("ra", "dec", "star_class", "feh", "e_feh", "dmod",
                        "gi0", "mag_g", "pmra", "pmdec", "ebv",
                        "obs_cat", "lit_known") if c in df.columns]
    tab = df.loc[mask, cols].copy()
    tab["in_simbad"] = np.asarray(ui["sim_mask"])[tab.index]  # sortable
    # empty until a SIMBAD query has run (sep stays numeric for Arrow/sorting)
    tab["simbad_main_id"] = ""
    tab["simbad_main_type"] = ""
    tab["simbad_sep_arcsec"] = np.nan
    hit = ui["sim"].index.intersection(tab.index)
    if len(hit):
        tab.loc[hit, SIMBAD_COLS] = ui["sim"].loc[hit, SIMBAD_COLS].values
    if st.checkbox("show only SIMBAD matches", key=ui["key"] + ":tab_sim",
                   disabled=not tab["in_simbad"].any()):
        tab = tab[tab["in_simbad"]]

    # click-to-highlight: selected stars jump to the top with a detail line
    sel_rows = st.session_state.get(ui["sel_key"], [])
    tab, sel = move_selected_first(tab, sel_rows)
    for r in sel[:5]:
        st.markdown(star_detail(df, r, ui["sim"]))
    if len(sel) > 5:
        st.caption(f"... and {len(sel) - 5} more selected rows")
    if ui.get("click_note") and sel:
        st.caption(ui["click_note"])

    disp = tab.head(5000)
    sel_set = set(sel)
    if 0 < len(disp) <= 2000 and (sel_set or disp["in_simbad"].any()):
        # row tints — Styler is cheap at this size, skipped above it
        def _tint(r):
            if r.name in sel_set:
                return ["background-color: rgba(255,193,7,0.30)"] * len(r)
            if r["in_simbad"]:
                return ["background-color: rgba(44,160,44,0.15)"] * len(r)
            return [""] * len(r)
        st.dataframe(disp.style.apply(_tint, axis=1), use_container_width=True)
    else:
        st.dataframe(disp, use_container_width=True)
    if n > 5000:
        st.caption("showing the first 5,000 rows — the download has all of them")
    st.download_button("Download filtered targets CSV",
                       tab.to_csv(index=False).encode(),
                       "explorer_targets.csv", "text/csv")


PANELS = [
    ("On-sky", panel_sky),
    ("Distance modulus", panel_dmod),
    ("[Fe/H] vs its uncertainty", panel_feh),
    ("Filtered targets", panel_table),
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
