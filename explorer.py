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
LVDB_DIRS = [d for d in (
    os.environ.get("MAGIC_LVDB_DIR"),
    "~/Documents/MIT_Work/Research/magic_scratch/dwarf_outskirts/pipeline/"
    "local_volume_database",
    os.path.join(APP_DIR, "data", "lvdb")) if d]
LVDB_DWARF_FILES = ["dwarf_mw.csv"]
LVDB_CLUSTER_FILES = ["gc_harris.csv", "gc_mw_new.csv", "gc_dwarf_hosted.csv"]
LVDB_MAX_DIST_KPC = 300.0     # drop local-volume systems beyond the MW halo

MATCH_RADIUS_ARCSEC = 1.0        # ledger cross-match — the 1" MAGIC pipeline
                                 # convention (make_targets.py, SIMBAD query);
                                 # the app's interactive checker stays at 2"
OBSERVED_CATEGORIES = ("MAGIC_Magellan", "nonMAGIC_Magellan", "Gemini")
SIMBAD_RADIUS_ARCSEC = 1.0       # CDS X-Match radius (make_targets.py convention)
SIMBAD_MAX_ROWS = 50_000         # refuse to upload more rows than this to CDS
# _v2: results from live SIMBAD TAP. The v1 file held CDS X-Match mirror
# results whose no-matches are known-wrong (the mirror snapshot is incomplete
# vs live SIMBAD), so the cache name is versioned to bust it.
SIMBAD_CACHE_CSV = os.path.join(APP_DIR, "data", "simbad_cache_v2.csv")
SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap"
SIMBAD_TAP_CHUNK = 10_000   # rows per synchronous upload-join query
LMC = (80.89, -69.76, 5.0)       # ra, dec, default excision radius (deg)
SMC = (13.19, -72.83, 3.0)
SCATTER_MAX = 150_000            # above this, scatter layers become 2D histograms
CHUNK = 2_000_000                # FITS -> Parquet conversion chunk (rows)
SCHEMA_VERSION = 8

# slider bounds = catalog percentiles clipped to these physical windows,
# so a handful of junk-photometry rows can't stretch a slider to uselessness
HARD_BOUNDS = {"pmra": (-30, 30), "pmdec": (-30, 30), "ebv": (0, 1),
               "gi0": (-2, 5), "feh": (-5, 2), "e_feh": (0, 5),
               "dmod": (0, 25), "mag_g": (10, 25), "magerr_cahk": (0, 2)}


def dmod_to_pc(dmod):
    """Distance modulus -> heliocentric distance in parsecs."""
    return 10.0 ** (np.asarray(dmod, float) / 5.0 + 1.0)


def pc_to_dmod(pc):
    """Heliocentric distance in parsecs -> distance modulus. Guards pc <= 0,
    which the slider's padded lower bound can reach."""
    return 5.0 * np.log10(np.maximum(np.asarray(pc, float), 1e-6) / 10.0)


# ── Cache pre-selection ─────────────────────────────────────────────────
# Applied once in build_cache, BEFORE anything reaches the explorer, so the
# UI cuts always operate on an already-cleaned sample. Each entry is
# (column, human-readable rule, predicate) — the note under "Cuts" on the
# target-selection page is rendered straight from this list, so editing a rule
# here updates both the filtering and what the page claims it did.
PRESELECT = [
    # real Gaia DR3 ids are >= 2^35, so "> 999999" rejects every no-match
    # encoding seen so far: 999999 (laptop copy), 0, and the masked-int64
    # fill INT64_MIN that astropy surfaces from the mpflags lite files
    ("source_id", "has a Gaia source_id (no-match sentinels dropped)",
     lambda d: np.asarray(d["source_id"], dtype=np.int64) > GAIA_NO_MATCH),
    ("extended_class_g", "extended_class_g in (0, 1) — drops galaxies and -9",
     lambda d: np.isin(d["extended_class_g"], (0, 1))),
    ("mag_psf_cahk", "valid CaHK photometry: 0 < mag_psf_cahk < 30",
     # upper bound rejects the ~1e20 no-measurement sentinel and the >90
     # placeholders; lower bound rejects unphysical negative magnitudes
     lambda d: np.isfinite(d["mag_psf_cahk"])
     & (d["mag_psf_cahk"] > CAHK_MIN) & (d["mag_psf_cahk"] < CAHK_MAX)),
    ("ebv_sfd98", "E(B-V) <= 0.2 (SFD98)",
     lambda d: np.isfinite(d["ebv_sfd98"]) & (d["ebv_sfd98"] <= EBV_MAX)),
]
GAIA_NO_MATCH = 999999   # ids at or below this are "no Gaia cross-match"
EBV_MAX = 0.2
CAHK_MIN = 0.0
CAHK_MAX = 30.0


def apply_preselect(d, names, n_rows, counts=None):
    """Boolean keep-mask from every PRESELECT rule whose column exists in
    `names`, applied in order to the column mapping `d` (a FITS record chunk
    or any dict of arrays). If `counts` is given, it accumulates the
    cumulative survivor count after each rule — the four-cut chain."""
    keep = np.ones(n_rows, dtype=bool)
    for col, rule, pred in PRESELECT:
        if col not in names:
            continue
        keep &= np.asarray(pred(d), dtype=bool)
        if counts is not None:
            counts[rule] = counts.get(rule, 0) + int(keep.sum())
    return keep


# UI wording for the pre-selection note: plain meaning only. The exact
# predicates (sentinel encodings, bounds) live in the PRESELECT lambdas and
# the technical rule strings above, which the build log and cache meta keep.
PRESELECT_PLAIN = {
    "source_id": "has a Gaia DR3 counterpart",
    "extended_class_g": "point-like sources only (galaxies removed)",
    "mag_psf_cahk": "has a valid CaHK measurement",
    "ebv_sfd98": "low reddening: E(B-V) ≤ 0.2 (SFD98)",
}

# human wording for known cloud-subset row cuts; anything else renders as the
# raw expression, and a subset with no embedded cut info shows nothing
SUBSET_CUT_PLAIN = {
    "(feh_rgb < -2) | (feh_ms < -2)":
        "metal-poor pre-cut: [Fe/H] < −2.0 under either the RGB or MS "
        "assumption (fehs_rgb < −2 | fehs_ms < −2)",
    "feh == feh and star_class in ['RGB', 'MS']":
        "metal-poor pre-cut: valid adopted [Fe/H] and star_class RGB or MS",
}


def preselect_note():
    """The pre-selection as markdown bullets for the target-selection page."""
    return "\n".join(f"- {PRESELECT_PLAIN.get(col, rule)}"
                      for col, rule, _ in PRESELECT)


def subset_cut_from_parquet(pq_path):
    """The row-cut expression a cloud subset was built with, embedded in the
    Parquet schema metadata by build_cloud_subset.py; None when absent."""
    try:
        import pyarrow.parquet as papq
        md = papq.read_schema(pq_path).metadata or {}
        cut = md.get(b"magic_subset_cut")
        return cut.decode() if cut else None
    except Exception:
        return None


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
    # Targeting context: distant low-metallicity halo giant follow-up.
    # Ambiguous stars are INCLUDED and evaluated under the RGB assumption —
    # the [Fe/H]_RGB targeting logic behind PicII-503 — because the distant
    # metal-poor giants this program exists for often classify as ambiguous.
    # MS is excluded: a measured parallax says dwarf, not distant giant.
    # This is preset-only context: the explorer's DEFAULT stays
    # 'Matching star_class'; only the [Fiducial cuts] button sets Assumed=RGB.
    "population": ("RGB", "ambiguous"),   # star_class checkboxes
    "assumed": "RGB",        # cut on the RGB-assumed feh/e_feh/dmod columns
    "pmra": (-3.5, 3.5),      # mas/yr
    "pmdec": (-3.5, 3.5),     # mas/yr
    "feh": (-5.0, -3.0),      # dex
    "gi0": (0.2, 1.5),        # overview paper quality cut (magic.tex ~l.447):
                              # "0.2 < (g-i)_0 < 1.5 to exclude the coolest
                              # stars and regions significantly bluer than
                              # the main-sequence turnoff"
    "mag_g": 18.5,            # keep stars brighter than this
    "dist_pc": (30_000.0, 1_000_000.0),   # d > 30 kpc; the upper bound is
                              # the old dmod=25 hard limit, i.e. no far cut
    "ebv": 0.05,              # stricter than the catalog's baked-in E(B-V)<0.2
    # overview-paper quality cuts: broadband color-color validity required,
    # Gaia-flagged variables excluded (unsuitable for photometric [Fe/H])
    "broadband_valid": True,  # tick the 'broadband_valid only' quality flag
    "gaia_var_flag": True,    # tick the 'exclude Gaia variables' flag
    "feh_ext": True,          # exclude grid-extrapolated [Fe/H] — unreliable
                              # for targeting (fehs_ext_rgb under Assumed=RGB)
    "sep_lmc": 5.0,           # excision radius around the LMC (deg)
    "sep_smc": 3.0,           # excision radius around the SMC (deg)
}
# every cut column a Clear-all must switch off
CUT_COLS = ("pmra", "pmdec", "gi0", "feh", "dist_pc", "ebv", "e_feh",
            "mag_g", "sep_lmc", "sep_smc",
            "broadband_valid", "gaia_var_flag", "lvdbcut", "feh_ext")

# every column a built cache / cloud subset carries
SCHEMA_COLUMNS = ["ra", "dec", "pmra", "pmdec", "ebv", "feh", "e_feh", "dmod",
                  "mag_g", "gi0", "star_class", "l", "b", "sep_lmc", "sep_smc",
                  "obs_cat", "lit_known",
                  "feh_ext",
                  "feh_rgb", "e_feh_rgb", "dmod_rgb", "feh_ext_rgb",
                  "feh_ms", "e_feh_ms", "dmod_ms", "feh_ext_ms",
                  "broadband_valid", "gaia_var_flag", "magerr_cahk",
                  "lvdb_host", "lvdb_host_type", "obs_instrument",
                  "source_id"]
RANGE_COLS = ("pmra", "pmdec", "ebv", "gi0", "feh", "e_feh", "dmod", "mag_g",
              "feh_rgb", "e_feh_rgb", "dmod_rgb",
              "feh_ms", "e_feh_ms", "dmod_ms",
              "broadband_valid", "gaia_var_flag", "magerr_cahk")

# canonical column -> catalog column candidates (first match wins)
CANDS = {
    "ra": ["ra"], "dec": ["dec"],
    "pmra": ["pmra"], "pmdec": ["pmdec"],
    "ebv": ["ebv_sfd98", "ebv"],
    "feh": ["feh"], "e_feh": ["e_feh"], "dmod": ["dmod"],
    "mag_g": ["mag_psf_g"],
    # per-class values: the catalog solves each star under BOTH an RGB and an
    # MS assumption, and `feh`/`e_feh`/`dmod` hold whichever matches
    # star_class. Carrying both lets the UI re-assume a class (see
    # ASSUMED_SUFFIX). Absent from pre-v5 caches and older release subsets.
    "feh_rgb": ["fehs_rgb"], "e_feh_rgb": ["fehs_errs_rgb"],
    "dmod_rgb": ["dmod_rgb"], "feh_ext_rgb": ["fehs_ext_rgb"],
    "feh_ms": ["fehs_ms"], "e_feh_ms": ["fehs_errs_ms"],
    "dmod_ms": ["dmod_ms"], "feh_ext_ms": ["fehs_ext_ms"],
    # [Fe/H] extrapolation flag: also per-class (fehs_ext_rgb / fehs_ext_ms),
    # so it swaps with the assumed class like feh/e_feh/dmod do
    "feh_ext": ["feh_extrapolation_flag"],
    # mpflags quality flags, carried as 1.0/0.0 (NaN where the catalog has
    # no such column, which disables the corresponding UI cut)
    "broadband_valid": ["broadband_valid"],
    "gaia_var_flag": ["gaia_var_flag"],
    # CaHK magnitude error: the mpflags lite catalogs call it MAGERR_PSF
    "magerr_cahk": ["magerr_psf_cahk", "MAGERR_PSF"],
}

# "Assumed [Fe/H], dmod values" -> column suffix ("" = as stored, i.e. the
# values matching each star's own star_class)
ASSUMED_SUFFIX = {"Matching star_class": "", "RGB": "_rgb", "MS": "_ms"}
PER_CLASS_COLS = ("feh", "e_feh", "dmod", "feh_ext")
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

    Returns (obs_cat, obs_inst, lit_known): obs_cat is the
    OBSERVED_CATEGORIES category of the nearest observation within the radius
    ("" if none — an observation always beats a literature entry), obs_inst
    that row's instrument (GMOS/GHOST for Gemini, MagE/MIKE for Magellan) for
    per-star display labels, lit_known flags a Literature entry within the
    radius.
    """
    from scipy.spatial import cKDTree
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
    xyz = _unit_vectors(ra, dec)
    obs_cat = np.full(len(xyz), "", dtype=object)
    obs_inst = np.full(len(xyz), "", dtype=object)
    lit_known = np.zeros(len(xyz), dtype=bool)
    obs = ledger[ledger["category"].isin(OBSERVED_CATEGORIES)]
    if len(obs):
        d, i = cKDTree(_unit_vectors(obs["ra"].values, obs["dec"].values)).query(
            xyz, k=1, distance_upper_bound=chord)
        hit = d <= chord
        ii = np.clip(i[hit], 0, len(obs) - 1)
        obs_cat[hit] = obs["category"].values[ii]
        if "instrument" in obs.columns:
            obs_inst[hit] = obs["instrument"].fillna("").astype(str).values[ii]
    lit = ledger[ledger["category"] == "Literature"]
    if len(lit):
        d, _ = cKDTree(_unit_vectors(lit["ra"].values, lit["dec"].values)).query(
            xyz, k=1, distance_upper_bound=chord)
        lit_known = d <= chord
    return obs_cat, obs_inst, lit_known


def build_cache(cat_path, progress=None):
    """Convert one catalog to the pruned Parquet cache + metadata sidecar."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy import units as u

    ledger = (pd.read_csv(EXCLUSION_CSV) if os.path.exists(EXCLUSION_CSV)
              else pd.DataFrame(columns=["ra", "dec", "category"]))
    lvdb_dwarfs, lvdb_clusters = load_lvdb()

    frames, missing = [], []
    pres_counts, n_total = {}, 0
    with fits.open(cat_path, memmap=True) as hdul:
        hdu = hdul[1]
        n = hdu.header["NAXIS2"]
        names = set(hdu.columns.names)
        pres_skipped = [rule for col, rule, _ in PRESELECT if col not in names]
        for start in range(0, n, CHUNK):
            rec = hdu.data[start:start + CHUNK]
            n_total += len(rec)
            # pre-selection happens HERE, before the schema frame exists,
            # so the page's "already applied" note is actually true
            keep = apply_preselect(rec, names, len(rec), pres_counts)
            rec = rec[keep]
            if not len(rec):
                if progress:
                    progress(min(1.0, (start + CHUNK) / n))
                continue
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
            if "source_id" in names:
                sid = np.asarray(rec["source_id"], dtype=np.int64)
                ids = pd.array(sid, dtype="Int64")
                ids[sid <= GAIA_NO_MATCH] = pd.NA   # sentinels -> blank
                cols["source_id"] = ids
            else:
                cols["source_id"] = pd.array([pd.NA] * len(rec), dtype="Int64")
            if "star_class" in names:
                sc = np.char.strip(rec["star_class"].astype(str))
            elif "is_rgb" in names:
                # is_rgb alone cannot classify a star with no [Fe/H] solution
                # (both branch values are NaN there) — those must not be
                # mislabeled RGB/MS, so they get their own class
                sc = np.where(~np.isfinite(cols["feh"]), "no-feh",
                              np.where(rec["is_rgb"], "RGB", "MS"))
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
            obs_cat, obs_inst, lit_known = classify_against_ledger(
                ra64, dec64, ledger)
            cols["obs_cat"] = obs_cat
            cols["obs_instrument"] = obs_inst
            cols["lit_known"] = lit_known
            cols["lvdb_host"], cols["lvdb_host_type"] = lvdb_host_typed(
                ra64, dec64, lvdb_dwarfs, lvdb_clusters)
            frames.append(pd.DataFrame(cols))
            if progress:
                progress(min(1.0, (start + len(rec)) / n))

    if not frames:
        raise RuntimeError("pre-selection removed every row — wrong catalog?")
    df = pd.concat(frames, ignore_index=True)
    df["star_class"] = df["star_class"].astype("category")
    df["obs_cat"] = df["obs_cat"].astype("category")
    df["obs_instrument"] = df["obs_instrument"].astype("category")
    df["lvdb_host"] = df["lvdb_host"].astype("category")
    df["lvdb_host_type"] = df["lvdb_host_type"].astype("category")

    print(f"pre-selection: {n_total:,} rows in")
    for col, rule, _ in PRESELECT:
        if col in names:
            print(f"  + {rule}: {pres_counts[rule]:,}")
    for rule in pres_skipped:
        print(f"  ! skipped (column absent): {rule}")

    meta = _make_meta(df, cat_path, missing)
    meta["preselect"] = {"n_input": int(n_total), "chain": pres_counts,
                         "skipped": pres_skipped}
    meta["lvdb_n_rh"] = DEFAULT_N_RH   # aperture the host columns were built with
    meta["lvdb_hosted"] = {t: int((df["lvdb_host_type"] == t).sum())
                           for t in ("dwarf", "cluster")}
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
    """(dwarfs, clusters) DataFrames [name, ra, dec, rhalf] from the first hit
    per file, keeping only systems within LVDB_MAX_DIST_KPC (heliocentric).
    rhalf is LVDB's half-light radius along the MAJOR AXIS in ARCMIN, NaN
    where the system has no structural fit — such rows still plot on the sky
    overlay but are skipped by the r_h proximity flag."""
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
                    "dec": pd.to_numeric(t["dec"], errors="coerce"),
                    "rhalf": pd.to_numeric(t.get("rhalf"), errors="coerce"),
                    "ellipticity": pd.to_numeric(t.get("ellipticity"),
                                                 errors="coerce"),
                    }).dropna(subset=["name", "ra", "dec"]))
                break
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["name", "ra", "dec", "rhalf",
                                           "ellipticity"]))
    return gather(LVDB_DWARF_FILES), gather(LVDB_CLUSTER_FILES)


DEFAULT_N_RH = 10.0  # default aperture for the LVDB proximity flag, in r_half
                     # (on-sky only: no distance term, by design)

# Excluded from the r_h proximity flag (still drawn on the sky overlay): the
# Clouds have their own dedicated sep_lmc/sep_smc excision cuts, and their
# r_half is so large (LMC 193', SMC 59') that a 5 r_h aperture reaches 16 deg
# for the LMC and swamps every compact dwarf in the flag.
LVDB_FLAG_EXCLUDE = ("LMC", "SMC")


def rhalf_circular(systems):
    """Circularized (spherically averaged) half-light radius in arcmin:
    r_h * sqrt(1 - ellipticity). LVDB's `rhalf` is the MAJOR-AXIS value, so
    using it directly as a circular aperture over-covers flattened systems —
    Sagittarius (e = 0.64) shrinks 342' -> 205'. This reproduces LVDB's own
    rhalf_sph_physical exactly. Systems with no ellipticity are treated as
    round (no change)."""
    rh = pd.to_numeric(systems["rhalf"], errors="coerce")
    e = pd.to_numeric(systems.get("ellipticity"), errors="coerce").fillna(0.0)
    return rh * np.sqrt(np.clip(1.0 - e, 0.0, 1.0))


def assumed_available(df, ranges=None):
    """Which "Assumed [Fe/H], dmod values" options this frame supports.

    A pre-v5 cache, an older release subset, or a catalog whose FITS simply
    lacks the per-class columns cannot re-assume a class. Presence alone is
    not enough: build_cache materializes a missing CANDS column as all-NaN,
    so a range from the metadata sidecar (None when nothing is finite) is
    what actually proves the values are there."""
    out = ["Matching star_class"]
    for label, sfx in ASSUMED_SUFFIX.items():
        if not sfx:
            continue
        cols = [f"{c}{sfx}" for c in PER_CLASS_COLS]
        if not all(c in df.columns for c in cols):
            continue
        if ranges is not None and not all(
                ranges.get(f"{c}{sfx}") for c in ("feh", "dmod")):
            continue
        out.append(label)
    return out


def assume_class(df, assumed):
    """View of df with feh/e_feh/dmod replaced by the values solved under the
    `assumed` class. "Matching star_class" returns df untouched. Only the
    three swapped columns are copied; the rest share storage with the cached
    frame, which must never be mutated (it is st.cache_resource-shared)."""
    sfx = ASSUMED_SUFFIX.get(assumed, "")
    if not sfx:
        return df
    src = {c: f"{c}{sfx}" for c in PER_CLASS_COLS}
    if not all(v in df.columns for v in src.values()):
        return df
    return df.assign(**{c: df[v] for c, v in src.items()})


def lvdb_host(star_ra, star_dec, systems, n_rh=DEFAULT_N_RH,
              exclude=LVDB_FLAG_EXCLUDE, circularize=True):
    """Name of the LVDB system whose circular n_rh * r_half aperture contains
    each star, "" where none does.

    Matching runs on 3D unit vectors, so the RA wraparound and the cos(dec)
    convergence near the poles are exact — several of these systems sit at
    high |dec| where an RA/Dec box would be wrong. Systems are visited
    largest-aperture-first so that where apertures overlap the most compact
    (most specific) host wins. Systems with no catalogued rhalf, and any named
    in `exclude` (the Clouds by default), are skipped. With circularize=True
    the aperture uses the spherically averaged radius (see rhalf_circular);
    pass False for the raw major-axis value.

    The aperture is circular: LVDB also carries ellipticity/position_angle,
    so for flattened systems this over-covers the minor axis and under-covers
    the major one."""
    from scipy.spatial import cKDTree
    star_ra = np.asarray(star_ra, float)
    host = np.full(len(star_ra), "", dtype=object)
    if not len(star_ra) or not len(systems):
        return host
    tree = cKDTree(_unit_vectors(star_ra, star_dec))
    ok = systems[~systems["name"].isin(list(exclude))].copy()
    ok["_r"] = rhalf_circular(ok) if circularize else pd.to_numeric(
        ok["rhalf"], errors="coerce")
    ok = ok.dropna(subset=["_r"]).sort_values("_r", ascending=False)
    for _, sy in ok.iterrows():
        theta = np.radians(float(n_rh) * float(sy["_r"]) / 60.0)  # arcmin
        if not theta > 0:
            continue
        # chord length subtending theta on the unit sphere
        idx = tree.query_ball_point(_unit_vectors([sy["ra"]], [sy["dec"]])[0],
                                    2.0 * np.sin(theta / 2.0))
        if idx:
            host[idx] = str(sy["name"])
    return host


def lvdb_host_typed(star_ra, star_dec, dwarfs, clusters, n_rh=DEFAULT_N_RH):
    """(host, host_type) per star from the precomputable LVDB proximity flag:
    host_type is "dwarf" or "cluster" ("" = near neither). Where a star sits
    inside both a dwarf and a cluster aperture (gc_dwarf_hosted clusters live
    inside their dwarfs), the cluster wins — it is the more specific host.
    Same defaults as lvdb_host: circularized r_half, Clouds excluded."""
    dh = lvdb_host(star_ra, star_dec, dwarfs, n_rh=n_rh)
    ch = lvdb_host(star_ra, star_dec, clusters, n_rh=n_rh)
    host = np.where(ch != "", ch, dh)
    htype = np.where(ch != "", "cluster", np.where(dh != "", "dwarf", ""))
    return host.astype(object), htype.astype(object)


LVDB_CUT_MODES = {   # "near an LVDB system" cut -> allowed lvdb_host_type
    "Near dwarf": ["dwarf"],
    "Near cluster": ["cluster"],
    "Near either": ["dwarf", "cluster"],
    "Isolated (near neither)": [""],
}
LVDB_NEAR_COLOR = "#e377c2"   # on-sky ring for stars near an LVDB system


# ──────────────────────── SIMBAD cross-match ────────────────────────
# On-demand only (button press): the filtered stars are cross-matched against
# LIVE SIMBAD at SIMBAD_RADIUS_ARCSEC via a TAP upload-join (sim-tap). The
# CDS X-Match mirror used previously is demonstrably incomplete — e.g.
# [MFW2011] 26017 at (14.689932, -33.706028) exists in live SIMBAD at 0.197"
# but is absent from the mirror out to 30" — so X-Match remains only as a
# clearly-labeled automatic fallback when TAP errors.

SIMBAD_COLS = ["simbad_main_id", "simbad_main_type", "simbad_sep_arcsec"]
SIMBAD_COLOR = "#2ca02c"   # green used for every "In SIMBAD" overlay
                           # (violet was hard to read on the dmod histogram;
                           # green kept everywhere for consistency)


def run_simbad_tap(ra, dec, radius_arcsec=SIMBAD_RADIUS_ARCSEC,
                   chunk=SIMBAD_TAP_CHUNK, query_fn=None):
    """Cross-match against LIVE SIMBAD: TAP upload-join on `basic` at
    radius_arcsec, chunked at `chunk` rows per synchronous query (the
    explorer's own SIMBAD_MAX_ROWS=50k cap means at most 5 chunks, so the
    async endpoint is unnecessary). Returns idx + SIMBAD_COLS, best (nearest)
    match per star. query_fn(adql, upload_table) -> astropy Table is
    injectable for tests; the default posts to sim-tap via astroquery."""
    if query_fn is None:
        from astroquery.simbad import Simbad
        def query_fn(adql, up):
            return Simbad.query_tap(adql, maxrec=2 * chunk, up=up)
    from astropy.table import Table
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    adql = ("SELECT up.idx, b.main_id, b.otype, "
            "DISTANCE(POINT('ICRS', up.ra, up.dec), "
            "POINT('ICRS', b.ra, b.dec)) * 3600.0 AS sep "
            "FROM TAP_UPLOAD.up AS up "
            "JOIN basic AS b ON 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), "
            f"CIRCLE('ICRS', up.ra, up.dec, {radius_arcsec / 3600.0:.10f}))")
    frames = []
    for s in range(0, len(ra), chunk):
        up = Table({"idx": np.arange(s, min(s + chunk, len(ra))),
                    "ra": ra[s:s + chunk], "dec": dec[s:s + chunk]})
        res = query_fn(adql, up)
        if res is not None and len(res):
            frames.append(res.to_pandas())
    if not frames:
        return pd.DataFrame(columns=["idx"] + SIMBAD_COLS)
    r = pd.concat(frames, ignore_index=True).sort_values("sep")
    r = r.drop_duplicates("idx")
    return pd.DataFrame({
        "idx": r["idx"].astype(int).values,
        "simbad_main_id": r["main_id"].astype(str).values,
        "simbad_main_type": r["otype"].astype(str).values,
        "simbad_sep_arcsec": r["sep"].astype(float).round(2).values})


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


def merge_simbad(df, mask, xmatch_fn=run_simbad_tap):
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


# ──────────────────────── selection manifest ────────────────────────
def _git_commit():
    """The running checkout's commit, 'unknown' outside a git checkout
    (e.g. a cloud container built from an archive)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=APP_DIR, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def selection_bundle(tab, manifest_text, stem):
    """One click, both files: an in-memory ZIP holding the plain CSV
    (targets_<stem>.csv) and the manifest (README_<stem>.txt)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"targets_{stem}.csv", tab.to_csv(index=False))
        z.writestr(f"README_{stem}.txt", manifest_text)
    return buf.getvalue()


def selection_manifest(ctx):
    """Plain-text reproducibility manifest: same catalog + repo commit +
    this file => the exact same filtered table. Field order is fixed so
    diffs between two manifests are meaningful."""
    L = ["MAGIC target explorer — selection manifest",
         f"generated_utc: {ctx['generated_utc']}",
         f"app_commit: {ctx['app_commit']}",
         f"catalog: {ctx['catalog']}",
         f"catalog_kind: {ctx['catalog_kind']}",
         f"cache_file: {ctx['cache_file']}",
         f"cache_schema: v{ctx['cache_schema']}",
         f"subset_cut: {ctx.get('subset_cut') or '(none — full catalog cache)'}",
         "preselection (baked into the cache):"]
    L += [f"  - {r}" for r in ctx["preselection"]]
    L += [f"population_classes: "
          f"{', '.join(ctx['population_classes']) or '(none)'}",
          f"assumed_mode: {ctx['assumed_mode']}",
          "cuts:"]
    for c in ctx["cuts"]:
        if c["col"] == "star_class":
            continue   # covered by population_classes above
        state = "enabled" if c.get("enabled") else "disabled"
        L.append(f"  - {c['col']}: {c['kind']} {c['value']} [{state}]")
    L += [f"lvdb_runtime_flag: {ctx['lvdb_runtime']}",
          f"ledger: {ctx['ledger']}",
          f"literature_counts_as_observed: {ctx['lit_is_observed']}",
          f"simbad: {ctx['simbad']}",
          "counts:"]
    for k, v in ctx["counts"].items():
        L.append(f"  - {k}: {v:,}")
    return "\n".join(L) + "\n"


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
            vals = (val,) if isinstance(val, str) else tuple(val)
            for cls in ("RGB", "MS", "ambiguous"):
                st.session_state[f"{key}:cls:{cls}"] = cls in vals
        elif col == "assumed":
            # reset to "Matching star_class" later if this catalog has no
            # per-class columns (the radio validates against its own options)
            st.session_state[key + ":assumed"] = val
        elif col in ("broadband_valid", "gaia_var_flag", "feh_ext"):
            # quality-flag checkboxes: True just switches the cut on
            st.session_state[f"{key}:{col}:on"] = bool(val)
        elif col in ("sep_lmc", "sep_smc") or rng.get(col):
            st.session_state[f"{key}:{col}:pending"] = val


def _clear_cuts(st, key):
    for cls, on in (("RGB", True), ("MS", True), ("ambiguous", False)):
        st.session_state[f"{key}:cls:{cls}"] = on
    st.session_state[key + ":assumed"] = "Matching star_class"
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
            meta_rel = _make_meta(dfr, f"{ref}@{rel['tag']}")
            cut = subset_cut_from_parquet(pq)
            if cut:
                meta_rel["subset_cut"] = cut
            with open(js, "w") as f:
                json.dump(meta_rel, f, indent=1)
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
    if rng.get("dmod"):   # distance is selected in pc, stored as a modulus
        rng["dist_pc"] = [float(dmod_to_pc(rng["dmod"][0])),
                          float(dmod_to_pc(rng["dmod"][1]))]
    key = os.path.basename(pq)  # widget namespace per catalog version

    ctrl, view = st.columns([1, 3], gap="medium")

    # ── cuts (each with an enable toggle; disabled cuts filter nothing) ──
    cuts = []
    with ctrl:
        st.subheader("Cuts")
        with st.expander("Pre-selection already applied to this catalog"):
            note = ("Applied when the cache was built, before any cut "
                    "below:\n\n" + preselect_note())
            cut = meta.get("subset_cut")
            if cut:
                note += "\n- " + SUBSET_CUT_PLAIN.get(cut, f"row pre-cut: `{cut}`")
            st.markdown(note
                        + "\n\nThe counts and sliders on this page all "
                          "describe the post-pre-selection sample.")
        b1, b2 = st.columns(2)
        if b1.button("Fiducial cuts", use_container_width=True,
                     help="Standard MAGIC low-metallicity giant selection — "
                          "edit the FIDUCIAL dict at the top of explorer.py"):
            _queue_fiducial(st, key, rng)
        if b2.button("Clear all", use_container_width=True):
            _clear_cuts(st, key)

        st.markdown("**Classification in catalog (star_class)**")
        cb = st.columns(3)
        keep_cls = []
        for i, (label, val, default) in enumerate(
                (("RGB", "RGB", True), ("MS", "MS", True),
                 ("Ambiguous", "ambiguous", False))):
            st.session_state.setdefault(f"{key}:cls:{val}", default)
            if cb[i].checkbox(label, key=f"{key}:cls:{val}"):
                keep_cls.append(val)
        if not keep_cls:
            st.warning("No star_class selected — nothing will pass the cuts.")
        cuts.append({"col": "star_class", "kind": "isin", "value": keep_cls,
                     "enabled": True})

        opts = assumed_available(df, meta.get("ranges"))
        st.session_state.setdefault(key + ":assumed", opts[0])
        if st.session_state[key + ":assumed"] not in opts:
            st.session_state[key + ":assumed"] = opts[0]
        assumed = st.radio(
            "Assumed [Fe/H], dmod values", opts, horizontal=True,
            key=key + ":assumed",
            help="The catalog solves every star under both an RGB and an MS "
                 "assumption. 'Matching star_class' uses the values for each "
                 "star's own classification (feh / e_feh / dmod); RGB or MS "
                 "forces that assumption for every star "
                 "(fehs_rgb / dmod_rgb, fehs_ms / dmod_ms).")
        if len(opts) == 1:
            st.caption("This catalog cache predates the per-class columns — "
                       "rebuild it to re-assume RGB / MS.")
        df = assume_class(df, assumed)
        sfx = ASSUMED_SUFFIX.get(assumed, "")
        if sfx:   # re-range the sliders onto the assumed-class percentiles
            for c in PER_CLASS_COLS:
                if rng.get(f"{c}{sfx}"):
                    rng[c] = rng[f"{c}{sfx}"]
            if rng.get("dmod"):
                rng["dist_pc"] = [float(dmod_to_pc(rng["dmod"][0])),
                                  float(dmod_to_pc(rng["dmod"][1]))]

        def add(cut_on, col, kind, value):
            cuts.append({"col": col, "kind": kind, "value": value, "enabled": cut_on})

        for col, label, fmt in (("pmra", "pmra (mas/yr)", "%.1f"),
                                ("pmdec", "pmdec (mas/yr)", "%.1f"),
                                ("gi0", "(g-i)₀ color", "%.2f"),
                                ("feh", "[Fe/H]", "%.2f")):
            if rng.get(col):
                on, val = _range_cut(st, label, rng[col], f"{key}:{col}", fmt)
                add(on, col, "range", val)
        # selected in parsecs, cut on the stored `dmod` column so the fiducial
        # marker line and every downstream dmod consumer keep working
        if rng.get("dist_pc"):
            on, val = _range_cut(st, "distance (pc)", rng["dist_pc"],
                                 f"{key}:dist_pc", "%.0f")
            add(on, "dmod", "range",
                (float(pc_to_dmod(val[0])), float(pc_to_dmod(val[1]))))
        for col, label, fmt, lo in (("ebv", "E(B-V) max", "%.3f", 0.0),
                                    ("e_feh", "[Fe/H] error max", "%.2f", 0.0),
                                    ("mag_g", "depth: g max (mag_psf_g)",
                                     "%.2f", None),
                                    ("magerr_cahk", "depth: σ(CaHK) max",
                                     "%.3f", 0.0)):
            if rng.get(col):
                bounds = (rng[col][0] if lo is None else lo, rng[col][1])
                on, val = _thresh_cut(st, label, bounds, f"{key}:{col}", fmt)
                add(on, col, "max", val)

        # footgun guard: ambiguous stars have no ADOPTED feh/e_feh/dmod
        # (all NaN), so a per-class cut under 'Matching star_class' silently
        # drops every one of them
        if (assumed == "Matching star_class" and "ambiguous" in keep_cls
                and any(c["enabled"]
                        and c["col"] in ("feh", "e_feh", "dmod", "feh_ext")
                        for c in cuts)):
            st.warning(
                "Ambiguous stars have no adopted [Fe/H]/dmod, so with "
                "Assumed = 'Matching star_class' they will ALL fail the "
                "enabled [Fe/H] / error / distance cut. Switch 'Assumed' to "
                "RGB or MS to cut them by their branch values instead.")

        st.subheader("Quality flags")
        # only offered when the catalog actually carries the column (a missing
        # CANDS column is materialized as all-NaN, and rng() is None for it)
        any_flag = False
        for col, label, keep, helptext in (
                ("broadband_valid", "broadband_valid only", 1.0,
                 "Keep only stars inside the g-r vs r-i color-color polygon."),
                ("gaia_var_flag", "exclude Gaia variables", 0.0,
                 "Drop stars matched to Gaia DR3 variables (I/358/varisum, 1\")."),
        ):
            if not rng.get(col):
                continue
            any_flag = True
            on = st.checkbox(label, key=f"{key}:{col}:on", help=helptext)
            # equality on a 1.0/0.0 column; NaN never passes an enabled cut
            add(on, col, "range", (keep, keep))
        # mode-aware extrapolation cut: feh_ext is post-assume_class, so it
        # already IS the adopted / RGB-assumed / MS-assumed flag. NaN fails
        # the enabled cut — deliberate and faithful: in the catalogs feh_ext
        # is NaN exactly where the mode's [Fe/H] is NaN (verified on v260810:
        # zero NaN-ext rows with a finite mode [Fe/H]), so no star with a
        # usable [Fe/H] is ever dropped by this box.
        if "feh_ext" in df.columns and bool(
                np.isfinite(df["feh_ext"].to_numpy()).any()):
            any_flag = True
            on = st.checkbox(
                "Exclude feh extrapolation flag", key=f"{key}:feh_ext:on",
                help="Keep only stars whose [Fe/H] in the active assumption "
                     "mode is not an extrapolation (flag == 0). Follows the "
                     "'Assumed' selector; stars with no [Fe/H] in that mode "
                     "fail while enabled.")
            add(on, "feh_ext", "range", (0.0, 0.0))
        if not any_flag:
            st.caption("This catalog carries no mpflags quality columns.")

        if "lvdb_host_type" in df.columns:
            on = st.checkbox(
                "LVDB proximity cut", key=f"{key}:lvdbcut:on",
                help="Precomputed at cache build: a star is 'near' a system "
                     f"when it falls inside {meta.get('lvdb_n_rh', DEFAULT_N_RH):g} "
                     "circularized r_half of an LVDB dwarf (dwarf_mw) or MW "
                     "star cluster (gc_* tables); Clouds excluded — use the "
                     "Excise Clouds cuts for those.")
            mode = st.selectbox("LVDB proximity", list(LVDB_CUT_MODES),
                                key=f"{key}:lvdbcut", disabled=not on,
                                label_visibility="collapsed")
            add(on, "lvdb_host_type", "isin", LVDB_CUT_MODES[mode])

        st.subheader("Excise Clouds")
        for name, (cra, cdec, rdef), col in (("LMC", LMC, "sep_lmc"),
                                             ("SMC", SMC, "sep_smc")):
            on, r = _thresh_cut(st, f"cut {name} ({cra}, {cdec}) — radius (deg)",
                                (0.5, 15.0), f"{key}:{col}", "%.1f", default=rdef)
            add(on, col, "min", r)

        st.subheader("LVDB proximity")
        near_on = st.checkbox("flag stars near an LVDB dwarf / star cluster",
                              key=key + ":nearlvdb")
        n_rh = st.number_input("aperture (× r_half)", 0.5, 50.0, DEFAULT_N_RH,
                               0.5, key=key + ":nrh", disabled=not near_on,
                               help="Circular aperture around each LVDB system, "
                                    "in half-light radii. Adds an lvdb_host "
                                    "column to the target table.")
        drop_near = st.checkbox("exclude flagged stars from the selection",
                                key=key + ":droplvdb", disabled=not near_on)

    mask = apply_cuts(df, cuts)

    # LVDB proximity flag — evaluated on the filtered rows only, so the cost
    # tracks the selection rather than the full catalog
    lvdb_flag = np.full(len(df), "", dtype=object)
    if near_on:
        dwarfs_rh, clusters_rh = load_lvdb()
        systems = pd.concat([dwarfs_rh, clusters_rh], ignore_index=True)
        sel = np.flatnonzero(mask)
        lvdb_flag[sel] = lvdb_host(df["ra"].to_numpy()[sel],
                                   df["dec"].to_numpy()[sel],
                                   systems, n_rh=float(n_rh))
        n_near = int((lvdb_flag != "").sum())
        n_skip = int(systems["rhalf"].isna().sum())
        st.caption(f"{n_near:,} of {int(mask.sum()):,} filtered stars fall "
                   f"within {float(n_rh):g} r_half of an LVDB system "
                   f"({', '.join(LVDB_FLAG_EXCLUDE)} excised — use the "
                   "Excise Clouds cuts for those)"
                   + (f"; {n_skip} of {len(systems)} systems skipped for "
                      "having no catalogued r_half" if n_skip else ""))
        if drop_near:
            mask = mask & (lvdb_flag == "")

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
                res = None
                try:
                    with st.spinner("Querying live SIMBAD (TAP upload join) ..."):
                        res = merge_simbad(df, mask)
                except Exception as e_tap:
                    st.warning(f"Live SIMBAD TAP failed ({e_tap}) — falling "
                               "back to the CDS X-Match mirror, which is "
                               "KNOWN INCOMPLETE: treat no-matches with care.")
                    try:
                        with st.spinner("Querying the CDS X-Match mirror ..."):
                            res = merge_simbad(df, mask,
                                               xmatch_fn=run_simbad_xmatch)
                    except Exception as e_xm:
                        st.warning(f"X-Match fallback failed too (offline or "
                                   f"CDS error) — the app keeps working: {e_xm}")
                if res is not None:
                    append_simbad_cache(df, res)
                    sim = st.session_state[skey]
                    st.session_state[skey] = pd.concat(
                        [sim[~sim.index.isin(res.index)], res]).sort_index()
                    st.session_state[skey + ":queried"].add(h)
                    st.success(f"{len(res):,} of {n_sel:,} filtered stars "
                               f"are in SIMBAD.")
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
        from datetime import datetime, timezone
        try:
            led_n = sum(1 for _ in open(EXCLUSION_CSV)) - 1
            led_mtime = datetime.fromtimestamp(
                os.path.getmtime(EXCLUSION_CSV),
                tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            ledger_line = (f"data/master_exclusion.csv rows={led_n} "
                           f"mtime={led_mtime}")
        except OSError:
            ledger_line = "(no ledger file)"
        manifest_ctx = {
            "generated_utc": datetime.now(timezone.utc)
                             .strftime("%Y-%m-%d %H:%M:%S UTC"),
            "app_commit": _git_commit(),
            "catalog": choice,
            "catalog_kind": ("release asset" if kind == "release"
                             else "local file"),
            "cache_file": os.path.basename(pq),
            "cache_schema": SCHEMA_VERSION,
            "subset_cut": meta.get("subset_cut"),
            "preselection": (list(meta["preselect"]["chain"])
                             if meta.get("preselect")
                             else [r for _, r, _ in PRESELECT]),
            "population_classes": keep_cls,
            "assumed_mode": assumed,
            "cuts": cuts,
            "lvdb_runtime": (f"on, n_rh={float(n_rh):g}, "
                             f"exclude_flagged={bool(drop_near)}"
                             if near_on else "off"),
            "ledger": ledger_line,
            "lit_is_observed": bool(lit_obs),
            "simbad": (f"live SIMBAD TAP ({SIMBAD_TAP_URL}), radius="
                       f"{SIMBAD_RADIUS_ARCSEC:g} arcsec, X-Match mirror as "
                       "labeled fallback, queried_this_session="
                       f"{bool(st.session_state[skey + ':queried'])}"),
            "counts": {"pass": m["selected"],
                       "observed_by_us": m["observed_us"],
                       "literature_known": m["literature"],
                       "remaining": m["remaining"],
                       "in_simbad": n_simbad},
        }
        stem = os.path.splitext(os.path.basename(ref))[0]
        ui = {"st": st, "key": key, "sim": sim, "sim_mask": sim_mask,
              "cuts": cuts, "lvdb_flag": lvdb_flag, "assumed_sfx": sfx,
              "manifest": manifest_ctx, "catalog_stem": stem,
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


def star_detail(df, row, sim, sfx=""):
    """One-line summary of a single star (df row position) for the table.
    sfx labels the metallicity/distance values by assumption mode ("_rgb",
    "_ms", or "" for values matching each star's own star_class)."""
    r = df.iloc[row]
    if str(r["obs_cat"]):
        # per-star provenance reads as the instrument (GMOS/GHOST/MagE/MIKE);
        # category-level accounting elsewhere stays Gemini/Magellan
        inst = (str(r["obs_instrument"])
                if "obs_instrument" in df.columns else "")
        status = f"observed: {inst or r['obs_cat']}"
    else:
        status = "literature-known" if bool(r["lit_known"]) else "unobserved"
    line = (f"**({r['ra']:.5f}, {r['dec']:.5f})** · g = {r['mag_g']:.2f} · "
            f"[Fe/H]{sfx} = {r['feh']:.2f} ± {r['e_feh']:.2f} · "
            f"dmod{sfx} = {r['dmod']:.2f} · "
            f"distance{sfx} = {dmod_to_pc(r['dmod']):,.0f} pc · "
            f"{r['star_class']} · {status}")
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
    # same marker spec as the e_feh panel's targets, so both read alike
    return fig_go.Scattergl(x=x, y=y, mode="markers", name=name,
                            marker=dict(size=5, color="#3a5f8a", opacity=0.75))


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
    if "lvdb_host" in df.columns:
        near = mask & (df["lvdb_host"].astype(str).to_numpy() != "")
        n_near = int(near.sum())
        if n_near:
            pos = np.flatnonzero(near)
            sampled = n_near > SCATTER_MAX
            if sampled:   # display-only decimation, like the density layer
                pos = np.random.default_rng(0).choice(pos, SCATTER_MAX,
                                                      replace=False)
            fig.add_trace(go.Scattergl(
                x=df[xc].to_numpy()[pos], y=df[yc].to_numpy()[pos],
                mode="markers",
                name="Near LVDB system" + (" (sampled)" if sampled else ""),
                customdata=df["lvdb_host"].astype(str).to_numpy()[pos],
                hovertemplate="%{customdata}<extra>Near LVDB system</extra>",
                marker=dict(symbol="circle-open", size=12,
                            color=LVDB_NEAR_COLOR, line=dict(width=1))))
    inst_all = (df["obs_instrument"].astype(str).to_numpy()
                if "obs_instrument" in df.columns else None)
    for sel, name, marker, cdata in (
            (obs_us, "observed by us",
             dict(symbol="x", size=9, color="#e45756"), inst_all),
            (lit, "literature-known",
             dict(symbol="circle-open", size=9, color="#f58518"), None),
            (ui["sim_mask"], "In SIMBAD",
             dict(symbol="diamond-open", size=10, color=SIMBAD_COLOR), None)):
        if sel.any():
            kw = {}
            if cdata is not None:   # hover names the instrument (GMOS/GHOST/
                kw = dict(customdata=cdata[sel],   # MagE/MIKE), not the bucket
                          hovertemplate="%{customdata}"
                                        "<extra>observed by us</extra>")
            fig.add_trace(go.Scattergl(
                x=df[xc].to_numpy()[sel], y=df[yc].to_numpy()[sel],
                mode="markers", name=name, marker=marker, **kw))

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
    cols = [c for c in ("ra", "dec", "source_id", "star_class",
                        "feh", "e_feh", "dmod",
                        "gi0", "mag_g", "pmra", "pmdec", "ebv",
                        "obs_cat", "obs_instrument", "lit_known",
                        "lvdb_host", "lvdb_host_type")
            if c in df.columns]
    tab = df.loc[mask, cols].copy()
    if "dmod" in tab.columns:   # heliocentric distance of the mode's dmod
        tab.insert(tab.columns.get_loc("dmod") + 1, "distance",
                   np.round(dmod_to_pc(tab["dmod"]), 1))
    sfx = ui.get("assumed_sfx", "")
    if sfx:   # values already ARE this mode's values — labeling only
        tab = tab.rename(columns={c: c + sfx
                                  for c in ("feh", "e_feh", "dmod", "distance")
                                  if c in tab.columns})
    tab["in_simbad"] = np.asarray(ui["sim_mask"])[tab.index]  # sortable
    flag = ui.get("lvdb_flag")   # the runtime custom-aperture flag, if on
    if flag is not None and (np.asarray(flag) != "").any():
        tab["lvdb_host_custom"] = np.asarray(flag, dtype=object)[tab.index]
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
        st.markdown(star_detail(df, r, ui["sim"], sfx=ui.get("assumed_sfx", "")))
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
    stem = ui.get("catalog_stem", "selection")
    if ui.get("manifest"):
        txt = selection_manifest(ui["manifest"])
        # stashed for tests: button payloads are not introspectable
        st.session_state[ui["key"] + ":manifest_txt"] = txt
        st.session_state[ui["key"] + ":bundle_stem"] = stem
        st.download_button("Download filtered targets (.zip)",
                           selection_bundle(tab, txt, stem),
                           f"targets_{stem}.zip", "application/zip")
    else:   # no manifest context (should not happen) — plain CSV fallback
        st.download_button("Download filtered targets CSV",
                           tab.to_csv(index=False).encode(),
                           f"targets_{stem}.csv", "text/csv")


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
