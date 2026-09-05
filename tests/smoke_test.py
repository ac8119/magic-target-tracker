#!/usr/bin/env python3
"""
End-to-end smoke test for the workstation ingestion.

Run after build_followup_progress.py:
    python3 tests/smoke_test.py

Checks that data/target_runs.csv is well-formed, that the explorer's
filter + metric logic is correct on a synthetic frame, and that the app
renders the "Target explorer" and "Follow-up progress" pages
(via streamlit.testing).
"""
import os
import sys

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
CSV = os.path.join(BASE, "data", "target_runs.csv")

# ── 1. the ingestion output is sane ──
df = pd.read_csv(CSV)
required = {"run", "source_file", "name", "gaia_source_id", "ra", "dec",
            "instrument", "status", "match_detail", "sep_arcsec", "in_earlier_run"}
assert required <= set(df.columns), f"missing columns: {required - set(df.columns)}"
assert len(df) > 0, "target_runs.csv is empty"
assert df["ra"].between(0, 360).all() and df["dec"].between(-90, 90).all()
assert set(df["status"].unique()) <= {"observed", "literature-known", "proposed"}
assert (df.loc[df["status"] != "proposed", "sep_arcsec"] <= 1.0).all(), \
    "a match exceeds the 1 arcsec radius"
n_runs = df["run"].nunique()
print(f"OK  target_runs.csv: {len(df)} rows, {n_runs} runs, "
      f"{(df['status'] == 'observed').sum()} observed, "
      f"{(df['status'] == 'literature-known').sum()} literature-known")

# ── 2. explorer filter + metric logic on a synthetic frame ──
import explorer

syn = pd.DataFrame({
    "ra":       [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    "dec":      [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    "feh":      [-3.0, -2.0, np.nan, -3.5, -1.0, -2.8],
    "e_feh":    [0.2, 0.3, np.nan, 0.1, 0.9, 0.4],
    "ebv":      [0.05, 0.15, 0.02, 0.01, 0.30, 0.08],
    "mag_g":    [17.0, 18.0, 16.0, 18.4, 19.5, 17.5],
    "star_class": pd.Categorical(["RGB", "MS", "ambiguous", "RGB", "MS", "RGB"]),
    "sep_lmc":  [10.0, 2.0, 8.0, 20.0, 1.0, 30.0],
    "obs_cat":  ["MAGIC_Magellan", "", "", "Gemini", "", ""],
    "obs_instrument": ["MagE", "", "", "GHOST", "", ""],
    "lit_known": [True, False, False, False, False, True],
})
cuts = [
    {"col": "feh", "kind": "range", "value": (-4.0, -2.5), "enabled": True},
    {"col": "ebv", "kind": "max", "value": 0.1, "enabled": True},
    {"col": "mag_g", "kind": "max", "value": 18.5, "enabled": True},
    {"col": "star_class", "kind": "isin", "value": ["RGB"], "enabled": True},
    {"col": "sep_lmc", "kind": "min", "value": 5.0, "enabled": True},
]
mask = explorer.apply_cuts(syn, cuts)
# rows 0,3,5 are RGB with feh in range, ebv<=0.1, outside the LMC circle;
# row 2 has NaN feh and must fail the enabled range cut
assert mask.tolist() == [True, False, False, True, False, True], mask.tolist()
# category split: row 0 observed by us (and lit — observation wins),
# row 3 observed by us, row 5 literature-only
m = explorer.metrics(syn, mask, lit_is_observed=True)
assert m == {"selected": 3, "observed_us": 2, "literature": 1,
             "remaining": 0}, m
m = explorer.metrics(syn, mask, lit_is_observed=False)
assert m["remaining"] == 1, m
# disabled cuts must filter nothing
for c in cuts:
    c["enabled"] = False
assert explorer.apply_cuts(syn, cuts).all()

# LVDB occupancy: a marker is kept only with surviving stars in its own or
# an 8-adjacent ~2 deg pixel (RA wraparound included)
stars_ra = np.array([10.0, 359.9])
stars_dec = np.array([-1.0, -30.0])
m_ra = np.array([10.5, 0.5, 200.0, 10.5])
m_dec = np.array([-0.5, -30.5, 50.0, -80.0])
keep = explorer.occupied(stars_ra, stars_dec, m_ra, m_dec)
# marker 0: same/adjacent pixel; marker 1: adjacent across the RA wrap;
# markers 2, 3: nowhere near any star
assert keep.tolist() == [True, True, False, False], keep.tolist()
assert not explorer.occupied(np.array([]), np.array([]), m_ra, m_dec).any()

# classify_against_ledger on a synthetic ledger: an observation within 2"
# sets its category and beats a literature entry at the same position
ledger = pd.DataFrame({
    "ra":       [10.0, 10.0, 20.0, 30.0],
    "dec":      [-1.0, -1.0, -2.0, -3.0],
    "category": ["MAGIC_Magellan", "Literature", "Literature", "Gemini"],
    "instrument": ["MagE", "", "", "GHOST"]})
oc, oi, lk = explorer.classify_against_ledger(
    np.array([10.0, 20.0, 30.0]), np.array([-1.0, -2.0, -3.0]), ledger)
# Gemini (GMOS or GHOST rows alike) counts as observed-by-us; the
# per-star label carries the instrument through
assert oc.tolist() == ["MAGIC_Magellan", "", "Gemini"], oc.tolist()
assert oi.tolist() == ["MagE", "", "GHOST"], oi.tolist()
assert lk.tolist() == [True, True, False], lk.tolist()
assert "Gemini" in explorer.OBSERVED_CATEGORIES
assert "GMOS" not in explorer.OBSERVED_CATEGORIES

# SIMBAD path with a stubbed X-Match (no network) + persistent-cache roundtrip
def fake_xmatch(ra, dec, radius_arcsec=1.0):
    return pd.DataFrame({"idx": [1], "simbad_main_id": ["HD 1"],
                         "simbad_main_type": ["Star"],
                         "simbad_sep_arcsec": [0.3]})
sim = explorer.merge_simbad(syn, mask, xmatch_fn=fake_xmatch)
# filtered rows are positions 0,3,5 -> local idx 1 is df row 3
assert sim.index.tolist() == [3] and sim.iloc[0]["simbad_main_id"] == "HD 1"
import tempfile
_orig = explorer.SIMBAD_CACHE_CSV
explorer.SIMBAD_CACHE_CSV = os.path.join(tempfile.mkdtemp(), "simbad_cache.csv")
try:
    explorer.append_simbad_cache(syn, sim)
    explorer.append_simbad_cache(syn, sim)          # idempotent
    back = explorer.load_simbad_cache(syn)
    assert back.index.tolist() == [3], back
    assert back.iloc[0]["simbad_main_id"] == "HD 1"
finally:
    explorer.SIMBAD_CACHE_CSV = _orig
# click-to-highlight helpers: selection state -> table ordering + detail
tab = syn.loc[[0, 3, 5], ["ra", "dec", "feh"]]
tab2, sel = explorer.move_selected_first(tab, [5, 999])
assert sel == [5] and tab2.index.tolist() == [5, 0, 3], tab2.index.tolist()
tab3, sel3 = explorer.move_selected_first(tab, [])
assert sel3 == [] and tab3.index.tolist() == [0, 3, 5]
syn2 = syn.assign(mag_g=17.0, dmod=16.5)
detail = explorer.star_detail(syn2, 5, sim)
assert "[Fe/H] = -2.80" in detail and "literature-known" in detail, detail
assert "SIMBAD" not in detail
detail3 = explorer.star_detail(syn2, 3, sim)   # sim has row 3 (HD 1)
# the per-star label is the instrument, not the Gemini umbrella
assert "SIMBAD: HD 1" in detail3 and "observed: GHOST" in detail3, detail3
assert "Gemini" not in detail3, detail3
assert "observed: MagE" in explorer.star_detail(syn2, 0, sim)
d_sfx = explorer.star_detail(syn2, 0, sim, sfx="_rgb")
assert "[Fe/H]_rgb =" in d_sfx and "dmod_rgb =" in d_sfx, d_sfx
# per-user catalog globs + allowlisted runtime paths
fake_secrets = {"catalogs": {
    "globs": ["/data/magic/*.fits"],
    "users": {"guy": ["/home/guy/cats/*.fits"]},
    "allowed_roots": ["/data/magic"]}}
assert explorer.resolve_globs(fake_secrets, user="guy") == \
    ["/home/guy/cats/*.fits", "/data/magic/*.fits"]      # user first
assert explorer.resolve_globs(fake_secrets, user="ani") == \
    ["/data/magic/*.fits"]                               # no user entry
assert explorer.resolve_globs({}, env_value="/a/*.fits:/b/*.fits") == \
    ["/a/*.fits", "/b/*.fits"]                           # env when no secrets
assert explorer.resolve_globs(fake_secrets, env_value="/a/*.fits") == \
    ["/data/magic/*.fits"]                               # secrets beat env
assert explorer.resolve_globs({}) == explorer.LOCAL_FALLBACK_GLOBS
roots = fake_secrets["catalogs"]["allowed_roots"]
assert explorer.path_allowed("/data/magic/deep/*.fits", roots)
assert explorer.path_allowed("/data/magic", roots)
assert not explorer.path_allowed("/etc/passwd", roots)
assert not explorer.path_allowed("/data/magic/../../etc", roots)   # traversal
assert not explorer.path_allowed("/data/magicother/x.fits", roots) # prefix trick
# PRESELECT: predicates + cumulative chain on a synthetic record chunk
INT64_MIN = np.iinfo(np.int64).min   # masked-int64 fill in the mpflags files
rec = {
    "source_id":        np.array([999999, 10**12, 10**12, 10**12, 10**12,
                                  10**12, INT64_MIN]),
    "extended_class_g": np.array([1, -9, 0, 1, 1, 2, 1]),
    "mag_psf_cahk":     np.array([18.0, 18.0, 1e20, 17.0, 95.0, 18.0, 18.0]),
    "ebv_sfd98":        np.array([0.05, 0.05, 0.05, 0.30, 0.05, 0.05, 0.05]),
}
counts = {}
keep = explorer.apply_preselect(rec, set(rec), 7, counts)
# row0 999999 sentinel; row1 extended -9; row2 CaHK 1e20 sentinel;
# row3 ebv 0.30; row4 CaHK >90 placeholder; row5 extended 2; row6 masked id
assert keep.tolist() == [False] * 7, keep.tolist()
assert list(counts.values()) == [5, 3, 1, 0], counts
# a rule whose column is absent is skipped, not fatal
keep2 = explorer.apply_preselect(rec, {"ebv_sfd98"}, 7, None)
assert keep2.tolist() == [True, True, True, False, True, True, True]
print("OK  PRESELECT predicates + cumulative chain")

# LVDB typed host flag: aperture membership, cluster-beats-dwarf, Clouds out
dw = pd.DataFrame({"name": ["Dwarfy", "LMC"], "ra": [10.0, 80.89],
                   "dec": [-30.0, -69.76], "rhalf": [10.0, 193.0],
                   "ellipticity": [0.0, np.nan]})
cl = pd.DataFrame({"name": ["Clusty"], "ra": [10.0], "dec": [-30.05],
                   "rhalf": [1.0], "ellipticity": [0.0]})
# 10 r_h apertures: Dwarfy 100' = 1.67 deg, Clusty 10' = 0.17 deg
host, typ = explorer.lvdb_host_typed(
    [10.0, 10.0, 10.0, 80.89, 40.0],        # star2 at 2 deg: outside Dwarfy
    [-29.5, -30.04, -28.0, -69.76, 0.0], dw, cl)   # star3 = LMC center: excluded
assert host.tolist() == ["Dwarfy", "Clusty", "", "", ""], host.tolist()
assert typ.tolist() == ["dwarf", "cluster", "", "", ""], typ.tolist()
lv = pd.DataFrame({"lvdb_host_type": pd.Categorical(
    ["dwarf", "cluster", "", ""])})
for mode, expect in (("Near dwarf", [1, 0, 0, 0]),
                     ("Near cluster", [0, 1, 0, 0]),
                     ("Near either", [1, 1, 0, 0]),
                     ("Isolated (near neither)", [0, 0, 1, 1])):
    got = explorer.apply_cuts(lv, [{"col": "lvdb_host_type", "kind": "isin",
                                    "value": explorer.LVDB_CUT_MODES[mode],
                                    "enabled": True}])
    assert got.tolist() == [bool(x) for x in expect], (mode, got.tolist())
print("OK  LVDB typed host flag + proximity cut modes")

# mode-aware extrapolation cut: post-assume feh_ext == 0, NaN fails
ext = pd.DataFrame({
    "feh":         [-3.0, -2.0, np.nan],
    "e_feh":       [0.2, 0.2, np.nan],
    "dmod":        [17.0, 17.0, np.nan],
    "feh_ext":     [0.0, 1.0, np.nan],
    "feh_rgb":     [-3.1, -2.2, -2.9],
    "e_feh_rgb":   [0.2, 0.2, 0.3],
    "dmod_rgb":    [17.1, 17.2, 17.3],
    "feh_ext_rgb": [1.0, 0.0, 0.0],
    "feh_ms":      [np.nan] * 3, "e_feh_ms": [np.nan] * 3,
    "dmod_ms":     [np.nan] * 3, "feh_ext_ms": [np.nan] * 3,
})
ext_cut = [{"col": "feh_ext", "kind": "range", "value": (0.0, 0.0),
            "enabled": True}]
assert explorer.apply_cuts(ext, ext_cut).tolist() == [True, False, False]
assert explorer.apply_cuts(
    explorer.assume_class(ext, "RGB"), ext_cut).tolist() == [False, True, True]
assert explorer.apply_cuts(
    explorer.assume_class(ext, "MS"), ext_cut).tolist() == [False] * 3
print("OK  mode-aware feh extrapolation cut (NaN fails)")

# build_exclusion_master additions ingestion: 1" same-category+program dedup
import tempfile as _tf2
from build_exclusion_master import merge_additions
_addir = _tf2.mkdtemp()
with open(os.path.join(_addir, "ledger_additions_test.csv"), "w") as f:
    f.write("name,ra,dec,category,detail,instrument\n"
            "dup_star,50.0,-30.0,Gemini,GS-1,GMOS\n"       # same cat+prog at 0"
            "multi_star,60.0,-40.0,Gemini,GS-1,GMOS\n"     # Magellan there: keep
            "fresh_star,70.0,-50.0,Gemini,GS-2,GHOST\n")
rows = [["dup_star", 50.0, -30.0, "Gemini", "GS-1", "GMOS"],
        ["mag_star", 60.0, -40.0, "MAGIC_Magellan", "240101_MagE", "MagE"]]
n = merge_additions(rows, _addir)
assert n == 2 and len(rows) == 4, (n, len(rows))
names = [r[0] for r in rows]
assert "multi_star" in names and "fresh_star" in names
assert names.count("dup_star") == 1, "same-category+program duplicate re-added"
assert rows[-1][5] == "GHOST" and rows[-1][3] == "Gemini"
print("OK  ledger additions ingestion + same-category dedup")

# cloud subset maker: cut + schema columns + roundtrip through the loader
import tempfile as _tf
from build_cloud_subset import make_subset
big = syn.assign(mag_g=17.0, dmod=16.5, junk_col=1.0,
                 source_id=pd.array([4906349572689836160, pd.NA,
                                     6499744678154128640, 5, 6, 7],
                                    dtype="Int64"))
sub = make_subset(big, "feh == feh and star_class in ['RGB', 'MS']")
assert len(sub) == 5, len(sub)   # row 2 (NaN feh AND ambiguous) dropped
assert "junk_col" not in sub.columns and "obs_cat" in sub.columns
from build_cloud_subset import write_subset
pq_tmp = os.path.join(_tf.mkdtemp(), "sub.parquet")
write_subset(sub, pq_tmp, "feh == feh and star_class in ['RGB', 'MS']")
back = pd.read_parquet(pq_tmp)
# 19-digit Gaia ids survive exactly (Int64, never float-mangled), NA blank
assert str(back["source_id"].dtype) == "Int64", back["source_id"].dtype
assert back["source_id"].iloc[0] == 4906349572689836160
csv_txt = back.to_csv(index=False)
assert "4906349572689836160" in csv_txt and "4.9063" not in csv_txt
# the row cut travels inside the parquet; absent -> None, never a guess
assert explorer.subset_cut_from_parquet(pq_tmp) == \
    "feh == feh and star_class in ['RGB', 'MS']"
bare = os.path.join(_tf.mkdtemp(), "bare.parquet")
sub.to_parquet(bare, index=False)
assert explorer.subset_cut_from_parquet(bare) is None
# and the UI note renders plain language, no sentinel talk
note = explorer.preselect_note()
assert "Gaia DR3 counterpart" in note and "point-like" in note, note
for jargon in ("999999", "sentinel", "masked", "1e20"):
    assert jargon not in note, (jargon, note)
meta_rt = explorer._make_meta(back, "sub@test")
assert meta_rt["n_rows"] == 5 and meta_rt["ranges"]["feh"] is not None
assert explorer.apply_cuts(
    back, [{"col": "feh", "kind": "max", "value": -2.9, "enabled": True}]
).sum() == 2   # rows at -3.0 and -3.5

# release-asset source: spec parsing + mocked-API download (no network)
spec = explorer.release_spec({"catalogs": {"release": {
    "repo": "o/r", "tag": "v1", "asset": "sub.parquet", "token": "SECRET"}}})
assert spec["assets"] == ["sub.parquet"]          # str -> list
assert explorer.release_spec({"catalogs": {"release": {"repo": "o/r"}}}) is None
assert explorer.release_spec({}) is None

class FakeResp:
    def __init__(self, status, js=None, chunks=None):
        self.status_code, self._js, self._chunks = status, js, chunks or []
    def json(self):
        return self._js
    def iter_content(self, chunk_size):
        return iter(self._chunks)

class FakeSession:
    def __init__(self):
        self.calls = []
    def get(self, url, headers=None, **kw):
        self.calls.append((url, headers or {}))
        if url.endswith("/releases/tags/v1"):
            return FakeResp(200, js={"assets": [
                {"name": "sub.parquet", "id": 77, "size": 8}]})
        if url.endswith("/releases/assets/77"):
            return FakeResp(200, chunks=[b"PARQ", b"UET!"])
        return FakeResp(404)

fs = FakeSession()
dest = os.path.join(_tf.mkdtemp(), "dl.parquet")
explorer.fetch_release_asset("o/r", "v1", "sub.parquet", "SECRET", dest,
                             session=fs)
assert open(dest, "rb").read() == b"PARQUET!"
assert fs.calls[0][1]["Authorization"] == "Bearer SECRET"
assert fs.calls[1][1]["Accept"] == "application/octet-stream"
try:
    explorer.fetch_release_asset("o/r", "v9", "sub.parquet", "SECRET",
                                 dest + "2", session=fs)
    raise SystemExit("expected a lookup failure")
except RuntimeError as e:
    assert "404" in str(e) and "SECRET" not in str(e), e
try:
    explorer.fetch_release_asset("o/r", "v1", "nope.parquet", "SECRET",
                                 dest + "3", session=fs)
    raise SystemExit("expected a missing-asset failure")
except RuntimeError as e:
    assert "nope.parquet" in str(e) and "SECRET" not in str(e), e

print("OK  explorer cuts/metrics, category split, occupancy, SIMBAD stub, "
      "selection helpers, per-user globs + path allowlist, cloud subset + "
      "release source")

# ── 3. the app renders the progress page ──
at = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=60)
at.secrets["credentials"] = {"smoketest": "pw"}
at.secrets["features"] = {"explorer": True}
at.session_state["user"] = "smoketest"
at.run()
assert not at.exception, at.exception

radio = at.sidebar.radio[0]
assert "Follow-up progress" in radio.options, \
    f"progress page not offered; options = {radio.options}"
radio.set_value("Follow-up progress").run()
assert not at.exception, at.exception

page_metrics = {m.label: m.value for m in at.metric}
assert "Unique targets proposed" in page_metrics, f"metrics rendered: {page_metrics}"
uniq = df[~df["in_earlier_run"]]
assert page_metrics["Unique targets proposed"] == f"{len(uniq):,}"
assert page_metrics["Observed"] == f"{(uniq['status'] == 'observed').sum():,}"
assert len(at.dataframe) >= 2, "per-run summary / target tables not rendered"
side = " ".join(str(c.value) for c in at.sidebar.caption)
assert "Gemini: 98" in side and "GMOS 93" in side and "GHOST 5" in side, side
print(f"OK  app 'Follow-up progress' page renders: {page_metrics}")

# ── 4. the explorer page renders on the cached real catalog (if present) ──
import glob as _glob
if explorer.find_catalogs() and _glob.glob(os.path.join(explorer.CACHE_DIR, "*.parquet")):
    at2 = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=300)
    at2.secrets["credentials"] = {"smoketest": "pw"}
    at2.secrets["features"] = {"explorer": True}
    at2.session_state["user"] = "smoketest"
    at2.run()
    radio = at2.sidebar.radio[0]
    assert "Target explorer" in radio.options, radio.options
    radio.set_value("Target explorer").run()
    assert not at2.exception, at2.exception
    em = {m.label: m.value for m in at2.metric}
    assert "Passing cuts" in em and "Observed by us" in em, em
    assert "Literature-known" in em and "In SIMBAD" in em, em
    print(f"OK  app 'Target explorer' page renders: {em}")

    # classification UI: star_class checkboxes (RGB/MS on, ambiguous off)
    # plus the per-class "Assumed" radio; no multiselect
    ck = {c.key: c.value for c in at2.checkbox}
    for cls, default in (("RGB", True), ("MS", True), ("ambiguous", False)):
        k = next(k for k in ck if k.endswith(f":cls:{cls}"))
        assert ck[k] == default, (k, ck[k])
    assumed = [r for r in at2.radio if "Assumed" in (r.label or "")]
    assert len(assumed) == 1, "expected the Assumed [Fe/H], dmod radio"
    assert list(assumed[0].options) == ["Matching star_class", "RGB", "MS"], \
        assumed[0].options
    assert not at2.multiselect, "star_class multiselect should be gone"
    # every cut has typed number boxes; depth cuts are mag_g + sigma(CaHK)
    keys = {n.key for n in at2.number_input}
    assert any(k.endswith(":feh:lo") for k in keys), keys
    assert any(k.endswith(":mag_g:box") for k in keys), keys
    assert any(k.endswith(":magerr_cahk:box") for k in keys), keys
    assert not any(":magerr_g" in k for k in keys), "old magerr_g widget back?"

    # the Fiducial preset must reproduce apply_cuts with the FIDUCIAL values
    # (derive the cache path exactly as the app does — a stale cache from an
    # older schema may coexist in CACHE_DIR, e.g. from a running app session)
    pqf, _ = explorer.cache_paths(explorer.find_catalogs()[explorer.DEFAULT_CATALOG])
    assert os.path.exists(pqf), f"cache missing for default catalog: {pqf}"
    cat = pd.read_parquet(pqf)
    # mirror the render pipeline: re-assume the fiducial class first, cut on
    # the swapped columns, and turn the pc range into a dmod range
    cat_a = explorer.assume_class(cat, explorer.FIDUCIAL["assumed"])
    fid = [{"col": "star_class", "kind": "isin",
            "value": list(explorer.FIDUCIAL["population"]), "enabled": True}]
    for col, val in explorer.FIDUCIAL.items():
        if col in ("population", "assumed"):
            continue
        if col == "dist_pc":
            fid.append({"col": "dmod", "kind": "range",
                        "value": (float(explorer.pc_to_dmod(val[0])),
                                  float(explorer.pc_to_dmod(val[1]))),
                        "enabled": True})
            continue
        if col == "broadband_valid":     # quality flags: require-valid /
            fid.append({"col": col, "kind": "range", "value": (1.0, 1.0),
                        "enabled": bool(val)})
            continue
        if col in ("gaia_var_flag", "feh_ext"):   # exclude-variable /
            fid.append({"col": col, "kind": "range",  # no-extrapolation
                        "value": (0.0, 0.0), "enabled": bool(val)})
            continue
        kind = "min" if col.startswith("sep_") else (
            "range" if isinstance(val, tuple) else "max")
        fid.append({"col": col, "kind": kind, "value": val, "enabled": True})
    fid_mask = explorer.apply_cuts(cat_a, fid)
    want = explorer.metrics(cat_a, fid_mask, lit_is_observed=True)

    # seed a stubbed SIMBAD match store (5 fiducial stars) so every panel
    # must show the "In SIMBAD" overlay after the preset is applied
    seed_pos = np.flatnonzero(fid_mask)[:5]
    ekey = os.path.basename(pqf)
    at2.session_state[f"{ekey}:simbad"] = pd.DataFrame(
        {"simbad_main_id": [f"FAKE {i}" for i in range(5)],
         "simbad_main_type": ["Star"] * 5,
         "simbad_sep_arcsec": [0.1] * 5}, index=seed_pos)
    at2.session_state[f"{ekey}:simbad:queried"] = set()
    # seed a click-selection: the third stubbed SIMBAD star
    at2.session_state[f"{ekey}:sel_rows"] = [int(seed_pos[2])]

    # default mode is 'Matching star_class' and no ambiguous-cut warning
    assert assumed[0].value == "Matching star_class", assumed[0].value
    assert not any("ambiguous" in (w.value or "").lower() for w in at2.warning)

    fbtn = next(b for b in at2.button if b.label == "Fiducial cuts")
    fbtn.click().run()
    assert not at2.exception, at2.exception
    # corrected fiducial: classes {RGB, ambiguous}, MS off, Assumed = RGB
    ck2 = {c.key: c.value for c in at2.checkbox}
    for cls, expect in (("RGB", True), ("ambiguous", True), ("MS", False)):
        k = next(k for k in ck2 if k.endswith(f":cls:{cls}"))
        assert ck2[k] == expect, (cls, ck2[k])
    a2 = [r for r in at2.radio if "Assumed" in (r.label or "")][0]
    assert a2.value == "RGB", a2.value
    # fiducial also switches both quality flags on
    ck2 = {c.key: c.value for c in at2.checkbox}
    for col in ("broadband_valid", "gaia_var_flag", "feh_ext"):
        k = next(k for k in ck2 if k.endswith(f":{col}:on"))
        assert ck2[k] is True, (col, ck2[k])
    # Assumed = RGB means no ambiguous-cut warning despite the feh cut
    assert not any("ambiguous" in (w.value or "").lower() for w in at2.warning)
    em2 = {m.label: m.value for m in at2.metric}
    assert em2["In SIMBAD"] == "5", em2
    assert em2["Passing cuts"] == f"{want['selected']:,}", (em2, want)
    assert em2["Observed by us"] == f"{want['observed_us']:,}", (em2, want)
    assert em2["Literature-known"] == f"{want['literature']:,}", (em2, want)
    assert em2["Remaining to observe"] == f"{want['remaining']:,}", (em2, want)
    print(f"OK  Fiducial preset applies correctly: {em2}")

    # the SIMBAD overlay must propagate to the dmod and e_feh panels
    def chart_spec(el):
        p = el.proto
        spec = getattr(getattr(p, "figure", p), "spec", "")
        return spec or getattr(p, "spec", "")
    charts = [chart_spec(el) for el in at2.get("plotly_chart")]
    assert len(charts) >= 3, f"expected sky/dmod/feh charts, got {len(charts)}"
    assert "In SIMBAD" in charts[1], "dmod panel lacks the SIMBAD overlay"
    assert "distance (kpc)" in charts[1], "dmod panel lacks the kpc top axis"
    assert "In SIMBAD" in charts[2], "e_feh panel lacks the SIMBAD overlay"
    # and the filtered-target table gains a sortable in_simbad column
    tab = at2.dataframe[0].value
    assert "in_simbad" in tab.columns and int(tab["in_simbad"].sum()) == 5,         tab.columns.tolist()
    assert "lvdb_host" in tab.columns and "lvdb_host_type" in tab.columns, \
        tab.columns.tolist()
    assert "obs_instrument" in tab.columns, tab.columns.tolist()
    assert "source_id" in tab.columns, tab.columns.tolist()
    # fiducial state has Assumed = RGB, so the value columns are labeled by
    # the assumption mode in the table (and hence the CSV download)
    assert "feh_rgb" in tab.columns and "e_feh_rgb" in tab.columns \
        and "dmod_rgb" in tab.columns, tab.columns.tolist()
    assert "feh" not in tab.columns and "dmod" not in tab.columns, \
        tab.columns.tolist()
    print("OK  SIMBAD overlay propagates to dmod, e_feh, and the table")

    # the seeded selection is first in the table, with a detail line above it
    assert tab.index[0] == int(seed_pos[2]), (tab.index[:3], seed_pos)
    assert bool(tab.iloc[0]["in_simbad"]) and \
        tab.iloc[0]["simbad_main_id"] == "FAKE 2", tab.iloc[0]
    details = " ".join(str(m.value) for m in at2.markdown)
    assert "SIMBAD: FAKE 2" in details, "detail line missing"
    print("OK  click-selection state puts the star first with a detail line")

    # footgun guard: Matching star_class + ambiguous checked + feh cut on
    # -> warning appears (and it names the fix)
    a2.set_value("Matching star_class").run()
    warns = [w.value for w in at2.warning if "ambiguous" in (w.value or "").lower()]
    assert warns and "Assumed" in warns[0], at2.warning
    a2 = [r for r in at2.radio if "Assumed" in (r.label or "")][0]
    a2.set_value("RGB").run()   # restore the fiducial state
    assert not any("ambiguous" in (w.value or "").lower() for w in at2.warning)
    print("OK  ambiguous-cut footgun warning in the trap state only")

    # feature flag is default-closed: without it neither workstation page
    # is offered, with it both are (data exists on this machine)
    at4 = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=300)
    at4.secrets["credentials"] = {"smoketest": "pw"}
    at4.secrets["features"] = {}          # flag absent = hidden
    at4.session_state["user"] = "smoketest"
    at4.run()
    opts = list(at4.sidebar.radio[0].options)
    assert "Target explorer" not in opts and "Follow-up progress" not in opts, opts
    both = list(at2.sidebar.radio[0].options)
    assert "Target explorer" in both and "Follow-up progress" in both, both
    print("OK  feature flag: pages hidden without it, present with it")

    # Clear all disables every cut again
    next(b for b in at2.button if b.label == "Clear all").click().run()
    em3 = {m.label: m.value for m in at2.metric}
    assert em3["Passing cuts"] != em2["Passing cuts"]
    print(f"OK  Clear all restores the unfiltered view: {em3['Passing cuts']} pass")
else:
    print("SKIP explorer page render (no catalog/cache on this machine)")
print("SMOKE TEST PASSED")
