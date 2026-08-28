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
    "obs_cat":  ["MAGIC_Magellan", "", "", "GMOS", "", ""],
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
    "ra":       [10.0, 10.0, 20.0],
    "dec":      [-1.0, -1.0, -2.0],
    "category": ["MAGIC_Magellan", "Literature", "Literature"]})
oc, lk = explorer.classify_against_ledger(
    np.array([10.0, 20.0, 30.0]), np.array([-1.0, -2.0, -3.0]), ledger)
assert oc.tolist() == ["MAGIC_Magellan", "", ""], oc.tolist()
assert lk.tolist() == [True, True, False], lk.tolist()

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
assert "SIMBAD: HD 1" in detail3 and "GMOS" in detail3, detail3
print("OK  explorer cuts/metrics, category split, occupancy, SIMBAD stub, "
      "selection helpers")

# ── 3. the app renders the progress page ──
at = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=60)
at.secrets["credentials"] = {"smoketest": "pw"}
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
print(f"OK  app 'Follow-up progress' page renders: {page_metrics}")

# ── 4. the explorer page renders on the cached real catalog (if present) ──
import glob as _glob
if explorer.find_catalogs() and _glob.glob(os.path.join(explorer.CACHE_DIR, "*.parquet")):
    at2 = AppTest.from_file(os.path.join(BASE, "app.py"), default_timeout=300)
    at2.secrets["credentials"] = {"smoketest": "pw"}
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

    # exactly one population control (radio), no star_class multiselect
    pop_radios = [r for r in at2.radio if "Population" in (r.label or "")]
    assert len(pop_radios) == 1, "expected a single population radio"
    assert list(pop_radios[0].options) == ["RGB", "MS", "both",
                                           "include ambiguous"], pop_radios[0].options
    assert not at2.multiselect, "star_class multiselect should be gone"
    # every cut has typed number boxes; the depth cut is now mag_psf_g
    keys = {n.key for n in at2.number_input}
    assert any(k.endswith(":feh:lo") for k in keys), keys
    assert any(k.endswith(":mag_g:box") for k in keys), keys
    assert not any("magerr" in k for k in keys), "magerr widgets should be gone"

    # the Fiducial preset must reproduce apply_cuts with the FIDUCIAL values
    pqf = sorted(_glob.glob(os.path.join(explorer.CACHE_DIR, "*.parquet")))[0]
    cat = pd.read_parquet(pqf)
    fid = [{"col": "star_class", "kind": "isin",
            "value": [explorer.FIDUCIAL["population"]], "enabled": True}]
    for col, val in explorer.FIDUCIAL.items():
        if col == "population":
            continue
        kind = "min" if col.startswith("sep_") else (
            "range" if isinstance(val, tuple) else "max")
        fid.append({"col": col, "kind": kind, "value": val, "enabled": True})
    fid_mask = explorer.apply_cuts(cat, fid)
    want = explorer.metrics(cat, fid_mask, lit_is_observed=True)

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

    fbtn = next(b for b in at2.button if b.label == "Fiducial cuts")
    fbtn.click().run()
    assert not at2.exception, at2.exception
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
    assert "In SIMBAD" in charts[2], "e_feh panel lacks the SIMBAD overlay"
    # and the filtered-target table gains a sortable in_simbad column
    tab = at2.dataframe[0].value
    assert "in_simbad" in tab.columns and int(tab["in_simbad"].sum()) == 5,         tab.columns.tolist()
    print("OK  SIMBAD overlay propagates to dmod, e_feh, and the table")

    # the seeded selection is first in the table, with a detail line above it
    assert tab.index[0] == int(seed_pos[2]), (tab.index[:3], seed_pos)
    assert bool(tab.iloc[0]["in_simbad"]) and \
        tab.iloc[0]["simbad_main_id"] == "FAKE 2", tab.iloc[0]
    details = " ".join(str(m.value) for m in at2.markdown)
    assert "SIMBAD: FAKE 2" in details, "detail line missing"
    print("OK  click-selection state puts the star first with a detail line")

    # Clear all disables every cut again
    next(b for b in at2.button if b.label == "Clear all").click().run()
    em3 = {m.label: m.value for m in at2.metric}
    assert em3["Passing cuts"] != em2["Passing cuts"]
    print(f"OK  Clear all restores the unfiltered view: {em3['Passing cuts']} pass")
else:
    print("SKIP explorer page render (no catalog/cache on this machine)")
print("SMOKE TEST PASSED")
