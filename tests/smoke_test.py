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
    "observed": [True, False, False, True, False, False],
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
m = explorer.metrics(syn, mask)
assert m == {"selected": 3, "observed": 2, "remaining": 1}, m
# disabled cuts must filter nothing
for c in cuts:
    c["enabled"] = False
assert explorer.apply_cuts(syn, cuts).all()
print("OK  explorer apply_cuts/metrics on synthetic frame")

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
    assert "Passing cuts" in em and "Already observed" in em, em
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
    want = explorer.metrics(cat, explorer.apply_cuts(cat, fid))

    fbtn = next(b for b in at2.button if b.label == "Fiducial cuts")
    fbtn.click().run()
    assert not at2.exception, at2.exception
    em2 = {m.label: m.value for m in at2.metric}
    assert em2["Passing cuts"] == f"{want['selected']:,}", (em2, want)
    assert em2["Already observed"] == f"{want['observed']:,}", (em2, want)
    print(f"OK  Fiducial preset applies correctly: {em2}")

    # Clear all disables every cut again
    next(b for b in at2.button if b.label == "Clear all").click().run()
    em3 = {m.label: m.value for m in at2.metric}
    assert em3["Passing cuts"] != em2["Passing cuts"]
    print(f"OK  Clear all restores the unfiltered view: {em3['Passing cuts']} pass")
else:
    print("SKIP explorer page render (no catalog/cache on this machine)")
print("SMOKE TEST PASSED")
