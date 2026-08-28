#!/usr/bin/env python3
"""
End-to-end smoke test for the workstation ingestion.

Run after build_followup_progress.py:
    python3 tests/smoke_test.py

Checks that data/target_runs.csv is well-formed and that the app's
"Follow-up progress" page actually renders it (via streamlit.testing).
"""
import os
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ── 2. the app renders the progress page ──
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

metrics = {m.label: m.value for m in at.metric}
assert "Unique targets proposed" in metrics, f"metrics rendered: {metrics}"
uniq = df[~df["in_earlier_run"]]
assert metrics["Unique targets proposed"] == f"{len(uniq):,}"
assert metrics["Observed"] == f"{(uniq['status'] == 'observed').sum():,}"
assert len(at.dataframe) >= 2, "per-run summary / target tables not rendered"
print(f"OK  app 'Follow-up progress' page renders: {metrics}")
print("SMOKE TEST PASSED")
