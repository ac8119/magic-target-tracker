"""
MAGIC Target Tracker — shared observed-star database + GMOS/MagE queue.

Run locally:   streamlit run target_tracker/app.py
Deploy:        Streamlit Community Cloud (see README_deploy.md)

Login credentials live in Streamlit secrets ([credentials] table), never in code.
Queue persistence: Google Sheet if [gsheets] secrets are configured (recommended
for cloud deployment, where the local filesystem is wiped on restart),
otherwise a local CSV next to this file.
"""
import hmac
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUSION_CSV = os.path.join(APP_DIR, "data", "master_exclusion.csv")
TARGET_RUNS_CSV = os.path.join(APP_DIR, "data", "target_runs.csv")
QUEUE_CSV = os.path.join(APP_DIR, "data", "queue.csv")
QUEUE_COLS = ["name", "ra", "dec", "instrument", "priority", "notes",
              "added_by", "added_utc", "status"]
DEFAULT_RADIUS_ARCSEC = 2.0

st.set_page_config(page_title="MAGIC Target Tracker", page_icon="🔭", layout="wide")


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ──────────────────────────── auth ────────────────────────────
def check_login():
    """Password gate. Users/passwords come from st.secrets['credentials']."""
    if st.session_state.get("user"):
        return True

    try:
        creds = dict(st.secrets["credentials"])
    except (KeyError, FileNotFoundError):
        st.error("No [credentials] section found in Streamlit secrets. "
                 "See README_deploy.md for setup.")
        st.stop()

    st.title("🔭 MAGIC Target Tracker")
    with st.form("login"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Log in")
    if ok:
        if user in creds and hmac.compare_digest(str(creds[user]), pw):
            st.session_state["user"] = user
            _rerun()
        else:
            st.error("Unknown username or wrong password.")
    return False


# ──────────────────────── coordinate utils ────────────────────────
def parse_coord(text):
    """Parse 'ra dec' in decimal degrees or HMS/DMS (colon- or space-separated).
    Returns (ra_deg, dec_deg) or raises ValueError."""
    s = text.strip().replace(",", " ")
    # sexagesimal with colons: 12:34:56.7 -12:34:56
    m = re.match(r"^(\d{1,2}):(\d{1,2}):([\d.]+)\s+([+-]?\d{1,3}):(\d{1,2}):([\d.]+)$", s)
    if m:
        h, mi, se, d, dm, ds = m.groups()
        ra = (int(h) + int(mi) / 60 + float(se) / 3600) * 15.0
        sign = -1.0 if d.strip().startswith("-") else 1.0
        dec = sign * (abs(int(d)) + int(dm) / 60 + float(ds) / 3600)
        return ra, dec
    parts = s.split()
    if len(parts) == 2:  # decimal degrees
        return float(parts[0]), float(parts[1])
    if len(parts) == 6:  # space-separated sexagesimal
        return parse_coord("{}:{}:{} {}:{}:{}".format(*parts))
    raise ValueError(f"Could not parse coordinates: '{text}'")


def angsep_arcsec(ra1, dec1, ra2, dec2):
    """Vectorized angular separation (arcsec). ra2/dec2 may be arrays."""
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(np.asarray(ra2, dtype=float)), np.radians(np.asarray(dec2, dtype=float))
    sd = np.sin((dec2 - dec1) / 2) ** 2
    sr = np.sin((ra2 - ra1) / 2) ** 2
    a = sd + np.cos(dec1) * np.cos(dec2) * sr
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * 3600.0


# ──────────────────────────── data ────────────────────────────
@st.cache_data
def load_exclusion():
    df = pd.read_csv(EXCLUSION_CSV)
    df["name"] = df["name"].fillna("")
    return df


@st.cache_data
def load_progress():
    """Per-run target status table written by build_followup_progress.py,
    or None if the ingestion has not been run on this machine."""
    if not os.path.exists(TARGET_RUNS_CSV):
        return None
    df = pd.read_csv(TARGET_RUNS_CSV)
    df["name"] = df["name"].fillna("")
    return df


# ──────────────────────────── queue backend ────────────────────────────
def _gsheet():
    """Return the gspread worksheet if [gsheets] secrets are configured, else None."""
    try:
        cfg = st.secrets["gsheets"]
    except (KeyError, FileNotFoundError):
        return None
    import gspread
    gc = gspread.service_account_from_dict(dict(cfg["service_account"]))
    return gc.open_by_key(cfg["sheet_key"]).sheet1


def load_queue():
    ws = _gsheet()
    if ws is not None:
        rows = ws.get_all_records()
        df = pd.DataFrame(rows, columns=QUEUE_COLS) if rows else pd.DataFrame(columns=QUEUE_COLS)
    elif os.path.exists(QUEUE_CSV):
        df = pd.read_csv(QUEUE_CSV)
    else:
        df = pd.DataFrame(columns=QUEUE_COLS)
    for c in ("ra", "dec"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def append_queue_row(row):
    ws = _gsheet()
    if ws is not None:
        if not ws.get_all_values():
            ws.append_row(QUEUE_COLS)
        ws.append_row([row[c] for c in QUEUE_COLS])
    else:
        df = load_queue()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        os.makedirs(os.path.dirname(QUEUE_CSV), exist_ok=True)
        df.to_csv(QUEUE_CSV, index=False)


# ──────────────────────────── matching ────────────────────────────
def match_one(ra, dec, excl, queue, radius):
    """Return (exclusion matches df, queue matches df) within radius arcsec."""
    sep = angsep_arcsec(ra, dec, excl["ra"].values, excl["dec"].values)
    em = excl[sep < radius].copy()
    em["sep_arcsec"] = np.round(sep[sep < radius], 2)
    qm = pd.DataFrame()
    if len(queue):
        qsep = angsep_arcsec(ra, dec, queue["ra"].values, queue["dec"].values)
        qm = queue[qsep < radius].copy()
        qm["sep_arcsec"] = np.round(qsep[qsep < radius], 2)
    return em.sort_values("sep_arcsec"), qm


# ──────────────────────────── pages ────────────────────────────
def page_check(excl, queue):
    st.header("Check targets against everything already observed")
    st.markdown(
        "Paste coordinates below **or** upload a CSV. Each target is cross-matched "
        "against **{:,} observed positions** (MAGIC Magellan runs, non-MAGIC Magellan, "
        "GMOS programs, and the literature: SAGA / Roederer+24 / JINAbase) plus the "
        "current queue.".format(len(excl)))
    radius = st.number_input("Match radius (arcsec)", 0.5, 30.0, DEFAULT_RADIUS_ARCSEC, 0.5)

    tab1, tab2 = st.tabs(["Single target", "Upload a list (CSV)"])

    with tab1:
        txt = st.text_input(
            "Coordinates — decimal degrees or sexagesimal",
            placeholder="e.g.  152.113 -1.614   or   10:08:27.1 -01:36:50")
        if txt:
            try:
                ra, dec = parse_coord(txt)
            except ValueError as e:
                st.error(str(e))
                return
            em, qm = match_one(ra, dec, excl, queue, radius)
            st.caption(f"Parsed as RA = {ra:.5f}°, Dec = {dec:.5f}°")
            if len(em):
                st.error(f"⛔ ALREADY OBSERVED — {len(em)} match(es) within {radius}\"")
                st.dataframe(em[["name", "ra", "dec", "category", "detail",
                                 "instrument", "sep_arcsec"]])
            if len(qm):
                st.warning(f"⚠️ Already in the queue ({len(qm)} match(es))")
                st.dataframe(qm[["name", "ra", "dec", "instrument", "added_by",
                                 "status", "sep_arcsec"]])
            if not len(em) and not len(qm):
                st.success(f"✅ CLEAR — no observed star or queue entry within {radius}\"")

    with tab2:
        st.markdown("CSV must have `ra` and `dec` columns in **decimal degrees** "
                    "(a `name` column is optional).")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                df.columns = [c.strip().lower() for c in df.columns]
                assert "ra" in df.columns and "dec" in df.columns
            except Exception:
                st.error("Could not read CSV, or it lacks 'ra'/'dec' columns.")
                return
            out = []
            for _, r in df.iterrows():
                em, qm = match_one(float(r["ra"]), float(r["dec"]), excl, queue, radius)
                if len(em):
                    verdict = "OBSERVED"
                    top = em.iloc[0]
                    info = f"{top['category']}/{top['detail']} ({top['sep_arcsec']}\")"
                elif len(qm):
                    verdict, info = "IN QUEUE", f"added by {qm.iloc[0]['added_by']}"
                else:
                    verdict, info = "CLEAR", ""
                out.append({"name": r.get("name", ""), "ra": r["ra"], "dec": r["dec"],
                            "verdict": verdict, "match": info})
            res = pd.DataFrame(out)
            n_bad = int((res["verdict"] != "CLEAR").sum())
            (st.success if n_bad == 0 else st.warning)(
                f"{len(res)} targets checked — {n_bad} already observed or queued, "
                f"{len(res) - n_bad} clear.")
            st.dataframe(res)
            st.download_button("Download results CSV",
                               res.to_csv(index=False).encode(),
                               "check_results.csv", "text/csv")


def page_queue(excl, queue):
    st.header("Observation queue")
    st.markdown("Targets proposed for upcoming GMOS / MagE / MIKE observations. "
                "Everyone sees the same list; duplicates are flagged at entry.")

    with st.expander("➕ Add a target", expanded=len(queue) == 0):
        with st.form("addq", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input("Target name *")
            coord = c2.text_input("Coordinates * (decimal or sexagesimal)")
            c3, c4 = st.columns(2)
            inst = c3.selectbox("Instrument", ["GMOS", "MagE", "MIKE", "other"])
            prio = c4.selectbox("Priority", ["normal", "high", "low"])
            notes = st.text_input("Notes (selection source, [Fe/H], mag, ...)")
            force = st.checkbox("Add even if flagged as duplicate")
            sub = st.form_submit_button("Check & add")
        if sub:
            if not name.strip() or not coord.strip():
                st.error("Name and coordinates are required.")
            else:
                try:
                    ra, dec = parse_coord(coord)
                except ValueError as e:
                    st.error(str(e))
                    return
                em, qm = match_one(ra, dec, excl, queue, DEFAULT_RADIUS_ARCSEC)
                if (len(em) or len(qm)) and not force:
                    st.error("⛔ Not added — this position matches an existing entry. "
                             "Tick the override box if this is intentional.")
                    if len(em):
                        st.dataframe(em[["name", "category", "detail", "sep_arcsec"]])
                    if len(qm):
                        st.dataframe(qm[["name", "added_by", "status", "sep_arcsec"]])
                else:
                    append_queue_row({
                        "name": name.strip(), "ra": round(ra, 6), "dec": round(dec, 6),
                        "instrument": inst, "priority": prio, "notes": notes.strip(),
                        "added_by": st.session_state["user"],
                        "added_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "status": "queued"})
                    st.success(f"Added {name.strip()} to the queue.")
                    _rerun()

    if len(queue):
        st.dataframe(queue, use_container_width=True)
        st.download_button("Download queue CSV", queue.to_csv(index=False).encode(),
                           "queue.csv", "text/csv")
    else:
        st.info("Queue is empty.")


def page_progress(prog):
    st.header("Low-metallicity follow-up progress")
    st.markdown(
        "Every target proposed in the dated `magic_targets/` runs, cross-matched "
        "(1\" — the MAGIC convention) against the observed database and the "
        "literature. Regenerate with `python3 build_followup_progress.py` after "
        "each observing run or new selection.")

    uniq = prog[~prog["in_earlier_run"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique targets proposed", f"{len(uniq):,}")
    c2.metric("Observed", f"{(uniq['status'] == 'observed').sum():,}")
    c3.metric("Literature-known", f"{(uniq['status'] == 'literature-known').sum():,}")
    c4.metric("Still to observe", f"{(uniq['status'] == 'proposed').sum():,}")

    st.subheader("Per run")
    summ = (prog.groupby("run", sort=False)["status"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["observed", "literature-known", "proposed"], fill_value=0))
    summ.insert(0, "targets", summ.sum(axis=1))
    summ["% observed"] = (100 * summ["observed"] / summ["targets"]).round(1)
    st.dataframe(summ, use_container_width=True)

    st.subheader("Targets")
    c1, c2, c3 = st.columns(3)
    runs = c1.multiselect("Run", list(prog["run"].unique()))
    stats = c2.multiselect("Status", ["observed", "literature-known", "proposed"])
    search = c3.text_input("Name contains")
    view = prog
    if runs:
        view = view[view["run"].isin(runs)]
    if stats:
        view = view[view["status"].isin(stats)]
    if search:
        view = view[view["name"].str.contains(search, case=False, na=False)]
    st.caption(f"{len(view):,} rows")
    st.dataframe(view, use_container_width=True)
    st.download_button("Download CSV", view.to_csv(index=False).encode(),
                       "followup_progress.csv", "text/csv")


def page_browse(excl):
    st.header("Browse the observed-star database")
    c1, c2 = st.columns(2)
    cats = c1.multiselect("Category", sorted(excl["category"].unique()),
                          default=["MAGIC_Magellan", "nonMAGIC_Magellan", "GMOS"])
    search = c2.text_input("Name contains")
    view = excl[excl["category"].isin(cats)] if cats else excl
    if search:
        view = view[view["name"].str.contains(search, case=False, na=False)]
    st.caption(f"{len(view):,} rows")
    st.dataframe(view, use_container_width=True)
    st.download_button("Download CSV", view.to_csv(index=False).encode(),
                       "observed_selection.csv", "text/csv")


# ──────────────────────────── main ────────────────────────────
if check_login():
    excl = load_exclusion()
    queue = load_queue()
    prog = load_progress()
    st.sidebar.title("🔭 MAGIC Target Tracker")
    st.sidebar.markdown(f"Logged in as **{st.session_state['user']}**")
    pages = ["Check targets", "Queue", "Browse observed"]
    if prog is not None:
        pages.append("Follow-up progress")
    page = st.sidebar.radio("Page", pages)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Observed database: {:,} positions\n\n"
        "MAGIC Magellan: {}\n\n"
        "non-MAGIC Magellan: {}\n\n"
        "GMOS: {}\n\n"
        "Literature: {:,}".format(
            len(excl),
            (excl["category"] == "MAGIC_Magellan").sum(),
            (excl["category"] == "nonMAGIC_Magellan").sum(),
            (excl["category"] == "GMOS").sum(),
            (excl["category"] == "Literature").sum()))
    if page == "Check targets":
        page_check(excl, queue)
    elif page == "Queue":
        page_queue(excl, queue)
    elif page == "Follow-up progress":
        page_progress(prog)
    else:
        page_browse(excl)
