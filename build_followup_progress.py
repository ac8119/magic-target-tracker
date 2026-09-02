#!/usr/bin/env python3
"""
Ingest the workstation targeting data into the target tracker: read every dated
target run in magic_targets/ and every MAGIC observed catalog in magic_obs/,
cross-match them, and write one flat per-target status table for the app.

Inputs (read-only, never modified):
  $MAGIC_FOLLOWUP_DIR/magic_targets/<run>/*.fits   proposed-target lists
                                                   (columns: objid_1, ra, dec, source_id, ...)
  $MAGIC_FOLLOWUP_DIR/magic_obs/magic_*.fits       MAGIC observed catalogs
  $MAGIC_FOLLOWUP_DIR/magic_obs/*_observed.fits      "       "        "
  data/master_exclusion.csv                        tracker's observed + literature ledger

MAGIC_FOLLOWUP_DIR defaults to ~/Documents/Research/magic-low-metallicity-followup.

Output: data/target_runs.csv — one row per unique target per run, with
  status = observed / literature-known / proposed
derived by positional cross-match (1.0 arcsec — the MAGIC convention, same as
scripts/make_targets.py). The app's "Follow-up progress" page reads this file.

Requires astropy (workstation only — the deployed app does not need it).
Re-run after every observing run or new target selection.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings("ignore")

FOLLOWUP_DIR = os.environ.get(
    "MAGIC_FOLLOWUP_DIR",
    os.path.expanduser("~/Documents/Research/magic-low-metallicity-followup"))
TARGETS_DIR = os.path.join(FOLLOWUP_DIR, "magic_targets")
OBS_DIR = os.path.join(FOLLOWUP_DIR, "magic_obs")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUSION_CSV = os.path.join(APP_DIR, "data", "master_exclusion.csv")
OUT = os.path.join(APP_DIR, "data", "target_runs.csv")

MATCH_RADIUS = 1.0 * u.arcsec  # MAGIC convention (see make_targets.py)

OBSERVED_CATEGORIES = ("MAGIC_Magellan", "nonMAGIC_Magellan", "GMOS")
NAME_COLS = ("name", "Name", "ID", "id", "objid", "objid_1")


def sky(df):
    return SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)


def infer_instrument(*tokens):
    """Guess the instrument from run-directory / file-name tokens."""
    s = " ".join(tokens).lower()
    for key, inst in (("mike", "MIKE"), ("mage", "MagE"), ("gmos", "GMOS")):
        if key in s:
            return inst
    return ""


def run_date_key(run):
    """Sort key for run directories: leading YYMMDD digits (e.g. tmp230914 -> 230914)."""
    m = re.search(r"(\d{6})", run)
    return m.group(1) if m else run


# ──────────────── observed + literature reference positions ────────────────
def load_reference():
    """Union of the tracker ledger and the workstation magic_obs/ catalogs.

    Returns a DataFrame (ra, dec, category, detail) where category decides the
    status: OBSERVED_CATEGORIES -> observed, 'Literature' -> literature-known.
    """
    ref = pd.read_csv(EXCLUSION_CSV)[["ra", "dec", "category", "detail"]]
    print(f"Tracker ledger: {len(ref)} positions "
          f"({sum(ref['category'].isin(OBSERVED_CATEGORIES))} observed, "
          f"{sum(ref['category'] == 'Literature')} literature)")

    # Workstation MAGIC observed catalogs (magic_*.fits, *_observed.fits).
    # SAGA_* / JINAbase / Roederer files are skipped: the ledger's Literature
    # rows were built from those same catalogs.
    local = sorted(glob.glob(os.path.join(OBS_DIR, "magic_*.fits")) +
                   glob.glob(os.path.join(OBS_DIR, "*_observed.fits")))
    rows = []
    for f in local:
        stem = os.path.splitext(os.path.basename(f))[0]
        t = Table.read(f)
        for r in t:
            rows.append([float(r["ra"]), float(r["dec"]), "MAGIC_Magellan", stem])
        print(f"  magic_obs/{os.path.basename(f)}: {len(t)} observed")
    localdf = pd.DataFrame(rows, columns=["ra", "dec", "category", "detail"])

    # Keep only local positions not already in the ledger (1" match).
    if len(localdf):
        _, sep, _ = sky(localdf).match_to_catalog_sky(sky(ref))
        new = localdf[sep > MATCH_RADIUS]
        print(f"magic_obs catalogs: {len(localdf)} positions, "
              f"{len(new)} not already in the ledger")
        ref = pd.concat([ref, new], ignore_index=True)
    return ref


# ──────────────────────── proposed target runs ────────────────────────
def load_run(run_dir):
    """All unique target positions proposed in one dated run directory.

    Every readable FITS table with ra/dec columns counts (base selections,
    *_new subsets, session lists, proposal candidate lists); duplicates within
    the run are removed at 1", keeping the first file (alphabetical) that
    lists the star.
    """
    run = os.path.basename(run_dir.rstrip("/"))
    frames = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.fits"))):
        fname = os.path.basename(f)
        try:
            t = Table.read(f)
        except Exception as e:
            print(f"  ! skipping unreadable {run}/{fname}: {e}")
            continue
        lower = {c.lower(): c for c in t.colnames}
        if "ra" not in lower or "dec" not in lower:
            print(f"  ! skipping {run}/{fname}: no ra/dec columns")
            continue
        namecol = next((lower[c.lower()] for c in NAME_COLS if c.lower() in lower), None)
        df = pd.DataFrame({
            "ra": np.asarray(t[lower["ra"]], dtype=float),
            "dec": np.asarray(t[lower["dec"]], dtype=float),
            "name": [str(v).strip() for v in t[namecol]] if namecol else "",
            "gaia_source_id": (np.asarray(t[lower["source_id"]]).astype(str)
                               if "source_id" in lower else ""),
            "source_file": fname,
            "instrument": infer_instrument(run, fname),
        })
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["ra", "dec"], ignore_index=True)
    # remove remaining near-duplicates (same star from files with slightly
    # different astrometry) at the match radius, keeping the first occurrence
    coords = sky(df)
    i1, i2, _, _ = coords.search_around_sky(coords, MATCH_RADIUS)
    keep = np.ones(len(df), dtype=bool)
    for a, b in zip(i1, i2):
        if a < b and keep[a]:
            keep[b] = False
    df = df[keep].reset_index(drop=True)
    df.insert(0, "run", run)
    return df


def main():
    ref = load_reference()
    obs_ref = ref[ref["category"].isin(OBSERVED_CATEGORIES)].reset_index(drop=True)
    lit_ref = ref[ref["category"] == "Literature"].reset_index(drop=True)
    obs_coords, lit_coords = sky(obs_ref), sky(lit_ref)

    run_dirs = sorted((d for d in glob.glob(os.path.join(TARGETS_DIR, "*"))
                       if os.path.isdir(d)),
                      key=lambda d: run_date_key(os.path.basename(d)))
    print(f"\n{'run':<38}{'targets':>8}{'observed':>9}{'lit':>6}{'proposed':>9}")
    all_runs, seen = [], None
    for run_dir in run_dirs:
        df = load_run(run_dir)
        if df is None:
            print(f"  ! no target tables in {os.path.basename(run_dir)}")
            continue
        coords = sky(df)

        # status: observed beats literature-known beats proposed (1" match)
        oi, osep, _ = coords.match_to_catalog_sky(obs_coords)
        li, lsep, _ = coords.match_to_catalog_sky(lit_coords)
        observed, lit = osep < MATCH_RADIUS, lsep < MATCH_RADIUS
        df["status"] = np.where(observed, "observed",
                                np.where(lit, "literature-known", "proposed"))
        df["match_detail"] = np.where(
            observed, obs_ref["detail"].values[oi],
            np.where(lit, lit_ref["detail"].values[li], ""))
        df["sep_arcsec"] = np.where(
            observed, osep.arcsec.round(2),
            np.where(lit, lsep.arcsec.round(2), np.nan))

        # was this star already proposed in an earlier run?
        if seen is not None and len(seen):
            _, psep, _ = coords.match_to_catalog_sky(sky(seen))
            df["in_earlier_run"] = psep < MATCH_RADIUS
        else:
            df["in_earlier_run"] = False
        seen = pd.concat([seen, df[["ra", "dec"]]], ignore_index=True)

        n = len(df)
        print(f"{df['run'].iloc[0]:<38}{n:>8}{int(observed.sum()):>9}"
              f"{int((~observed & lit).sum()):>6}"
              f"{int((df['status'] == 'proposed').sum()):>9}")
        all_runs.append(df)

    out = pd.concat(all_runs, ignore_index=True)
    out.to_csv(OUT, index=False)

    uniq = out[~out["in_earlier_run"]]
    print(f"\nTotals: {len(out)} run-targets across {len(all_runs)} runs; "
          f"{len(uniq)} unique stars")
    for s in ("observed", "literature-known", "proposed"):
        print(f"  unique {s}: {(uniq['status'] == s).sum()}")
    print(f"\nWrote {OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
