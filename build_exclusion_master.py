#!/usr/bin/env python3
"""
Merge all "already observed" ledgers into one flat table for the target tracker app.

Sources:
  1. MagE_MIKE_observations/master_observed.fits   (MAGIC Magellan runs)
  2. non_MAGIC_observations/master_observed.fits   (non-MAGIC Magellan)
  3. GMOS_observed/master_GMOS_observed.csv        (Gemini GMOS programs)
  4. lit_obs/master_lit_obs.fits                   (SAGA + Roederer24 + JINAbase)
  5. target_tracker/data/ledger_additions_*.csv    (workstation-side additions in
     the full ledger schema — e.g. Gemini exports absorbed away from the laptop)

Output: target_tracker/data/master_exclusion.csv
Columns: name, ra, dec, category, detail, instrument
Category vocabulary: MAGIC_Magellan / nonMAGIC_Magellan / Gemini / Literature
(the instrument column carries the within-category split, e.g. MagE/MIKE for
MAGIC_Magellan and GMOS/GHOST for Gemini).
Re-run this script whenever any source ledger is updated; the additions files
are ingested last with a 1" same-category+program de-dup, so rows already
absorbed into a source ledger drop out automatically.
"""
import csv
import glob
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "target_tracker", "data")
OUT = os.path.join(DATA_DIR, "master_exclusion.csv")


def merge_additions(rows, data_dir, radius_arcsec=1.0):
    """Ingest every data/ledger_additions_*.csv (full ledger schema) into
    `rows`, skipping additions that already exist as the same category AND
    detail (program/run) within radius_arcsec. Position alone is deliberately
    NOT a duplicate: multi-instrument coverage of one star (Magellan + Gemini
    rows at the same coordinates) is legitimate and must survive. Returns the
    number of rows added."""
    files = sorted(glob.glob(os.path.join(data_dir, "ledger_additions_*.csv")))
    if not files:
        return 0

    by_key = {}
    for r in rows:
        by_key.setdefault((r[3], r[4]), []).append((float(r[1]), float(r[2])))

    def is_new(ra, dec, coords):
        if not coords:
            return True
        arr = np.radians(np.asarray(coords, float))
        ra1, dec1 = np.radians(ra), np.radians(dec)
        h = (np.sin((arr[:, 1] - dec1) / 2) ** 2
             + np.cos(dec1) * np.cos(arr[:, 1]) * np.sin((arr[:, 0] - ra1) / 2) ** 2)
        sep = np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0
        return float(sep.min()) >= radius_arcsec

    n_added = 0
    for path in files:
        n_file, n_dup = 0, 0
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["category"].strip(), r["detail"].strip())
                ra, dec = float(r["ra"]), float(r["dec"])
                if not is_new(ra, dec, by_key.get(key, [])):
                    n_dup += 1
                    continue
                rows.append([r["name"].strip(), ra, dec, key[0], key[1],
                             r["instrument"].strip()])
                by_key.setdefault(key, []).append((ra, dec))
                n_file += 1
        n_added += n_file
        print(f"additions {os.path.basename(path)}: +{n_file}"
              + (f" ({n_dup} already in a source ledger)" if n_dup else ""))
    return n_added


def main():
    from astropy.table import Table

    rows = []

    t = Table.read(os.path.join(BASE, "MagE_MIKE_observations", "master_observed.fits"))
    for r in t:
        rows.append([str(r['name']).strip(), float(r['ra']), float(r['dec']),
                     "MAGIC_Magellan", str(r['run']).strip(), str(r['instrument']).strip()])
    print(f"MAGIC Magellan: {len(t)}")

    t = Table.read(os.path.join(BASE, "non_MAGIC_observations", "master_observed.fits"))
    for r in t:
        rows.append([str(r['name']).strip(), float(r['ra']), float(r['dec']),
                     "nonMAGIC_Magellan", str(r['run']).strip(), str(r['instrument']).strip()])
    print(f"non-MAGIC Magellan: {len(t)}")

    with open(os.path.join(BASE, "GMOS_observed", "master_GMOS_observed.csv")) as f:
        n = 0
        for r in csv.DictReader(f):
            rows.append([r['object'].strip(), float(r['ra']), float(r['dec']),
                         "Gemini", r['program_id'].strip(), "GMOS"])
            n += 1
    print(f"Gemini (GMOS source ledger): {n}")

    t = Table.read(os.path.join(BASE, "lit_obs", "master_lit_obs.fits"))
    for r in t:
        rows.append(["", float(r['ra']), float(r['dec']),
                     "Literature", str(r['source']).strip(), ""])
    print(f"Literature: {len(t)}")

    n_add = merge_additions(rows, DATA_DIR)
    print(f"workstation additions: {n_add}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ra", "dec", "category", "detail", "instrument"])
        w.writerows(rows)
    print(f"\nWrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
