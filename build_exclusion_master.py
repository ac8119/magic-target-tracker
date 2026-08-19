#!/usr/bin/env python3
"""
Merge all "already observed" ledgers into one flat table for the target tracker app.

Sources:
  1. MagE_MIKE_observations/master_observed.fits   (MAGIC Magellan runs)
  2. non_MAGIC_observations/master_observed.fits   (non-MAGIC Magellan)
  3. GMOS_observed/master_GMOS_observed.csv        (Gemini GMOS programs)
  4. lit_obs/master_lit_obs.fits                   (SAGA + Roederer24 + JINAbase)

Output: target_tracker/data/master_exclusion.csv
Columns: name, ra, dec, category, detail, instrument
Re-run this script whenever any source ledger is updated.
"""
import os
import csv
import numpy as np
from astropy.table import Table

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "target_tracker", "data", "master_exclusion.csv")

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
                     "GMOS", r['program_id'].strip(), "GMOS"])
        n += 1
print(f"GMOS: {n}")

t = Table.read(os.path.join(BASE, "lit_obs", "master_lit_obs.fits"))
for r in t:
    rows.append(["", float(r['ra']), float(r['dec']),
                 "Literature", str(r['source']).strip(), ""])
print(f"Literature: {len(t)}")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "ra", "dec", "category", "detail", "instrument"])
    w.writerows(rows)
print(f"\nWrote {OUT}: {len(rows)} rows")
