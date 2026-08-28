#!/usr/bin/env python3
"""
Build a compressed explorer-schema Parquet subset of a MAGIC catalog, sized
for a Streamlit Community Cloud deploy (host it as a private GitHub release
asset; see README_deploy.md for the end-to-end recipe).

    python3 build_cloud_subset.py                          # default catalog + cut
    python3 build_cloud_subset.py --catalog NAME_OR_PATH \
        --cut "feh == feh and star_class in ['RGB', 'MS'] and feh < -1.0" \
        --out magic_cloud_subset.parquet

The subset is cut from the explorer's Parquet cache (built first if missing),
so it inherits every derived column: Galactic l/b, LMC/SMC separations, and
the ledger obs_cat / lit_known flags. Only the explorer's schema columns are
kept. Upload with, e.g.:
    gh release create v1 magic_cloud_subset.parquet --repo you/magic-data
Aim to keep the printed in-RAM estimate comfortably under ~1 GB.
"""
import argparse
import os
import sys

import pandas as pd

import explorer

DEFAULT_CUT = "feh == feh and star_class in ['RGB', 'MS']"


def make_subset(df, cut=DEFAULT_CUT):
    """Apply a pandas query cut and keep only the explorer schema columns."""
    sub = df.query(cut) if cut and cut.strip() else df
    cols = [c for c in explorer.SCHEMA_COLUMNS if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=explorer.DEFAULT_CATALOG,
                    help="catalog basename (from the configured globs) or a "
                         "FITS path (default: %(default)s)")
    ap.add_argument("--cut", default=DEFAULT_CUT,
                    help="pandas query expression (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output Parquet (default: <catalog>_cloud_subset.parquet)")
    a = ap.parse_args()

    path = (a.catalog if os.path.exists(a.catalog)
            else explorer.find_catalogs().get(a.catalog))
    if not path:
        sys.exit(f"catalog '{a.catalog}' not found on the configured search paths")
    pq, _ = explorer.cache_paths(path)
    if not os.path.exists(pq):
        print(f"building the explorer cache for {os.path.basename(path)} first ...")
        pq, _ = explorer.build_cache(
            path, progress=lambda f: print(f"  {f:5.0%}", end="\r"))
        print()
    df = pd.read_parquet(pq)
    sub = make_subset(df, a.cut)
    out = a.out or (os.path.splitext(os.path.basename(path))[0]
                    + "_cloud_subset.parquet")
    sub.to_parquet(out, compression="zstd", index=False)

    mem = int(sub.memory_usage(deep=True).sum())
    print(f"cut:   {a.cut}")
    print(f"rows:  {len(sub):,} of {len(df):,}")
    print(f"file:  {out}  ({os.path.getsize(out) / 1e6:.0f} MB, zstd)")
    print(f"RAM:   ~{mem / 1e6:.0f} MB loaded"
          + ("  [OK for the ~1 GB Streamlit Cloud budget]" if mem < 8e8 else ""))
    if mem >= 8e8:
        print("WARNING: near/over the ~1 GB Streamlit Cloud budget — tighten "
              "the cut, e.g. append: and feh < -1.0")


if __name__ == "__main__":
    main()
