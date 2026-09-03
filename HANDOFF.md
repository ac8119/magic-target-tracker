# Handoff: new-catalog ingestion for the target explorer

Written 2026-09-03. Task list for a workstation agent picking this up.

Everything below was worked out against a **laptop copy** of the catalog:
`MAGIC_allcat_mpflags_lite_v260810.fits`, 59,904,346 rows, 53 columns, at
`~/Documents/MIT_Work/Research/magic_scratch/magic_pipeline/mpflags/output/`.
The workstation has its **own, possibly different** version — the pipeline ran
there first. Every row count here is a *reference value to check against*, not
a target to assume. Step 0 exists for that reason.

Parquet caches are keyed on `getmtime(catalog)` + `getmtime(master_exclusion.csv)`
(`explorer.cache_paths`), so the workstation builds its own cache regardless of
whether the data is byte-identical. Nothing is transferable; don't try to copy
caches between machines.

---

## 0. Verify the local catalog first

Before anything else, establish what the workstation's file actually contains:

```python
from astropy.io import fits
with fits.open(CATALOG, memmap=True) as h:
    print(h[1].header["NAXIS2"], len(h[1].columns.names))
    print(h[1].columns.names)
```

Report: row count, column count, and specifically whether **`star_class`**
exists. That single answer decides step 2.

---

## 1. Wire `PRESELECT` into `build_cache` — BUG, do this first

`explorer.PRESELECT` (a list of `(column, rule, predicate)` triples) and
`preselect_note()` exist, and the target-selection page renders an expander
titled *"Pre-selection already applied to this catalog"* from them.

**But `build_cache` performs no row filtering at all.** The page currently
claims cuts were applied that were not. Fix by applying every `PRESELECT`
predicate in `build_cache` before the schema frame is assembled, then confirm
the surviving count matches step 3's expectation.

The four rules, as confirmed with the catalog owner:

| rule | detail |
|---|---|
| has a Gaia `source_id` | sentinel for no-match is **999999** (not 0, not NaN) |
| `extended_class_g in (0, 1)` | drops galaxy-like 2/3 **and** the `-9` no-measurement value |
| `0 < mag_psf_cahk < 30` | upper bound rejects the ~1e20 sentinel and >90 placeholders; lower bound rejects unphysical negative mags |
| `ebv_sfd98 <= 0.2` | SFD98 reddening |

Note the CaHK column is **100% finite** — "missing" is encoded as the 1e20
sentinel, so a NaN check alone catches nothing. The catalog README claims these
sentinels were cleaned to NaN; in this file they were not.

## 2. Decide `star_class` — BLOCKING

The laptop's catalog has **no `star_class`**, and no `chi2_rgb`/`chi2_ms`/
`rgb_excluded`/`ms_excluded` either. It carries only `is_rgb` (bool) plus the
two per-branch families (`fehs_rgb`/`fehs_errs_rgb`/`fehs_ext_rgb`/`loggs_rgb`/
`dmod_rgb` and the `_ms` equivalents).

`is_rgb` is stale — it should have been superseded by `star_class`. Measured
facts, so nobody re-derives them:

- `is_rgb` agrees with the branch that `feh`/`dmod` adopted in **1,500,006 of
  1,500,006** sampled rows (zero disagreements). The staleness therefore
  propagates into the adopted `feh`/`e_feh`/`dmod` — the columns the explorer
  actually cuts on. There is no fresher signal in the file to recover.
- `is_rgb` is True for 49,858,991 rows (83%), and ~23 M of those have no `feh`
  at all, so it is not usable as an RGB label on its own.
- Whenever `feh` is NaN, **both** branch values are NaN. There is no
  "ambiguous" state where a branch exists but was not adopted.
- `dmod` is never NaN, even where `feh` is.

Pick one:

- **(a) If the workstation catalog has `star_class`** — use it, nothing to
  decide. Confirm its distinct values before wiring it up.
- **(b) Re-run the classification stage** so the file carries `star_class`.
  Cleanest; blocks the build until done.
- **(c) Reconstruct** `RGB` = `is_rgb & isfinite(feh)`, `MS` =
  `~is_rgb & isfinite(feh)`, `no-feh` = `~isfinite(feh)`. Explicitly **not**
  the old four-way RGB/MS/ambiguous/invalid.

Whichever is chosen, `explorer.py` line ~371 currently does
`np.where(is_rgb, "RGB", "MS")`, which mislabels every no-`feh` star as one or
the other and yields no third category. It must change.

## 3. Update `CANDS`, then build the cache

`explorer.CANDS` maps canonical name → candidate FITS columns (first match
wins; a missing column is materialized as all-NaN and recorded in `missing`).
Differences to handle in the new catalog:

- CaHK magnitude error is **`MAGERR_PSF`** (not `magerr_psf_cahk`)
- no `star_class` (see step 2)
- `broadband_valid` / `gaia_var_flag` are present and are **live
  discriminators** here (65% and 278,363 respectively), unlike the old pruned
  catalog where both were already applied upstream. They are deliberately
  **not** in `PRESELECT` — they are UI checkboxes under "Quality flags".

Then build. Reference counts from the laptop copy, in order:

| step | remaining |
|---|---|
| all rows | 59,904,346 |
| + has Gaia `source_id` | 30,760,030 |
| + `extended_class_g in (0,1)` | 28,784,735 |
| + `0 < mag_psf_cahk < 30` | 28,434,061 |
| + `ebv_sfd98 <= 0.2` | **26,148,419** (43.65%) |

Of those survivors: `is_rgb` True 17,135,080 / False 9,013,339; finite `feh`
24,264,129; `broadband_valid` 21,913,614; `gaia_var_flag` 255,294; and
1,884,290 have both branch metallicities NaN.

Sanity check that carried over correctly: finite `feh` across the *whole*
catalog is 35,597,361, which matches the catalog README exactly.

## 4. Build and upload the cloud subset

Cut, chosen because it does not depend on the untrustworthy classification —
a star is kept if **either** branch is metal-poor:

```
(fehs_rgb < -2) | (fehs_ms < -2)
```

Applied to the pre-selected sample this gives **4,496,769 rows → 0.29 GB on
disk (zstd), 0.49 GB in RAM** with the full 28-column schema. That fits
Streamlit Cloud's ~1 GB budget with room to spare, so **no column pruning is
needed** and the per-class columns stay — meaning RGB/MS re-assumption works
on the cloud too.

Why the OR rather than a cut on the adopted `feh`: it keeps 1,564,162 stars a
`feh < -2` cut would discard (1,943,587 qualify only via the RGB branch,
526,617 only via MS).

For scale, if a different threshold is wanted (rows / disk / RAM at 28 cols):

| threshold | rows | disk | RAM |
|---|---|---|---|
| −1.5 | 7,706,162 | 0.49 GB | 0.85 GB |
| **−2.0** | **4,496,769** | **0.29 GB** | **0.49 GB** |
| −2.5 | 2,621,170 | 0.17 GB | 0.29 GB |
| −3.0 | 1,231,173 | 0.08 GB | 0.14 GB |

Do **not** try to ship the full 26.1 M pre-selected sample: 2.88 GB RAM at 28
columns, and still 1.52 GB even pruned to 15 columns. Row count is the binding
constraint, not schema width.

```bash
python3 build_cloud_subset.py --catalog <CATALOG>.fits \
    --cut "(fehs_rgb < -2) | (fehs_ms < -2)" \
    --out <name>_cloud_subset.parquet
gh release upload v1 <name>_cloud_subset.parquet --repo ac8119/magic-target-data
```

Then add the asset to `[catalogs.release]` in Streamlit Cloud → Settings →
Secrets. `asset` accepts a list, and each entry becomes its own
`name@tag` entry in the catalog dropdown, so old and new can coexist.

Caveat: `build_cloud_subset.make_subset` currently passes `--cut` to
`DataFrame.query`, and its `DEFAULT_CUT` uses `and`/`in` syntax. Verify the
`|` form above parses under `query` (it should — `query` accepts `|`), or
switch to a boolean mask.

## 5. Delete the orphaned caches

`SCHEMA_VERSION` went 4 → 5, so pre-existing caches are unreachable by name.
On the laptop that is 986 MB across two files in `data/explorer_cache/`.
Check for the equivalent on the workstation and delete.

## 6. Commit the pending work

Uncommitted in the working tree:

- `explorer.py` — LVDB `r_h` proximity flag, distance-in-pc slider,
  `star_class` checkboxes + "Assumed [Fe/H], dmod values" selector, quality
  flags section, `PRESELECT` + page note
- `data/lvdb/dwarf_mw.csv` — **required**; without it `load_lvdb()` returns no
  dwarfs and the dwarf overlay and the `r_h` flag are silently empty, locally
  and on the cloud

## 7. Rotate the exposed GitHub PAT

The `[catalogs.release]` token for the private `ac8119/magic-target-data` repo
was pasted into a chat transcript and must be considered compromised. Revoke
at github.com/settings/tokens, issue a new one, and update **both** the
Streamlit Cloud Secrets box and the local `.streamlit/secrets.toml`. The
release-fetch path itself is verified working (HTTP 200, asset name matches),
so a fresh token drops straight in.

---

## Context worth not rediscovering

- **`has_companion_2arcsec` is unusable** in this catalog — `True` for 100% of
  rows. The catalog README documents the cause (companion self-match done on
  sky coordinates but joined back by non-unique `objid`) and says
  `flag_companions.py` was fixed on 2026-08-10, so it should be correct the
  next time the mpflags stage runs.
- **`feh_ext` is carried and swaps with the assumed class, but nothing reads
  it yet** — no filter, no table column. It is also vanishingly rare:
  sampling 600 k rows, `fehs_ext_rgb` was nonzero 6 times and `fehs_ext_ms` 9
  times.
- **In the old catalog, `rgb_excluded` was exactly `chi2_rgb > 9`** and
  `ms_excluded` exactly `chi2_ms > 9` — 100.0000% agreement over 640 k rows, a
  3σ fit-quality rejection, *not* "the other branch fits better" (that reading
  only agrees 84%). Those columns do not exist in the new catalog; recorded
  only so the question isn't reopened.
- **LVDB `rhalf` is the major-axis radius.** The proximity flag circularizes
  it as `r_h * sqrt(1 - ellipticity)`, which reproduces LVDB's own
  `rhalf_sph_physical` to 0.000 arcmin across all 65 dwarfs. Sagittarius goes
  342' → 205'. The Clouds are excluded from the flag via `LVDB_FLAG_EXCLUDE`
  since they have their own `sep_lmc`/`sep_smc` excision cuts. Default
  aperture is 10 r_h, on-sky only — no distance term, deliberately, because
  the catalog distances are not reliable enough.
- **`resolve_globs` treats `globs = []` as unset** and falls through to
  hardcoded laptop paths in `LOCAL_FALLBACK_GLOBS`. To restrict the dropdown
  to release assets only, set a deliberately non-matching glob rather than an
  empty list. Not fixed.
