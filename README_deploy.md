# MAGIC Target Tracker — run & deploy

A password-protected Streamlit app that lets anyone on the team:
1. **Check targets** (single coords or an uploaded CSV) against every star already
   observed — MAGIC Magellan runs, non-MAGIC Magellan, GMOS programs, and the
   literature (SAGA / Roederer+24 / JINAbase) — ~20,000 positions, 2" cone match.
2. **Add targets to a shared queue**, with automatic duplicate blocking at entry.
3. **Browse/download** the observed-star database.

## Files
- `app.py` — the app
- `build_exclusion_master.py` — regenerates `data/master_exclusion.csv` from the
  four source ledgers. **Re-run after every observing run / GMOS export update.**
- `data/master_exclusion.csv` — merged observed table the app reads
- `build_followup_progress.py` — workstation ingestion: reads the dated target
  runs in `magic_targets/` and the observed catalogs in `magic_obs/` (from
  `$MAGIC_FOLLOWUP_DIR`, default `~/Documents/Research/magic-low-metallicity-followup`),
  cross-matches at 1", and writes `data/target_runs.csv` for the
  **Follow-up progress** page. Re-run after every observing run / target selection.
- `data/target_runs.csv` — per-run target status table (git-ignored: the repo is
  public and this holds unpublished proposed-target coordinates)
- `tests/smoke_test.py` — end-to-end check of the ingestion + progress page
- `secrets.toml.example` — template for logins + optional Google Sheet queue

## Follow-up progress (workstation)
```bash
python3 build_followup_progress.py   # needs astropy; ~30 s
python3 tests/smoke_test.py          # optional sanity check
streamlit run app.py                 # new sidebar page: "Follow-up progress"
```
The page appears only when `data/target_runs.csv` exists, so the cloud deploy
is unaffected until you decide to commit that file.

## Run locally
```bash
cd target_tracker
streamlit run app.py        # login: ani / changeme (edit .streamlit/secrets.toml)
```
Queue entries go to `data/queue.csv` when no Google Sheet is configured.

## Deploy (free, ~15 min)
1. Put this `target_tracker/` folder in a GitHub repo — private, or public if
   you're OK with the target coordinates in `data/master_exclusion.csv` being
   world-readable (the app login does NOT protect files in a public repo)
   (only `app.py`, `requirements.txt`, `data/master_exclusion.csv` are needed;
   `.gitignore` already excludes secrets).
2. Go to https://share.streamlit.io -> "Create app" -> pick the repo,
   main file `app.py`.
3. In the app's **Settings -> Secrets**, paste your `[credentials]` block
   (see `secrets.toml.example`) — one username/password per collaborator.
4. **Set up the Google Sheet queue** (strongly recommended: without it, queue
   entries are lost when the app restarts). Follow the `[gsheets]` comments in
   `secrets.toml.example`; run `python3 json_key_to_toml.py your-key.json SHEET_KEY`
   to generate the block, then paste it into the app secrets too.
5. Send collaborators the app URL + their password. That's it — the URL is
   public but everything is behind the login.

## Updating the observed database
After a new observing run or GMOS OT export:
```bash
python3 target_tracker/build_exclusion_master.py
git commit -am "update observed db" && git push   # Streamlit Cloud auto-redeploys
```
