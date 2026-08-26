# Fantasy Draft Assistant

A live fantasy football draft tool: mark players Drafted/Mine as your draft
happens, get dynamically-updating recommendations as the pool shrinks, and
track your roster against your league's slot requirements. Vanilla
JS/HTML/CSS front end, no build step, backed by a data pipeline that blends
real analyst research, real 2025 season stats, and a trained ML model into
a single per-player value score.

## Quick start (just run the app)

Everything needed to run the app is already committed -- no pipeline rerun
required.

```
pip install -r requirements.txt
cd ui
python serve.py 8765
```

Then open `http://localhost:8765` in a browser.

## League settings (hardcoded in `ui/app.js`)

- PPR scoring, custom passing/rushing/receiving/kicking/defense weights
  (see `data/README.md` for the exact scoring table).
- Roster: 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX (RB/WR/TE), 1 K, 1 DEF, 5 BENCH.
- League size is configurable in the app (defaults to 12).

## Project structure

```
data/    -- the scoring pipeline (see data/README.md for full detail)
  pull.py, join.py, score.py, build_json.py   -- pipeline stages, run in order
  train_ml_model.py, retrain_with_real2025.py -- ML model training
  models/                                     -- trained per-position models
  guide_*.csv, real2025_*.csv                 -- hand-collected analyst/real-2025 data
  players.json                                -- pipeline output (source of truth)
ui/      -- the app itself
  index.html, app.js, style.css               -- the front end
  serve.py                                    -- static file server (adds no-cache headers)
  players.json                                -- copy of data/players.json served to the app
```

## Rebuilding the data pipeline

Only needed if you want to regenerate `players.json` from scratch (new
season, new guide data, retrained models). Not required just to run the
app -- see Quick start above.

```
cd data
python pull.py
python build_defense_stats.py
python join.py
python score.py
python build_json.py
cp players.json ../ui/players.json
```

Full detail on data sources, known limitations, and the scoring model is
in `data/README.md`.

## Notes for anyone cloning this

- No secrets/API keys are committed -- safe to clone and run as-is.
- The real-2025 stats (`data/real2025_*.csv`) and hand-transcribed guide
  data (`data/guide_*.csv`) were collected manually (browser scraping +
  PDF transcription), not by an automated script -- refreshing those for a
  future season is manual work, not a command to rerun.
- `data/README.md` documents one open assumption worth knowing about: the
  21-27 points-allowed defensive scoring tier was never confirmed against
  the actual league rules and currently defaults to 0.
