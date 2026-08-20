# Fantasy Draft Assistant -- Data Pipeline

Pipeline: `pull.py` -> `join.py` -> `score.py` -> `build_json.py` -> `players.json`

Run it end to end:

```
pip install nfl_data_py rapidfuzz requests pandas numpy
python pull.py
python join.py
python score.py
python build_json.py
```

## Data sources actually used (and why)

**Seasonal stats** -- `nfl_data_py.import_seasonal_data([2024])`.
`import_seasonal_data([2025])` currently 404s: the nflverse hosted release
that `nfl_data_py` reads from hasn't published a complete 2025 seasonal
rollup yet, even though we're now into the 2026 preseason. So "last
season" in this pipeline is **2024**, the latest season with a complete
seasonal file. Player name/position/team come from
`import_seasonal_rosters([2024])` (seasonal data alone is stat columns
keyed by `player_id` only, no name/team). Snap share comes from
`import_snap_counts([2024])`, averaged per player across the season and
joined in via `pfr_id`.

Important gap: `import_seasonal_data` only covers offensive skill
positions with tracked play-by-play stats (QB/RB/WR/TE). It has **no
rows for kickers or team defense/special teams units** -- those aren't
part of that dataset at all. So `raw_stats.csv` has zero K or DEF rows.

**ADP / consensus rankings** -- FantasyPros' actual CSV export endpoint
(`...ppr-cheatsheets.php?export=xls`) requires a logged-in session; hit
anonymously it just serves back the rankings page's HTML instead of a
CSV. Rather than substitute a lower-quality source, `pull.py` parses the
same data FantasyPros' own page uses to render its table: the page embeds
a `var ecrData = {...}` JSON blob with the full consensus rankings
(519 players covering QB/RB/WR/TE/K/DST for 2026 PPR drafts), served with
no auth. This gives per-player: average expert rank (used as `adp`),
overall ECR rank, position rank, bye week, and rough ownership %. It's
the same underlying data a CSV export would contain, for PPR draft
rankings specifically. Source URL:
`https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php`

## Handling positions with no 2024 stats

Two groups end up in `joined.csv` without a real production signal:

1. **K and DEF** -- no per-player stats exist in `nfl_data_py` at all for
   these positions. `score.py` falls back to a pure ADP-based
   `value_score` (z-score of ADP, inverted) for both groups. `value_gap`
   for K/DEF will therefore always be close to 0 -- there's no
   independent signal to disagree with ADP, so don't read anything into
   K/DEF value_gap.
2. **2025 draft rookies and other players with no 2024 stat line**
   (e.g. Ashton Jeanty, Omarion Hampton, Tetairoa McMillan) -- these are
   real, highly-drafted players this year but have nothing to measure
   against. They're kept in the dataset (ADP-only) rather than dropped,
   but are deliberately scored *below every stat-based player in their
   position group*, which mechanically makes them look "overpriced" in
   `value_gap`. **This is a modeling artifact, not a real signal** -- do
   not use a rookie's negative value_gap as a "fade this rookie" talking
   point. The real sleeper/overpriced signal is only meaningful for
   players who have both a 2024 stat line and a 2026 ADP.

Similarly, a handful of established players carry an ADP but list
`team: FA` (currently unsigned free agents as of this pull, e.g. Austin
Ekeler, Joe Mixon at pull time) -- their `value_gap` looks huge because
they produced well in 2024 but are barely being drafted now. That's
mechanically correct given the inputs but not a useful "sleeper" --
they don't have a starting job. For draft-relevant sleeper/overpriced
calls, filter to players on a real team (`team != "FA"`).

## Fuzzy name matching (join.py)

Names are normalized (lowercased, periods/apostrophes/hyphens stripped,
Jr./Sr./II/III/IV/V suffixes dropped) and matched exactly first. Leftovers
are fuzzy-matched within the same position group using `rapidfuzz`, with
a deliberately strict rule: the **last name token must match exactly**,
and the **first name token must be a strong partial match** (catches
nicknames/truncation like Scotty/Scott Miller, Joshua/Josh Palmer,
Kenny/Kenneth Gainwell, Gabe/Gabriel Davis). A naive whole-name similarity
score was tried first and produced a false positive (matched rookie
"Kevin Coleman" to established WR "Keon Coleman" -- different players,
same surname) -- the stricter first/last split fixes that while still
catching real nickname variants.

## Value model (score.py)

Within each of QB/RB/WR/TE, four stats are z-scored (within position) and
combined:
- fantasy points/game (PPR) -- weight 0.5
- volume/game: carries+targets for RB/WR/TE, attempts for QB -- weight 0.2
- average offensive snap share -- weight 0.2
- efficiency: PPR points per opportunity -- weight 0.1

If a component is missing for >70% of a position group, its weight is
redistributed proportionally across the remaining components (keeps
weights summing to 1.0 without ever silently zeroing a group out).

`position_rank` = rank by `value_score` within position (1 = best).
`adp_position_rank` = rank by ADP within position (1 = earliest drafted).
`value_gap = adp_position_rank - position_rank`. Positive = player
outproduces where ADP has them within their position (sleeper/undervalued
by the market); negative = ADP is pricier than production supports
(overpriced/fade).

## Output

`players.json` -- 519 players across QB/RB/WR/TE/K/DEF, matching the UI
schema exactly (see `build_json.py` docstring).
