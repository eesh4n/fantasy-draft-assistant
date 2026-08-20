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

This was re-verified on 2026-08-20 (well into 2026 preseason -- the 2025
season, Sept 2025 through Feb 2026, is fully complete), specifically
*because* "2025 isn't published yet" was the original reason for using
2024 and that reason needed to actually be re-checked rather than assumed
to still be true a year later. It's still true: `import_seasonal_data([2025])`
still raises `HTTP Error 404: Not Found`. This isn't a stale-cache or
timing fluke -- `import_seasonal_data` reads
`player_stats_{year}.parquet` directly from the `player_stats` release at
`github.com/nflverse/nflverse-data`, and hitting that release's asset
list directly
(`https://api.github.com/repos/nflverse/nflverse-data/releases/tags/player_stats`)
confirms there is no `player_stats_2025.*` asset published there at all
-- the file list tops out at `*_2024.*`. `import_weekly_data([2025])`
was also tried directly as a second check and 404s the same way. So "last
season" in this pipeline is still **2024**, the latest season with a
complete seasonal file -- kept because it's still genuinely the latest
available, not because it was left unchecked. **Re-run this same check
before next season's draft** (`import_seasonal_data([2025])`, or list the
release assets at the URL above) -- don't assume last year's "not
published yet" still holds.

Player name/position/team come from
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

Within QB and within RB/WR/TE, a handful of per-game rate stats are
z-scored (within position) and combined into `value_score`. The two
groups use **different components**, not just different weights on the
same numbers, because QBs don't receive and because RB/WR/TE touchdown
equity and receiving work needed to become visible on their own instead
of being averaged into one "volume" number (the old model's biggest
blind spot -- it couldn't tell a receiving back from a between-the-tackles
runner with the same total touches, or a dual-threat QB from a pure
pocket passer with the same pass-attempt volume).

**RB/WR/TE:**
| component | weight | what it is |
|---|---|---|
| `ppg` | 0.30 | total PPR fantasy points/game (overall anchor) |
| `rushing_ppg` | 0.15 | rushing-only fantasy pts/game (0.1/yd + 6/TD) |
| `receiving_ppg` | 0.20 | receiving-only fantasy pts/game (1/rec + 0.1/yd + 6/TD) |
| `td_rate` | 0.10 | (rush TD + rec TD) / (carries + targets) |
| `snap_share` | 0.15 | average offensive snap % |
| `efficiency` | 0.10 | PPR points per opportunity (carries+targets) |

`receiving_ppg` is weighted above `rushing_ppg` on purpose -- this is a
**PPR league**, so a unit of receiving work (which already includes the
reception bonus in how it's calculated) is worth more than the same unit
of rushing work, and a pass-catching back or slot receiver shouldn't get
buried under a pure runner with similar total touches. This is the
concrete fix for the "receiving back undervalued" problem: e.g. in the
2024 data, Alvin Kamara's `receiving_ppg` (9.6) actually exceeds his
`rushing_ppg` (9.4) -- the old single "volume" number couldn't show that
split at all, and his overall touch count alone undersold how much of
his value came from the more-valuable receiving side.

`td_rate` is a **touchdown-rate proxy, not a real red-zone stat**.
nfl_data_py's seasonal data has no red-zone-specific carries/targets
column (no "red zone touches" or "goal-to-go carries" field exists in
`import_seasonal_data`'s output) -- so this uses TDs scored per *total*
opportunity as a rough stand-in for goal-line role. That's a real
limitation: a back who scores efficiently because of a great offensive
line or soft schedule will show the same `td_rate` as one who scores
because he specifically gets goal-line work. Don't read `td_rate` as
literal red-zone usage -- it's the closest available proxy, deliberately
labeled as such rather than presented as something more precise than it
is.

**QB:**
| component | weight | what it is |
|---|---|---|
| `ppg` | 0.45 | total PPR fantasy points/game |
| `volume` | 0.15 | pass attempts/game |
| `rushing_ppg` | 0.15 | rushing-only fantasy pts/game (0.1/yd + 6/TD) |
| `snap_share` | 0.15 | average offensive snap % |
| `efficiency` | 0.10 | PPR points per opportunity (attempts+carries) |

`rushing_ppg` is new for QB and is the direct fix for the model
previously missing rushing production entirely: two QBs with identical
passing lines are not equal fantasy assets if one of them also runs for
6+ points/game. In the 2024 data this is visible directly -- e.g. Lamar
Jackson and Josh Allen both carry real `rushing_ppg` (6.8 and 7.8
respectively) that a passing-only model would have completely missed.
`rushing_ppg` is legitimately close to zero for most pocket passers (e.g.
Patrick Mahomes: 2.7) -- that's a real, low value reflecting their actual
role, not a data gap, so unlike other components it is never a candidate
for the missing-data redistribution below (a QB position group is never
going to be >70% null on this column, since every QB has *some*
carries/rushing_yards value, even if it's small).

If a component is missing (non-null for less than 30% of a position
group -- i.e. missing for >70% of it), its weight is redistributed
proportionally across the remaining components (keeps weights summing to
1.0 without ever silently zeroing a group out). In practice this only
ever fires for `snap_share`, if the snap-count pull fails.

`position_rank` = rank by `value_score` within position (1 = best).
`adp_position_rank` = rank by ADP within position (1 = earliest drafted).
`value_gap = adp_position_rank - position_rank`. Positive = player
outproduces where ADP has them within their position (sleeper/undervalued
by the market); negative = ADP is pricier than production supports
(overpriced/fade).

## Output

`players.json` -- 519 players across QB/RB/WR/TE/K/DEF, matching the UI
schema exactly (see `build_json.py` docstring).
