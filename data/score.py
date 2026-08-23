"""
score.py -- Compute a within-position value model from joined.csv.

All points (ppg, rushing_ppg, receiving_ppg, efficiency) are computed from
this league's EXACT custom scoring rules, not nfl_data_py's generic
fantasy_points_ppr column:
  Passing:   0.04 pt/yard, 4 pt/TD, -1 per interception
  Rushing:   0.1 pt/yard, 6 pt/TD
  Receiving: 1 pt/reception (PPR), 0.1 pt/yard, 6 pt/TD
  (Two-point conversions are part of the league's rules but not reflected
  here -- no such column exists in nfl_data_py's seasonal data, and they're
  rare enough not to meaningfully move rankings. See README.)

For QB/RB/WR/TE, value_score is now PRIMARILY an ML-model-derived signal,
not a hand-weighted formula -- see "ML value model" section below for the
full design. The composite mechanism (z-score each available component,
weight, sum, with missing-data redistribution) is unchanged from before;
what changed is WHICH components feed it: the cluster of raw per-game
production stats that used to be individually weighted (ppg, rushing_ppg,
receiving_ppg, red_zone_share, snap_share, efficiency, and for QB, volume
instead of receiving_ppg/red_zone_share) is now replaced by a single
ml_predicted_ppg component -- the output of a real, cross-validated
scikit-learn model (trained in train_ml_model.py on 7 seasons of
historical nfl_data_py data, 2018-2024) -- carrying the SAME combined
weight those raw components used to hold. The guide_*/playcaller_*
components below are unchanged EXCEPT where noted in the "ROUND 2" section
immediately below, for the reason explained in "ML value model": most
can't be part of the trained model (no historical equivalent exists), but
they're real analyst signal, so they stay in as a secondary blend on top
of the model's own prediction.

-------------------------------------------------------------------------
ROUND 2 UPDATE -- iterative model improvement + honest blend-weight fitting
-------------------------------------------------------------------------
Responding to explicit pushback that a single training pass wasn't "max
accuracy" and that the guide-derived signals should be taken into account
IN the model wherever honestly possible, not just hand-blended:

  1. train_ml_model.py now runs a real iterative search per position:
     wider Ridge/ElasticNet grids, + Lasso, + a small regularized
     RandomForest, + a refit-per-fold averaging ensemble of the best
     linear + best tree model, PLUS a feature-set search (see that file's
     module docstring "ROUND 2" section for the full method and the
     "don't switch away from the simpler choice unless it clears a real
     margin" discipline applied twice -- once for feature-set selection,
     once for model-class selection). Before/after CV R^2 per position:
       QB: 0.3116 -> 0.3116 (base feature set + ridge win again -- no
           change found that clears the improvement margin; reported
           honestly, not forced).
       RB: 0.5081 -> 0.5282 (+0.0201) -- adopted `age` + two genuine
           historical-analog features, team_rb_ppg_hist and
           team_ol_rb_ybc_att (see point 2), with Lasso replacing Ridge.
       WR: 0.5207 -> 0.5409 (+0.0202) -- adopted `age` (Lasso replacing
           ElasticNet); the team-context historical analog was tested but
           did NOT clear the margin for WR (see point 2) and was not
           adopted.
       TE: 0.4911 -> 0.4911 (unchanged to 4 decimals; Lasso and
           ElasticNet landed on essentially the same fit -- no real
           change).
     Exact numbers, every candidate tried, and the full grid are recorded
     in data/models/{pos}_metadata.json's "feature_set_variants_tried" and
     "candidates" keys.

  2. Historical analogs for the guide-derived RB/WR blend signals: checked
     whether nfl_data_py's OWN historical data has a genuine equivalent to
     each guide_*/playcaller_* signal (not fabricated -- only adopted where
     real). Two were found and tested as REAL TRAINED FEATURES (not blend
     terms):
       - team_rb_ppg_hist / team_wr_ppg_hist: this SAME team's own
         historical average fantasy PPG at the position (computed
         identically for every 2018-2024 season from this file's own `ppg`
         column) -- a genuine historical analog to
         playcaller_rb_ppg_rank / playcaller_wr_ppg_rank (the guide's
         forward-looking "how much does this playcaller's system feed this
         position" framing measures the same underlying thing).
       - team_ol_rb_ybc_att: team-level rushing yards-before-contact per
         carry (PFR data via nfl_data_py's import_seasonal_pfr) -- a
         standard, honest historical proxy for O-line push, analogous to
         guide_ol_run_block_rank_2025.
     RESULT: for RB, both cleared the improvement margin and were adopted
     into the model -- so ol_run_block_rank and playcaller_rb_ppg_rank_inv
     are REMOVED from RB's hand-picked blend below (their old weight is
     folded into ml_predicted_ppg's weight instead, since the model itself
     now captures that signal -- see the RB components dict for the exact
     math). For WR, team_wr_ppg_hist was tested but provided no
     measurable improvement over `age` alone, so it was NOT adopted, and
     playcaller_wr_ppg_rank_inv stays exactly as it was.
     NO historical analog was found (or attempted) for guide_adj_ppg
     (one analyst's current-season subjective adjustment),
     pct_pts_lost_to_luck (a current-season-only variance audit -- "this
     specific analyst's luck metric obviously has no historical
     equivalent"), guide_proj_volume_rank (a purely forward-looking
     projection with nothing to validate against in past seasons), or
     pct_rb1_rank (would need a full historical backfield-committee-share
     reconstruction from play-by-play -- out of scope this round, an
     honest limitation). These all stay hand-picked post-prediction blend
     terms, EXCEPT guide_adj_ppg's WEIGHT specifically -- see point 3.

  3. guide_adj_ppg's blend weight is now EMPIRICALLY FIT, not hand-picked
     (data/fit_blend_weight.py). Method: for the subset of QB/RB/WR
     players who have BOTH ml_predicted_ppg (from their 2024 profile) AND
     guide_adj_ppg (the analyst's pre-season adjusted estimate), grid-
     search the alpha in [0,1] blending z-scored ml_predicted_ppg and
     guide_adj_ppg that maximizes Spearman rank correlation against REAL
     KNOWN 2025 outcomes (real2025_total_pts, actual nfl.com season
     totals -- genuinely known, not fabricated, and not circular: both
     inputs are pre-season information, the target is what actually
     happened). Result, honestly reported even though it's a big swing:
     guide_adj_ppg alone predicts real 2025 outcomes MUCH better than
     ml_predicted_ppg alone for all three positions (QB: 0.686 vs 0.143
     Spearman; RB: 0.644 vs 0.420; WR: 0.718 vs 0.342), so the fitted
     blend puts the large majority of weight on guide_adj_ppg -- see each
     position's components dict below for the exact fitted numbers.
     IMPORTANT HONEST CAVEAT: this comparison is not fully apples-to-apples.
     guide_adj_ppg is the analyst's OWN estimate of the exact 2025 season
     being validated against (a same-season nowcast, likely informed by
     real depth-chart/injury news) -- of course it tracks that season's
     actual outcome closely. ml_predicted_ppg is a genuine year-ahead
     forecast built purely from 2024 profile data, a fundamentally harder
     task, and its historical CV R^2 (0.31-0.54, all clearing the naive
     "last year's ppg" baseline by a wide margin) shows it has real skill
     at THAT task. So the fitted weight is the honest answer to "how
     should these two specific signals be blended for predicting the
     season right after 2024," not proof ml_predicted_ppg is weak in
     general. TE was explicitly NOT refit this way -- only n=11 TEs have
     all three needed columns, too small to trust a fit (see
     fit_blend_weight.py's MIN_N_TO_TRUST=20) -- TE keeps its original
     hand-picked guide_adj_ppg weight, documented as such below.

  ML value model (replaces the old hand-weighted RB/WR/TE and QB
  production clusters):
        train_ml_model.py pulls nfl_data_py seasonal + play-by-play data
        for 2018-2024 (7 seasons -> 6 consecutive season-pairs) and, for
        each position, builds a supervised (season Y features) ->
        (season Y+1 actual ppg) training table -- a genuine forecasting
        setup matching the real draft-ranking use case (use THIS season's
        profile to predict NEXT season's output). Features are computed by
        directly reusing compute_stat_group_scores() (this same function,
        below) against each historical season's raw stats, so the
        historical training features and this file's live inference
        features are guaranteed to be computed by the same code path.
        Per position, Ridge, ElasticNet, and a shallow/regularized
        GradientBoostingRegressor are compared via k-fold cross-validation
        against each other AND against a naive "last year's ppg predicts
        next year's ppg" baseline; the best-generalizing model (tree only
        wins if it beats the best linear model's CV R^2 by a real margin,
        given the small per-position n) is refit on the full training set
        and saved to data/models/{position}_model.joblib, with feature
        order/hyperparameters/CV performance recorded in
        data/models/{position}_metadata.json. See train_ml_model.py's
        module docstring for the complete methodology, leak-free
        season-pair construction, and minimum-games filters.

        This file (score.py) loads that saved model+metadata (see
        load_ml_model() below) and applies it to the CURRENT season's
        (2024 base) features -- the exact same feature columns, in the
        exact same order the model was trained on -- producing
        ml_predicted_ppg per player. This becomes the PRIMARY signal in
        value_score, replacing the hand-picked weights that used to sit on
        ppg/rushing_ppg/receiving_ppg/red_zone_share/snap_share/efficiency
        (RB/WR/TE) or ppg/volume/rushing_ppg/snap_share/efficiency (QB)
        directly. The replacement weight for ml_predicted_ppg is exactly
        the SUM of the weights those raw components used to hold in the
        pre-ML composite (see each position's components dict below for
        the exact numbers) -- i.e. "the model's own prediction now stands
        in for the whole raw-production cluster it was trained to
        summarize," nothing more, nothing redistributed elsewhere.
        ml_predicted_ppg, along with the raw feature columns that feed it,
        is also surfaced in stats.ml_predicted_ppg (build_json.py) for
        transparency/detail views.
        Players who never appear in a saved model's feature set as
        non-null (e.g. missing snap_pct) get their missing feature
        median-imputed before prediction, matching this file's existing
        median-impute convention for every other optional component.
        If a position's model file is missing entirely (train_ml_model.py
        not yet run), ml_predicted_ppg falls back to the player's own raw
        ppg (a safe, documented degenerate case -- see load_ml_model()).

  RB/WR/TE components:
        (ml_predicted_ppg weight below is the SUM of the old ppg .30 +
        rushing_ppg .15 + receiving_ppg .20 + red_zone_share .10 +
        snap_share .15 + efficiency .10 weights = 1.00 for the ORIGINAL
        pre-guide-data composite; see the guide-adjusted math further down
        for the actual current weights after guide_adj_ppg etc. were
        folded in -- the RB/WR/TE-specific components dicts in the code
        below show the final numbers actually used.)
        ml_predicted_ppg -- the model's own next-season PPG prediction,
                                 trained on ppg, rushing_ppg, receiving_ppg,
                                 red_zone_share, snap_share, efficiency
                                 (see "ML value model" above). Replaces all
                                 six of those as individually-weighted
                                 composite inputs; each remains available
                                 individually in stats.* for detail views.
        red_zone_share (feature, not composite weight) -- REAL red-zone
                                 usage: (red-zone rush attempts + red-zone
                                 targets) / (total carries + targets),
                                 computed from 2024 play-by-play data
                                 (yardline_100 <= 20) by redzone.py and
                                 joined in by join.py via player_id
                                 (gsis_id). Per an analyst's claim that
                                 red-zone touches are one of the two most
                                 predictive stats for RB touchdown
                                 production (~65% of RB TDs), this
                                 replaces the old td_rate proxy, which only
                                 measured TDs-per-opportunity over ALL
                                 opportunities (not red-zone-specific)
                                 because nfl_data_py's seasonal aggregate
                                 data has no red-zone columns. See
                                 redzone.py's docstring for the exact
                                 red-zone/goal-line definitions used and
                                 the goal_line_share stat (not part of the
                                 model's feature set, but surfaced in
                                 stats.goal_line_share for detail views).

  QB components:
        ml_predicted_ppg -- the model's own next-season PPG prediction,
                                 trained on ppg, volume (pass attempts/
                                 game), rushing_ppg, snap_share, efficiency
                                 -- replacing all five as individually-
                                 weighted composite inputs (same mechanism
                                 as RB/WR/TE above). rushing_ppg being part
                                 of the trained feature set means the model
                                 itself learns how much a dual-threat QB's
                                 rushing production is worth, rather than
                                 that being a hand-picked weight.

  guide_adj_ppg (ALL FOUR positions -- QB/RB/WR/TE):
        REAL analyst data hand-transcribed from Joel Smyth's Draft Guide
        2026 (data/guide_adjusted_ppg.csv), joined onto joined.csv by name
        (join.py, reusing the same normalize_name + last-name/first-name
        fuzzy-match approach already used for the raw_stats join). This is
        the analyst's own context-adjusted 2025 PPG estimate (e.g.
        stripping out backup-QB games, adding back missed-time production)
        -- a full season MORE RECENT than this pipeline's base 2024
        play-by-play stats, and a human-refined production read rather
        than a raw counting stat. Given both of those, it's arguably more
        informative than the pipeline's own base `ppg` (stale 2024 raw
        stats), so it's weighted meaningfully -- second only to `ppg`
        itself for every position. The guide only covers ~30-50 players
        per position (a big-board preview, not every player); for players
        with no guide match, guide_adj_ppg is null and gets dropped by the
        existing missing-data threshold below, same as any other missing
        component -- NOT defaulted to 0 (which would wrongly penalize
        players simply outside the guide's coverage).

        New weights, redistributed proportionally FROM the original
        weights above to make room for guide_adj_ppg while keeping every
        group's total at 1.0:
          QB:      ppg .38, volume .13, rushing_ppg .13, snap_share .13,
                    efficiency .08, guide_adj_ppg .15
          WR/TE:   ppg .25, rushing_ppg .12, receiving_ppg .16,
                    red_zone_share .08, snap_share .12, efficiency .08,
                    guide_adj_ppg .19
          RB:      see below -- RB carves further room out of the WR/TE
                    split to also fit its two guide-sourced components.

        ML REPLACEMENT NOTE: everything above this point in the docstring
        (the weight numbers through the rest of this section, including
        the RB/playcaller sections below) describes the historical
        evolution of the composite's weights, kept for the paper trail.
        The CURRENT composite (see "ML value model" above and the
        `components` dicts in compute_stat_group_scores()) takes each
        position's raw-production cluster from these numbers -- QB:
        ppg+volume+rushing_ppg+snap_share+efficiency; RB/WR/TE:
        ppg+rushing_ppg+receiving_ppg+red_zone_share+snap_share+
        efficiency -- and replaces that whole cluster with a single
        ml_predicted_ppg entry at the SUM of those weights. Every
        guide_*/playcaller_*/luck weight below (guide_adj_ppg,
        ol_run_block_rank, proj_volume_rank, playcaller_rb_ppg_rank_inv,
        pct_rb1_rank_inv, playcaller_wr_ppg_rank_inv,
        pct_pts_lost_to_luck) is UNCHANGED from the numbers documented
        below -- only the raw-production cluster was swapped out for the
        model's prediction.

  RB-only additions (on top of guide_adj_ppg above):
        0.07  ol_run_block_rank -- REAL 2025 offensive-line run-blocking
                                    rank for the RB's team (1 = best in the
                                    NFL), from data/guide_ol_stats.csv,
                                    joined onto RB rows by team in join.py.
                                    Inverted before z-scoring (lower/better
                                    rank -> higher z-score) so it points the
                                    same direction as every other component
                                    here. Real context for a runner's
                                    efficiency/opportunity quality, but kept
                                    to a modest weight since it's a team-
                                    level signal, not the player's own
                                    production.
        0.11  proj_volume_rank    -- the analyst's own projected 2026 RB
                                    volume rank (data/guide_rb_volume.csv,
                                    joined by name in join.py), "fully
                                    weighted with targets + goal-line
                                    attempts" per the guide. This is
                                    legitimately distinct from the existing
                                    red_zone_share component: red_zone_share
                                    is backward-looking (real 2024 red-zone
                                    opportunity share from play-by-play),
                                    while proj_volume_rank is the analyst's
                                    forward-looking 2026 projection -- real
                                    additional signal, not a duplicate.
                                    Inverted before z-scoring (lower rank =
                                    better) like ol_run_block_rank above.
                                    Weighted just under guide_adj_ppg since
                                    it's the analyst's own explicit "how
                                    much volume will this back get" call,
                                    but still meaningfully below it since
                                    it's a rank (ordinal), not a rate.
        Resulting RB weights: ppg .20, rushing_ppg .10, receiving_ppg .13,
          red_zone_share .07, snap_share .10, efficiency .07,
          guide_adj_ppg .15, ol_run_block_rank .07, proj_volume_rank .11
          (sums to 1.0).
        adj_volume_25_rank and confidence from guide_rb_volume.csv are NOT
        part of the composite -- they're carried through to
        stats.adj_volume_25_rank / stats.guide_volume_confidence in
        build_json.py as informational/detail-view fields only.
        ol_run_block_rank/proj_volume_rank are null for non-RB rows (WR/TE/
        QB never get these columns populated) and for RBs whose team/name
        didn't match the guide files -- same missing-data handling as
        everything else in this file.

  RB-only playcaller/scheme additions, hand-transcribed from the
  "Playcaller Table" page of Joel Smyth's Draft Guide 2026
  (data/guide_playcaller_stats.csv), joined onto RB rows by team in
  join.py:
        0.04  playcaller_rb_ppg_rank -- REAL rank (1=best) of this team's
                                    average fantasy RB PPG across the
                                    current playcaller's tenure (up to their
                                    last 5 seasons). A team-level offensive-
                                    system signal, distinct from
                                    ol_run_block_rank (blocking quality) and
                                    proj_volume_rank (the analyst's player-
                                    volume projection) -- this captures how
                                    much the SCHEME itself has historically
                                    fed the RB spot. Inverted before
                                    z-scoring (lower/better rank -> higher
                                    value) like every other rank column
                                    here.
        0.04  pct_rb1_rank        -- REAL rank (1=best) of %RB1, the share
                                    of the team's RB fantasy points that go
                                    to the starter (higher %RB1 = more of a
                                    true bellcow role, lower = committee).
                                    Rank 1 = highest %RB1 = most bellcow-
                                    friendly, so a player on a high-%RB1
                                    team should score BETTER -- inverted
                                    before z-scoring the same direction as
                                    playcaller_rb_ppg_rank above.
        Both are kept to a small, modest weight (0.04 each) since they are
        team-level context, not the player's own production, same
        philosophy as ol_run_block_rank's 0.07.

        Redistribution math: the previous RB weights (ppg .20,
        rushing_ppg .10, receiving_ppg .13, red_zone_share .07,
        snap_share .10, efficiency .07, guide_adj_ppg .15,
        ol_run_block_rank .07, proj_volume_rank .11 -- sums to 1.0) are each
        scaled down by a factor of (1 - 0.08) = 0.92 to free up exactly
        0.08 of total weight (0.04 + 0.04) for the two new components,
        preserving their relative proportions:
          ppg:               .20 * .92 = .184
          rushing_ppg:       .10 * .92 = .092
          receiving_ppg:     .13 * .92 = .1196
          red_zone_share:    .07 * .92 = .0644
          snap_share:        .10 * .92 = .092
          efficiency:        .07 * .92 = .0644
          guide_adj_ppg:     .15 * .92 = .138
          ol_run_block_rank: .07 * .92 = .0644
          proj_volume_rank:  .11 * .92 = .1012
          playcaller_rb_ppg_rank: .04
          pct_rb1_rank:           .04
          (sum = 0.92 + 0.08 = 1.0, confirmed by summing the scaled block:
          .184+.092+.1196+.0644+.092+.0644+.138+.0644+.1012 = .92)
        playcaller_rb_ppg_rank/pct_rb1_rank are null for non-RB rows and for
        RBs whose team didn't match guide_playcaller_stats.csv (the four
        teams with a brand-new, 0-season playcaller -- PHI/SEA/WAS/BAL --
        which the guide itself leaves blank rather than assigning a made-up
        rank) -- same missing-data threshold handling as everything else in
        this file. personnel/run_scheme/motion_rank/formation_spread/
        rb_screen_rank from the same guide file are NOT part of the
        composite -- too qualitative/hard to weight confidently under time
        pressure -- and are carried through to build_json.py purely as
        informational/detail-view fields.

  WR-only playcaller addition, same source file, joined onto WR rows by
  team in join.py:
        0.05  playcaller_wr_ppg_rank -- REAL rank (1=best) of this team's
                                    average fantasy WR PPG across the
                                    current playcaller's tenure. Inverted
                                    before z-scoring like the RB version
                                    above.
        Redistribution math: the WR/TE weights (ppg .25, rushing_ppg .12,
        receiving_ppg .16, red_zone_share .08, snap_share .12,
        efficiency .08, guide_adj_ppg .19 -- sums to 1.0) are each scaled
        down by a factor of (1 - 0.05) = 0.95 to free up 0.05 for
        playcaller_wr_ppg_rank:
          ppg:              .25 * .95 = .2375
          rushing_ppg:      .12 * .95 = .114
          receiving_ppg:    .16 * .95 = .152
          red_zone_share:   .08 * .95 = .076
          snap_share:       .12 * .95 = .114
          efficiency:       .08 * .95 = .076
          guide_adj_ppg:    .19 * .95 = .1805
          playcaller_wr_ppg_rank: .05
          (sum = 0.95 + 0.05 = 1.0, confirmed:
          .2375+.114+.152+.076+.114+.076+.1805 = .95)
        Applies to WR only -- TE keeps the original (unmodified) WR/TE
        weights above, since guide_playcaller_stats.csv's WR PPG rank
        column is specifically a WR-context signal, not joined onto TE
        rows in join.py. playcaller_wr_ppg_rank is null for non-WR rows and
        for WRs on the four brand-new-playcaller teams (same as the RB
        case above), handled by the same missing-data threshold.

  If a component is missing (non-null) for less than 30% of a position
  group -- i.e. missing for >70% of it -- its weight is redistributed
  proportionally across the remaining available components, so weights
  still sum to 1.0. (Previously this only ever fired for snap_share when
  snap count data failed to pull; now it's also the normal path for
  guide_adj_ppg/ol_run_block_rank/proj_volume_rank on the ~50-80% of
  players per position who aren't covered by the guide.)

For K, there are now THREE tiers, checked in this order in main():
  1. Guide-matched kickers (data/guide_kicker_stats.csv, hand-transcribed
     REAL 2025 data from Joel Smyth's Draft Guide 2026 -- see
     compute_kicker_scores_with_guide()): real data beats proxy, so these
     kickers get a new composite where guide_ppg_25/guide_fg_acc dominate.
  2. Kickers with a 2024 play-by-play proxy match (data/kicker_stats.csv)
     but NO guide match: UNCHANGED from before -- same
     compute_kicker_scores() / K_COMPONENTS as always. This is the "proxy
     is a reasonable fallback for anyone the guide doesn't rank" case; the
     old logic is deliberately left untouched (not de-weighted) here, only
     de-weighted when it's blended alongside guide data in tier 1.
  3. Neither: same ADP-only fallback as DEF/QB/RB/WR/TE's "without_stats"
     path.

  Tier 2, UNCHANGED (real per-kicker stats from 2024 play-by-play, via
  data/kicker_stats.csv / pull_kicker_stats.py -- nfl_data_py's seasonal
  aggregate table itself has zero kicker rows, so this is a systematic
  stats-based build, not named-player overrides):
  An analyst's video identified what actually drives kicker fantasy value,
  in priority order: (1) overall FG accuracy, (2) FG accuracy from 50+
  yards specifically, (3) dome/weather-independent home stadium, (4) team
  offense quality (more scoring drives = more kick attempts). Composite:
        0.40  fg_pct         -- overall FG accuracy (z-scored)
        0.25  fg_pct_50plus  -- 50+ yard FG accuracy (z-scored; falls back
                                 to fg_pct when null for a kicker with <3
                                 attempts from 50+, see pull_kicker_stats.py
                                 MIN_50PLUS_ATTEMPTS -- reusing fg_pct
                                 keeps the weight meaningful instead of
                                 silently dropping to 0 for most kickers,
                                 since long-range attempts are infrequent)
        0.10  is_dome         -- 0/1, weather-independent home stadium
        0.10  team_offense_ppg -- z-scored team scoring rate (more drives
                                 = more kick volume)
        0.15  adp_anchor      -- inverted z-score of ADP, blended in so a
                                 kicker with almost no 2024 attempts (e.g.
                                 an injury replacement/new signing with a
                                 thin or all-null stat line) doesn't get
                                 wildly mis-ranked by noisy/absent
                                 accuracy numbers alone; ADP already
                                 reflects real-world expert judgment about
                                 projected role. Accuracy (0.40+0.25=0.65)
                                 dominates by design per the analyst's
                                 stated priority order; dome/offense are
                                 clearly secondary (0.10 each); ADP is a
                                 stabilizer, not the primary signal (unlike
                                 the pure-ADP fallback this replaces).
  Kickers with no 2024 play-by-play match at all (no fg_pct) AND no guide
  match skip straight to the same ADP-only fallback described below for
  DEF, same as QB/RB/WR/TE's "without_stats" path.

  Tier 1 (real data beats proxy -- see compute_kicker_scores_with_guide()
  and GUIDE_K_COMPONENTS below): guide_ppg_25 (real 2025 PPG) and
  guide_fg_acc (real 2025 FG%) are REAL, more recent, human-verified
  numbers -- not a 2024-pbp reconstruction -- so they take over as the
  dominant signal (0.25 + 0.20 = 0.45 of the composite). The tier-2 proxy
  components (fg_pct, fg_pct_50plus, is_dome, team_offense_ppg) are kept,
  not deleted, but DE-WEIGHTED from their tier-2 weights (0.40/0.25/0.10/
  0.10 -> 0.10/0.07/0.02/0.05) to a secondary/tie-breaking role in this
  blended tier, since the guide doesn't cover 100% of kickers and this
  proxy logic remains the fallback for whoever it misses (tier 2 above).
  guide_off_rank (team's projected 2026 offensive rank, 1=best -- inverted
  before z-scoring), guide_go_pct_rank (team's 4th-down go-for-it rate rank
  in FG range; NOT inverted -- a HIGH rank number here means a
  conservative team that kicks more field goals, which is good for a
  kicker, per the guide file's own column semantics), guide_is_50plus_good,
  and guide_is_value are secondary analyst-judgment signals. See
  GUIDE_K_COMPONENTS for the exact weights (sums to 1.0 including the
  de-weighted tier-2 proxy terms and a small adp_anchor stabilizer).

For DEF (REAL analyst data from Joel Smyth's Draft Guide 2026, hand-
transcribed into data/guide_def_stats.csv, now PRIMARY; the play-by-play-
derived proxy in data/defense_stats.csv / build_defense_stats.py is kept
as a SECONDARY/tie-breaking signal, not deleted, since the guide doesn't
cover every situation the proxy could theoretically be regenerated for):
  Real data beats proxy: pr_roe_rank (real Pressure Rate over Expected
  rank -- the actual metric the old def_pressure_rate_proxy could only
  approximate via sacks-per-dropback) and adj_ppg_25_rank (the analyst's
  own 2025 adjusted-PPG rank, built from real box scores rather than this
  pipeline's 2024-pbp custom-scoring reconstruction) are both REAL,
  analyst-computed numbers now available from guide_def_stats.csv, so they
  take over as the dominant signal. The old def_custom_adjusted_ppg /
  def_pressure_rate_proxy pair is de-weighted (0.60+0.20 -> 0.15+0.10) to a
  secondary/tie-breaking role rather than removed -- guide_def_stats.csv
  only covers this season's 32 teams as ranked by one analyst, so keeping
  the pbp-derived proxy alive is a reasonable fallback/blend, not dead
  weight. bottom10_offense and trend are two additional REAL analyst
  judgments (strength of opposing offenses faced, offseason roster
  trajectory) that build_defense_stats.py's docstring explicitly flagged
  as gaps it could NOT derive from play-by-play data alone -- the guide
  fills both gaps directly, so they're folded in as a small flag-based
  bonus/malus (not z-scored/weighted like the rank-based components, since
  they're binary/categorical, not continuous). Composite:
        0.35  guide_pr_roe_rank      -- REAL pressure-rate rank (1=best),
                                         inverted (z-score of -rank) so a
                                         lower/better rank scores higher.
                                         Co-primary signal: this is the
                                         actual metric the old proxy only
                                         approximated.
        0.35  guide_adj_ppg_25_rank  -- REAL analyst adjusted-PPG rank
                                         (1=best), inverted the same way.
                                         Co-primary signal alongside
                                         pressure rank -- both come
                                         straight from the analyst's own
                                         published numbers, not a 2024-pbp
                                         reconstruction.
        0.15  def_custom_adjusted_ppg -- DE-WEIGHTED (was 0.60). Still
                                         this league's exact custom
                                         scoring rules averaged per game,
                                         kept as a secondary/tie-breaking
                                         signal and as the fallback engine
                                         for any team the guide doesn't
                                         cover.
        0.10  def_pressure_rate_proxy -- DE-WEIGHTED (was 0.20). Sacks per
                                         opponent pass attempt faced --
                                         now redundant with the REAL
                                         pr_roe_rank above, kept at low
                                         weight purely as a tie-breaker.
        0.05  turnover/sack volume    -- DE-WEIGHTED (was 0.20). Same
                                         def_sacks + def_interceptions +
                                         def_forced_fumbles z-score as
                                         before, now a minor tie-breaker.
  Plus two small additive (not z-scored) modifiers layered on top of the
  weighted composite above, each on the same rough +/-1-z-unit scale so
  neither dominates the rank-based signal:
        -0.15  bottom10_offense flag -- REAL analyst red flag: this
                                         defense faces a lot of bad
                                         offenses, per Joel Smyth's Draft
                                         Guide 2026 (their stated
                                         reasoning, not re-derived here).
                                         Applied when bottom10_offense==1.
        +0.05 / -0.05  trend         -- REAL offseason-trajectory signal:
                                         up = +0.05, down = -0.05,
                                         flat = 0. Fills the "offseason
                                         roster change" gap
                                         build_defense_stats.py flagged
                                         as undoable from pbp data alone.
  If guide_pr_roe_rank is null for a team (not in guide_def_stats.csv),
  that team's composite runs on def_custom_adjusted_ppg /
  def_pressure_rate_proxy / turnover volume alone (their now-secondary
  weights, renormalized) rather than losing the 0.70 of composite weight
  the guide columns would otherwise own -- see compute_defense_scores().
  If def_custom_adjusted_ppg is ALSO null for a team (e.g.
  defense_stats.csv hasn't been generated -- build_defense_stats.py not
  run), that team falls back further to the pure ADP-based rank (inverted
  so lower ADP = higher score) previously used for all of DEF. Documented
  in data/README.md.

Output: joined.csv + value_score, position_rank, adp_position_rank,
value_gap (adp_position_rank - position_rank; positive = sleeper /
undervalued by ADP relative to actual production, negative = ADP is
pricier than production supports).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

STAT_POSITIONS = {"QB", "RB", "WR", "TE"}
# K has real per-kicker stats now (data/kicker_stats.csv), handled by its
# own branch in main() below -- not part of STAT_POSITIONS since its
# columns/composite are entirely different from QB/RB/WR/TE's.
NO_STAT_POSITIONS = {"DEF"}

# League's EXACT custom scoring rules (not nfl_data_py's generic
# fantasy_points_ppr column, which uses its own standard formula and does
# not match this league -- e.g. this league's interception penalty is -1,
# not the more common -2). All QB/RB/WR/TE points in this file are computed
# from raw stat columns using these rates. Two-point conversions are part
# of the league's rules but are NOT reflected here -- nfl_data_py's
# seasonal data has no two-point-conversion column, and they're rare
# enough not to meaningfully change rankings. Documented in README.
PASS_YDS_TO_PTS = 0.04
PASS_TD_TO_PTS = 4
INT_TO_PTS = -1
YDS_TO_PTS = 0.1       # rushing AND receiving yards, per this league's rules
TD_TO_PTS = 6          # rushing AND receiving TDs
REC_TO_PTS = 1         # PPR: 1 pt/reception

MODELS_DIR = DATA_DIR / "models"

# The 2024-base "live" season this pipeline runs inference on -- matches
# pull.py's STATS_SEASON. Used only by the team-context historical-analog
# features below (team_ol_rb_ybc_att needs to know which season's PFR data
# to pull for the CURRENT live composite); bump alongside pull.py's
# STATS_SEASON when a new season's data is published.
LIVE_SEASON = 2024

# NOTE: the per-position feature list a saved model actually expects is no
# longer hardcoded here -- it's read straight from each position's
# data/models/{pos}_metadata.json ("features" key), which train_ml_model.py
# writes out after its feature-set-selection search (see that file's module
# docstring "FEATURE VARIANTS tested per position"). This guarantees
# score.py can never drift out of sync with whatever feature set a given
# training run actually picked (base, base+age, base+team_ctx, etc.) --
# previously this was a hand-copied duplicate of train_ml_model.py's dict
# and the two could silently disagree.
_ML_MODEL_CACHE = {}
_ML_FEATURES_CACHE = {}


def load_ml_model(pos: str):
    """Loads (and caches) the trained sklearn Pipeline for `pos`, saved by
    train_ml_model.py to data/models/{pos}_model.joblib. Returns None if
    the model file doesn't exist (e.g. train_ml_model.py hasn't been run
    yet) -- callers must handle that by falling back to raw ppg, see
    compute_ml_predicted_ppg() below.
    """
    if pos in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[pos]
    model_path = MODELS_DIR / f"{pos}_model.joblib"
    if not model_path.exists():
        _ML_MODEL_CACHE[pos] = None
        return None
    import joblib
    model = joblib.load(model_path)
    _ML_MODEL_CACHE[pos] = model
    return model


def load_ml_features(pos: str) -> list:
    """Loads (and caches) the exact feature list/order for `pos`'s saved
    model, from data/models/{pos}_metadata.json's "features" key -- see
    note above. Falls back to the original 2024-era baseline feature set
    if metadata is missing (shouldn't happen once train_ml_model.py has
    been run, but keeps this file from crashing on a stale/partial
    checkout)."""
    if pos in _ML_FEATURES_CACHE:
        return _ML_FEATURES_CACHE[pos]
    meta_path = MODELS_DIR / f"{pos}_metadata.json"
    fallback = {
        "QB": ["ppg", "volume", "rushing_ppg", "snap_share", "efficiency"],
        "RB": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
        "WR": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
        "TE": ["ppg", "rushing_ppg", "receiving_ppg", "red_zone_share", "snap_share", "efficiency"],
    }[pos]
    if not meta_path.exists():
        _ML_FEATURES_CACHE[pos] = fallback
        return fallback
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta.get("features", fallback)
    _ML_FEATURES_CACHE[pos] = features
    return features


def pull_team_ol_rb_ybc(season: int) -> pd.DataFrame:
    """Team-level offensive-line run-blocking proxy: mean rushing
    yards-before-contact per carry (PFR's ybc_att), aggregated to the team
    level from nfl_data_py's import_seasonal_pfr('rush', ...). A real,
    honestly-computable historical analog to the guide's
    guide_ol_run_block_rank_2025 -- computed the SAME way for every
    historical training season (via train_ml_model.py, which calls this
    same function) and for the live season, so training features and live
    inference features are guaranteed consistent. Excludes the '2TM'/'3TM'
    multi-team PFR rows (ambiguous team attribution for a traded player).
    Weighted by attempts so a handful of backup carries doesn't swing a
    team's number as much as its starter's larger sample."""
    import nfl_data_py as nfl

    pfr_rush = nfl.import_seasonal_pfr("rush", [season])
    pfr_rush = pfr_rush[~pfr_rush["tm"].isin(["2TM", "3TM"])].copy()
    pfr_rush = pfr_rush.dropna(subset=["att", "ybc_att"])
    pfr_rush = pfr_rush[pfr_rush["att"] > 0]

    def weighted_mean(g):
        w = g["att"]
        return float((g["ybc_att"] * w).sum() / w.sum()) if w.sum() > 0 else np.nan

    team_ybc = pfr_rush.groupby("tm").apply(weighted_mean).reset_index()
    team_ybc.columns = ["team", "team_ol_rb_ybc_att"]
    return team_ybc


_TEAM_OL_YBC_CACHE = {}


def add_team_context_features(df: pd.DataFrame, pos: str, season: int) -> pd.DataFrame:
    """Adds team_rb_ppg_hist / team_wr_ppg_hist (this season's own
    team-level average fantasy PPG at the position, aggregated from the
    per-player `ppg` column already computed earlier in
    compute_stat_group_scores) and, for RB, team_ol_rb_ybc_att (pulled via
    pull_team_ol_rb_ybc above) -- see train_ml_model.py's module docstring
    "ROUND 2" section for why these are genuine historical analogs to the
    guide's playcaller_rb_ppg_rank / playcaller_wr_ppg_rank /
    guide_ol_run_block_rank_2025 blend terms, tested as real trained-model
    features rather than hand-picked post-prediction blend weights.
    Called identically for every historical training season (via
    train_ml_model.py) and for the live inference season, so there is one
    code path computing this feature both times.
    """
    df = df.copy()
    games = df["games"].fillna(0)
    usable = games >= 1  # include committee/backup work, matching the
    # guide's own team-level PPG-under-this-playcaller framing (whole
    # depth chart's production, not just the starter).
    if pos == "RB":
        team_ppg = df.loc[usable].groupby("team")["ppg"].mean().rename("team_rb_ppg_hist")
        df = df.merge(team_ppg, on="team", how="left")
        if season not in _TEAM_OL_YBC_CACHE:
            _TEAM_OL_YBC_CACHE[season] = pull_team_ol_rb_ybc(season)
        df = df.merge(_TEAM_OL_YBC_CACHE[season], on="team", how="left")
    elif pos == "WR":
        team_ppg = df.loc[usable].groupby("team")["ppg"].mean().rename("team_wr_ppg_hist")
        df = df.merge(team_ppg, on="team", how="left")
    return df


def compute_ml_predicted_ppg(df: pd.DataFrame, pos: str) -> pd.Series:
    """Applies the trained per-position model (see "ML value model" in the
    module docstring) to this season's already-computed feature columns
    (ppg, rushing_ppg, receiving_ppg, red_zone_share, snap_share,
    efficiency, volume -- computed earlier in compute_stat_group_scores by
    this same function file) to produce ml_predicted_ppg: the model's own
    projection of next season's PPG for each player, given their CURRENT
    (2024 base) production profile.

    Missing individual features are median-imputed within this position
    group before prediction -- the same missing-data convention used
    everywhere else in this file, so a player missing e.g. snap_pct still
    gets a prediction rather than being silently dropped.

    Falls back to the player's own raw ppg (a safe, clearly-documented
    degenerate case -- not a crash) if this position has no saved model
    yet (train_ml_model.py hasn't been run), if a required feature column
    isn't present on `df` at all, or if the saved model's expected input
    shape doesn't match load_ml_features(pos) (e.g. a stale/partially
    written models/{pos}_model.joblib + {pos}_metadata.json pair from a
    run that didn't finish, or from a different, unrelated script writing
    to the same models/ directory -- this file's feature set is no longer
    a single hardcoded constant (see load_ml_features()), so this
    consistency check matters more than it used to).
    """
    model = load_ml_model(pos)
    feature_cols = load_ml_features(pos)
    if model is None:
        print(f"WARNING: no saved ML model for {pos} (run train_ml_model.py) "
              f"-- falling back to raw ppg for ml_predicted_ppg.")
        return df["ppg"].copy()

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"WARNING: {pos} model expects feature(s) {missing_cols} not present on "
              f"this dataframe -- falling back to raw ppg for ml_predicted_ppg.")
        return df["ppg"].copy()

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())
    # A position group that's entirely missing one feature (e.g. all-null
    # snap_pct) would median-impute to NaN; fall back to 0 in that
    # unlikely edge case so predict() doesn't error.
    X = X.fillna(0)
    try:
        preds = pd.Series(model.predict(X.to_numpy(dtype=float)), index=df.index)
    except (ValueError, AttributeError) as e:
        print(f"WARNING: {pos} model.predict() failed ({e}) -- this usually means "
              f"models/{pos}_model.joblib doesn't match models/{pos}_metadata.json's "
              f"feature list (stale/mismatched files from an interrupted or unrelated "
              f"training run). Falling back to raw ppg for ml_predicted_ppg. Re-run "
              f"train_ml_model.py to regenerate a consistent pair.")
        return df["ppg"].copy()

    # Low-snap-share shrinkage: a backup with a tiny 2024 snap share (e.g.
    # a QB2 who played a handful of series) produces a noisy, unreliable
    # per-game RATE -- the model can extrapolate that small, hot sample
    # into an inflated next-season prediction (confirmed live: Marcus
    # Mariota at 26.8% snap share was predicted to QB8 off a small sample,
    # well above his actual 2024 ppg). Shrink ml_predicted_ppg toward the
    # player's own raw ppg in proportion to how little we actually saw of
    # them -- full trust in the model at snap_share >= FULL_CONF, linearly
    # down to full distrust (pure raw ppg) at snap_share <= NO_CONF. This
    # does not discard the model's signal for anyone with a real sample --
    # it only tempers it for players whose 2024 rate stat is itself noisy.
    FULL_CONF, NO_CONF = 0.5, 0.1
    snap = df["snap_share"].fillna(0)
    confidence = ((snap - NO_CONF) / (FULL_CONF - NO_CONF)).clip(0, 1)
    preds = confidence * preds + (1 - confidence) * df["ppg"].fillna(preds)
    return preds


def compute_passing_pts(df: pd.DataFrame) -> pd.Series:
    return (
        df["passing_yards"].fillna(0) * PASS_YDS_TO_PTS
        + df["passing_tds"].fillna(0) * PASS_TD_TO_PTS
        + df["interceptions"].fillna(0) * INT_TO_PTS
    )


ZSCORE_CLIP = 2.0  # was 3.0 -- still let ml_predicted_ppg peg at the ceiling
# for Taysom Hill (TE) and Anthony Richardson (QB), both unusual rushing-
# heavy statistical profiles the model has little/no in-distribution
# training data for, so it extrapolates to an implausible prediction that
# a 3.0 clip still let dominate. 2.0 is still a genuine outlier bar
# (only ~2.3% of a normal distribution exceeds it) while meaningfully
# reducing how much any single maxed-out component can dominate.

def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score, clipped to +-ZSCORE_CLIP.

    Confirmed bug this fixes: Taysom Hill (TE) has real rushing_ppg (~8
    pts/game) that's a genuine, massive outlier for the position -- real
    TEs almost never rush. Because the TE group's rushing_ppg has very
    low variance (nearly everyone is near 0), his single real data point
    produced an extreme unclipped z-score that dominated his composite
    and ranked him TE1, despite an ADP of ~370 showing the market
    correctly treats him as a low-value gadget player, not a true
    receiving TE. Clipping every z-score to a fixed range prevents any
    single stat outlier -- Hill or a future similar case -- from
    single-handedly overwhelming the weighted composite, without needing
    a player-specific carve-out.
    """
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - s.mean()) / std
    return z.clip(lower=-ZSCORE_CLIP, upper=ZSCORE_CLIP)


# Sample-size guardrail: a player with very few 2024 games can produce a
# huge single-game rate stat (e.g. one spot start) that, un-shrunk, ranks
# ABOVE legitimate full-season starters -- confirmed bug: Joe Milton III
# (Cowboys backup QB) had 1 game at a 95% snap share and a big garbage-
# time-adjacent stat line, and with no games-played floor anywhere in this
# pipeline he ranked QB4 overall with a +43 value_gap. Two-part fix:
#   1. MIN_GAMES_FOR_STATS: below this, a player is treated as having NO
#      usable stat sample at all and falls into the same ADP-only fallback
#      path as a rookie with zero 2024 games (see the has_stats gate below).
#   2. FULL_CONFIDENCE_GAMES: between MIN_GAMES_FOR_STATS and this, the
#      composite z-score is linearly shrunk toward 0 (league-average) by
#      games/FULL_CONFIDENCE_GAMES, so a 4-game outlier stat line pulls
#      most of the way toward "unremarkable" rather than fully counting --
#      a standard small-sample shrinkage estimator, not just a hard cutoff.
MIN_GAMES_FOR_STATS = 3
FULL_CONFIDENCE_GAMES = 10


def sample_size_shrinkage(games_series: pd.Series) -> pd.Series:
    # Confirmed bug: this only ever runs inside compute_stat_group_scores()
    # on rows that already passed the has_stats gate (has_2024_stats |
    # has_guide_signal), so a NaN games value here can ONLY mean "no 2024
    # stat line, real guide/real2025 signal instead" (has_2024_stats
    # requires games >= MIN_GAMES_FOR_STATS, which NaN never satisfies) --
    # e.g. Ashton Jeanty, guide_adj_ppg 14.5 + real2025_total_pts 127.5 +
    # proj_volume_rank #4. Shrinking toward 0 by games/FULL_CONFIDENCE_GAMES
    # with fillna(0) multiplied their ENTIRE composite (including the real
    # guide/real2025 signal that got them into this function at all) by
    # exactly 0, silently wiping every rookie with real signal down to a
    # tie with players who have NOTHING -- the has_stats gate fix let them
    # in the door, then this zeroed them right back out. games-based
    # shrinkage only makes sense as a penalty for a THIN 2024 sample; a
    # player with no 2024 sample at all isn't thin, they're relying on a
    # different (non-games-based) signal entirely, so they get full
    # confidence (factor=1) instead of the old effective disqualification.
    factor = (games_series / FULL_CONFIDENCE_GAMES).clip(upper=1.0)
    return factor.fillna(1.0)


def compute_stat_group_scores(df: pd.DataFrame, season: int = LIVE_SEASON) -> pd.DataFrame:
    df = df.copy()
    games = df["games"].replace(0, np.nan)
    pos = df["position"].iloc[0]

    # Rushing-only fantasy production, per game. Computed for every
    # position (including QB) so we can separate rushing production from
    # everything else instead of lumping it into one "volume" number.
    rushing_pts = (
        df["rushing_yards"].fillna(0) * YDS_TO_PTS
        + df["rushing_tds"].fillna(0) * TD_TO_PTS
    )
    df["rushing_ppg"] = rushing_pts / games

    df["snap_share"] = df["snap_pct"]  # already 0-1 average

    if pos == "QB":
        passing_pts = compute_passing_pts(df)
        df["ppg"] = (passing_pts + rushing_pts) / games
        df["volume"] = df["attempts"] / games  # passing volume/game
        opportunities = df["attempts"].fillna(0) + df["carries"].fillna(0)
        df["efficiency"] = (passing_pts + rushing_pts) / opportunities.replace(0, np.nan)

        # ML value model -- see module docstring "ML value model". Replaces
        # the old individually-weighted ppg/volume/rushing_ppg/snap_share/
        # efficiency cluster (which summed to 0.80) with a single
        # ml_predicted_ppg component at that same combined weight.
        df["ml_predicted_ppg"] = compute_ml_predicted_ppg(df, pos)

        components = {
            "ml_predicted_ppg": 0.36 + 0.12 + 0.12 + 0.12 + 0.08,  # = 0.80, sum of old ppg/volume/rushing_ppg/snap_share/efficiency weights
            # guide_adj_ppg weight -- EMPIRICALLY FIT (data/fit_blend_weight.py),
            # not hand-picked. See that script + this file's module docstring
            # "BLEND WEIGHT FITTING" for the full method: grid-search the
            # ml_predicted_ppg<->guide_adj_ppg mix that best Spearman-
            # correlates with REAL known 2025 outcomes (real2025_total_pts,
            # actual nfl.com season totals) for the n=32 QBs with both
            # signals. Result: guide_adj_ppg alone (Spearman 0.686) beats
            # ml_predicted_ppg alone (0.143) by a wide margin on this
            # validation, and the fitted mix puts ~92% of the two-way weight
            # on guide_adj_ppg (fitted ratio ml:guide = 0.081, i.e.
            # guide_weight = ml_weight / 0.081 = 9.877). See the docstring
            # section for the honest caveat on why this isn't fully
            # apples-to-apples (guide_adj_ppg is a same-season nowcast with
            # real depth-chart/injury information; ml_predicted_ppg is a
            # true year-ahead forecast from 2024 data alone) -- kept anyway,
            # per this fit, because it's still the best available honest
            # answer to "how should these two signals actually be weighted."
            "guide_adj_ppg": 9.877,
            # Real per-player "Luck Metric" (data/guide_luck_metric.csv):
            # pct of season fantasy points lost/gained to variance (OT
            # points, penalty-nullified plays, dropped TDs, tackled-at-1,
            # etc). Positive = unlucky = positive-regression candidate, so
            # this is added WITHOUT inverting -- a higher (more positive)
            # value should raise the composite. Small weight since it only
            # covers a top/bottom-25 subset (~50 players total across all
            # 4 positions) and the existing >=30%-non-null threshold below
            # will drop it entirely for positions where coverage is too
            # sparse to trust, same protection as every other optional
            # component in this file.
            "pct_pts_lost_to_luck": 0.06,
            # REAL 2025 season-total fantasy points, pulled directly from
            # nfl.com's live 2025 stats pages (data/real2025_stats.csv --
            # see real2025_score.py). This is ACTUAL 2025 production, not
            # a projection or last year's stats -- the single most direct
            # real-2025 signal in this whole model. Weighted comparably to
            # guide_adj_ppg since both are real current-season data; covers
            # 155 players (every real starter + most fantasy-relevant
            # backups) so the existing >=30%-non-null threshold will keep
            # it active for QB/RB/WR/TE.
            "real2025_total_pts": 0.16,
        }
    else:
        # Receiving-only fantasy production, per game -- kept separate from
        # rushing_ppg above so a receiving-work-heavy player (pass-catching
        # RB, possession WR, receiving TE) is visibly rewarded for it,
        # rather than that signal getting averaged away inside one
        # combined "volume" number.
        receiving_pts = (
            df["receptions"].fillna(0) * REC_TO_PTS
            + df["receiving_yards"].fillna(0) * YDS_TO_PTS
            + df["receiving_tds"].fillna(0) * TD_TO_PTS
        )
        df["receiving_ppg"] = receiving_pts / games
        df["ppg"] = (rushing_pts + receiving_pts) / games

        opportunities = df["carries"].fillna(0) + df["targets"].fillna(0)
        df["volume"] = opportunities / games  # combined touches+targets/game
        df["efficiency"] = (rushing_pts + receiving_pts) / opportunities.replace(0, np.nan)

        # Team-context historical-analog features (team_rb_ppg_hist /
        # team_wr_ppg_hist / team_ol_rb_ybc_att) -- see
        # add_team_context_features() above and train_ml_model.py's module
        # docstring "ROUND 2" section. Computed here (RB/WR only) so
        # they're available to compute_ml_predicted_ppg() below whenever a
        # position's saved model actually uses them (per its metadata's
        # feature list) -- harmless no-op columns otherwise.
        if pos in ("RB", "WR"):
            df = add_team_context_features(df, pos, season)

        # ML value model -- see module docstring "ML value model". Replaces
        # the old individually-weighted ppg/rushing_ppg/receiving_ppg/
        # red_zone_share/snap_share/efficiency cluster with a single
        # ml_predicted_ppg component at that cluster's combined weight
        # (computed per position below, right where that old cluster's
        # weights used to be spelled out).
        df["ml_predicted_ppg"] = compute_ml_predicted_ppg(df, pos)

        # REAL red-zone usage share, not a TD-rate proxy. red_zone_share
        # comes from data/redzone_stats.csv (built by redzone.py from 2024
        # play-by-play data: red-zone rush attempts + red-zone targets,
        # divided by total carries+targets) and is joined into joined.csv
        # by join.py on player_id. This captures touchdown/goal-line
        # equity directly from actual red-zone opportunity share, per an
        # analyst's claim that red-zone touches are one of the two most
        # predictive stats for RB touchdown production (~65% of RB TDs) --
        # replacing the old td_rate proxy (TDs per opportunity over ALL
        # opportunities, not red-zone-specific, since nfl_data_py's
        # seasonal aggregate data has no red-zone columns).
        # goal_line_share (yardline_100 <= 5, a stricter cut) is also
        # computed by redzone.py and carried through to stats.
        # output/players.json, but is not part of the composite weight --
        # red_zone_share is the broader, less noisy signal for volume-
        # weighted RB/WR/TE comparisons; goal_line_share is exposed for
        # detail views/spot-checking instead.
        if pos == "RB":
            # RB carves out room for RB-only guide-sourced components on
            # top of guide_adj_ppg -- see module docstring "RB-only
            # additions" for the original reasoning/redistribution math.
            #
            # ml_predicted_ppg NOW ABSORBS TWO OF THESE (2026 retrain --
            # see train_ml_model.py's module docstring "ROUND 2"): the RB
            # model's feature-set search found that team_rb_ppg_hist (a
            # real historical analog of playcaller_rb_ppg_rank -- this same
            # team's own average fantasy RB PPG, computed identically for
            # every 2018-2024 season) and team_ol_rb_ybc_att (a real
            # historical analog of guide_ol_run_block_rank_2025 -- team
            # rushing yards-before-contact per carry from PFR) genuinely
            # improved CV R^2 (0.5081 -> 0.5282, clearing
            # FEATURE_IMPROVEMENT_MARGIN) when added as TRAINED model
            # features. So ol_run_block_rank and playcaller_rb_ppg_rank_inv
            # are REMOVED from this hand-picked post-prediction blend --
            # keeping them here too would double-count the same underlying
            # signal once as a learned feature and once as a hand weight --
            # and their old weight (.061 + .038 = .099) is folded into
            # ml_predicted_ppg's weight instead, below.
            #
            # proj_volume_rank: the analyst's own forward-looking 2026 RB
            # volume projection (targets + goal-line attempts weighted),
            # from data/guide_rb_volume.csv joined by name in join.py. This
            # one has NO honest historical analog (it's a pure forward
            # projection with nothing to validate against in past seasons),
            # so it stays a hand-picked post-prediction blend term.
            # Inverted (lower/better rank -> higher value) so it points the
            # same direction as every other z-scored component.
            df["proj_volume_rank"] = -df["guide_proj_volume_rank"]

            # pct_rb1_rank: REAL %RB1 bellcow-share rank from
            # data/guide_playcaller_stats.csv (joined by team in join.py).
            # No historical analog was built for this one this round (would
            # need a full backfield-committee-share reconstruction from
            # play-by-play for every historical season -- out of scope this
            # pass, an honest limitation, not attempted) -- stays hand-
            # picked. Inverted before z-scoring like every other rank here.
            df["pct_rb1_rank_inv"] = -df["pct_rb1_rank"]

            components = {
                # sum of old ppg .174 + rushing_ppg .087 + receiving_ppg .113
                # + red_zone_share .061 + snap_share .087 + efficiency .061
                # + ol_run_block_rank .061 + playcaller_rb_ppg_rank_inv .038
                # (both now absorbed into the model itself -- see comment
                # above) = 0.682
                "ml_predicted_ppg": 0.174 + 0.087 + 0.113 + 0.061 + 0.087 + 0.061 + 0.061 + 0.038,  # = 0.682
                # guide_adj_ppg weight -- EMPIRICALLY FIT, not hand-picked
                # (data/fit_blend_weight.py; see QB block above for the full
                # method). n=45 RBs with both ml_predicted_ppg and
                # guide_adj_ppg. guide_adj_ppg alone: Spearman 0.644 vs real
                # 2025 outcomes; ml_predicted_ppg alone: 0.420. Fitted ratio
                # ml:guide = 0.053 -> guide_weight = ml_weight(.682)/.053 =
                # 12.958. Same nowcast-vs-forecast caveat as QB applies.
                "guide_adj_ppg": 12.958,
                "proj_volume_rank": 0.096,
                "pct_rb1_rank_inv": 0.038,
                "pct_pts_lost_to_luck": 0.054,  # see QB block above for reasoning
                "real2025_total_pts": 0.12,  # see QB block above for reasoning
            }
        elif pos == "WR":
            # playcaller_wr_ppg_rank: REAL team-level playcaller context
            # from data/guide_playcaller_stats.csv (joined by team in
            # join.py) -- see module docstring "WR-only playcaller
            # addition" for the redistribution math. Inverted before
            # z-scoring like the RB version above.
            #
            # UNLIKE RB's playcaller_rb_ppg_rank, this one STAYS a
            # hand-picked blend term: the WR model's feature-set search
            # (train_ml_model.py "ROUND 2") DID test the honest historical
            # analog (team_wr_ppg_hist, this team's own historical average
            # fantasy WR PPG) as a trained feature, but it produced no real
            # improvement over `age` alone (CV R^2 0.5409 either way,
            # base+age+team_ctx == base+age to 4 decimals) -- so it was NOT
            # adopted into the model, and playcaller_wr_ppg_rank_inv is kept
            # here unchanged. Honestly reported as "tested, not adopted,"
            # not silently dropped.
            df["playcaller_wr_ppg_rank_inv"] = -df["playcaller_wr_ppg_rank"]

            components = {
                # sum of old ppg .223 + rushing_ppg .107 + receiving_ppg .143
                # + red_zone_share .071 + snap_share .107 + efficiency .071
                "ml_predicted_ppg": 0.223 + 0.107 + 0.143 + 0.071 + 0.107 + 0.071,  # = 0.722
                # guide_adj_ppg weight -- EMPIRICALLY FIT, not hand-picked
                # (data/fit_blend_weight.py; see QB block above for the full
                # method). n=29 WRs with both signals. guide_adj_ppg alone:
                # Spearman 0.718 vs real 2025 outcomes; ml_predicted_ppg
                # alone: 0.342. Fitted ratio ml:guide = 0.111 -> guide_weight
                # = ml_weight(.722)/.111 = 6.498. Same nowcast-vs-forecast
                # caveat as QB applies.
                "guide_adj_ppg": 6.498,
                "playcaller_wr_ppg_rank_inv": 0.047,
                "pct_pts_lost_to_luck": 0.061,  # see QB block above for reasoning
                "real2025_total_pts": 0.14,  # see QB block above for reasoning
            }
        else:
            # TE: guide_adj_ppg weight LEFT HAND-PICKED, deliberately NOT
            # empirically refit -- data/fit_blend_weight.py found only
            # n=11 TEs with both ml_predicted_ppg and guide_adj_ppg AND a
            # real2025_total_pts outcome to validate against (MIN_N_TO_TRUST
            # = 20 in that script). Fitting a blend weight off 11 rows would
            # be noise dressed up as precision, not a real answer -- kept at
            # its original hand-picked value instead. Documented explicitly
            # rather than silently reusing another position's fitted ratio.
            components = {
                # sum of old ppg .235 + rushing_ppg .113 + receiving_ppg .150
                # + red_zone_share .075 + snap_share .113 + efficiency .075
                "ml_predicted_ppg": 0.235 + 0.113 + 0.150 + 0.075 + 0.113 + 0.075,  # = 0.761
                "guide_adj_ppg": 0.179,
                "pct_pts_lost_to_luck": 0.06,  # see QB block above for reasoning
                "real2025_total_pts": 0.14,  # see QB block above for reasoning
            }

    available = {}
    for col, weight in components.items():
        non_null_frac = df[col].notna().mean()
        # require a meaningfully populated column to use it at all
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        # degenerate fallback: use ppg alone
        available = {"ppg": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    composite = composite * sample_size_shrinkage(df["games"])
    df["value_score"] = composite
    return df


# K composite weights -- see module docstring for the full reasoning behind
# each weight. Accuracy (fg_pct + fg_pct_50plus) intentionally dominates
# per the analyst's stated priority order; dome/team-offense are secondary;
# adp_anchor is a stabilizer against noisy/thin stat lines, not the primary
# signal (unlike the pure-ADP fallback this replaces for K).
K_COMPONENTS = {
    "fg_pct": 0.40,
    "fg_pct_50plus": 0.25,
    "is_dome": 0.10,
    "team_offense_ppg": 0.10,
    "adp_anchor": 0.15,
}


# Bayesian-shrinkage pseudo-counts for fg_pct / fg_pct_50plus, used only to
# stabilize the COMPOSITE (raw fg_pct/fg_pct_50plus in stats/players.json
# are untouched). Without this, a kicker with a handful of attempts and a
# lucky 100% (e.g. 5-for-5) out-z-scores a proven, high-volume kicker at
# 93% on 44 attempts -- confirmed happening in practice (Spencer Shrader,
# 5/5, ranked #1 overall above Chris Boswell, 41/44). Shrinking each
# kicker's rate toward the position's attempt-weighted average FG%,
# proportional to a virtual prior of FG_PCT_PRIOR_WEIGHT / FG50_PRIOR_WEIGHT
# attempts at that average, fixes this the standard way (a "regressed"
# batting-average-style estimator) without discarding the signal entirely
# the way a hard attempts cutoff would. 50+ yard attempts are rarer per
# kicker, so it gets a smaller prior weight than overall fg_pct.
FG_PCT_PRIOR_WEIGHT = 10
FG50_PRIOR_WEIGHT = 4


def _shrink_pct(made: pd.Series, attempts: pd.Series, prior_weight: float) -> pd.Series:
    attempts = attempts.fillna(0)
    made = made.fillna(0)
    league_avg = made.sum() / attempts.sum() if attempts.sum() > 0 else 0.0
    return (made + prior_weight * league_avg) / (attempts + prior_weight)


def compute_kicker_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Real per-kicker composite for K, replacing the old ADP-only
    fallback. Requires fg_pct to be non-null (i.e. the kicker had a 2024
    play-by-play match via kicker_stats.csv) -- callers should route
    kickers without a match to the ADP-only fallback instead, same as
    QB/RB/WR/TE's "without_stats" path.
    """
    df = df.copy()

    # Shrink both accuracy rates toward the group's attempt-weighted mean
    # before z-scoring, so low-volume kickers' noisy percentages don't
    # dominate the ranking (see FG_PCT_PRIOR_WEIGHT comment above). The
    # RAW fg_pct/fg_pct_50plus values (not these shrunk ones) are what get
    # written to joined.csv/players.json -- this shrinkage only affects
    # the composite's internal ranking math.
    fg_pct_shrunk = _shrink_pct(df["fg_made"], df["fg_attempts"], FG_PCT_PRIOR_WEIGHT)

    # fg_pct_50plus falls back to fg_pct when null (too few 50+ attempts to
    # trust the rate -- see pull_kicker_stats.py's MIN_50PLUS_ATTEMPTS).
    # This keeps the 0.25 weight meaningful for the majority of kickers who
    # simply don't see many 50+ yard tries in a season, instead of that
    # weight effectively vanishing. Also shrunk toward the group mean for
    # the same small-sample reason as fg_pct above.
    fg50_made = df["fg_made_50plus"].fillna(0)
    fg50_attempts = df["fg_attempts_50plus"].fillna(0)
    fg_pct_50plus_shrunk = _shrink_pct(fg50_made, fg50_attempts, FG50_PRIOR_WEIGHT)
    # kickers with literally zero 50+ attempts get no information from that
    # bucket at all -- fall back fully to their (shrunk) overall fg_pct.
    fg_pct_50plus_shrunk = fg_pct_50plus_shrunk.where(fg50_attempts > 0, fg_pct_shrunk)

    # is_dome can arrive as a real bool, a numpy bool, or (after a CSV
    # round-trip) the literal strings "True"/"False" -- normalize all three
    # before casting to float, rather than assuming a clean bool dtype.
    is_dome_raw = df["is_dome"]
    if is_dome_raw.dtype == object:
        is_dome_raw = is_dome_raw.map({"True": True, "False": False, True: True, False: False})
    df["is_dome_numeric"] = is_dome_raw.astype(float)
    # Lower ADP = better; invert so higher adp_anchor = more valuable,
    # matching every other z-scored component here.
    df["adp_anchor"] = -df["adp"]

    z_fg_pct = zscore(fg_pct_shrunk)
    z_fg_pct_50plus = zscore(fg_pct_50plus_shrunk)
    z_dome = zscore(df["is_dome_numeric"].fillna(df["is_dome_numeric"].median()))
    z_offense = zscore(df["team_offense_ppg"].fillna(df["team_offense_ppg"].median()))
    z_adp = zscore(df["adp_anchor"].fillna(df["adp_anchor"].median()))

    composite = (
        z_fg_pct * K_COMPONENTS["fg_pct"]
        + z_fg_pct_50plus * K_COMPONENTS["fg_pct_50plus"]
        + z_dome * K_COMPONENTS["is_dome"]
        + z_offense * K_COMPONENTS["team_offense_ppg"]
        + z_adp * K_COMPONENTS["adp_anchor"]
    )
    df["value_score"] = composite
    return df


# Guide-blended K composite weights -- see module docstring "Tier 1" for
# the full reasoning. guide_ppg_25/guide_fg_acc (REAL 2025 numbers from
# Joel Smyth's Draft Guide 2026) dominate; the tier-2 play-by-play proxy
# terms (fg_pct/fg_pct_50plus/is_dome/team_offense_ppg) are kept but
# DE-WEIGHTED to secondary/tie-breaking weight rather than removed, per
# "real data beats proxy, but the proxy is a reasonable fallback." Sums to
# 1.0.
GUIDE_K_COMPONENTS = {
    "guide_ppg_25": 0.25,
    "guide_fg_acc": 0.20,
    "guide_off_rank_inv": 0.08,
    "guide_go_pct_rank": 0.05,
    "guide_is_50plus_good": 0.05,
    "guide_is_value": 0.05,
    "guide_is_dome": 0.03,
    "fg_pct": 0.10,           # de-weighted from 0.40 (tier 2)
    "fg_pct_50plus": 0.07,    # de-weighted from 0.25 (tier 2)
    "is_dome": 0.02,          # de-weighted from 0.10 (tier 2); mostly
                               # redundant with guide_is_dome above
    "team_offense_ppg": 0.05,  # de-weighted from 0.10 (tier 2)
    "adp_anchor": 0.05,        # de-weighted from 0.15 (tier 2) -- still a
                               # stabilizer, now a minor one since real
                               # guide data is doing most of the work
}


def compute_kicker_scores_with_guide(df: pd.DataFrame) -> pd.DataFrame:
    """Blended real-guide-data + play-by-play-proxy composite for K rows
    matched in data/guide_kicker_stats.csv (guide_off_rank non-null; see
    join.py). A kicker in this subset may or may not ALSO have a
    kicker_stats.csv proxy match -- if not, the proxy components below are
    simply null and get median-imputed via the same missing-data handling
    used everywhere else in this file (compute_stat_group_scores'
    fillna(median) + <30%-non-null weight redistribution pattern), so a
    guide-only kicker's score still leans entirely on real guide signal
    instead of being dragged toward a meaningless imputed proxy value.
    """
    df = df.copy()

    # Same Bayesian shrinkage as the proxy-only path above, reused here so
    # the de-weighted fg_pct/fg_pct_50plus terms are still stabilized
    # against small samples the same way.
    fg_pct_shrunk = _shrink_pct(df["fg_made"], df["fg_attempts"], FG_PCT_PRIOR_WEIGHT)
    fg50_made = df["fg_made_50plus"].fillna(0)
    fg50_attempts = df["fg_attempts_50plus"].fillna(0)
    fg_pct_50plus_shrunk = _shrink_pct(fg50_made, fg50_attempts, FG50_PRIOR_WEIGHT)
    fg_pct_50plus_shrunk = fg_pct_50plus_shrunk.where(fg50_attempts > 0, fg_pct_shrunk)
    df["fg_pct"] = fg_pct_shrunk
    df["fg_pct_50plus"] = fg_pct_50plus_shrunk

    df["is_dome"] = df["is_dome"].astype(float)
    df["guide_is_dome"] = df["guide_is_dome"].astype(float)
    df["adp_anchor"] = -df["adp"]

    # guide_off_rank: 1 = best projected 2026 offense. Invert so a better
    # (lower-number) offense rank scores higher, matching every other
    # z-scored component here.
    df["guide_off_rank_inv"] = -df["guide_off_rank"]
    # guide_go_pct_rank is used AS-IS, not inverted -- per
    # guide_kicker_stats.csv's own column semantics, a HIGH rank number
    # here means a conservative team that goes for the field goal more
    # often in FG range, which is good for the kicker (more attempts).

    available = {}
    for col, weight in GUIDE_K_COMPONENTS.items():
        non_null_frac = df[col].notna().mean()
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        available = {"guide_ppg_25": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    df["value_score"] = composite
    return df


# DEF composite weights -- see module docstring for the full reasoning
# behind each weight. guide_pr_roe_rank/guide_adj_ppg_25_rank (REAL analyst
# data from Joel Smyth's Draft Guide 2026) are now co-primary; the old
# play-by-play-derived proxy trio is kept but DE-WEIGHTED to a secondary/
# tie-breaking role (real data beats proxy, but the proxy stays alive as a
# fallback for any team the guide doesn't cover). Sums to 1.0.
DEF_COMPONENTS = {
    "guide_pr_roe_rank": 0.35,       # REAL pressure-rate rank (co-primary)
    "guide_adj_ppg_25_rank": 0.35,   # REAL adjusted-PPG rank (co-primary)
    "def_custom_adjusted_ppg": 0.15,  # de-weighted from 0.60
    "def_pressure_rate_proxy": 0.10,  # de-weighted from 0.20
    "def_turnover_volume": 0.05,      # de-weighted from 0.20
}

# Additive (not z-scored/weighted) modifiers layered on top of the
# DEF_COMPONENTS composite -- both are REAL analyst judgments from the
# guide that build_defense_stats.py's docstring flagged as gaps it could
# NOT derive from 2024 play-by-play data alone (opponent-offense strength
# faced, offseason roster trajectory). Binary/categorical, so they're
# applied as flat bonuses/maluses on the same rough +/-1-z-unit scale as
# the weighted components, rather than being z-scored themselves.
DEF_BOTTOM10_OFFENSE_MALUS = -0.15
DEF_TREND_MODIFIER = {"up": 0.05, "down": -0.05, "flat": 0.0}


def compute_defense_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Real per-team composite for DEF, replacing the old pure-ADP
    fallback. Requires def_custom_adjusted_ppg to be non-null (i.e. the
    team had a defense_stats.csv row) -- callers should route teams
    without a match to the ADP-only fallback instead, same as
    QB/RB/WR/TE's "without_stats" path. guide_def_stats.csv columns
    (guide_pr_roe_rank etc.) may be null for a team not in the guide --
    handled the same way as any other missing component (median-imputed,
    or its weight redistributed if missing for >70% of this group).
    """
    df = df.copy()

    df["def_turnover_volume"] = (
        df["def_sacks"].fillna(0)
        + df["def_interceptions"].fillna(0)
        + df["def_forced_fumbles"].fillna(0)
    )

    # guide_pr_roe_rank / guide_adj_ppg_25_rank: 1 = best. Invert so a
    # better (lower-number) rank scores higher, matching every other
    # z-scored component here.
    df["guide_pr_roe_rank_inv"] = -df["guide_pr_roe_rank"]
    df["guide_adj_ppg_25_rank_inv"] = -df["guide_adj_ppg_25_rank"]

    components = {
        "guide_pr_roe_rank_inv": DEF_COMPONENTS["guide_pr_roe_rank"],
        "guide_adj_ppg_25_rank_inv": DEF_COMPONENTS["guide_adj_ppg_25_rank"],
        "def_custom_adjusted_ppg": DEF_COMPONENTS["def_custom_adjusted_ppg"],
        "def_pressure_rate_proxy": DEF_COMPONENTS["def_pressure_rate_proxy"],
        "def_turnover_volume": DEF_COMPONENTS["def_turnover_volume"],
    }

    available = {}
    for col, weight in components.items():
        non_null_frac = df[col].notna().mean()
        if non_null_frac >= 0.3:
            available[col] = weight

    weight_sum = sum(available.values())
    if weight_sum == 0:
        available = {"def_custom_adjusted_ppg": 1.0}
        weight_sum = 1.0

    composite = pd.Series(np.zeros(len(df)), index=df.index)
    for col, weight in available.items():
        z = zscore(df[col].fillna(df[col].median()))
        composite += z * (weight / weight_sum)

    # Flat additive modifiers -- see DEF_BOTTOM10_OFFENSE_MALUS /
    # DEF_TREND_MODIFIER comments above. Teams with no guide match get 0
    # for both (fillna(0) / unmapped trend -> 0), i.e. no modifier applied.
    bottom10 = df["guide_bottom10_offense"].fillna(0)
    composite = composite + bottom10 * DEF_BOTTOM10_OFFENSE_MALUS
    trend_modifier = df["guide_trend"].map(DEF_TREND_MODIFIER).fillna(0.0)
    composite = composite + trend_modifier

    df["value_score"] = composite
    return df


def main():
    df = pd.read_csv(DATA_DIR / "joined.csv")

    scored_parts = []
    for pos, group in df.groupby("position"):
        group = group.copy()
        if pos == "K":
            # Three tiers -- see module docstring "For K, there are now
            # THREE tiers":
            #   1. guide-matched (guide_off_rank non-null): real-data-
            #      dominant blended composite, regardless of whether a
            #      proxy (fg_pct) match also exists.
            #   2. proxy-only (fg_pct non-null, no guide match): UNCHANGED
            #      compute_kicker_scores() -- the guide doesn't cover every
            #      kicker, so this stays a reasonable fallback.
            #   3. neither: ADP-only fallback, same as everywhere else.
            has_guide = group["guide_off_rank"].notna()
            has_stats = group["fg_pct"].notna()

            guide_rows = group[has_guide]
            proxy_only_rows = group[~has_guide & has_stats]
            no_stats_rows = group[~has_guide & ~has_stats]

            scored_pieces = []
            if len(guide_rows) > 0:
                scored_pieces.append(compute_kicker_scores_with_guide(guide_rows))
            if len(proxy_only_rows) > 0:
                scored_pieces.append(compute_kicker_scores(proxy_only_rows))

            with_stats = (
                pd.concat(scored_pieces, ignore_index=True)
                if scored_pieces
                else proxy_only_rows.copy()
            )

            if len(no_stats_rows) > 0:
                no_stats_rows = no_stats_rows.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                adp_rank = no_stats_rows["adp"].rank(method="min", ascending=True)
                no_stats_rows["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, no_stats_rows], ignore_index=True)
        elif pos in STAT_POSITIONS:
            # fantasy_points_ppr is used ONLY as an existence check here --
            # "did this player have a 2024 stat row at all" -- not as a
            # points value. Every actual point calculation in this file
            # uses the league's custom scoring (compute_passing_pts, plus
            # the rushing/receiving math above), never this column's value.
            # MIN_GAMES_FOR_STATS (not just > 0): a 1-2 game sample is not
            # a real signal -- see MIN_GAMES_FOR_STATS / sample_size_shrinkage
            # docstring above for the confirmed bug this fixes.
            has_2024_stats = group["fantasy_points_ppr"].notna() & (group["games"] >= MIN_GAMES_FOR_STATS)

            # Confirmed bug: a rookie with real 2025 guide data (e.g. Ashton
            # Jeanty -- guide big-board rank 16, guide_adj_ppg 14.5,
            # proj_volume_rank #4, high confidence) was being routed straight
            # into the pure-ADP fallback below purely because they have no
            # 2024 stat line, completely IGNORING that real guide signal.
            # A player with real guide data counts as having "stats" even
            # with zero 2024 games -- they go through
            # compute_stat_group_scores() same as everyone else; every
            # column built from 2024 raw stats will be NaN for them and get
            # median-imputed to a neutral (average) z-score by the existing
            # per-component fillna(median) logic below -- same missing-data
            # handling already used for any player missing any one
            # component -- while the real guide_adj_ppg/ol_run_block_rank/
            # proj_volume_rank columns contribute a genuine, non-neutral
            # signal. Only players with NEITHER 2024 stats NOR any guide
            # match remain in the pure ADP-only fallback.
            # Every real signal that feeds `components` in
            # compute_stat_group_scores() below MUST also be listed here --
            # a signal used in the composite but missing from this gate is
            # exactly the bug class that let Jaxson Dart (real 2025 data,
            # no 2024 stats) fall through to the ADP-only path despite
            # having real signal. This list includes every guide-derived
            # column that appears in ANY position's `components` dict:
            # guide_adj_ppg/real2025_total_pts/pct_pts_lost_to_luck (all
            # positions), guide_proj_volume_rank/pct_rb1_rank (RB),
            # playcaller_wr_ppg_rank (WR). guide_ol_run_block_rank_2025 is
            # no longer a components entry (absorbed into the RB ML model's
            # trained features -- see "ROUND 2" in the module docstring)
            # but is kept here too since it's still real signal a player
            # could have with nothing else.
            guide_cols_present = [c for c in [
                "guide_adj_ppg", "guide_ol_run_block_rank_2025", "guide_proj_volume_rank",
                "real2025_total_pts", "pct_pts_lost_to_luck", "pct_rb1_rank",
                "playcaller_wr_ppg_rank",
            ] if c in group.columns]
            has_guide_signal = pd.Series(False, index=group.index)
            for c in guide_cols_present:
                has_guide_signal = has_guide_signal | group[c].notna()

            has_stats = has_2024_stats | has_guide_signal
            with_stats = group[has_stats]
            without_stats = group[~has_stats]

            if len(with_stats) > 0:
                with_stats = compute_stat_group_scores(with_stats)
            if len(without_stats) > 0:
                # rookies/no-2024-stats players: ADP-only fallback so they
                # still show up in the file, ranked by ADP within position,
                # scored below the stat-based players in that group.
                without_stats = without_stats.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                # rank purely by adp (lower adp = better), map into a
                # value_score range strictly below the stat-based players
                adp_rank = without_stats["adp"].rank(method="min", ascending=True)
                without_stats["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, without_stats], ignore_index=True)
        else:
            # DEF: real stats-based composite for teams with a
            # defense_stats.csv match (def_custom_adjusted_ppg non-null);
            # any team without a match falls back to the ADP-only
            # treatment (lower ADP => higher value_score), same pattern as
            # K/QB/RB/WR/TE's "without_stats" path.
            has_stats = group["def_custom_adjusted_ppg"].notna()
            with_stats = group[has_stats]
            without_stats = group[~has_stats]

            if len(with_stats) > 0:
                with_stats = compute_defense_scores(with_stats)
            if len(without_stats) > 0:
                without_stats = without_stats.copy()
                min_score = with_stats["value_score"].min() if len(with_stats) else 0
                adp_rank = without_stats["adp"].rank(method="min", ascending=True)
                without_stats["value_score"] = min_score - 0.01 * adp_rank

            group = pd.concat([with_stats, without_stats], ignore_index=True)

        scored_parts.append(group)

    scored = pd.concat(scored_parts, ignore_index=True)

    scored["position_rank"] = (
        scored.groupby("position")["value_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    scored["adp_position_rank"] = (
        scored.groupby("position")["adp"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    scored["value_gap"] = scored["adp_position_rank"] - scored["position_rank"]

    scored = scored.sort_values(["position", "position_rank"]).reset_index(drop=True)

    out_path = DATA_DIR / "joined.csv"
    scored.to_csv(out_path, index=False)
    print(f"Scored {len(scored)} players. Wrote back to {out_path}")

    for pos in sorted(scored["position"].unique()):
        sub = scored[scored["position"] == pos]
        print(f"  {pos}: {len(sub)} players")


if __name__ == "__main__":
    sys.exit(main())
