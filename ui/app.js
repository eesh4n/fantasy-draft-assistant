/* ============================================================
   Fantasy Draft Assistant — vanilla JS, no build step, no server.
   ============================================================ */

// ---------- League config (hardcoded) ----------
const CONFIG = {
  scoring: "PPR",
  roster: {
    QB: 1,
    RB: 2,
    WR: 2,
    TE: 1,
    FLEX: 2,     // eligible: RB, WR, TE
    K: 1,
    DEF: 1,
    BENCH: 5
  },
  flexEligible: ["RB", "WR", "TE"],
  // order specific-slot-first, flex fallback, bench last
  slotPriority: {
    QB: ["QB", "BENCH"],
    RB: ["RB", "FLEX", "BENCH"],
    WR: ["WR", "FLEX", "BENCH"],
    TE: ["TE", "FLEX", "BENCH"],
    K: ["K", "BENCH"],
    DEF: ["DEF", "BENCH"]
  }
};

const SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BENCH"];
const STORAGE_KEY = "draftAssistant.state.v1";
const LEAGUE_SIZE_KEY = "draftAssistant.leagueSize.v1";

// ---------- Draft strategy (from Joel Smyth's Draft Guide 2026) ----------
// Round-by-round target is a MEDIAN plan, not a hard rule -- "Best Player
// Available still most important" per the guide. Shown as guidance only.
const ROUND_TARGETS = [
  "RB", "RB", "WR", "BPA (best player available)", "WR", "BPA", "BPA",
  "QB", "Upside WR", "Punt TE (or best value)", "Top Handcuff", "Upside QB",
  "Favorite Deep Sleeper", "D/ST", "Kicker / IR stash"
];

const POSITION_STRATEGY = {
  QB: {
    main: "Main target: ADP QB7-11 range -- snipe a favorite that falls (usually 2-3 still there in Round 8).",
    secondary: "Secondary: a late rushing QB (works ~70% of the time) -- Purdy/Nix count as volume plays here too.",
    note: "QB3-6 are going way later this year -- still worth waiting, but if one falls far it's likely a real value (QB3 2026 ADP ~55th overall; QB6 2025 ADP was ~59th)."
  },
  RB: {
    main: "Main target: leave the draft with 3 RBs from the top ~25-30. RB/RB in rounds 1-2 is how ~90% of league winners are built.",
    secondary: "RB30-40 is mostly a waste of a pick compared to that same range at QB/WR/TE.",
    note: "Prefer hunting backup/handcuff RBs late over a TE2 or early K/D-ST in most league formats."
  },
  WR: {
    main: "Main target: Round 3 & 5 WR range -- prefer WR over most RBs as a whole this range.",
    secondary: "The WR5-12 range specifically isn't a favorite compared to the RBs in that same round or the WRs available a round later.",
    note: "WR is the best position for hunting late-round upside."
  },
  TE: {
    main: "Main target: wait for best value -- take TE whenever it's genuinely the best player available, any range.",
    secondary: "TE2-4 is fine if you're stuck with no favorite RB/WR left. Round 7/8 is a good spot to grab the last mid-tier TE once RB/WR is dried up.",
    note: "Punting TE completely (and streaming/waiting) is a legitimate, working strategy."
  }
};

const OVERALL_RULES = [
  "Don't draft off rankings without understanding ADP -- use the ranking-vs-ADP gap to find real value, not just to follow a list.",
  "\"Don't beat ADP\": the goal isn't a safe RB30 who finishes RB29 -- take the riskier, higher-upside player if their true value potential is higher.",
  "No K or D/ST until the last two rounds, unless you're trying to intimidate the room.",
  "Good \"process\" picks late: rookie WRs, rushing QBs, talent on good offenses, and cemented RB2s/handcuffs.",
  "Balance risk across your roster -- avoid stacking several boom/bust or injury-prone profiles onto the same team.",
  "Waivers matter as much as the draft, all season -- stay active early and even right after the draft."
];

function getLeagueSize() {
  const stored = Number(localStorage.getItem(LEAGUE_SIZE_KEY));
  return stored && stored >= 4 && stored <= 20 ? stored : 12;
}

// Top picks matching the CURRENT round's target position (from
// ROUND_TARGETS), so the panel doesn't just say "target: RB" -- it names
// the 3 actual best available players for that target right now. For a
// non-positional round (BPA / Upside WR / Punt TE / etc), falls back to
// the existing overall recommendation ranking, since there's no single
// position to filter to.
function getRoundTargetPicks(target, n = 3) {
  const posMap = { RB: "RB", WR: "WR", QB: "QB", TE: "TE", "D/ST": "DEF" };
  const pos = posMap[target];
  const available = getAvailablePlayers().filter(p => p.team !== "FA");
  const openCounts = getOpenSlotCounts();
  const pool = pos ? available.filter(p => p.position === pos) : available;
  const scored = pool.map(p => ({ player: p, score: computeRecommendationScore(p, openCounts) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, n);
}

function renderStrategyPanel() {
  const totalPicks = draftedIds.size + mineIds.size;
  const leagueSize = getLeagueSize();
  const round = Math.floor(totalPicks / leagueSize) + 1;
  const target = ROUND_TARGETS[round - 1] || "Best player available";
  const roundEl = document.getElementById("strategyRound");
  const posNote = POSITION_STRATEGY[target] ? POSITION_STRATEGY[target].main : null;
  const picks = getRoundTargetPicks(target, 3);
  const picksHtml = picks.length
    ? `<div class="strategy-picks">
        <div class="strategy-picks-label">Top picks for this round:</div>
        ${picks.map(({ player }) => `
          <div class="strategy-pick-row">
            <span class="pos-pill pos-${player.position}">${player.position}</span>
            <span class="strategy-pick-name">${escapeHtml(player.name)}</span>
            <span class="strategy-pick-adp">ADP ${player.adp}</span>
          </div>
        `).join("")}
      </div>`
    : `<div class="strategy-picks-label">No available players match this round's target.</div>`;
  roundEl.innerHTML = `
    <div class="strategy-round-num">Round ${round}</div>
    <div class="strategy-round-target">Target: <strong>${escapeHtml(target)}</strong></div>
    ${posNote ? `<div class="strategy-round-note">${escapeHtml(posNote)}</div>` : ""}
    ${picksHtml}
  `;

  const bodyEl = document.getElementById("strategyRulesBody");
  const posBlocks = Object.entries(POSITION_STRATEGY).map(([pos, s]) => `
    <div class="strategy-pos-block">
      <div class="strategy-pos-name">${pos}</div>
      <div>${escapeHtml(s.main)}</div>
      <div class="strategy-pos-secondary">${escapeHtml(s.secondary)}</div>
      <div class="strategy-pos-note">${escapeHtml(s.note)}</div>
    </div>
  `).join("");
  bodyEl.innerHTML = `
    <div class="strategy-rules-list">
      ${OVERALL_RULES.map(r => `<div class="strategy-rule">${escapeHtml(r)}</div>`).join("")}
    </div>
    ${posBlocks}
  `;
}

// ---------- State ----------
// allPlayers: full player list as loaded from json (never mutated)
// draftedIds: Set of player ids taken by someone else
// mineIds: Set of player ids on your roster
// rosterSlots: { QB: [id|null], RB: [id|null, id|null], ... }
let allPlayers = [];
let draftedIds = new Set();
let mineIds = new Set();
let rosterSlots = buildEmptyRosterSlots();
let searchTerm = "";
let activePosFilter = "ALL";

function buildEmptyRosterSlots() {
  const slots = {};
  SLOT_ORDER.forEach(key => {
    slots[key] = new Array(CONFIG.roster[key]).fill(null);
  });
  return slots;
}

// ---------- Persistence ----------
function saveState() {
  const payload = {
    draftedIds: Array.from(draftedIds),
    mineIds: Array.from(mineIds),
    rosterSlots
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (e) {
    console.warn("Failed to save draft state", e);
  }
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const payload = JSON.parse(raw);
    draftedIds = new Set(payload.draftedIds || []);
    mineIds = new Set(payload.mineIds || []);
    rosterSlots = payload.rosterSlots || buildEmptyRosterSlots();
    // defensive: ensure all slot arrays exist with correct length
    SLOT_ORDER.forEach(key => {
      if (!Array.isArray(rosterSlots[key])) {
        rosterSlots[key] = new Array(CONFIG.roster[key]).fill(null);
      }
    });
    return true;
  } catch (e) {
    console.warn("Failed to load saved draft state", e);
    return false;
  }
}

function clearState() {
  localStorage.removeItem(STORAGE_KEY);
  draftedIds = new Set();
  mineIds = new Set();
  rosterSlots = buildEmptyRosterSlots();
}

// ---------- Data loading ----------
async function loadPlayers() {
  const badge = document.getElementById("dataSourceBadge");
  try {
    const res = await fetch("players.json", { cache: "no-store" });
    if (!res.ok) throw new Error("players.json not found");
    const data = await res.json();
    allPlayers = data;
    badge.textContent = "live data: players.json";
    return;
  } catch (e) {
    // fall back to mock data
    try {
      const res2 = await fetch("mock_players.json", { cache: "no-store" });
      const data2 = await res2.json();
      allPlayers = data2;
      badge.textContent = "mock data (standalone test)";
    } catch (e2) {
      badge.textContent = "no data found";
      allPlayers = [];
    }
  }
}

// ---------- Slot filling logic ----------
function findOpenSlotIndex(slotKey) {
  const arr = rosterSlots[slotKey];
  if (!arr) return -1;
  return arr.findIndex(v => v === null);
}

function assignPlayerToRoster(player) {
  const priority = CONFIG.slotPriority[player.position] || ["BENCH"];
  for (const slotKey of priority) {
    const idx = findOpenSlotIndex(slotKey);
    if (idx !== -1) {
      rosterSlots[slotKey][idx] = player.id;
      return slotKey;
    }
  }
  return null; // roster completely full, no slot available
}

function removePlayerFromRoster(playerId) {
  SLOT_ORDER.forEach(key => {
    rosterSlots[key] = rosterSlots[key].map(v => (v === playerId ? null : v));
  });
}

// ---------- Actions ----------
function markDrafted(playerId) {
  draftedIds.add(playerId);
  mineIds.delete(playerId);
  removePlayerFromRoster(playerId);
  saveState();
  renderAll();
}

function markMine(playerId) {
  const player = allPlayers.find(p => p.id === playerId);
  if (!player) return;
  mineIds.add(playerId);
  draftedIds.delete(playerId);
  assignPlayerToRoster(player);
  saveState();
  renderAll();
}

// Two modals (reset-confirm, player-detail) share the same full-screen
// overlay pattern. If both are ever open at once — e.g. Tab-key focus
// reaching an element behind an open modal and Enter/Space activating it,
// since neither modal traps focus — they stack, and the top one eats all
// clicks meant for the one behind it, making everything look "broken".
// closeAllOverlays() is called before opening any modal so only one can
// ever be visible at a time.
function closeAllOverlays() {
  document.getElementById("resetConfirmOverlay").hidden = true;
  document.getElementById("playerDetailOverlay").hidden = true;
}

// Native confirm() is unreliable here — some browsers/embedded webviews
// silently suppress it (click does nothing, no error), which is exactly
// what made the Reset button appear broken. Using an in-page overlay
// instead removes that dependency entirely.
function openResetConfirm() {
  closeAllOverlays();
  document.getElementById("resetConfirmOverlay").hidden = false;
}

function closeResetConfirm() {
  document.getElementById("resetConfirmOverlay").hidden = true;
}

function resetDraft() {
  clearState();
  closeResetConfirm();
  renderAll();
}

// ---------- Player detail / rationale ----------
function fmtStat(v, suffix = "") {
  if (v === null || v === undefined) return "—";
  const rounded = suffix === "%" ? Math.round(v) : Math.round(v * 10) / 10;
  return `${rounded}${suffix}`;
}

function buildRationale(player) {
  const lines = [];
  const s = player.stats || {};
  const hasStats = s.ppg !== null && s.ppg !== undefined;

  if (s.chart_note) {
    lines.push(`Chart read (${s.chart_source}): ${s.chart_note}`);
  }

  if (s.stat_note) {
    lines.push(`Analyst insight: ${s.stat_note}`);
  }

  if (player.position === "K" || player.position === "DEF") {
    lines.push(
      `${player.position === "K" ? "Kickers" : "Team defenses"} don't have individual production stats to model, so this ranking is based on ADP alone — it just mirrors expert consensus rather than an independent read.`
    );
  } else if (!hasStats) {
    lines.push(
      `No 2024 stat line exists for ${player.name} (rookie or no games played), so there's nothing to compare production against. This ranking is ADP-only and deliberately placed below every stat-based player at the position — a low or negative value gap here is a data limitation, not a real "fade" signal.`
    );
  } else {
    lines.push(
      `In 2024, ${player.name} averaged ${fmtStat(s.ppg)} PPR points/game on ${fmtStat(s.volume)} touches or targets/game, playing ${fmtStat(s.snap_share !== null ? Math.round(s.snap_share * 100) : null, "%")} of offensive snaps — producing ${fmtStat(s.efficiency)} points per opportunity.`
    );
    if (s.ml_predicted_ppg !== null && s.ml_predicted_ppg !== undefined) {
      lines.push(
        `The model, trained on 7 past seasons of NFL data (2018–2024), projects them for ${fmtStat(s.ml_predicted_ppg)} points/game next season — this prediction, not a hand-picked formula, now drives most of their ranking here.`
      );
    }
    if (s.rushing_ppg !== null && s.rushing_ppg !== undefined && s.receiving_ppg !== null && s.receiving_ppg !== undefined) {
      lines.push(
        `Of that, ${fmtStat(s.rushing_ppg)} pts/game came on the ground and ${fmtStat(s.receiving_ppg)} pts/game came through the air (receptions count extra in this PPR league) — with a ${fmtStat(s.td_rate !== null && s.td_rate !== undefined ? s.td_rate * 100 : null, "%")} touchdown rate per touch/target, a rough proxy for goal-line role.`
      );
    } else if (s.rushing_ppg !== null && s.rushing_ppg !== undefined) {
      lines.push(
        `Of that, ${fmtStat(s.rushing_ppg)} pts/game came from rushing — a real signal for dual-threat value beyond pure pass-attempt volume.`
      );
    }
    if (player.position === "RB" && s.playcaller_rb_ppg_rank !== null && s.playcaller_rb_ppg_rank !== undefined) {
      lines.push(
        `Their offense's playcaller has historically produced the #${Math.round(s.playcaller_rb_ppg_rank)} RB PPG in the league — real system context on top of ${player.name}'s own numbers.`
      );
    } else if (player.position === "WR" && s.playcaller_wr_ppg_rank !== null && s.playcaller_wr_ppg_rank !== undefined) {
      lines.push(
        `Their offense's playcaller has historically produced the #${Math.round(s.playcaller_wr_ppg_rank)} WR PPG in the league — real system context on top of ${player.name}'s own numbers.`
      );
    }
    if (player.value_gap > 10) {
      lines.push(
        `That production ranks #${player.position_rank} among ${player.position}s, but ADP has them going like the #${player.adp_position_rank} — a gap of ${player.value_gap} spots. The market is pricing them behind what their own numbers support: a value pick.`
      );
    } else if (player.value_gap < -10) {
      lines.push(
        `That production only ranks #${player.position_rank} among ${player.position}s, yet ADP has them going like the #${player.adp_position_rank} — priced ${Math.abs(player.value_gap)} spots ahead of what their numbers support. Could be a name/hype premium, or a real role change since 2024 the model can't see.`
      );
    } else {
      lines.push(
        `Production rank (#${player.position_rank}) and ADP rank (#${player.adp_position_rank}) are close — the market has this one about right.`
      );
    }
  }
  return lines;
}

function openPlayerDetail(player) {
  closeAllOverlays();
  const content = document.getElementById("playerDetailContent");
  const gapClass = player.value_gap > 0 ? "gap-positive" : (player.value_gap < 0 ? "gap-neg" : "");
  const gapText = player.value_gap > 0 ? `+${player.value_gap}` : `${player.value_gap}`;
  const rationale = buildRationale(player);
  const s = player.stats || {};

  content.innerHTML = `
    <h2 class="detail-title">${escapeHtml(player.name)}
      <span class="pos-pill pos-${player.position}">${player.position}</span>
    </h2>
    <div class="detail-sub">${escapeHtml(player.team)} · ADP ${player.adp} (position #${player.adp_position_rank}) · value rank #${player.position_rank} · gap <span class="${gapClass}">${gapText}</span></div>

    <div class="detail-stat-grid">
      <div class="detail-stat"><span class="detail-stat-label">PPG (PPR)</span><span class="detail-stat-value">${fmtStat(s.ppg)}</span></div>
      <div class="detail-stat"><span class="detail-stat-label">Volume/game</span><span class="detail-stat-value">${fmtStat(s.volume)}</span></div>
      <div class="detail-stat"><span class="detail-stat-label">Snap share</span><span class="detail-stat-value">${fmtStat(s.snap_share !== null && s.snap_share !== undefined ? Math.round(s.snap_share * 100) : null, "%")}</span></div>
      <div class="detail-stat"><span class="detail-stat-label">Pts/opportunity</span><span class="detail-stat-value">${fmtStat(s.efficiency)}</span></div>
      ${s.rushing_ppg !== null && s.rushing_ppg !== undefined ? `<div class="detail-stat"><span class="detail-stat-label">Rushing pts/game</span><span class="detail-stat-value">${fmtStat(s.rushing_ppg)}</span></div>` : ""}
      ${s.receiving_ppg !== null && s.receiving_ppg !== undefined ? `<div class="detail-stat"><span class="detail-stat-label">Receiving pts/game</span><span class="detail-stat-value">${fmtStat(s.receiving_ppg)}</span></div>` : ""}
      ${s.td_rate !== null && s.td_rate !== undefined ? `<div class="detail-stat"><span class="detail-stat-label">TD rate/opportunity</span><span class="detail-stat-value">${fmtStat(s.td_rate * 100, "%")}</span></div>` : ""}
    </div>

    <div class="detail-rationale">
      ${rationale.map(l => `<p>${escapeHtml(l)}</p>`).join("")}
    </div>

    <p class="detail-note">Stats are 2024 season per-game averages (last completed season). "Gap" = ADP position rank minus value-model position rank.</p>
  `;

  document.getElementById("playerDetailOverlay").hidden = false;
}

function closePlayerDetail() {
  document.getElementById("playerDetailOverlay").hidden = true;
}

// ---------- Derived data ----------
function getAvailablePlayers() {
  return allPlayers.filter(p => !draftedIds.has(p.id) && !mineIds.has(p.id));
}

function getOpenSlotCounts() {
  const open = {};
  SLOT_ORDER.forEach(key => {
    open[key] = rosterSlots[key].filter(v => v === null).length;
  });
  return open;
}

// Need score per position: how many currently-open slots a player
// of this position could fill (specific slot + FLEX if eligible).
// Higher need score => that position is scarce in your roster right now.
function positionNeedScore(position, openCounts) {
  if (position === "QB") return openCounts.QB;
  if (position === "K") return openCounts.K;
  if (position === "DEF") return openCounts.DEF;
  if (CONFIG.flexEligible.includes(position)) {
    return openCounts[position] + openCounts.FLEX;
  }
  return 0;
}

// Recommendation score:
//  base = value_score weighted with value_gap (sleepers get a boost)
//  needMultiplier = scales the base score by how badly you need this
//                   position right now. Positions with open starting
//                   slots are weighted much higher than ones already
//                   filled (bench-only need gets a small bump too).
function computeRecommendationScore(player, openCounts) {
  // Recommendations must track overall draft order first (ADP), with
  // value_gap as a small nudge — not the reverse. An earlier version
  // scaled value_score/value_gap directly, which let a round-6/7 sleeper
  // (e.g. WR rank #3 with a +31 gap) outscore the actual #1 overall pick
  // (gap 0, since ADP already prices them correctly) — nonsense advice at
  // the top of a draft, since you'd never draft a round-7 player 1st
  // overall for a "value" argument. Anchoring on -adp keeps recommendations
  // in realistic draft order; value_gap only breaks ties/nudges within a
  // similar tier, clamped so it can never flip the order across rounds.
  const adpScore = 600 - Math.min(player.adp, 600); // higher = earlier ADP
  const clampedGap = Math.max(-20, Math.min(20, player.value_gap));
  const base = adpScore + clampedGap * 2;
  const needScore = positionNeedScore(player.position, openCounts);
  let needMultiplier;
  if (needScore > 0) {
    // each open relevant starting slot adds 35% weight, capped
    needMultiplier = 1 + Math.min(needScore, 3) * 0.35;
    // RB-scarcity premium: per analyst commentary (data/README.md), RB
    // positional value dries up much faster than WR early in a draft --
    // last season's top-20 PPG players skewed heavily RB, flipping to
    // mostly WR by picks 21-25. A small flat boost while an RB/FLEX slot
    // is still open nudges recommendations toward "take RBs early" without
    // ever overriding a clearly better player at a different position.
    if (player.position === "RB") {
      needMultiplier += 0.15;
    }
  } else if (openCounts.BENCH > 0) {
    // no starting slot open, but bench space exists — mild interest
    needMultiplier = 1.0;
  } else {
    // roster fully built out for this position, low priority
    needMultiplier = 0.55;
  }
  return base * needMultiplier;
}

function getTopRecommendations(n = 5) {
  // team "FA" = unsigned free agent, no active roster spot on a real NFL
  // team. Their value_gap is a data artifact (good 2024 stats, but no
  // 2026 role) — see data/README.md. Excluded from recommendations since
  // they're not a real draft pick, even though they stay visible in the
  // full player table for reference.
  const available = getAvailablePlayers().filter(p => p.team !== 'FA');
  const openCounts = getOpenSlotCounts();
  const scored = available.map(p => ({
    player: p,
    score: computeRecommendationScore(p, openCounts)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, n);
}

// ---------- Rendering ----------
function renderAll() {
  renderTable();
  renderRoster();
  renderRecommendations();
  renderStrategyPanel();
}

function matchesFilters(player) {
  if (activePosFilter !== "ALL" && player.position !== activePosFilter) return false;
  if (!searchTerm) return true;
  const term = searchTerm.toLowerCase();
  return (
    player.name.toLowerCase().includes(term) ||
    player.position.toLowerCase().includes(term) ||
    player.team.toLowerCase().includes(term)
  );
}

function renderTable() {
  const tbody = document.getElementById("playerTableBody");
  const emptyState = document.getElementById("emptyState");
  tbody.innerHTML = "";

  // Default (ALL) view sorts by overall ADP, like a normal draft board —
  // interleaving by position_rank there produced a confusing round-robin
  // (every position's #1, then every position's #2, ...). Within a single
  // position filter, position_rank is more useful, so keep that sort there.
  const visible = allPlayers
    .filter(matchesFilters)
    .sort((a, b) => {
      if (activePosFilter === "ALL") {
        return a.adp - b.adp;
      }
      return a.position_rank - b.position_rank || b.value_score - a.value_score;
    });

  if (visible.length === 0) {
    emptyState.style.display = "block";
  } else {
    emptyState.style.display = "none";
  }

  const frag = document.createDocumentFragment();

  visible.forEach(player => {
    const isDrafted = draftedIds.has(player.id);
    const isMine = mineIds.has(player.id);

    const tr = document.createElement("tr");
    if (isDrafted) tr.classList.add("drafted-row");
    if (isMine) tr.classList.add("mine-row");

    const gapClass = player.value_gap > 0 ? "gap-positive" : (player.value_gap < 0 ? "gap-neg" : "");
    const gapText = player.value_gap > 0 ? `+${player.value_gap}` : `${player.value_gap}`;

    tr.innerHTML = `
      <td class="col-name"><span class="player-name player-name-link">${escapeHtml(player.name)}</span></td>
      <td><span class="pos-pill pos-${player.position}">${player.position}</span></td>
      <td>${escapeHtml(player.team)}</td>
      <td>#${player.position_rank}</td>
      <td>${player.adp}</td>
      <td class="gap-cell"><span class="${gapClass}">${gapText}</span></td>
      <td>${player.stats && player.stats.guide_adj_ppg != null ? player.stats.guide_adj_ppg : "—"}</td>
      <td class="actions-cell"></td>
    `;

    tr.querySelector(".player-name-link").addEventListener("click", () => openPlayerDetail(player));

    const actionsCell = tr.querySelector(".actions-cell");

    if (isDrafted) {
      const span = document.createElement("span");
      span.textContent = "Drafted";
      span.style.color = "var(--text-dim)";
      span.style.fontSize = "12px";
      actionsCell.appendChild(span);
    } else if (isMine) {
      const span = document.createElement("span");
      span.textContent = "✓ Mine";
      span.style.color = "var(--accent)";
      span.style.fontWeight = "700";
      span.style.fontSize = "12px";
      actionsCell.appendChild(span);
    } else {
      const mineBtn = document.createElement("button");
      mineBtn.className = "action-btn mine-btn";
      mineBtn.textContent = "Mine";
      mineBtn.addEventListener("click", () => markMine(player.id));

      const draftedBtn = document.createElement("button");
      draftedBtn.className = "action-btn drafted-btn";
      draftedBtn.textContent = "Drafted";
      draftedBtn.addEventListener("click", () => markDrafted(player.id));

      actionsCell.appendChild(mineBtn);
      actionsCell.appendChild(draftedBtn);
    }

    frag.appendChild(tr);
  });

  tbody.appendChild(frag);
}

const SLOT_LABELS = {
  QB: "QB", RB: "RB", WR: "WR", TE: "TE",
  FLEX: "FLEX", K: "K", DEF: "DEF", BENCH: "BENCH"
};

function renderRoster() {
  const container = document.getElementById("rosterSlots");
  container.innerHTML = "";

  SLOT_ORDER.forEach(slotKey => {
    rosterSlots[slotKey].forEach((playerId, idx) => {
      const row = document.createElement("div");
      row.className = "roster-slot" + (playerId ? " filled" : "");

      const label = document.createElement("span");
      label.className = "slot-label";
      const count = CONFIG.roster[slotKey] > 1 ? ` ${idx + 1}` : "";
      label.textContent = SLOT_LABELS[slotKey] + count;

      const value = document.createElement("span");
      if (playerId) {
        const p = allPlayers.find(pl => pl.id === playerId);
        value.className = "slot-player";
        value.textContent = p ? `${p.name} (${p.position})` : playerId;
      } else {
        value.className = "slot-empty";
        value.textContent = "open";
      }

      row.appendChild(label);
      row.appendChild(value);
      container.appendChild(row);
    });
  });
}

function renderRecommendations() {
  const container = document.getElementById("recList");
  container.innerHTML = "";

  const recs = getTopRecommendations(5);

  if (recs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No available players.";
    container.appendChild(empty);
    return;
  }

  recs.forEach(({ player, score }) => {
    const item = document.createElement("div");
    item.className = "rec-item";

    const main = document.createElement("div");
    main.className = "rec-main";

    const name = document.createElement("div");
    name.className = "rec-name rec-name-link";
    name.textContent = `${player.name}`;
    name.addEventListener("click", () => openPlayerDetail(player));

    const meta = document.createElement("div");
    meta.className = "rec-meta";
    const gapText = player.value_gap > 0 ? ` · +${player.value_gap} value` : "";
    meta.textContent = `${player.position} · ${player.team} · rank #${player.position_rank}${gapText}`;

    main.appendChild(name);
    main.appendChild(meta);

    const scoreEl = document.createElement("div");
    scoreEl.className = "rec-score";
    scoreEl.textContent = score.toFixed(0);

    const btn = document.createElement("button");
    btn.className = "rec-mine-btn";
    btn.textContent = "Mine";
    btn.addEventListener("click", () => markMine(player.id));

    item.appendChild(main);
    item.appendChild(scoreEl);
    item.appendChild(btn);
    container.appendChild(item);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------- Wiring ----------
function wireControls() {
  const searchInput = document.getElementById("searchInput");
  searchInput.addEventListener("input", e => {
    searchTerm = e.target.value.trim();
    renderTable();
  });

  const posFilters = document.getElementById("posFilters");
  posFilters.addEventListener("click", e => {
    const btn = e.target.closest(".pos-chip");
    if (!btn) return;
    activePosFilter = btn.dataset.pos;
    posFilters.querySelectorAll(".pos-chip").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    renderTable();
  });

  document.getElementById("resetBtn").addEventListener("click", openResetConfirm);
  document.getElementById("resetCancelBtn").addEventListener("click", closeResetConfirm);
  document.getElementById("resetConfirmBtn").addEventListener("click", resetDraft);
  const resetOverlay = document.getElementById("resetConfirmOverlay");
  resetOverlay.addEventListener("click", e => {
    if (e.target === resetOverlay) closeResetConfirm();
  });

  document.getElementById("playerDetailCloseBtn").addEventListener("click", closePlayerDetail);
  const detailOverlay = document.getElementById("playerDetailOverlay");
  detailOverlay.addEventListener("click", e => {
    if (e.target === detailOverlay) closePlayerDetail();
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeAllOverlays();
  });

  const leagueSizeInput = document.getElementById("leagueSizeInput");
  leagueSizeInput.value = getLeagueSize();
  leagueSizeInput.addEventListener("input", () => {
    const v = Number(leagueSizeInput.value);
    if (v >= 4 && v <= 20) {
      localStorage.setItem(LEAGUE_SIZE_KEY, String(v));
      renderStrategyPanel();
    }
  });

  const rulesToggle = document.getElementById("strategyRulesToggle");
  const rulesBody = document.getElementById("strategyRulesBody");
  rulesToggle.addEventListener("click", () => {
    rulesBody.hidden = !rulesBody.hidden;
    rulesToggle.textContent = rulesBody.hidden
      ? "Show full rules & position notes"
      : "Hide full rules & position notes";
  });
}

// ---------- Init ----------
(async function init() {
  wireControls();
  loadState();
  await loadPlayers();
  renderAll();
})();
