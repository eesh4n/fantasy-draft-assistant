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
  const roundPlanHtml = `
    <div class="strategy-round-plan">
      <div class="strategy-pos-name">Round-by-round plan</div>
      ${ROUND_TARGETS.map((t, i) => `
        <div class="strategy-round-plan-row${i + 1 === round ? " current" : ""}">
          <span class="strategy-round-plan-num">R${i + 1}</span>
          <span>${escapeHtml(t)}</span>
        </div>
      `).join("")}
    </div>
  `;
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
    ${roundPlanHtml}
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
let compareIds = new Set();
let favoriteIds = new Set();
let sortColumn = "adp"; // "adp" | "valrank" | "pos" | "team" | "gap" | "ppg"
let sortDir = 1; // 1 = ascending (best first for both adp and valrank)
// True once the user has actually clicked a sort header. Before that, the
// table uses an implicit default (ADP asc in the ALL view, position_rank
// asc within a position filter) rather than whatever sortColumn happens to
// be initialized to -- keeps the header arrow from pointing at a column
// that isn't actually driving the current sort.
let userSorted = false;

const FAVORITES_KEY = "draftAssistant.favorites.v1";
// Seeded once on first load only -- the user's own personal late-round
// sleeper picks (their call, not the model's), per an explicit request to
// save these as a watchlist. If the user un-favorites one, it stays
// removed (seeding only fires when no saved favorites exist yet at all).
const DEFAULT_FAVORITE_NAMES = [
  "Mike Washington", "De'Zhaun Stribling", "KC Concepcion", "Travis Etienne",
  "Jonah Coleman", "Keaton Mitchell", "Malik Willis", "Jalen Coker",
];

function loadFavorites() {
  try {
    const raw = localStorage.getItem(FAVORITES_KEY);
    if (raw) {
      favoriteIds = new Set(JSON.parse(raw));
      return;
    }
  } catch (e) { /* fall through to seeding */ }
  favoriteIds = new Set();
  DEFAULT_FAVORITE_NAMES.forEach(n => {
    const p = allPlayers.find(p => p.name.toLowerCase().includes(n.toLowerCase().split(" ")[0]) && p.name.toLowerCase().includes(n.toLowerCase().split(" ").slice(-1)[0]));
    if (p) favoriteIds.add(p.id);
  });
  saveFavorites();
}

function saveFavorites() {
  try {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(Array.from(favoriteIds)));
  } catch (e) { console.warn("Failed to save favorites", e); }
}

function toggleFavorite(playerId) {
  if (favoriteIds.has(playerId)) favoriteIds.delete(playerId);
  else favoriteIds.add(playerId);
  saveFavorites();
  renderTable();
}

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

// Undo a mistaken Mine/Drafted click. Before this existed, the ONLY way to
// walk back a wrong click was the nuclear "Reset Draft" (wipes everything).
// removePlayerFromRoster() already correctly frees the roster slot -- it
// was just never wired to a button for this direction.
function unmarkPlayer(playerId) {
  draftedIds.delete(playerId);
  mineIds.delete(playerId);
  removePlayerFromRoster(playerId);
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
  document.getElementById("compareOverlay").hidden = true;
  document.getElementById("tradeOverlay").hidden = true;
}

// ---------- Compare players ----------
function renderCompareBar() {
  const bar = document.getElementById("compareBar");
  const count = document.getElementById("compareBarCount");
  if (compareIds.size < 2) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  count.textContent = `${compareIds.size} selected`;
}

const COMPARE_ROWS = [
  { label: "Position", get: p => p.position, higherBetter: null },
  { label: "Team", get: p => p.team, higherBetter: null },
  { label: "Value rank (pos)", get: p => p.position_rank, higherBetter: false },
  { label: "ADP", get: p => p.adp, higherBetter: false },
  // Players with NEITHER a 2024 stat line NOR any real guide/real-2025
  // signal get an artificial (often deeply negative) value_gap purely as an
  // ADP-fallback ranking mechanism (see score.py / the "hasNoStats"
  // handling in renderTable) -- it's a data limitation, not a real
  // "overpriced" signal. Mirror renderTable's hasNoStats check (ppg OR
  // guide_adj_ppg OR real2025_total_pts OR pct_pts_lost_to_luck OR
  // proj_volume_rank -- score.py's full PLAYER_LEVEL_GUIDE_COLS gate)
  // instead of ppg alone, so a rookie with real guide/2025 signal (e.g.
  // Ashton Jeanty, or Jeremiyah Love who qualifies via proj_volume_rank
  // alone) shows their real gap.
  { label: "Value gap", get: p => (p.stats && (p.stats.ppg != null || p.stats.guide_adj_ppg != null || p.stats.real2025_total_pts != null || p.stats.pct_pts_lost_to_luck != null || p.stats.proj_volume_rank != null) ? p.value_gap : null), higherBetter: true },
  { label: "Model value score", get: p => p.value_score, higherBetter: true },
  { label: "ML predicted PPG", get: p => (p.stats && p.stats.ml_predicted_ppg != null ? p.stats.ml_predicted_ppg : null), higherBetter: true },
  { label: "'25 adj PPG (guide)", get: p => (p.stats && p.stats.guide_adj_ppg != null ? p.stats.guide_adj_ppg : null), higherBetter: true },
  { label: "2024 PPG (base)", get: p => (p.stats ? p.stats.ppg : null), higherBetter: true },
  { label: "Snap share", get: p => (p.stats && p.stats.snap_share != null ? Math.round(p.stats.snap_share * 100) + "%" : null), higherBetter: null },
];

function openCompareModal() {
  closeAllOverlays();
  const players = allPlayers.filter(p => compareIds.has(p.id));
  if (players.length < 2) return;

  const content = document.getElementById("compareContent");
  const headerCells = players.map(p => `<th>${escapeHtml(p.name)}</th>`).join("");

  const rowsHtml = COMPARE_ROWS.map(row => {
    const values = players.map(p => row.get(p));
    let bestIdx = -1;
    if (row.higherBetter !== null) {
      const numeric = values.map(v => (typeof v === "number" ? v : null));
      if (numeric.some(v => v !== null)) {
        const best = row.higherBetter
          ? Math.max(...numeric.filter(v => v !== null))
          : Math.min(...numeric.filter(v => v !== null));
        bestIdx = numeric.indexOf(best);
      }
    }
    const cells = values.map((v, i) => {
      const text = v === null || v === undefined ? "—" : v;
      const cls = i === bestIdx ? "compare-best" : "";
      return `<td class="${cls}">${escapeHtml(String(text))}</td>`;
    }).join("");
    return `<tr><th>${escapeHtml(row.label)}</th>${cells}</tr>`;
  }).join("");

  // Verdict: same-position comparisons use value_score directly (it's
  // z-scored within position, so directly comparable there). Cross-
  // position comparisons are flagged as directional-only, since
  // value_score's z-score normalization isn't calibrated to be precisely
  // comparable ACROSS positions (a WR's z-score distribution isn't the
  // same shape as a QB's).
  const samePos = players.every(p => p.position === players[0].position);
  const sorted = [...players].sort((a, b) => b.value_score - a.value_score);
  const winner = sorted[0];
  let verdict = `The model prefers <strong>${escapeHtml(winner.name)}</strong> (highest value_score: ${winner.value_score.toFixed(2)}).`;
  if (!samePos) {
    verdict += ` Note: these players are at different positions -- value_score is normalized within each position separately, so this comparison is directional, not precisely calibrated across positions. Treat it as "the model rates this one higher relative to their own position peers," not a strict points comparison.`;
  }

  content.innerHTML = `
    <h2 style="margin:0 0 14px;">Compare Players</h2>
    <div style="overflow-x:auto;">
      <table class="compare-table">
        <thead><tr><th></th>${headerCells}</tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
    <div class="compare-verdict">${verdict}</div>
  `;

  document.getElementById("compareOverlay").hidden = false;
}

function clearCompare() {
  compareIds.clear();
  renderCompareBar();
  renderTable();
}

// ---------- Trade Calculator ----------
// Free, built-in alternative to paid tools (WalterFootball/RotoBot-style
// trade graders) using data this app already has -- no external API/auth.
//
// The core problem with a naive "sum value_score on each side" trade
// grader: value_score is z-scored WITHIN each position (see COMPARE_ROWS
// comment above), so a top TE and a top QB aren't on a truly comparable
// scale -- QB is deep (many playable options), TE craters fast after the
// first few names. Summing raw value_score would treat a TE-for-QB swap
// as if both positions were equally scarce, which they're not.
//
// Fix: convert every player's value_score into VALUE OVER REPLACEMENT
// (VOR) before comparing anything -- the standard fantasy-analytics way
// to make cross-position value comparable. Replacement level = the
// value_score of the last realistic starter at that position, derived
// from this league's actual roster requirements (CONFIG) and league
// size, not a hardcoded assumption. A position that craters hard after
// its starters (e.g. TE) will show a low replacement level relative to a
// deep position (e.g. WR), so a mid-tier TE naturally scores a bigger
// VOR than an equivalently-ranked WR -- scarcity falls out of the real
// data instead of being hand-tuned.
let tradeGiveIds = new Set();
let tradeReceiveIds = new Set();
// The trade partner's CURRENT full roster (not part of the trade itself)
// -- optional context so the calculator can judge whether a trade would
// actually make sense for THEM too, not just you. Without this, the
// personalized read only ever reasons about your own needs, which can't
// tell you whether the other manager would realistically accept.
let tradeTheirIds = new Set();

const TRADE_HISTORY_KEY = "draftAssistant.tradeHistory.v1";
const TRADE_HISTORY_MAX = 20;

// Lightweight save/load recall for evaluated trades -- NOT a permanent
// archive (capped at TRADE_HISTORY_MAX, oldest dropped first), just a way
// to revisit or compare a trade you already looked at. Names are stored
// alongside ids so a saved trade still displays sensibly even if a player
// is later removed from the dataset (e.g. offseason roster file refresh).
function loadTradeHistory() {
  try {
    const raw = localStorage.getItem(TRADE_HISTORY_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) { console.warn("Failed to load trade history", e); }
  return [];
}

function saveTradeHistory(list) {
  try {
    localStorage.setItem(TRADE_HISTORY_KEY, JSON.stringify(list));
  } catch (e) { console.warn("Failed to save trade history", e); }
}

function saveCurrentTrade() {
  if (tradeGiveIds.size === 0 || tradeReceiveIds.size === 0) return;
  const nameFor = id => {
    const p = allPlayers.find(pl => pl.id === id);
    return p ? p.name : id;
  };
  const entry = {
    id: `trade_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    giveIds: Array.from(tradeGiveIds),
    receiveIds: Array.from(tradeReceiveIds),
    theirIds: Array.from(tradeTheirIds),
    savedAt: new Date().toISOString(),
    giveNames: Array.from(tradeGiveIds).map(nameFor),
    receiveNames: Array.from(tradeReceiveIds).map(nameFor),
  };
  const history = loadTradeHistory();
  history.push(entry);
  while (history.length > TRADE_HISTORY_MAX) history.shift();
  saveTradeHistory(history);
  renderTradeCalculator();
}

function loadSavedTrade(entryId) {
  const history = loadTradeHistory();
  const entry = history.find(e => e.id === entryId);
  if (!entry) return;
  const validIds = new Set(allPlayers.map(p => p.id));
  tradeGiveIds = new Set((entry.giveIds || []).filter(id => validIds.has(id)));
  tradeReceiveIds = new Set((entry.receiveIds || []).filter(id => validIds.has(id)));
  tradeTheirIds = new Set((entry.theirIds || []).filter(id => validIds.has(id)));
  renderTradeCalculator();
}

function deleteSavedTrade(entryId) {
  const history = loadTradeHistory().filter(e => e.id !== entryId);
  saveTradeHistory(history);
  renderTradeCalculator();
}

// Simple day-granularity relative time (e.g. "today", "3d ago") falling
// back to a short date once it's old enough that "Nd ago" stops being
// useful at a glance.
function formatTradeSavedAt(iso) {
  const saved = new Date(iso);
  if (isNaN(saved.getTime())) return "";
  const now = new Date();
  const dayMs = 24 * 60 * 60 * 1000;
  const diffDays = Math.floor((new Date(now.toDateString()) - new Date(saved.toDateString())) / dayMs);
  if (diffDays <= 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return saved.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function tradeHistoryItemHtml(entry) {
  const give = (entry.giveNames || []).join(", ") || "—";
  const receive = (entry.receiveNames || []).join(", ") || "—";
  const when = formatTradeSavedAt(entry.savedAt);
  return `
    <div class="trade-history-item" data-id="${entry.id}">
      <div class="trade-history-summary">
        <span class="trade-history-line">Give: ${escapeHtml(give)} &rarr; Receive: ${escapeHtml(receive)}</span>
        <span class="trade-history-when">saved ${escapeHtml(when)}</span>
      </div>
      <div class="trade-history-actions">
        <button class="btn trade-history-load-btn" data-id="${entry.id}" type="button">Load</button>
        <button class="btn btn-danger trade-history-delete-btn" data-id="${entry.id}" type="button">Delete</button>
      </div>
    </div>
  `;
}

// Pure (non-mutating) version of assignPlayerToRoster/findOpenSlotIndex --
// simulates filling a SEPARATE roster (the trade partner's) using the
// exact same CONFIG.slotPriority logic your own roster uses, without
// touching the global rosterSlots. Order matters for a real draft (who
// picked what when), but for a need ESTIMATE from a final roster list,
// slot-filling order doesn't change the final open-slot counts as long as
// there's enough room for every real starter -- CONFIG's roster sizes are
// applied identically either way.
function simulateOpenSlotCounts(playerIds) {
  const slots = buildEmptyRosterSlots();
  const players = allPlayers.filter(p => playerIds.has(p.id));
  players.forEach(player => {
    const priority = CONFIG.slotPriority[player.position] || ["BENCH"];
    for (const slotKey of priority) {
      const idx = slots[slotKey] ? slots[slotKey].findIndex(v => v === null) : -1;
      if (idx !== -1) { slots[slotKey][idx] = player.id; break; }
    }
  });
  const open = {};
  SLOT_ORDER.forEach(key => { open[key] = slots[key].filter(v => v === null).length; });
  return open;
}

// How the league's 2 FLEX (RB/WR/TE-eligible) slots get split when
// computing each position's effective starter count -- a simplifying
// heuristic (real flex usage skews RB > WR > TE), not derived from data,
// since this app doesn't track league-wide flex-usage stats. Documented
// here as an assumption, same as other estimated constants in this file.
const FLEX_SHARE = { RB: 0.5, WR: 0.35, TE: 0.15 };

function effectiveStarterCount(position, leagueSize) {
  const base = CONFIG.roster[position] || 0;
  const flexCut = FLEX_SHARE[position] || 0;
  const starters = base + CONFIG.roster.FLEX * flexCut;
  return Math.max(1, Math.round(starters * leagueSize));
}

// Replacement value per position, memoized per call (allPlayers/leagueSize
// don't change mid-render). Falls back to the worst available value_score
// at that position if the league is small enough that the replacement
// rank exceeds the real player pool.
function computeReplacementLevels() {
  const leagueSize = getLeagueSize();
  const byPos = {};
  ["QB", "RB", "WR", "TE", "K", "DEF"].forEach(pos => {
    const pool = allPlayers
      .filter(p => p.position === pos && typeof p.value_score === "number")
      .sort((a, b) => b.value_score - a.value_score);
    if (!pool.length) { byPos[pos] = 0; return; }
    const rank = effectiveStarterCount(pos, leagueSize);
    const idx = Math.min(rank, pool.length - 1);
    byPos[pos] = pool[idx].value_score;
  });
  return byPos;
}

function vorFor(player, replacementLevels) {
  const repl = replacementLevels[player.position] ?? 0;
  return (player.value_score ?? 0) - repl;
}

// Reuses the same "how badly do you need this position" scale as the
// recommendation engine, so trade personalization and draft
// recommendations reason about need identically.
function tradeNeedMultiplier(position, openCounts) {
  const needScore = positionNeedScore(position, openCounts);
  if (needScore > 0) return 1 + Math.min(needScore, 3) * 0.35;
  if (openCounts.BENCH > 0) return 1.0;
  return 0.55;
}

// Consolidation/"best player" bonus: fantasy rosters are capped, so
// landing the single best player in a trade is worth more than the raw
// VOR sum implies -- you can't split a bench spot across two mediocre
// players into one great one after the fact. Applied symmetrically to
// both sides (whoever holds the single highest-VOR player in the whole
// trade gets the bonus on their side), not just declared for a "winner".
const CONSOLIDATION_WEIGHT = 0.15;

function weightedVorTotal(withVorArray, openCounts) {
  return withVorArray.reduce(
    (sum, x) => sum + x.vor * tradeNeedMultiplier(x.player.position, openCounts), 0
  );
}

function computeTradeSide(ids, replacementLevels, openCounts) {
  const players = allPlayers.filter(p => ids.has(p.id));
  const withVor = players.map(p => ({ player: p, vor: vorFor(p, replacementLevels) }));
  const rawTotal = withVor.reduce((sum, x) => sum + x.vor, 0);
  const personalizedTotal = weightedVorTotal(withVor, openCounts);
  const bestVor = withVor.length ? Math.max(...withVor.map(x => x.vor)) : 0;
  return { players: withVor, rawTotal, personalizedTotal, bestVor };
}

// Neutral (non-roster-weighted) VOR diff, including the consolidation
// bonus -- this is the same "rawDiff" buildTradeVerdict uses for its
// band() label. Pulled out to a standalone helper so findSweetener (and
// anything else that needs to test "what if we added player X") can
// recompute this exact number without duplicating/drifting from the
// verdict's own math.
function computeRawDiff(give, receive) {
  const giveAdj = give.rawTotal + CONSOLIDATION_WEIGHT * give.bestVor;
  const receiveAdj = receive.rawTotal + CONSOLIDATION_WEIGHT * receive.bestVor;
  return receiveAdj - giveAdj;
}

// ---------- Trade sweetener suggestions ----------
// If the trade currently favors the OTHER side (rawDiff < -0.15, matching
// buildTradeVerdict's own "slightly/clearly favors them" bands), the fix
// is for the user to ask to RECEIVE more -- not give more -- so this
// searches the trade partner's CURRENT roster (tradeTheirIds, the only
// visibility this app has into what they could offer as bait) for the
// smallest possible addition to "You Receive" that brings the trade back
// to roughly even. Smallest on purpose: a real trade partner is more
// likely to part with their least valuable spare piece than their best
// bench player, so we search from the low end and stop at the first (i.e.
// lowest-VOR) player that actually closes the gap, rather than suggesting
// whichever partner player is most valuable overall.
//
// If the trade instead clearly favors the USER (rawDiff > 0.6), there's
// no gap to close -- but a lopsided offer is also the kind of offer a
// real manager is likely to decline or lowball-counter. In that case,
// suggest sweetening FROM the user's own bench (rosterSlots) with their
// single lowest-VOR spare player, as a lighter-touch "make it more likely
// to get accepted" note.
function findSweetener(give, receive, replacementLevels, rawDiff) {
  if (rawDiff < -0.15) {
    if (tradeTheirIds.size === 0) {
      return { direction: "need_their_roster" };
    }

    const giveAdj = give.rawTotal + CONSOLIDATION_WEIGHT * give.bestVor;
    const candidates = allPlayers
      .filter(p => tradeTheirIds.has(p.id) && !tradeGiveIds.has(p.id) && !tradeReceiveIds.has(p.id))
      .map(p => ({ player: p, vor: vorFor(p, replacementLevels) }))
      .sort((a, b) => a.vor - b.vor);

    for (const cand of candidates) {
      const newRawTotal = receive.rawTotal + cand.vor;
      const newBestVor = Math.max(receive.bestVor, cand.vor);
      const newReceiveAdj = newRawTotal + CONSOLIDATION_WEIGHT * newBestVor;
      const newDiff = newReceiveAdj - giveAdj;
      if (newDiff >= -0.15) {
        return { direction: "add_to_receive", player: cand.player, newDiff };
      }
    }
    return null; // nothing on their roster (alone) closes the gap
  }

  if (rawDiff > 0.6) {
    const usedIds = new Set([...tradeGiveIds, ...tradeReceiveIds]);
    const benchIds = [];
    SLOT_ORDER.forEach(key => {
      rosterSlots[key].forEach(id => {
        if (id && !usedIds.has(id)) benchIds.push(id);
      });
    });
    const spare = allPlayers
      .filter(p => benchIds.includes(p.id))
      .map(p => ({ player: p, vor: vorFor(p, replacementLevels) }))
      .sort((a, b) => a.vor - b.vor)[0];
    if (!spare) return null;
    return { direction: "add_to_give", player: spare.player, newDiff: null };
  }

  return null;
}

function renderTradeSweetener(give, receive, replacementLevels, rawDiff) {
  // Only meaningful once both sides actually have players -- rawDiff is
  // undefined/garbage otherwise (buildTradeVerdict bails out earlier too).
  if (give.players.length === 0 || receive.players.length === 0) return "";

  const sweetener = findSweetener(give, receive, replacementLevels, rawDiff);
  if (!sweetener) return "";

  if (sweetener.direction === "need_their_roster") {
    return `<div class="trade-sweetener">💡 Add their roster below to see which of their bench players would even out this trade.</div>`;
  }
  if (sweetener.direction === "add_to_receive") {
    const vor = vorFor(sweetener.player, replacementLevels);
    return `<div class="trade-sweetener">💡 <strong>Fair-trade sweetener:</strong> ask them to also include ${escapeHtml(sweetener.player.name)} (${escapeHtml(sweetener.player.position)}, VOR ${vor.toFixed(2)}) in "You Receive" -- that's the smallest addition from their current roster that would bring this trade to roughly even (${sweetener.newDiff >= 0 ? "+" : ""}${sweetener.newDiff.toFixed(2)}).</div>`;
  }
  if (sweetener.direction === "add_to_give") {
    return `<div class="trade-sweetener">💡 This trade clearly favors you -- consider tossing in ${escapeHtml(sweetener.player.name)} (${escapeHtml(sweetener.player.position)}) from your bench to make the offer more likely to actually get accepted.</div>`;
  }
  return "";
}

// theirOpenCounts is null when no partner roster has been entered --
// two-sided evaluation is optional context, not required to use the
// calculator at all.
function buildTradeVerdict(give, receive, theirOpenCounts, replacementLevels) {
  const bothEmpty = give.players.length === 0 && receive.players.length === 0;
  if (bothEmpty) {
    return { verdict: "Add players to both sides to evaluate a trade.", detail: "" };
  }
  if (give.players.length === 0 || receive.players.length === 0) {
    return { verdict: "Add at least one player on each side.", detail: "" };
  }

  const rawDiff = computeRawDiff(give, receive);

  const givePers = give.personalizedTotal + CONSOLIDATION_WEIGHT * give.bestVor;
  const receivePers = receive.personalizedTotal + CONSOLIDATION_WEIGHT * receive.bestVor;
  const persDiff = receivePers - givePers;

  // Fairness bands scaled off the spread of real VOR values in this
  // dataset (top players land roughly 1.5-2.5 VOR above replacement) --
  // a diff under ~0.15 reads as a wash, not a real edge either way.
  const band = v => (v > 0.6 ? "clearly favors you" : v > 0.15 ? "slightly favors you" : v < -0.6 ? "clearly favors them" : v < -0.15 ? "slightly favors them" : "is roughly even");

  const giveNames = give.players.map(x => x.player.name).join(", ");
  const receiveNames = receive.players.map(x => x.player.name).join(", ");

  let verdict = `Scarcity-adjusted, this trade <strong>${band(rawDiff)}</strong> (neutral read, no roster context: ${rawDiff >= 0 ? "+" : ""}${rawDiff.toFixed(2)} value units).`;

  const lines = [];
  lines.push(`You give up ${escapeHtml(giveNames)} (VOR total ${give.rawTotal.toFixed(2)}) for ${escapeHtml(receiveNames)} (VOR total ${receive.rawTotal.toFixed(2)}).`);
  // Dynamic, not a canned line: names the ACTUAL positions in THIS trade
  // and their real computed replacement levels, instead of always citing
  // the same "mid-tier TE vs mid-tier QB" example regardless of what's
  // actually being traded.
  const allPositions = [...give.players, ...receive.players].map(x => x.player.position);
  const distinctPositions = [...new Set(allPositions)];
  if (distinctPositions.length <= 1) {
    lines.push(`Every player here is the same position (${distinctPositions[0] || "n/a"}), so this trade doesn't cross a scarcity gap -- VOR and raw value_score point the same direction in this specific case, unlike a cross-position trade where they can diverge.`);
  } else {
    const replParts = distinctPositions.map(pos => `${pos} ${(replacementLevels[pos] ?? 0).toFixed(2)}`).join(", ");
    lines.push(`This trade crosses positions (${distinctPositions.join(" / ")}) -- in this league, the last realistic starter's value_score sits at ${replParts}. That real gap between positions (not an assumption) is what VOR subtracts out before comparing anyone here.`);
  }
  if (give.bestVor > receive.bestVor + 0.05) {
    lines.push(`You're giving up the single best player in this trade (highest VOR) -- that consolidation cost is factored in, since one true stud is worth more than the sum-of-parts suggests.`);
  } else if (receive.bestVor > give.bestVor + 0.05) {
    lines.push(`You're picking up the single best player in this trade -- that consolidation value is factored in on your side.`);
  }
  if (Math.abs(persDiff - rawDiff) > 0.1) {
    // Bug found via live testing: describing this as "better/worse than
    // the neutral read" reads as contradicting the band() label right
    // next to it -- e.g. "clearly favors you (+1.55)... worse for you"
    // when persDiff (1.55) is smaller than rawDiff (2.65) but still
    // clearly positive. Only call out a genuine reversal (sign flip)
    // explicitly; otherwise describe the edge as growing/shrinking, which
    // can't contradict the band label since the direction word always
    // matches which way persDiff actually moved.
    const signFlipped = Math.sign(persDiff) !== Math.sign(rawDiff) && rawDiff !== 0 && persDiff !== 0;
    const movedTowardYou = persDiff > rawDiff;
    let roosterNote;
    if (signFlipped) {
      roosterNote = `this actually flips the verdict once your roster is factored in`;
    } else {
      roosterNote = movedTowardYou
        ? `the edge in your favor grows once your roster is factored in, since it fills an open need`
        : `the edge shrinks once your roster is factored in, since you already have that position covered`;
    }
    lines.push(`Factoring in YOUR current roster needs specifically, this trade <strong>${band(persDiff)}</strong> (${persDiff >= 0 ? "+" : ""}${persDiff.toFixed(2)}) -- ${roosterNote}.`);
  } else {
    lines.push(`Your current roster needs don't meaningfully change this read -- the personalized and neutral verdicts agree.`);
  }

  if (theirOpenCounts) {
    // From the trade partner's side: they're giving up the "You Receive"
    // set and getting the "You Give" set -- weighted by THEIR open
    // counts, not yours, so this actually estimates whether they'd have
    // real incentive to accept, not just whether the trade is good for you.
    const theirGain = weightedVorTotal(give.players, theirOpenCounts) + CONSOLIDATION_WEIGHT * give.bestVor;
    const theirCost = weightedVorTotal(receive.players, theirOpenCounts) + CONSOLIDATION_WEIGHT * receive.bestVor;
    const theirDiff = theirGain - theirCost;
    const theirBand = theirDiff > 0.6 ? "a clear win for them" : theirDiff > 0.15 ? "a mild win for them" : theirDiff < -0.6 ? "a clear loss for them" : theirDiff < -0.15 ? "a mild loss for them" : "roughly even for them";
    lines.push(`Based on the roster you entered for them, this trade looks like <strong>${theirBand}</strong> (${theirDiff >= 0 ? "+" : ""}${theirDiff.toFixed(2)} from their side) -- ${theirDiff < -0.15 ? "they may be reluctant to accept as-is, since it doesn't address their own roster needs well." : "they'd likely have real incentive to accept."}`);
  } else {
    lines.push(`Add their current roster below for a two-sided read -- right now this only evaluates whether the trade is good for YOU, not whether they'd actually want it.`);
  }

  // Playoff SOS isn't blended into the VOR math (see sosBadge() below for
  // why), but when a trade is otherwise too close to call on value alone,
  // a notably tough/easy Weeks 15-17 slate is a legitimate qualitative
  // tiebreaker worth flagging -- it just shouldn't drown out a trade
  // that's already a clear win or loss on value.
  if (Math.abs(rawDiff) <= 0.6) {
    const giveSos = mostExtremeSos(give.players);
    const receiveSos = mostExtremeSos(receive.players);
    const isNotable = x => x && (x.player.stats.sos_playoff_rank <= 8 || x.player.stats.sos_playoff_rank >= 25);
    const giveNotable = isNotable(giveSos);
    const receiveNotable = isNotable(receiveSos);
    if (receiveNotable && (!giveNotable || Math.abs(receiveSos.player.stats.sos_playoff_rank - 16.5) > Math.abs(giveSos.player.stats.sos_playoff_rank - 16.5))) {
      const rank = receiveSos.player.stats.sos_playoff_rank;
      const descriptor = rank <= 8 ? "a tough" : "an easy";
      lines.push(`This trade is close on value, so it's worth weighing playoff schedule: ${escapeHtml(receiveSos.player.name)} (you'd receive) has ${descriptor} Weeks 15-17 slate (#${rank}), which matters more in a close call like this than it would in a lopsided trade.`);
    } else if (giveNotable) {
      const rank = giveSos.player.stats.sos_playoff_rank;
      const descriptor = rank <= 8 ? "a tough" : "an easy";
      lines.push(`This trade is close on value, so it's worth weighing playoff schedule: ${escapeHtml(giveSos.player.name)} (you'd give up) has ${descriptor} Weeks 15-17 slate (#${rank}), which matters more in a close call like this than it would in a lopsided trade.`);
    }
  }

  return { verdict, detail: lines.join(" ") };
}

// Returns the {player, vor} entry (from a computeTradeSide().players array)
// whose sos_playoff_rank is most extreme (closest to 1 or 32), or null if
// none of the players have SOS data. Used to pick a single most-relevant
// callout rather than enumerating every player's schedule.
function mostExtremeSos(playersWithVor) {
  let best = null, bestDist = -1;
  for (const x of playersWithVor) {
    const rank = x.player.stats && x.player.stats.sos_playoff_rank;
    if (rank == null) continue;
    const dist = Math.abs(rank - 16.5);
    if (dist > bestDist) { best = x; bestDist = dist; }
  }
  return best;
}

// Team-level SOS context, not blended into VOR -- this pipeline has no
// weekly player projections to honestly combine schedule strength with,
// so it's shown as a separate informational flag only. Rank 1 = toughest
// schedule in the league, 32 = easiest.
function sosBadge(player) {
  const rank = player.stats && player.stats.sos_playoff_rank;
  if (rank == null) return "";
  let cls = "sos-neutral", label = `Playoff SOS #${rank}`;
  if (rank <= 8) { cls = "sos-tough"; label = `Tough playoff SOS #${rank}`; }
  else if (rank >= 25) { cls = "sos-easy"; label = `Easy playoff SOS #${rank}`; }
  return `<span class="sos-badge ${cls}" title="Strength of schedule for weeks 15-17, based on opponents' real points-allowed-per-game -- #1 = toughest, #32 = easiest">${label}</span>`;
}

function tradePlayerRow(player, vor, side) {
  // Shows the RAW numbers VOR is derived from, not just the abstracted
  // VOR total -- value_score (this app's core composite: ML prediction +
  // guide data + real 2025 production, same number shown everywhere else
  // in the app) and real2025_total_pts (actual box-score points this
  // season, when available) alongside the scarcity-adjusted VOR, so
  // nothing is hidden behind the derived metric.
  const rawScore = typeof player.value_score === "number" ? player.value_score.toFixed(2) : "—";
  const real2025 = player.stats && player.stats.real2025_total_pts != null ? player.stats.real2025_total_pts : null;
  const realLine = real2025 != null ? ` · ${real2025} real '25 pts` : "";
  return `
    <div class="trade-player-row" data-id="${player.id}" data-side="${side}">
      <span class="pos-pill pos-${player.position}">${player.position}</span>
      <span class="trade-player-name">${escapeHtml(player.name)}</span>
      <span class="trade-player-team">${escapeHtml(player.team)}</span>
      <span class="trade-player-values" title="value_score = this app's core composite (ML prediction + guide data + real 2025 production); VOR = value_score minus the replacement-level player at this position/league size">value_score ${rawScore} · VOR ${vor.toFixed(2)}${realLine}</span>
      ${sosBadge(player)}
      <button class="trade-remove-btn" data-id="${player.id}" data-side="${side}" type="button" aria-label="Remove">&times;</button>
    </div>
  `;
}

function tradeSideSet(side) {
  if (side === "give") return tradeGiveIds;
  if (side === "receive") return tradeReceiveIds;
  return tradeTheirIds;
}

// Lightweight row for the "their roster" context list -- no VOR/value
// display, since these players aren't part of the trade being evaluated,
// just context for estimating the OTHER team's positional needs.
function tradeTheirRosterRow(player) {
  return `
    <div class="trade-player-row" data-id="${player.id}" data-side="their">
      <span class="pos-pill pos-${player.position}">${player.position}</span>
      <span class="trade-player-name">${escapeHtml(player.name)}</span>
      <span class="trade-player-team">${escapeHtml(player.team)}</span>
      <button class="trade-remove-btn" data-id="${player.id}" data-side="their" type="button" aria-label="Remove">&times;</button>
    </div>
  `;
}

function renderTradeCalculator() {
  const replacementLevels = computeReplacementLevels();
  const openCounts = getOpenSlotCounts();
  const give = computeTradeSide(tradeGiveIds, replacementLevels, openCounts);
  const receive = computeTradeSide(tradeReceiveIds, replacementLevels, openCounts);
  const theirOpenCounts = tradeTheirIds.size > 0 ? simulateOpenSlotCounts(tradeTheirIds) : null;
  const { verdict, detail } = buildTradeVerdict(give, receive, theirOpenCounts, replacementLevels);
  const rawDiff = computeRawDiff(give, receive);
  const sweetenerHtml = renderTradeSweetener(give, receive, replacementLevels, rawDiff);
  const theirPlayers = allPlayers.filter(p => tradeTheirIds.has(p.id));
  const tradeHistory = loadTradeHistory();

  const content = document.getElementById("tradeContent");
  content.innerHTML = `
    <h2 style="margin:0 0 6px;">Trade Calculator</h2>
    <p class="trade-subtitle">Free, built into this app -- no external sync. Each player shows both the raw value_score (this app's core composite) and VOR (that same value_score adjusted for positional scarcity) -- the verdict below uses VOR, since raw value_score isn't directly comparable across positions, but nothing is hidden.</p>
    <div class="trade-feature-badges">
      <span class="trade-feature-badge" title="Converts every player's value_score into VOR, so a scarce position (TE) isn't compared unfairly against a deep one (QB).">📊 Position scarcity</span>
      <span class="trade-feature-badge" title="Whoever lands the single highest-VOR player gets a bonus -- one stud beats two mediocre players.">💎 Consolidation bonus</span>
      <span class="trade-feature-badge" title="Re-weighted using YOUR actual open roster slots -- not a generic fairness score.">🎯 Your roster, personalized</span>
      <span class="trade-feature-badge" title="Add the other team's roster to see if THEY'D actually want this trade too.">🔄 Two-sided read</span>
      <span class="trade-feature-badge" title="Close on value? We surface the player with the toughest or easiest Weeks 15-17 schedule as a tiebreaker.">🗓️ Playoff SOS tiebreaker</span>
      <span class="trade-feature-badge" title="If a trade's lopsided, we suggest the smallest sweetener that would even it out.">💡 Fair-trade finder</span>
      <span class="trade-feature-badge" title="Save any trade you're evaluating and reload it later to compare offers.">💾 Trade history</span>
    </div>
    <div class="trade-columns">
      <div class="trade-side">
        <h3>You Give</h3>
        <input class="trade-search" data-side="give" type="text" placeholder="Search a player to add…" autocomplete="off">
        <div class="trade-search-results" data-side="give"></div>
        <div class="trade-player-list">${give.players.map(x => tradePlayerRow(x.player, x.vor, "give")).join("") || '<div class="trade-empty">No players added</div>'}</div>
      </div>
      <div class="trade-side">
        <h3>You Receive</h3>
        <input class="trade-search" data-side="receive" type="text" placeholder="Search a player to add…" autocomplete="off">
        <div class="trade-search-results" data-side="receive"></div>
        <div class="trade-player-list">${receive.players.map(x => tradePlayerRow(x.player, x.vor, "receive")).join("") || '<div class="trade-empty">No players added</div>'}</div>
      </div>
    </div>
    <div class="trade-verdict">${verdict}</div>
    ${tradeGiveIds.size > 0 && tradeReceiveIds.size > 0 ? `<button class="btn trade-save-btn" type="button">Save this trade</button>` : ""}
    <div class="trade-detail">${detail}</div>
    ${sweetenerHtml}
    <div class="trade-their-roster">
      <h3>Their Current Roster <span class="trade-optional-tag">optional -- for a two-sided fairness read</span></h3>
      <input class="trade-search" data-side="their" type="text" placeholder="Add players on the OTHER team's current roster…" autocomplete="off">
      <div class="trade-search-results" data-side="their"></div>
      <div class="trade-player-list trade-their-list">${theirPlayers.map(tradeTheirRosterRow).join("") || '<div class="trade-empty">No players added -- the verdict above only evaluates the trade for you until this is filled in</div>'}</div>
    </div>
    <div class="trade-history">
      <h3>Saved Trades</h3>
      <div class="trade-history-list">${tradeHistory.length ? tradeHistory.slice().reverse().map(tradeHistoryItemHtml).join("") : '<div class="trade-empty">No saved trades yet</div>'}</div>
    </div>
    <div class="trade-scope-note">Playoff-week (15-17) strength of schedule is shown per player above, based on real opponent points-allowed data, and called out below when it's close enough to matter -- not blended into the VOR math itself (this app has no weekly player-projection model to combine it with honestly). Not covered here: Dynasty/keeper/draft-pick value (redraft-only model), and syncing a real league from Sleeper/ESPN/Yahoo -- evaluate any trade by adding the players manually above.</div>
  `;

  content.querySelectorAll(".trade-search").forEach(input => {
    input.addEventListener("input", () => renderTradeSearchResults(input.dataset.side, input.value));
  });
  content.querySelectorAll(".trade-remove-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tradeSideSet(btn.dataset.side).delete(btn.dataset.id);
      renderTradeCalculator();
    });
  });
  const saveBtn = content.querySelector(".trade-save-btn");
  if (saveBtn) {
    saveBtn.addEventListener("click", () => saveCurrentTrade());
  }
  content.querySelectorAll(".trade-history-load-btn").forEach(btn => {
    btn.addEventListener("click", () => loadSavedTrade(btn.dataset.id));
  });
  content.querySelectorAll(".trade-history-delete-btn").forEach(btn => {
    btn.addEventListener("click", () => deleteSavedTrade(btn.dataset.id));
  });
}

function renderTradeSearchResults(side, term) {
  const box = document.querySelector(`.trade-search-results[data-side="${side}"]`);
  if (!box) return;
  const q = term.trim().toLowerCase();
  if (!q) { box.innerHTML = ""; return; }
  const excludeIds = tradeSideSet(side);
  const matches = allPlayers
    .filter(p => !excludeIds.has(p.id) && p.name.toLowerCase().includes(q))
    .slice(0, 8);
  box.innerHTML = matches.map(p => `
    <div class="trade-search-result" data-id="${p.id}" data-side="${side}">
      <span class="pos-pill pos-${p.position}">${p.position}</span>
      ${escapeHtml(p.name)} <span class="trade-search-result-team">${escapeHtml(p.team)}</span>
    </div>
  `).join("");
  box.querySelectorAll(".trade-search-result").forEach(el => {
    el.addEventListener("click", () => {
      tradeSideSet(el.dataset.side).add(el.dataset.id);
      renderTradeCalculator();
    });
  });
}

function openTradeCalculator() {
  closeAllOverlays();
  renderTradeCalculator();
  document.getElementById("tradeOverlay").hidden = false;
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
      `No 2024 stat line exists for ${player.name} (rookie or no games played), so there's nothing to compare base production against. ${s.real2025_total_pts !== null && s.real2025_total_pts !== undefined ? `Real 2025 production does exist though — ${fmtStat(s.real2025_total_pts)} total fantasy points this season, pulled directly from actual box scores — and that's what's driving this ranking instead.` : `This ranking is ADP-only and deliberately placed below every stat-based player at the position — a low or negative value gap here is a data limitation, not a real "fade" signal.`}`
    );
  } else {
    if (s.real2025_total_pts !== null && s.real2025_total_pts !== undefined) {
      lines.push(
        `So far in 2025, ${player.name} has ${fmtStat(s.real2025_total_pts)} total fantasy points — real, actual production from this season's box scores (not a projection), pulled directly from NFL.com. This is now one of the strongest signals in their ranking.`
      );
    }
    lines.push(
      `Their 2024 base line: ${fmtStat(s.ppg)} PPR points/game on ${fmtStat(s.volume)} touches or targets/game, playing ${fmtStat(s.snap_share !== null ? Math.round(s.snap_share * 100) : null, "%")} of offensive snaps — producing ${fmtStat(s.efficiency)} points per opportunity.`
    );
    if (s.ml_predicted_ppg !== null && s.ml_predicted_ppg !== undefined) {
      lines.push(
        `The model, trained on 7 past seasons of NFL data (2018–2024), projects them for ${fmtStat(s.ml_predicted_ppg)} points/game — this prediction, blended with real 2025 production where available, now drives most of their ranking here.`
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
        `That production only ranks #${player.position_rank} among ${player.position}s, yet ADP has them going like the #${player.adp_position_rank} — priced ${Math.abs(player.value_gap)} spots ahead of what their numbers support. ${s.real2025_total_pts !== null && s.real2025_total_pts !== undefined ? "That gap already accounts for real 2025 production, so it's a genuine disagreement with the market, not just stale data." : "Could be a name/hype premium, or a real role change this season the model can't see without 2025 data for this player."}`
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

  const arrow = sortDir === 1 ? " ↑" : " ↓";
  const SORT_LABELS = { valrank: "Val Rank", adp: "ADP", pos: "Pos", team: "Team", gap: "Gap", ppg: "'25 PPG" };
  Object.entries({ sortValRank: "valrank", sortAdp: "adp", sortPos: "pos", sortTeam: "team", sortGap: "gap", sortPpg: "ppg" })
    .forEach(([elId, col]) => {
      // Only show an arrow once the user actually picked a column -- before
      // that, sortColumn still holds its init value ("adp") even though
      // the applied default sort may really be position_rank (filtered
      // view), so showing the arrow there would point at the wrong header.
      document.getElementById(elId).textContent = SORT_LABELS[col] + (userSorted && sortColumn === col ? arrow : "");
    });

  // Sortable by clicking the ADP or Val Rank column header (sortColumn/
  // sortDir). Default: ADP ascending in the ALL view (a normal draft
  // board -- interleaving by position_rank there produced a confusing
  // round-robin of every position's #1, then every position's #2, ...);
  // position_rank ascending within a single position filter. Either can
  // be overridden by clicking a header, which applies regardless of the
  // current position filter.
  // null/undefined always sort to the bottom regardless of direction, so
  // e.g. sorting by '25 PPG doesn't shove every guide-uncovered player to
  // the top when sorted ascending.
  const SORT_GETTERS = {
    valrank: p => p.position_rank,
    adp: p => p.adp,
    gap: p => p.value_gap,
    ppg: p => (p.stats ? p.stats.guide_adj_ppg : null),
    pos: p => p.position,
    team: p => p.team,
  };
  const visible = allPlayers
    .filter(matchesFilters)
    .sort((a, b) => {
      // Before the user clicks any header, use the implicit default
      // (see comment above): ADP asc in ALL, position_rank asc filtered.
      // sortColumn/sortDir are ignored here -- they still hold whatever
      // they were initialized/left at, not a real user choice.
      if (!userSorted) {
        if (activePosFilter === "ALL") return a.adp - b.adp;
        return a.position_rank - b.position_rank || b.value_score - a.value_score;
      }
      if (!SORT_GETTERS[sortColumn]) {
        return a.position_rank - b.position_rank || b.value_score - a.value_score;
      }
      const get = SORT_GETTERS[sortColumn];
      const av = get(a), bv = get(b);
      if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
      if (bv === null || bv === undefined) return -1;
      if (typeof av === "string") return sortDir * av.localeCompare(bv);
      return sortDir * (av - bv) || (b.value_score - a.value_score);
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

    // A player with NEITHER 2024 stats NOR any real guide/real-2025 signal
    // (see score.py's has_stats gate) is ranked purely by ADP in a fallback
    // tier BELOW every stat-based player at that position, so their raw
    // value_gap number looks like a real "way overpriced" red flag when
    // it's actually just "no production data exists yet to rank against."
    // This used to key off stats.ppg alone, which was accurate back when
    // no-2024-stats players were ALWAYS in the ADP-only fallback -- but
    // score.py now scores a rookie with real guide_adj_ppg/real2025_total_pts
    // (e.g. Ashton Jeanty) through the full composite despite having no
    // 2024 ppg, so their value_gap IS real signal now. Check for any of the
    // signals score.py's PLAYER_LEVEL_GUIDE_COLS gate actually uses, not
    // just ppg/guide_adj_ppg/real2025_total_pts -- confirmed bug: that
    // narrower 3-column check missed pct_pts_lost_to_luck and
    // proj_volume_rank, which alone can also qualify a zero-2024-stats
    // rookie for the full composite (e.g. Jeremiyah Love, Jadarian Price --
    // real proj_volume_rank, nothing else), wrongly showing "no data" for
    // players who do have real ranking signal.
    const hasNoStats = !player.stats || (
      player.stats.ppg == null &&
      player.stats.guide_adj_ppg == null &&
      player.stats.real2025_total_pts == null &&
      player.stats.pct_pts_lost_to_luck == null &&
      player.stats.proj_volume_rank == null
    );
    const gapClass = player.value_gap > 0 ? "gap-positive" : (player.value_gap < 0 ? "gap-neg" : "");
    const gapText = hasNoStats
      ? "no data"
      : (player.value_gap > 0 ? `+${player.value_gap}` : `${player.value_gap}`);
    const gapCellClass = hasNoStats ? "gap-nodata" : gapClass;

    const isFav = favoriteIds.has(player.id);
    const isCompared = compareIds.has(player.id);
    tr.innerHTML = `
      <td class="col-compare"><input type="checkbox" class="compare-checkbox" ${isCompared ? "checked" : ""}></td>
      <td class="col-name">
        <button class="fav-star${isFav ? " active" : ""}" title="Favorite / sleeper watchlist" type="button">★</button>
        <span class="player-name player-name-link">${escapeHtml(player.name)}</span>
      </td>
      <td><span class="pos-pill pos-${player.position}">${player.position}</span></td>
      <td>${escapeHtml(player.team)}</td>
      <td>#${player.position_rank}</td>
      <td>${player.adp}</td>
      <td class="gap-cell"><span class="${gapCellClass}">${gapText}</span></td>
      <td>${player.stats && player.stats.guide_adj_ppg != null ? player.stats.guide_adj_ppg : "—"}</td>
      <td class="actions-cell"></td>
    `;

    tr.querySelector(".player-name-link").addEventListener("click", () => openPlayerDetail(player));

    tr.querySelector(".fav-star").addEventListener("click", () => toggleFavorite(player.id));

    tr.querySelector(".compare-checkbox").addEventListener("change", e => {
      if (e.target.checked) compareIds.add(player.id);
      else compareIds.delete(player.id);
      renderCompareBar();
    });

    const actionsCell = tr.querySelector(".actions-cell");

    if (isDrafted) {
      const span = document.createElement("span");
      span.textContent = "Drafted";
      span.style.color = "var(--text-dim)";
      span.style.fontSize = "12px";
      actionsCell.appendChild(span);

      const undoBtn = document.createElement("button");
      undoBtn.className = "action-btn undo-btn";
      undoBtn.textContent = "Undo";
      undoBtn.addEventListener("click", () => unmarkPlayer(player.id));
      actionsCell.appendChild(undoBtn);
    } else if (isMine) {
      const span = document.createElement("span");
      span.textContent = "✓ Mine";
      span.style.color = "var(--accent)";
      span.style.fontWeight = "700";
      span.style.fontSize = "12px";
      actionsCell.appendChild(span);

      const undoBtn = document.createElement("button");
      undoBtn.className = "action-btn undo-btn";
      undoBtn.textContent = "Undo";
      undoBtn.addEventListener("click", () => unmarkPlayer(player.id));
      actionsCell.appendChild(undoBtn);
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
    meta.textContent = `${player.position} · ${player.team} · rank #${player.position_rank} · ADP ${player.adp}`;

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

  document.getElementById("compareBarOpenBtn").addEventListener("click", openCompareModal);
  document.getElementById("compareBarClearBtn").addEventListener("click", clearCompare);
  document.getElementById("compareCloseBtn").addEventListener("click", closeAllOverlays);
  const compareOverlay = document.getElementById("compareOverlay");
  compareOverlay.addEventListener("click", e => {
    if (e.target === compareOverlay) closeAllOverlays();
  });

  document.getElementById("playerDetailCloseBtn").addEventListener("click", closePlayerDetail);
  const detailOverlay = document.getElementById("playerDetailOverlay");
  detailOverlay.addEventListener("click", e => {
    if (e.target === detailOverlay) closePlayerDetail();
  });

  document.getElementById("tradeCalcBtn").addEventListener("click", openTradeCalculator);
  document.getElementById("tradeCloseBtn").addEventListener("click", closeAllOverlays);
  const tradeOverlay = document.getElementById("tradeOverlay");
  tradeOverlay.addEventListener("click", e => {
    if (e.target === tradeOverlay) closeAllOverlays();
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

  function setSort(col) {
    // userSorted-guard: sortColumn defaults to "adp" before any click, so
    // without checking userSorted here, the FIRST click on the ADP header
    // would read as "already sorted by adp" and flip straight to
    // descending instead of starting fresh at ascending.
    if (userSorted && sortColumn === col) {
      sortDir = -sortDir;
    } else {
      sortColumn = col;
      sortDir = 1;
    }
    userSorted = true;
    renderTable();
  }
  document.getElementById("sortValRank").addEventListener("click", () => setSort("valrank"));
  document.getElementById("sortAdp").addEventListener("click", () => setSort("adp"));
  document.getElementById("sortPos").addEventListener("click", () => setSort("pos"));
  document.getElementById("sortTeam").addEventListener("click", () => setSort("team"));
  document.getElementById("sortGap").addEventListener("click", () => setSort("gap"));
  document.getElementById("sortPpg").addEventListener("click", () => setSort("ppg"));

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
  loadFavorites();
  renderAll();
})();
