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

function resetDraft() {
  if (!confirm("Reset the entire draft? This clears drafted players, your roster, and cannot be undone.")) {
    return;
  }
  clearState();
  renderAll();
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
  // value_score is a small z-score (~-2..+2), but value_gap can swing into
  // the hundreds for deep-bench/FA-adjacent players (see data/README.md —
  // it's rank-distance, not points). Weighting them equally let huge gaps
  // on low-value players (e.g. WR rank #70 with a +100 gap) outrank clear
  // elite talent (e.g. a #1-ranked player with gap 0) at the top of the
  // draft. Scale value_score up and clamp value_gap's swing so gap acts as
  // a tiebreaker/boost among comparable players, not the dominant term.
  const clampedGap = Math.max(-20, Math.min(20, player.value_gap));
  const base = player.value_score * 8 + clampedGap * 0.4;
  const needScore = positionNeedScore(player.position, openCounts);
  let needMultiplier;
  if (needScore > 0) {
    // each open relevant starting slot adds 35% weight, capped
    needMultiplier = 1 + Math.min(needScore, 3) * 0.35;
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

  const visible = allPlayers
    .filter(matchesFilters)
    .sort((a, b) => a.position_rank - b.position_rank || b.value_score - a.value_score);

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
      <td class="col-name"><span class="player-name">${escapeHtml(player.name)}</span></td>
      <td><span class="pos-pill pos-${player.position}">${player.position}</span></td>
      <td>${escapeHtml(player.team)}</td>
      <td>#${player.position_rank}</td>
      <td>${player.adp}</td>
      <td class="gap-cell"><span class="${gapClass}">${gapText}</span></td>
      <td class="actions-cell"></td>
    `;

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
    name.className = "rec-name";
    name.textContent = `${player.name}`;

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

  document.getElementById("resetBtn").addEventListener("click", resetDraft);
}

// ---------- Init ----------
(async function init() {
  wireControls();
  loadState();
  await loadPlayers();
  renderAll();
})();
