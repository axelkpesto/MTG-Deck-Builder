(function () {
  "use strict";

  const page = document.body.dataset.page || "builder";
  const loginShell = document.getElementById("loginShell");
  const appShell = document.getElementById("appShell");
  if (!loginShell || !appShell) return;

  const CATEGORY_ORDER = [
    "Commander",
    "Creature",
    "Artifact",
    "Enchantment",
    "Instant",
    "Sorcery",
    "Land",
    "Planeswalker",
    "Battle",
    "Other",
  ];

  const CATEGORY_LABELS = Object.fromEntries(
    CATEGORY_ORDER.map((name) => [name.toLowerCase(), name]),
  );

  const MANA_COLORS = {
    W: { label: "White",     bg: "#f4edd0", fg: "#5a4a1a" },
    U: { label: "Blue",      bg: "#7ab8d9", fg: "#002b49" },
    B: { label: "Black",     bg: "#4a4a4a", fg: "#e0e0e0" },
    R: { label: "Red",       bg: "#d4622c", fg: "#fff"    },
    G: { label: "Green",     bg: "#4e9b4e", fg: "#fff"    },
    C: { label: "Colorless", bg: "#a8a8a8", fg: "#fff"    },
  };

  const state = {
    user: null,
    cards: [],
    deckTitle: "",
    sortBy: "category",
    filterCategory: "",
    search: "",
    commander: "",
    loading: false,
    analysis: null,
    analysisTimer: null,
    analysisRequestId: 0,
    metaCache: {},
    currentDeckId: "",
    activePreviewName: "",
  };

  const els = {
    userLabel: document.getElementById("userLabel"),
    commander: document.getElementById("commander"),
    generateBtn: document.getElementById("generateBtn"),
    completeBtn: document.getElementById("completeBtn"),
    importCards: document.getElementById("importCards"),
    importBtn: document.getElementById("importBtn"),
    saveBtn: document.getElementById("saveBtn"),
    searchInput: document.getElementById("searchInput"),
    sortSelect: document.getElementById("sortSelect"),
    filterSelect: document.getElementById("filterSelect"),
    deckBoard: document.getElementById("deckBoard"),
    cardCount: document.getElementById("cardCount"),
    statusText: document.getElementById("statusText"),
    analysisMeta: document.getElementById("analysisMeta"),
    analysisEmpty: document.getElementById("analysisEmpty"),
    analysisPanel: document.getElementById("analysisPanel"),
    analysisSummary: document.getElementById("analysisSummary"),
    analysisTags: document.getElementById("analysisTags"),
    analysisCurve: document.getElementById("analysisCurve"),
    analysisColors: document.getElementById("colorGrid"),
    costBar: document.getElementById("costBar"),
    productionBar: document.getElementById("productionBar"),
    pageDeckTitle: document.getElementById("pageDeckTitle"),
    heroSummary: document.getElementById("heroSummary"),
    savedDecksList: document.getElementById("savedDecksList"),
    savedDecksEmpty: document.getElementById("savedDecksEmpty"),
    savedStatus: document.getElementById("savedStatus"),
    cardModal: document.getElementById("cardModal"),
    cardModalClose: document.getElementById("cardModalClose"),
    cardModalImage: document.getElementById("cardModalImage"),
    cardModalName: document.getElementById("cardModalName"),
    cardModalCmc: document.getElementById("cardModalCmc"),
    cardModalTypeline: document.getElementById("cardModalTypeline"),
    cardModalBadges: document.getElementById("cardModalBadges"),
    cardModalOracle: document.getElementById("cardModalOracle"),
    cardModalFlavor: document.getElementById("cardModalFlavor"),
    cardModalPT: document.getElementById("cardModalPT"),
    cardModalRemove: document.getElementById("cardModalRemove"),
  };

  function uid() {
    return (
      (crypto.randomUUID && crypto.randomUUID()) ||
      `${Date.now()}_${Math.random().toString(16).slice(2)}`
    );
  }

  function setStatus(text, tone) {
    if (!els.statusText) return;
    els.statusText.textContent = text || "";
    els.statusText.dataset.tone = tone || "";
  }

  function setBusy(value, text) {
    state.loading = value;
    if (typeof text === "string") {
      setStatus(text, value ? "busy" : "");
    }
    [els.generateBtn, els.completeBtn, els.importBtn, els.saveBtn].forEach((btn) => {
      if (btn) btn.disabled = value;
    });
  }

  async function getSession() {
    const res = await fetch("/session.php", { method: "GET" });
    if (!res.ok) throw new Error(`Session check failed: ${res.status}`);
    return res.json();
  }

  async function callApi(payload) {
    const res = await fetch("/api.php", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await res.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      parsed = { raw: text };
    }
    if (!res.ok) {
      throw new Error(`API ${res.status}: ${JSON.stringify(parsed)}`);
    }
    return parsed;
  }

  async function callLocalJson(url, payload, method) {
    const resolvedMethod = method || (payload ? "POST" : "GET");
    const res = await fetch(url, {
      method: resolvedMethod,
      headers: payload ? { "Content-Type": "application/json" } : undefined,
      body: payload ? JSON.stringify(payload) : undefined,
    });
    const text = await res.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      parsed = { raw: text };
    }
    if (!res.ok) {
      throw new Error(`API ${res.status}: ${JSON.stringify(parsed)}`);
    }
    return parsed;
  }

  function reportError(context, err) {
    console.error(context, err);
  }

  function parseImportLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return null;
    const leadQty = trimmed.match(/^(\d+)\s+(.+)$/);
    if (leadQty) return { quantity: Number(leadQty[1]), name: leadQty[2].trim() };
    const trailQty = trimmed.match(/^(.+)\s+x(\d+)$/i);
    if (trailQty) return { quantity: Number(trailQty[2]), name: trailQty[1].trim() };
    return { quantity: 1, name: trimmed };
  }

  function normalizeTagPayload(predicted, scores) {
    const sortedScores = Array.isArray(scores)
      ? scores
          .filter((item) => item && typeof item.tag === "string")
          .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
      : [];
    const tags = sortedScores.length
      ? sortedScores.map((item) => item.tag)
      : Array.isArray(predicted)
        ? predicted.filter((tag) => typeof tag === "string")
        : [];
    return { tags, primaryTag: tags[0] || "" };
  }

  function formatTag(tag) {
    if (!tag) return "Untagged";
    return String(tag)
      .replace(/_/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }


  function parseListString(value) {
    if (Array.isArray(value)) return value.map((item) => String(item));
    if (typeof value !== "string") return [];
    const matches = value.match(/'([^']+)'/g);
    if (!matches) return [];
    return matches.map((item) => item.replace(/'/g, ""));
  }

  function normalizeMeta(name, meta, imageUrls) {
    const types = parseListString(meta.Types || meta.types || []);
    return {
      name:
        (typeof meta.Name === "string" && meta.Name.trim()) ||
        (typeof meta.card_name === "string" && meta.card_name.trim()) ||
        name,
      cardId: meta.card_id || "",
      types,
      raw: meta,
      imageUrls: Array.isArray(imageUrls) ? imageUrls : [],
    };
  }

  async function fetchCardImages(names) {
    if (!Array.isArray(names) || names.length === 0) {
      return { found: {}, missing: {} };
    }
    return callLocalJson("/card_images.php", { cards: names });
  }

  async function hydrateMetaForNames(names) {
    const uniqueNames = Array.from(
      new Set(
        (Array.isArray(names) ? names : [])
          .map((name) => String(name || "").trim())
          .filter(Boolean),
      ),
    ).filter((name) => !state.metaCache[name]);

    if (uniqueNames.length === 0) return;

    const [metaData, imageData] = await Promise.all([
      callApi({
        path: "/get_vector_descriptions",
        method: "POST",
        body: { cards: uniqueNames },
      }),
      fetchCardImages(uniqueNames),
    ]);

    uniqueNames.forEach((name) => {
      const meta = metaData.found?.[name];
      if (!meta) return;
      const normalized = normalizeMeta(name, meta, imageData.found?.[name] || []);
      state.metaCache[name] = normalized;
      state.metaCache[normalized.name] = normalized;
    });
  }

  async function loadCardMeta(name) {
    const cached = state.metaCache[name];
    if (cached && Array.isArray(cached.imageUrls)) return cached;

    const [metaData, imageData] = await Promise.all([
      callApi({
        path: "/get_vector_description",
        method: "POST",
        body: { id: name },
      }),
      fetchCardImages([name]),
    ]);

    const normalized = normalizeMeta(
      name,
      metaData,
      imageData.found?.[name] || imageData.found?.[metaData.card_name] || [],
    );
    state.metaCache[name] = normalized;
    state.metaCache[normalized.name] = normalized;
    return normalized;
  }

  function getCommander() {
    return (
      (els.commander && els.commander.value.trim()) ||
      state.commander ||
      state.cards.find((card) => card.primaryTag === "Commander")?.name ||
      ""
    );
  }

  function getDeckTitle() {
    const commander = getCommander();
    return state.deckTitle || (commander ? `${commander} Deck` : "Untitled Deck");
  }

  function imageUrlFromMeta(meta) {
    if (meta?.imageUrls?.[0]) return meta.imageUrls[0];
    if (meta?.cardId) {
      return `https://api.scryfall.com/cards/${encodeURIComponent(meta.cardId)}?format=image&version=normal`;
    }
    return "";
  }

  function getCardCategory(card) {
    if (!card) return "Other";
    if (card.primaryTag === "Commander" || card.name === getCommander()) return "Commander";
    const meta = state.metaCache[card.name];
    const types = Array.isArray(meta?.types) ? meta.types.map((type) => String(type).toLowerCase()) : [];
    for (const type of types) {
      if (CATEGORY_LABELS[type]) return CATEGORY_LABELS[type];
    }
    if (String(card.primaryTag || "").toLowerCase() === "land") return "Land";
    return "Other";
  }

  function getCardGroup(card) {
    if (!card) return "Other";
    if (card.primaryTag === "Commander" || card.name === getCommander()) return "Commander";
    // Always put land-typed cards in Land regardless of tag (e.g. lands tagged Ramp)
    if (getCardCategory(card) === "Land") return "Land";
    const tag = formatTag(card.primaryTag || "");
    return tag === "Untagged" ? getCardCategory(card) : tag;
  }

  function totalCardCount() {
    return state.cards.reduce((sum, card) => sum + Number(card.quantity || 0), 0);
  }

  function updateDeckHeader() {
    if (els.pageDeckTitle) {
      const commander = getCommander();
      els.pageDeckTitle.textContent = commander || "Commander Builder";
    }
    if (els.heroSummary) {
      const commander = getCommander();
      const total = totalCardCount();
      els.heroSummary.textContent = commander
        ? `${commander} leading ${total} cards in the current build.`
        : "";
    }
  }

  function updateCounts() {
    if (els.cardCount) els.cardCount.textContent = `${totalCardCount()} cards`;
    updateDeckHeader();
  }

  function buildAnalysisCards() {
    const cards = [];
    state.cards.forEach((card) => {
      for (let i = 0; i < card.quantity; i += 1) cards.push(card.name);
    });
    return cards;
  }

  function renderSummaryMetric(label, value) {
    const wrap = document.createElement("div");
    wrap.className = "metric";
    const val = document.createElement("div");
    val.className = "metric-value";
    val.textContent = String(value);
    const lab = document.createElement("div");
    lab.className = "metric-label";
    lab.textContent = label;
    wrap.appendChild(val);
    wrap.appendChild(lab);
    return wrap;
  }

  function renderAnalysis() {
    if (!els.analysisPanel || !els.analysisEmpty) return;
    if (!state.analysis) {
      els.analysisPanel.hidden = true;
      els.analysisEmpty.hidden = true;
      if (els.analysisSummary) els.analysisSummary.innerHTML = "";
      if (els.analysisTags) els.analysisTags.innerHTML = "";
      if (els.analysisCurve) els.analysisCurve.innerHTML = "";
      if (els.analysisColors) els.analysisColors.innerHTML = "";
      return;
    }

    const summary = state.analysis;
    const tags = summary.tags?.tag_counts || {};
    const colors = summary.color_distribution?.colors?.counts || {};
    const colorPercents = summary.color_distribution?.colors?.percent || {};
    const curve = summary.curve?.mana_curve?.counts || [];
    const lands = summary.lands?.lands || {};

    const BASIC_TO_COLOR = { Plains: "W", Island: "U", Swamp: "B", Mountain: "R", Forest: "G" };
    const basicTypes = lands.basic_types || {};
    const production = { W: 0, U: 0, B: 0, R: 0, G: 0 };
    Object.entries(basicTypes).forEach(([type, count]) => {
      const c = BASIC_TO_COLOR[type];
      if (c) production[c] = Number(count || 0);
    });
    const totalProduction = Object.values(production).reduce((a, b) => a + b, 0);
    const productionPercents = { W: 0, U: 0, B: 0, R: 0, G: 0 };
    if (totalProduction > 0) {
      Object.keys(productionPercents).forEach((c) => {
        productionPercents[c] = production[c] / totalProduction;
      });
    }

    const COLOR_IMAGE = { W: "/assets/plains.png", U: "/assets/island.png", B: "/assets/swamp.png", R: "/assets/mountain.png", G: "/assets/forest.png" };

    els.analysisEmpty.hidden = true;
    els.analysisPanel.hidden = false;
    els.analysisSummary.innerHTML = "";
    els.analysisSummary.appendChild(renderSummaryMetric("Cards", buildAnalysisCards().length));
    els.analysisSummary.appendChild(renderSummaryMetric("Unique", state.cards.length));
    els.analysisSummary.appendChild(renderSummaryMetric("Lands", lands.land_count || 0));
    els.analysisSummary.appendChild(renderSummaryMetric("Basics", lands.basic_count || 0));

    // Cost bar: segmented by color
    if (els.costBar) {
      els.costBar.innerHTML = "";
      const totalPct = ["W", "U", "B", "R", "G"].reduce((s, c) => s + Number(colorPercents[c] || 0), 0);
      const colorlessWidth = Math.max(0, 1 - totalPct);
      ["W", "U", "B", "R", "G"].forEach((c) => {
        const pct = Number(colorPercents[c] || 0);
        if (pct <= 0) return;
        const seg = document.createElement("div");
        seg.className = "stats-bar-segment";
        seg.style.width = `${Math.round(pct * 100)}%`;
        seg.style.background = MANA_COLORS[c]?.bg || "#888";
        els.costBar.appendChild(seg);
      });
      if (colorlessWidth > 0.005) {
        const seg = document.createElement("div");
        seg.className = "stats-bar-segment";
        seg.style.width = `${Math.round(colorlessWidth * 100)}%`;
        seg.style.background = MANA_COLORS.C?.bg || "#a8a8a8";
        els.costBar.appendChild(seg);
      }
    }

    // Production bar: segmented by basic land color
    if (els.productionBar) {
      els.productionBar.innerHTML = "";
      if (totalProduction > 0) {
        ["W", "U", "B", "R", "G"].forEach((c) => {
          const pct = productionPercents[c];
          if (pct <= 0) return;
          const seg = document.createElement("div");
          seg.className = "stats-bar-segment";
          seg.style.width = `${Math.round(pct * 100)}%`;
          seg.style.background = MANA_COLORS[c]?.bg || "#888";
          els.productionBar.appendChild(seg);
        });
      } else {
        const seg = document.createElement("div");
        seg.className = "stats-bar-segment";
        seg.style.width = "100%";
        seg.style.background = MANA_COLORS.C?.bg || "#a8a8a8";
        els.productionBar.appendChild(seg);
      }
    }

    // Color stat grid: one card per color with cost + production bars
    if (els.analysisColors) {
      els.analysisColors.innerHTML = "";
      ["W", "U", "B", "R", "G"].forEach((c) => {
        const costPct = Math.round(Number(colorPercents[c] || 0) * 100);
        const prodPct = Math.round(productionPercents[c] * 100);
        const mc = MANA_COLORS[c];

        const card = document.createElement("div");
        card.className = "color-stat-card";

        const icon = document.createElement("img");
        icon.className = "color-stat-icon";
        icon.src = COLOR_IMAGE[c];
        icon.alt = MANA_COLORS[c]?.label || c;

        const costLabel = document.createElement("div");
        costLabel.className = "color-stat-section-label";
        costLabel.textContent = "Cost";

        const costTrack = document.createElement("div");
        costTrack.className = "color-stat-bar-track";
        const costFill = document.createElement("div");
        costFill.className = "color-stat-bar-fill";
        costFill.style.width = `${costPct}%`;
        costFill.style.background = mc?.bg || "#888";
        costTrack.appendChild(costFill);

        const costText = document.createElement("div");
        costText.className = "color-stat-text";
        const pipCount = colors[c] || 0;
        costText.textContent = `${pipCount} pip${pipCount !== 1 ? "s" : ""} - ${costPct}%`;

        const prodLabel = document.createElement("div");
        prodLabel.className = "color-stat-section-label";
        prodLabel.textContent = "Production";

        const prodTrack = document.createElement("div");
        prodTrack.className = "color-stat-bar-track";
        const prodFill = document.createElement("div");
        prodFill.className = "color-stat-bar-fill";
        prodFill.style.width = `${prodPct}%`;
        prodFill.style.background = mc?.bg || "#888";
        prodTrack.appendChild(prodFill);

        const prodText = document.createElement("div");
        prodText.className = "color-stat-text";
        const basicCount = production[c] || 0;
        prodText.textContent = `${basicCount} mana - ${prodPct}%`;

        card.appendChild(icon);
        card.appendChild(costLabel);
        card.appendChild(costTrack);
        card.appendChild(costText);
        card.appendChild(prodLabel);
        card.appendChild(prodTrack);
        card.appendChild(prodText);
        els.analysisColors.appendChild(card);
      });
    }

    const topTags = Object.entries(tags)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    els.analysisTags.innerHTML = "";
    topTags.forEach(([tag, count]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${formatTag(tag)} ${count}`;
      els.analysisTags.appendChild(chip);
    });

    els.analysisCurve.innerHTML = "";
    const curveMax = Math.max(1, ...curve);
    curve.forEach((count, index) => {
      const bucket = document.createElement("div");
      bucket.className = "curve-bucket";

      const bar = document.createElement("div");
      bar.className = "curve-bar";
      bar.style.height = `${Math.max(12, (Number(count || 0) / curveMax) * 100)}px`;

      const countEl = document.createElement("span");
      countEl.className = "curve-count";
      countEl.textContent = String(count || 0);

      const label = document.createElement("span");
      label.className = "curve-label";
      label.textContent = index === 6 ? "6+" : String(index);

      bucket.appendChild(countEl);
      bucket.appendChild(bar);
      bucket.appendChild(label);
      els.analysisCurve.appendChild(bucket);
    });
  }

  function compareCards(a, b) {
    if (state.sortBy === "name") {
      return a.name.localeCompare(b.name);
    }
    if (state.sortBy === "quantity") {
      return b.quantity - a.quantity || a.name.localeCompare(b.name);
    }
    if (state.sortBy === "tag") {
      return formatTag(a.primaryTag).localeCompare(formatTag(b.primaryTag)) || a.name.localeCompare(b.name);
    }
    const ag = getCardGroup(a);
    const bg = getCardGroup(b);
    if (ag === "Commander" && bg !== "Commander") return -1;
    if (bg === "Commander" && ag !== "Commander") return 1;
    return ag.localeCompare(bg) || a.name.localeCompare(b.name);
  }

  function getDisplayCards() {
    let list = state.cards.slice();
    if (state.search) {
      const q = state.search.toLowerCase();
      list = list.filter((card) => card.name.toLowerCase().includes(q));
    }
    if (state.filterCategory) {
      list = list.filter((card) => getCardGroup(card) === state.filterCategory);
    }
    list.sort(compareCards);
    return list;
  }

  function uniqueGroups() {
    const groups = new Set();
    state.cards.forEach((card) => groups.add(getCardGroup(card)));
    const ordered = Array.from(groups).sort((a, b) => a.localeCompare(b));
    if (groups.has("Commander")) {
      return ["Commander", ...ordered.filter((group) => group !== "Commander")];
    }
    return ordered;
  }

  function renderCategoryFilter() {
    if (!els.filterSelect) return;
    const groups = uniqueGroups();
    els.filterSelect.innerHTML = "";

    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All Tags";
    els.filterSelect.appendChild(all);

    groups.forEach((group) => {
      const opt = document.createElement("option");
      opt.value = group;
      opt.textContent = group;
      if (state.filterCategory === group) opt.selected = true;
      els.filterSelect.appendChild(opt);
    });
  }

  function updateCardQuantity(card, nextValue) {
    const quantity = Math.max(1, Math.floor(Number(nextValue) || 1));
    card.quantity = quantity;
    updateCounts();
    renderDeckBoard();
    scheduleAnalysis(true);
  }

  function removeCard(card) {
    state.cards = state.cards.filter((item) => item.id !== card.id);
    renderCategoryFilter();
    renderDeckBoard();
    scheduleAnalysis(true);
  }

  function createStackCard(card, index) {
    const cardEl = document.createElement("article");
    cardEl.className = "column-stack-card";
    cardEl.style.setProperty("--stack-index", String(index));
    cardEl.dataset.state = "loading";
    cardEl.tabIndex = 0;

    const image = document.createElement("img");
    image.className = "column-preview-image";
    image.alt = card.name;
    cardEl.appendChild(image);

    const qtyBadge = document.createElement("button");
    qtyBadge.className = "card-qty-badge";
    qtyBadge.type = "button";
    qtyBadge.textContent = String(card.quantity);
    qtyBadge.title = `${card.name} ×${card.quantity} — click to change`;
    qtyBadge.addEventListener("click", (event) => {
      event.stopPropagation();
      const next = window.prompt(`Quantity for ${card.name}`, String(card.quantity));
      if (next === null) return;
      qtyBadge.textContent = String(Math.max(1, Math.floor(Number(next) || 1)));
      updateCardQuantity(card, next);
    });
    cardEl.appendChild(qtyBadge);

    const removeBtn = document.createElement("button");
    removeBtn.className = "card-remove-btn";
    removeBtn.type = "button";
    removeBtn.textContent = "×";
    removeBtn.title = `Remove ${card.name}`;
    removeBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      removeCard(card);
    });
    cardEl.appendChild(removeBtn);

    cardEl.addEventListener("mouseenter", () => cardEl.classList.add("is-active"));
    cardEl.addEventListener("mouseleave", () => cardEl.classList.remove("is-active"));
    cardEl.addEventListener("focusin", () => cardEl.classList.add("is-active"));
    cardEl.addEventListener("focusout", () => cardEl.classList.remove("is-active"));
    cardEl.addEventListener("click", () => {
      loadCardMeta(card.name).then((meta) => openCardModal(card, meta));
    });

    loadCardMeta(card.name)
      .then((resolved) => {
        const imageUrl = imageUrlFromMeta(resolved);
        if (!imageUrl) {
          cardEl.dataset.state = "missing";
          return;
        }
        image.src = imageUrl;
        cardEl.dataset.state = "ready";
      })
      .catch(() => {
        cardEl.dataset.state = "missing";
      });

    return cardEl;
  }

  function renderDeckBoard() {
    if (!els.deckBoard) return;
    const grouped = new Map();
    getDisplayCards().forEach((card) => {
      const group = getCardGroup(card);
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(card);
    });

    els.deckBoard.innerHTML = "";
    uniqueGroups().forEach((group) => {
      const cards = grouped.get(group) || [];
      if (cards.length === 0) return;

      const column = document.createElement("section");
      column.className = "deck-column";

      const totalQty = cards.reduce((sum, c) => sum + c.quantity, 0);
      const header = document.createElement("header");
      header.className = "deck-column-header";
      header.innerHTML = `<div class="deck-column-title">${group}<span class="deck-column-qty">Qty: ${cards.length}</span></div><div class="deck-column-count">${totalQty}</div>`;

      const sortedCards = cards.slice().sort((a, b) => b.quantity - a.quantity);

      const imageStack = document.createElement("div");
      imageStack.className = "column-image-stack";
      const cardNodes = sortedCards.map((card, index) => createStackCard(card, index));
      cardNodes.forEach((node) => imageStack.appendChild(node));

      const PEEK = 36;       // px visible per card
      cardNodes.forEach((node, index) => {
        node.style.top = `${index * PEEK}px`;
        node.style.zIndex = String(10 + index);
      });
      imageStack.style.minHeight = `${(sortedCards.length - 1) * PEEK + 340}px`;

      column.appendChild(header);
      column.appendChild(imageStack);
      els.deckBoard.appendChild(column);
    });

    updateCounts();
  }

  function ensureCommanderCardListed() {
    const commander = getCommander();
    if (!commander) return;
    state.commander = commander;

    const existing = state.cards.find((card) => card.name.toLowerCase() === commander.toLowerCase());
    if (existing) {
      existing.quantity = 1;
      existing.tags = ["Commander"];
      existing.primaryTag = "Commander";
      return;
    }

    state.cards.unshift({
      id: uid(),
      name: commander,
      quantity: 1,
      tags: ["Commander"],
      primaryTag: "Commander",
    });
  }

  function mergeCards(items) {
    items.forEach((item) => {
      const existing = state.cards.find(
        (card) => card.name.toLowerCase() === item.name.toLowerCase(),
      );
      if (existing) {
        existing.quantity += item.quantity;
        if (item.tags) existing.tags = item.tags;
        if (item.primaryTag) existing.primaryTag = item.primaryTag;
      } else {
        state.cards.push({
          id: uid(),
          name: item.name,
          quantity: item.quantity,
          tags: item.tags || [],
          primaryTag: item.primaryTag || "",
        });
      }
    });
  }

  async function validateAndTagCard(item) {
    const meta = await callApi({
      path: "/get_vector_description",
      method: "POST",
      body: { id: item.name },
    });
    const tagData = await callApi({
      path: "/get_tags",
      method: "POST",
      body: { id: item.name, threshold: 0.5, top_k: 8 },
    });
    const images = await fetchCardImages([item.name]);
    const normalized = normalizeTagPayload(tagData.predicted, tagData.predicted_scores || tagData.scores);
    const resolvedName =
      typeof meta.card_name === "string" && meta.card_name.trim()
        ? meta.card_name.trim()
        : item.name;
    const normalizedMeta = normalizeMeta(
      resolvedName,
      meta,
      images.found?.[resolvedName] || images.found?.[item.name] || [],
    );
    state.metaCache[resolvedName] = normalizedMeta;
    return {
      name: resolvedName,
      quantity: item.quantity,
      tags: normalized.tags,
      primaryTag: normalized.primaryTag,
    };
  }

  async function enrichCards(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return { cards: [], missing: [] };
    }

    const names = items.map((item) => item.name);
    const [metaData, tagData, imageData] = await Promise.all([
      callApi({
        path: "/get_vector_descriptions",
        method: "POST",
        body: { cards: names },
      }),
      callApi({
        path: "/get_tag_list",
        method: "POST",
        body: { cards: names, threshold: 0.5, top_k: 8 },
      }),
      fetchCardImages(names),
    ]);

    const cards = [];
    const missing = [];

    items.forEach((item) => {
      const meta = metaData.found?.[item.name];
      const tags = tagData.found?.[item.name];
      if (!meta || !tags) {
        missing.push(item.name);
        return;
      }

      const resolvedName =
        typeof meta.card_name === "string" && meta.card_name.trim()
          ? meta.card_name.trim()
          : tags.card_id || item.name;

      state.metaCache[resolvedName] = normalizeMeta(
        resolvedName,
        meta,
        imageData.found?.[item.name] || imageData.found?.[resolvedName] || [],
      );

      const normalized = normalizeTagPayload(
        tags.predicted,
        tags.predicted_scores || tags.scores,
      );

      cards.push({
        name: resolvedName,
        quantity: item.quantity,
        tags: normalized.tags,
        primaryTag: normalized.primaryTag,
      });
    });

    return { cards, missing };
  }

  async function runAnalysis(options) {
    if (!els.analysisMeta) return;
    const settings = Object.assign({ showBusy: false }, options);
    const commander = getCommander();
    const cards = buildAnalysisCards();
    if (!commander || cards.length === 0) {
      state.analysis = null;
      els.analysisMeta.textContent = "Add cards or generate a deck to see analysis.";
      renderAnalysis();
      return;
    }

    const requestId = ++state.analysisRequestId;
    if (settings.showBusy) {
      setBusy(true, "Analyzing...");
    } else {
      els.analysisMeta.textContent = "Refreshing analysis...";
    }

    try {
      const data = await callApi({
        path: "/analyze_deck",
        method: "POST",
        body: { commander, cards },
      });
      if (requestId !== state.analysisRequestId) return;
      state.analysis = data;
      els.analysisMeta.textContent = `Updated for ${cards.length} cards.`;
      renderAnalysis();
      if (!settings.showBusy) setStatus("Analysis updated.", "success");
    } catch (err) {
      if (requestId !== state.analysisRequestId) return;
      els.analysisMeta.textContent = "Analysis failed.";
      if (settings.showBusy) {
        reportError("runAnalysis failed", err);
        setStatus("Analysis failed.", "error");
      }
    } finally {
      if (settings.showBusy) setBusy(false);
    }
  }

  function scheduleAnalysis(immediate) {
    clearTimeout(state.analysisTimer);
    if (immediate) {
      runAnalysis({ showBusy: false });
      return;
    }
    state.analysisTimer = window.setTimeout(() => {
      runAnalysis({ showBusy: false });
    }, 500);
  }

  async function generateDeck() {
    const commander = getCommander();
    if (!commander) {
      setStatus("Commander is required.", "error");
      return;
    }
    state.commander = commander;
    setBusy(true, "Generating deck...");
    try {
      const data = await callApi({
        path: "/generate_deck",
        method: "POST",
        body: { id: commander },
      });
      const deckCounts = Array.isArray(data) ? data[0] : data;
      if (!deckCounts || typeof deckCounts !== "object") {
        throw new Error("Unexpected generate_deck response format");
      }

      state.currentDeckId = "";
      state.cards = [{
        id: uid(),
        name: commander,
        quantity: 1,
        tags: ["Commander"],
        primaryTag: "Commander",
      }];

      const generatedCards = [];
      Object.entries(deckCounts).forEach(([name, quantity]) => {
        if (name.toLowerCase() === commander.toLowerCase()) return;
        const q = Number(quantity);
        if (!Number.isFinite(q) || q <= 0) return;
        generatedCards.push({ name, quantity: Math.floor(q) });
      });

      const enriched = await enrichCards(generatedCards);
      enriched.cards.forEach((card) => {
        state.cards.push({
          id: uid(),
          name: card.name,
          quantity: card.quantity,
          tags: card.tags,
          primaryTag: card.primaryTag,
        });
      });

      renderCategoryFilter();
      renderDeckBoard();
      await runAnalysis({ showBusy: false });
      setStatus("Deck generated and analyzed.", "success");
    } catch (err) {
      reportError("generateDeck failed", err);
      setStatus("Deck generation failed.", "error");
    } finally {
      setBusy(false);
    }
  }

  async function completeDeck() {
    const commander = getCommander();
    if (!commander) {
      setStatus("Set a commander first.", "error");
      return;
    }
    setBusy(true, "Adding similar cards...");
    try {
      const data = await callApi({
        path: "/get_similar_vectors",
        method: "POST",
        body: { id: commander, num_vectors: 30 },
      });

      const existing = new Set(state.cards.map((card) => card.name.toLowerCase()));
      const additions = [];
      Object.values(data || {}).forEach((entry) => {
        if (!entry || typeof entry.card_name !== "string") return;
        const name = entry.card_name;
        if (existing.has(name.toLowerCase())) return;
        if (additions.length >= 10) return;
        additions.push({ name, quantity: 1 });
      });

      mergeCards(additions);
      const untagged = state.cards.filter((card) => !card.primaryTag);
      if (untagged.length > 0) {
        const enriched = await enrichCards(untagged);
        const taggedByName = new Map(enriched.cards.map((card) => [card.name.toLowerCase(), card]));
        state.cards = state.cards.map((card) => {
          const taggedCard = taggedByName.get(card.name.toLowerCase());
          if (!taggedCard) return card;
          return {
            ...card,
            name: taggedCard.name,
            tags: taggedCard.tags,
            primaryTag: taggedCard.primaryTag,
          };
        });
      }

      renderCategoryFilter();
      renderDeckBoard();
      scheduleAnalysis(true);
      setStatus(`Added ${additions.length} similar cards.`, "success");
    } catch (err) {
      reportError("completeDeck failed", err);
      setStatus("Adding similar cards failed.", "error");
    } finally {
      setBusy(false);
    }
  }

  async function importCards() {
    const lines = els.importCards.value.split(/\r?\n/);
    const parsed = lines.map(parseImportLine).filter(Boolean);
    if (parsed.length === 0) {
      setStatus("No cards to import.", "error");
      return;
    }

    setBusy(true, "Validating cards...");
    try {
      const results = await Promise.allSettled(parsed.map(validateAndTagCard));
      const validCards = [];
      const invalidNames = [];

      results.forEach((result, index) => {
        if (result.status === "fulfilled") {
          validCards.push(result.value);
        } else {
          invalidNames.push(parsed[index].name);
        }
      });

      if (validCards.length > 0) {
        mergeCards(validCards);
        ensureCommanderCardListed();
        els.importCards.value = "";
        renderCategoryFilter();
        renderDeckBoard();
        await runAnalysis({ showBusy: false });
      }

      if (invalidNames.length > 0) {
        setStatus(`Skipped invalid cards: ${invalidNames.join(", ")}`, "error");
      } else {
        setStatus(`Imported ${validCards.length} cards.`, "success");
      }
    } finally {
      setBusy(false);
    }
  }

  function buildPersistedDeck() {
    const commander = getCommander();
    return {
      id: state.currentDeckId || undefined,
      title: getDeckTitle(),
      format: "commander",
      commander,
      cards: {
        commander,
        cards: state.cards
          .filter((card) => card.name.toLowerCase() !== commander.toLowerCase())
          .map((card) => ({
            name: card.name,
            quantity: card.quantity,
          })),
      },
    };
  }

  async function saveDeck() {
    const commander = getCommander();
    if (!commander) {
      setStatus("Set a commander before saving.", "error");
      return;
    }
    if (state.cards.length === 0) {
      setStatus("Add cards before saving.", "error");
      return;
    }

    const nextTitle = window.prompt("Deck title", getDeckTitle());
    if (nextTitle === null) {
      setStatus("Save canceled.", "error");
      return;
    }
    state.deckTitle = nextTitle.trim() || `${commander} Deck`;
    updateDeckHeader();

    setBusy(true, state.currentDeckId ? "Updating saved deck..." : "Saving deck...");
    try {
      const result = await callLocalJson("/decks.php", {
        deck: buildPersistedDeck(),
      });
      const savedDeck = result.deck || {};
      state.currentDeckId = savedDeck.id || state.currentDeckId;
      const url = new URL(window.location.href);
      if (state.currentDeckId) {
        url.searchParams.set("deck", state.currentDeckId);
        window.history.replaceState({}, "", url.toString());
      }
      setStatus("Deck saved.", "success");
    } catch (err) {
      reportError("saveDeck failed", err);
      if (String(err.message || "").includes("401")) {
        window.location.href = "/login.php";
        return;
      }
      setStatus("Saving deck failed.", "error");
    } finally {
      setBusy(false);
    }
  }

  async function loadDeck(deckId) {
    const result = await callLocalJson(`/decks.php?id=${encodeURIComponent(deckId)}`);
    const savedDeck = result.deck;
    if (!savedDeck) throw new Error("Missing saved deck payload");

    state.currentDeckId = savedDeck.id || deckId;
    state.deckTitle = savedDeck.title || "";
    const savedCards = savedDeck.cards;
    const savedCommander = String(savedCards?.commander || "").trim();
    state.commander = savedCommander;
    if (els.commander) els.commander.value = savedCommander;

    const persistedCards = Array.isArray(savedCards?.cards) ? savedCards.cards : [];
    const cardRows = persistedCards
      .map((card) => ({
        name: String(card.name || "").trim(),
        quantity: Math.max(1, Number(card.quantity || 1)),
        tags: Array.isArray(card.tags) ? card.tags : [],
        primaryTag: String(card.primaryTag || "").trim(),
      }))
      .filter((card) => card.name && card.name.toLowerCase() !== savedCommander.toLowerCase());

    const needsEnrichment = cardRows.some((card) => !card.primaryTag);
    const hydratedCards = needsEnrichment ? (await enrichCards(cardRows)).cards : cardRows;

    state.cards = hydratedCards.map((card) => ({
      id: uid(),
      name: card.name,
      quantity: card.quantity,
      tags: Array.isArray(card.tags) ? card.tags : [],
      primaryTag: String(card.primaryTag || "").trim(),
    }));

    ensureCommanderCardListed();
    await hydrateMetaForNames(state.cards.map((card) => card.name));
    renderCategoryFilter();
    renderDeckBoard();
    await runAnalysis({ showBusy: false });
    setStatus("Saved deck loaded.", "success");
  }

  function formatDate(value) {
    if (!value) return "Unknown";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Unknown";
    return date.toLocaleString();
  }

  async function renderSavedDecks() {
    if (!els.savedDecksList) return;
    els.savedDecksList.innerHTML = "";
    if (els.savedStatus) els.savedStatus.textContent = "Loading saved decks...";

    try {
      const result = await callLocalJson("/decks.php");
      const decks = Array.isArray(result.decks) ? result.decks : [];
      if (els.savedStatus) {
        els.savedStatus.textContent = `${decks.length} saved deck${decks.length === 1 ? "" : "s"}.`;
      }
      if (els.savedDecksEmpty) {
        els.savedDecksEmpty.hidden = decks.length !== 0;
      }

      decks.forEach((deck) => {
        const cardCount = Number(deck.card_count || 0);
        const cardNames = Array.isArray(deck.cards)
          ? deck.cards.map((card) => String(card.name || "")).filter(Boolean).slice(0, 3)
          : [];

        const card = document.createElement("article");
        card.className = "saved-card";

        const title = document.createElement("a");
        title.className = "saved-card-title";
        title.href = `/?deck=${encodeURIComponent(deck.id)}`;
        title.textContent = deck.title || deck.commander || "Untitled Deck";

        const commander = document.createElement("div");
        commander.className = "saved-card-commander";
        commander.textContent = deck.commander || "Unknown Commander";

        const meta = document.createElement("div");
        meta.className = "saved-card-meta";
        meta.textContent = `${cardCount} cards - Updated ${formatDate(deck.updated_at)}`;

        const sample = document.createElement("div");
        sample.className = "saved-card-sample";
        sample.textContent = cardNames.join(" - ");

        const open = document.createElement("a");
        open.className = "button ghost";
        open.href = `/?deck=${encodeURIComponent(deck.id)}`;
        open.textContent = "Open in Builder";

        const del = document.createElement("button");
        del.className = "button ghost saved-card-delete";
        del.type = "button";
        del.textContent = "Delete";
        del.addEventListener("click", async () => {
          if (!window.confirm(`Delete "${deck.title || deck.commander || "this deck"}"? This cannot be undone.`)) return;
          del.disabled = true;
          del.textContent = "Deleting…";
          try {
            await callLocalJson(`/decks.php?id=${encodeURIComponent(deck.id)}`, null, "DELETE");
            card.remove();
            const remaining = els.savedDecksList.querySelectorAll(".saved-card").length;
            if (els.savedStatus) els.savedStatus.textContent = `${remaining} saved deck${remaining === 1 ? "" : "s"}.`;
            if (els.savedDecksEmpty) els.savedDecksEmpty.hidden = remaining !== 0;
          } catch (err) {
            reportError("deleteDeck failed", err);
            del.disabled = false;
            del.textContent = "Delete";
          }
        });

        const actions = document.createElement("div");
        actions.className = "saved-card-actions";
        actions.appendChild(open);
        actions.appendChild(del);

        card.appendChild(title);
        card.appendChild(commander);
        card.appendChild(meta);
        if (cardNames.length > 0) card.appendChild(sample);
        card.appendChild(actions);
        els.savedDecksList.appendChild(card);
      });
    } catch (err) {
      reportError("renderSavedDecks failed", err);
      if (els.savedStatus) els.savedStatus.textContent = "Failed to load saved decks.";
      if (els.savedDecksEmpty) {
        els.savedDecksEmpty.hidden = false;
        els.savedDecksEmpty.textContent = "Unable to load saved decks right now.";
      }
    }
  }

  // ── Card Detail Modal ────────────────────────────────────────────

  const COLOR_MODAL_STYLE = {
    W: { bg: "#f5f0e0", fg: "#5a4a00", label: "White" },
    U: { bg: "#1a3a6e", fg: "#b8d4f8", label: "Blue"  },
    B: { bg: "#2a1a3a", fg: "#c8b8e8", label: "Black" },
    R: { bg: "#6e1a1a", fg: "#f8c8b8", label: "Red"   },
    G: { bg: "#1a3a1a", fg: "#b8e8c8", label: "Green" },
    C: { bg: "#444",    fg: "#ccc",    label: "Colorless" },
  };

  const RARITY_STYLE = {
    common:   { bg: "#444",    fg: "#ccc"    },
    uncommon: { bg: "#607a8a", fg: "#e0eaf0" },
    rare:     { bg: "#7a6020", fg: "#f8e8a0" },
    mythic:   { bg: "#8a3010", fg: "#f8c880" },
  };

  async function fetchScryfallCard(cardId) {
    try {
      const resp = await fetch(`https://api.scryfall.com/cards/${encodeURIComponent(cardId)}`);
      if (!resp.ok) return null;
      return resp.json();
    } catch {
      return null;
    }
  }

  function showTagFallback(card) {
    const tags = Array.isArray(card.tags) ? card.tags.slice() : [];
    els.cardModalOracle.textContent = tags.map(formatTag).join("\n") || "";
  }

  function openCardModal(card, meta) {

    const raw = meta.raw || {};
    const imageUrl = imageUrlFromMeta(meta);

    // Image
    els.cardModalImage.src = imageUrl || "";
    els.cardModalImage.alt = card.name;

    // Name + CMC
    els.cardModalName.textContent = meta.name || card.name;
    const cmc = raw["Mana Cost"] ?? raw.mana_cost ?? "";
    els.cardModalCmc.textContent = cmc !== "" ? `CMC ${cmc}` : "";
    els.cardModalCmc.hidden = cmc === "";

    // Type line
    const supertypes = parseListString(raw.Supertypes || raw.supertypes || []);
    const types      = parseListString(raw.Types      || raw.types      || []);
    const subtypes   = parseListString(raw.Subtypes   || raw.subtypes   || []);
    const typeStr = [
      [...supertypes, ...types].join(" "),
      subtypes.length ? `— ${subtypes.join(" ")}` : "",
    ].filter(Boolean).join(" ");
    els.cardModalTypeline.textContent = typeStr || "—";

    // Badges: color identity + rarity
    els.cardModalBadges.innerHTML = "";
    const colors = parseListString(raw["Color Identity"] || raw.color_identity || []);
    colors.forEach((c) => {
      const key = c.toUpperCase();
      const style = COLOR_MODAL_STYLE[key] || COLOR_MODAL_STYLE.C;
      const badge = document.createElement("span");
      badge.className = "card-modal-badge";
      badge.textContent = style.label;
      badge.style.background = style.bg;
      badge.style.color = style.fg;
      els.cardModalBadges.appendChild(badge);
    });
    const rarity = (raw.Rarity || raw.rarity || "").toLowerCase();
    if (rarity) {
      const rs = RARITY_STYLE[rarity] || { bg: "#444", fg: "#ccc" };
      const rb = document.createElement("span");
      rb.className = "card-modal-badge";
      rb.textContent = rarity.charAt(0).toUpperCase() + rarity.slice(1);
      rb.style.background = rs.bg;
      rb.style.color = rs.fg;
      els.cardModalBadges.appendChild(rb);
    }

    // Oracle text — loading state while we fetch from Scryfall
    els.cardModalOracle.textContent = "Loading card text…";
    els.cardModalOracle.classList.add("is-loading");
    els.cardModalFlavor.textContent = "";
    els.cardModalPT.textContent = "";

    // Remove handler
    els.cardModalRemove.onclick = () => {
      removeCard(card);
      els.cardModal.close();
    };

    els.cardModal.showModal();

    // Fetch Scryfall data for oracle text + P/T
    if (meta.cardId) {
      fetchScryfallCard(meta.cardId).then((sf) => {
        els.cardModalOracle.classList.remove("is-loading");
        if (!sf) {
          showTagFallback(card);
          return;
        }
        els.cardModalOracle.textContent = sf.oracle_text || sf.card_faces?.[0]?.oracle_text || "";
        if (!els.cardModalOracle.textContent) showTagFallback(card);
        const flavor = sf.flavor_text || sf.card_faces?.[0]?.flavor_text || "";
        els.cardModalFlavor.textContent = flavor;
        if (sf.power && sf.toughness) {
          els.cardModalPT.textContent = `${sf.power} / ${sf.toughness}`;
        }
      });
    } else {
      els.cardModalOracle.classList.remove("is-loading");
      showTagFallback(card);
    }
  }

  function bindModalEvents() {
    els.cardModalClose.addEventListener("click", () => els.cardModal.close());
    els.cardModal.addEventListener("click", (e) => {
      if (e.target === els.cardModal) els.cardModal.close();
    });
  }

  function bindBuilderEvents() {
    els.generateBtn.addEventListener("click", generateDeck);
    els.completeBtn.addEventListener("click", completeDeck);
    els.importBtn.addEventListener("click", importCards);
    els.saveBtn.addEventListener("click", saveDeck);

    els.commander.addEventListener("change", () => {
      state.commander = els.commander.value.trim();
      updateDeckHeader();
      scheduleAnalysis();
    });
    els.searchInput.addEventListener("input", () => {
      state.search = els.searchInput.value.trim();
      renderDeckBoard();
    });
    els.sortSelect.addEventListener("change", () => {
      state.sortBy = els.sortSelect.value;
      renderDeckBoard();
    });
    els.filterSelect.addEventListener("change", () => {
      state.filterCategory = els.filterSelect.value;
      renderDeckBoard();
    });
  }

  async function initBuilder() {
    bindBuilderEvents();
    bindModalEvents();
    renderCategoryFilter();
    renderDeckBoard();
    renderAnalysis();
    updateDeckHeader();

    const params = new URLSearchParams(window.location.search);
    const deckId = params.get("deck");
    if (deckId) {
      setBusy(true, "Loading saved deck...");
      try {
        await loadDeck(deckId);
      } catch (err) {
        reportError("initBuilder loadDeck failed", err);
        setStatus("Unable to load that saved deck.", "error");
      } finally {
        setBusy(false);
      }
      return;
    }

    setStatus("Ready.", "success");
  }

  async function initSavedPage() {
    await renderSavedDecks();
  }

  async function initAuthenticated(user) {
    state.user = user;
    if (els.userLabel) {
      const label = user.email || user.name || user.id || "Authenticated User";
      els.userLabel.textContent = `Signed in as ${label}`;
    }

    if (page === "saved") {
      await initSavedPage();
      return;
    }

    await initBuilder();
  }

  async function init() {
    try {
      const session = await getSession();
      if (!session.authenticated) {
        loginShell.hidden = false;
        appShell.hidden = true;
        return;
      }

      loginShell.hidden = true;
      appShell.hidden = false;
      await initAuthenticated(session.user || {});
    } catch (err) {
      reportError("init failed", err);
      loginShell.hidden = false;
      appShell.hidden = true;
      setStatus("Initialization failed.", "error");
    }
  }

  init();
})();
