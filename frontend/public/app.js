(function () {
  "use strict";

  const loginShell = document.getElementById("loginShell");
  const app = document.getElementById("app");
  if (!app || !loginShell) return;

  const TAG_COLORS = {
    Commander: "#f6c86f",
    Ramp: "#8cffc6",
    Draw: "#87ccff",
    Removal: "#ff8e8e",
    Protection: "#c6b0ff",
    "Board Wipe": "#ffae70",
    Recursion: "#7dd1a8",
    Token: "#f8f28b",
    Land: "#c1a98a",
    Utility: "#9eb0c8",
    Untagged: "#6f8099",
  };

  const state = {
    cards: [],
    sortBy: "tag",
    filterTag: "",
    search: "",
    commander: "",
    loading: false,
    analysis: null,
    analysisTimer: null,
    analysisRequestId: 0,
    metaCache: {},
  };

  const els = {
    userLabel: document.getElementById("userLabel"),
    deckName: document.getElementById("deckName"),
    commander: document.getElementById("commander"),
    generateBtn: document.getElementById("generateBtn"),
    completeBtn: document.getElementById("completeBtn"),
    importCards: document.getElementById("importCards"),
    importBtn: document.getElementById("importBtn"),
    searchInput: document.getElementById("searchInput"),
    sortSelect: document.getElementById("sortSelect"),
    filterSelect: document.getElementById("filterSelect"),
    cardRows: document.getElementById("cardRows"),
    cardCount: document.getElementById("cardCount"),
    statusText: document.getElementById("statusText"),
    analysisMeta: document.getElementById("analysisMeta"),
    analysisEmpty: document.getElementById("analysisEmpty"),
    analysisPanel: document.getElementById("analysisPanel"),
    analysisSummary: document.getElementById("analysisSummary"),
    analysisTags: document.getElementById("analysisTags"),
    analysisCurve: document.getElementById("analysisCurve"),
    analysisColors: document.getElementById("analysisColors"),
    cardPreview: document.getElementById("cardPreview"),
    cardPreviewName: document.getElementById("cardPreviewName"),
    cardPreviewImage: document.getElementById("cardPreviewImage"),
  };

  function uid() {
    return (
      (crypto.randomUUID && crypto.randomUUID()) ||
      `${Date.now()}_${Math.random().toString(16).slice(2)}`
    );
  }

  function setStatus(text, tone) {
    els.statusText.textContent = text || "";
    els.statusText.dataset.tone = tone || "";
  }

  function setBusy(value, text) {
    state.loading = value;
    if (typeof text === "string") {
      setStatus(text, value ? "busy" : "");
    }
    [els.generateBtn, els.completeBtn, els.importBtn].forEach((btn) => {
      if (btn) btn.disabled = value;
    });
  }

  async function getSession() {
    const res = await fetch("/session.php", { method: "GET" });
    if (!res.ok) {
      throw new Error(`Session check failed: ${res.status}`);
    }
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

  function reportError(context, err) {
    console.error(context, err);
  }

  async function callLocalJson(url, payload) {
    const res = await fetch(url, {
      method: payload ? "POST" : "GET",
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

  function parseImportLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return null;

    const leadQty = trimmed.match(/^(\d+)\s+(.+)$/);
    if (leadQty) {
      return { quantity: Number(leadQty[1]), name: leadQty[2].trim() };
    }

    const trailQty = trimmed.match(/^(.+)\s+x(\d+)$/i);
    if (trailQty) {
      return { quantity: Number(trailQty[2]), name: trailQty[1].trim() };
    }

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

  function getTagColor(tag) {
    return TAG_COLORS[formatTag(tag)] || "#6f8099";
  }

  function mergeCards(items) {
    for (const item of items) {
      const existing = state.cards.find(
        (c) => c.name.toLowerCase() === item.name.toLowerCase(),
      );
      if (existing) {
        existing.quantity += item.quantity;
      } else {
        state.cards.push({
          id: uid(),
          name: item.name,
          quantity: item.quantity,
          tags: [],
          primaryTag: "",
        });
      }
    }
  }

  function ensureCommanderCardListed() {
    const commander = els.commander.value.trim();
    if (!commander) return;

    state.commander = commander;
    const existing = state.cards.find(
      (card) => card.name.toLowerCase() === commander.toLowerCase(),
    );

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

  function buildAnalysisCards() {
    const cards = [];
    state.cards.forEach((card) => {
      for (let i = 0; i < card.quantity; i += 1) {
        cards.push(card.name);
      }
    });
    return cards;
  }

  function getCommander() {
    return (
      els.commander.value.trim() ||
      state.commander ||
      state.cards.find((c) => c.primaryTag === "Commander")?.name ||
      ""
    );
  }

  function getDisplayCards() {
    let list = state.cards.slice();

    if (state.search) {
      const q = state.search.toLowerCase();
      list = list.filter((c) => c.name.toLowerCase().includes(q));
    }

    if (state.filterTag) {
      list = list.filter(
        (c) => (c.primaryTag || "Untagged") === state.filterTag,
      );
    }

    if (state.sortBy === "name") {
      list.sort((a, b) => a.name.localeCompare(b.name));
    } else if (state.sortBy === "quantity") {
      list.sort(
        (a, b) => b.quantity - a.quantity || a.name.localeCompare(b.name),
      );
    } else {
      list.sort((a, b) => {
        const at = a.primaryTag || "Untagged";
        const bt = b.primaryTag || "Untagged";
        if (at === "Commander") return -1;
        if (bt === "Commander") return 1;
        return at.localeCompare(bt) || a.name.localeCompare(b.name);
      });
    }

    return list;
  }

  function uniqueTags() {
    const tags = new Set();
    for (const card of state.cards) {
      tags.add(card.primaryTag || "Untagged");
    }
    return Array.from(tags).sort((a, b) => a.localeCompare(b));
  }

  function renderTagFilter() {
    const tags = uniqueTags();
    els.filterSelect.innerHTML = "";
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All Tags";
    els.filterSelect.appendChild(all);

    tags.forEach((tag) => {
      const opt = document.createElement("option");
      opt.value = tag;
      opt.textContent = formatTag(tag);
      if (state.filterTag === tag) opt.selected = true;
      els.filterSelect.appendChild(opt);
    });
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
    if (!state.analysis) {
      els.analysisPanel.hidden = true;
      els.analysisEmpty.hidden = false;
      els.analysisSummary.innerHTML = "";
      els.analysisTags.innerHTML = "";
      els.analysisCurve.innerHTML = "";
      els.analysisColors.innerHTML = "";
      return;
    }

    const summary = state.analysis;
    const tags = summary.tags?.tag_counts || {};
    const colors = summary.color_distribution?.colors?.counts || {};
    const colorPercents = summary.color_distribution?.colors?.percent || {};
    const curve = summary.curve?.mana_curve?.counts || [];
    const lands = summary.lands?.lands || {};

    els.analysisEmpty.hidden = true;
    els.analysisPanel.hidden = false;
    els.analysisSummary.innerHTML = "";
    els.analysisSummary.appendChild(
      renderSummaryMetric("Cards", buildAnalysisCards().length),
    );
    els.analysisSummary.appendChild(
      renderSummaryMetric("Unique", state.cards.length),
    );
    els.analysisSummary.appendChild(
      renderSummaryMetric("Lands", lands.land_count || 0),
    );
    els.analysisSummary.appendChild(
      renderSummaryMetric("Basics", lands.basic_count || 0),
    );

    const topTags = Object.entries(tags)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    els.analysisTags.innerHTML = "";
    if (topTags.length === 0) {
      const empty = document.createElement("div");
      empty.className = "subtle";
      empty.textContent = "No tag data available yet.";
      els.analysisTags.appendChild(empty);
    } else {
      topTags.forEach(([tag, count]) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = `${formatTag(tag)} ${count}`;
        els.analysisTags.appendChild(chip);
      });
    }

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

    els.analysisColors.innerHTML = "";
    ["W", "U", "B", "R", "G"].forEach((color) => {
      const row = document.createElement("div");
      row.className = "stack-row";

      const top = document.createElement("div");
      top.className = "stack-top";
      top.textContent = `${color}: ${colors[color] || 0}`;

      const track = document.createElement("div");
      track.className = "stack-track";

      const fill = document.createElement("div");
      fill.className = "stack-fill";
      fill.style.width = `${Math.round(Number(colorPercents[color] || 0) * 100)}%`;

      track.appendChild(fill);
      row.appendChild(top);
      row.appendChild(track);
      els.analysisColors.appendChild(row);
    });
  }

  function updateCounts() {
    const total = state.cards.reduce((sum, c) => sum + c.quantity, 0);
    els.cardCount.textContent = `${total} cards`;
  }

  function positionPreview(event) {
    const x = event.clientX + 18;
    const y = event.clientY + 18;
    els.cardPreview.style.left = `${x}px`;
    els.cardPreview.style.top = `${y}px`;
  }

  function hidePreview() {
    els.cardPreview.hidden = true;
  }

  function imageUrlFromCardId(cardId) {
    if (!cardId) return "";
    return `https://api.scryfall.com/cards/${encodeURIComponent(cardId)}?format=image&version=normal`;
  }

  async function fetchCardImages(names) {
    if (!Array.isArray(names) || names.length === 0) {
      return { found: {}, missing: {} };
    }
    return callLocalJson("/card_images.php", { cards: names });
  }

  async function loadCardMeta(name) {
    const cached = state.metaCache[name];
    if (cached && Array.isArray(cached.imageUrls)) {
      return cached;
    }

    let meta = cached;
    if (!meta) {
      const data = await callApi({
        path: "/get_vector_description",
        method: "POST",
        body: { id: name },
      });
      meta = {
        name: data.card_name || name,
        cardId: data.card_id || "",
        imageUrls: [],
      };
    }

    const imageData = await fetchCardImages([meta.name || name]);
    meta.imageUrls =
      imageData.found?.[meta.name] ||
      imageData.found?.[name] ||
      meta.imageUrls ||
      [];

    state.metaCache[name] = meta;
    state.metaCache[meta.name] = meta;
    return meta;
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
    const normalized = normalizeTagPayload(
      tagData.predicted,
      tagData.predicted_scores || tagData.scores,
    );
    const resolvedName =
      typeof meta.card_name === "string" && meta.card_name.trim()
        ? meta.card_name.trim()
        : item.name;
    state.metaCache[resolvedName] = {
      name: resolvedName,
      cardId: meta.card_id || "",
    };
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

      state.metaCache[resolvedName] = {
        name: resolvedName,
        cardId: meta.card_id || "",
        imageUrls: imageData.found?.[item.name] || [],
      };

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
  function showPreview(card, event) {
    els.cardPreview.hidden = false;
    els.cardPreview.dataset.cardName = card.name;
    positionPreview(event);
    els.cardPreviewName.textContent = card.name;
    els.cardPreviewImage.removeAttribute("src");
    els.cardPreviewImage.alt = card.name;
    els.cardPreviewImage.dataset.state = "loading";

    loadCardMeta(card.name)
      .then((meta) => {
        if (
          els.cardPreview.hidden ||
          els.cardPreview.dataset.cardName !== card.name
        )
          return;
        els.cardPreviewName.textContent = meta.name || card.name;
        const imageUrl = meta.imageUrls?.[0] || imageUrlFromCardId(meta.cardId);
        if (!imageUrl) {
          els.cardPreviewImage.dataset.state = "missing";
          return;
        }
        els.cardPreviewImage.src = imageUrl;
        els.cardPreviewImage.dataset.state = "ready";
      })
      .catch(() => {
        if (els.cardPreview.dataset.cardName !== card.name) return;
        els.cardPreviewImage.dataset.state = "missing";
      });
  }

  function renderRows() {
    const rows = getDisplayCards();
    els.cardRows.innerHTML = "";

    rows.forEach((card) => {
      const tr = document.createElement("tr");

      const qtyTd = document.createElement("td");
      const qtyInput = document.createElement("input");
      qtyInput.className = "input qty-input";
      qtyInput.type = "number";
      qtyInput.min = "1";
      qtyInput.value = String(card.quantity);
      qtyInput.addEventListener("change", () => {
        const v = Number(qtyInput.value);
        if (!Number.isFinite(v) || v <= 0) return;
        card.quantity = Math.floor(v);
        updateCounts();
        scheduleAnalysis(true);
      });
      qtyTd.appendChild(qtyInput);

      const nameTd = document.createElement("td");
      const nameBtn = document.createElement("button");
      nameBtn.className = "card-name-button";
      nameBtn.type = "button";
      nameBtn.textContent = card.name;
      nameBtn.addEventListener("mouseenter", (event) =>
        showPreview(card, event),
      );
      nameBtn.addEventListener("mousemove", positionPreview);
      nameBtn.addEventListener("mouseleave", hidePreview);
      nameBtn.addEventListener("focus", (event) => showPreview(card, event));
      nameBtn.addEventListener("blur", hidePreview);
      nameTd.appendChild(nameBtn);

      const tagTd = document.createElement("td");
      const wrap = document.createElement("div");
      wrap.className = "tag-control";

      const dot = document.createElement("span");
      dot.className = "tag-dot";
      dot.style.background = getTagColor(card.primaryTag || "Untagged");

      const tagLabel = document.createElement("span");
      tagLabel.textContent = formatTag(card.primaryTag);

      wrap.appendChild(dot);
      wrap.appendChild(tagLabel);
      tagTd.appendChild(wrap);

      const actionTd = document.createElement("td");
      const removeBtn = document.createElement("button");
      removeBtn.className = "button danger";
      removeBtn.type = "button";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", () => {
        state.cards = state.cards.filter((c) => c.id !== card.id);
        renderTagFilter();
        renderRows();
        scheduleAnalysis(true);
      });
      actionTd.appendChild(removeBtn);

      tr.appendChild(qtyTd);
      tr.appendChild(nameTd);
      tr.appendChild(tagTd);
      tr.appendChild(actionTd);
      els.cardRows.appendChild(tr);
    });

    updateCounts();
  }

  async function runAnalysis(options) {
    const settings = Object.assign(
      { immediate: false, showBusy: false },
      options,
    );
    const commander = getCommander();
    const cards = buildAnalysisCards();

    if (!commander || cards.length === 0) {
      state.analysis = null;
      els.analysisMeta.textContent =
        "Add cards or generate a deck to see analysis.";
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
      runAnalysis({ immediate: true, showBusy: false });
      return;
    }
    state.analysisTimer = window.setTimeout(() => {
      runAnalysis({ immediate: false, showBusy: false });
    }, 500);
  }

  async function generateDeck() {
    const commander = els.commander.value.trim();
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

      state.cards = [
        {
          id: uid(),
          name: commander,
          quantity: 1,
          tags: ["Commander"],
          primaryTag: "Commander",
        },
      ];

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

      if (enriched.missing.length > 0) {
        console.warn(
          "Skipped generated cards missing metadata or tags.",
          enriched.missing,
        );
      }
      renderTagFilter();
      renderRows();
      await runAnalysis({ immediate: true, showBusy: false });
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

      const existing = new Set(state.cards.map((c) => c.name.toLowerCase()));
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
        const taggedByName = new Map(
          enriched.cards.map((card) => [card.name.toLowerCase(), card]),
        );
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
        if (enriched.missing.length > 0) {
          console.warn(
            "Skipped similar cards missing metadata or tags.",
            enriched.missing,
          );
        }
      }
      renderTagFilter();
      renderRows();
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
        state.cards = state.cards.map((card) => {
          const validCard = validCards.find(
            (item) => item.name.toLowerCase() === card.name.toLowerCase(),
          );
          if (!validCard) return card;
          return {
            ...card,
            name: validCard.name,
            tags: validCard.tags,
            primaryTag: validCard.primaryTag,
          };
        });
        els.importCards.value = "";
        renderTagFilter();
        renderRows();
        await runAnalysis({ immediate: true, showBusy: false });
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

  function bindEvents() {
    els.generateBtn.addEventListener("click", generateDeck);
    els.completeBtn.addEventListener("click", completeDeck);
    els.importBtn.addEventListener("click", importCards);

    els.commander.addEventListener("change", () => {
      state.commander = els.commander.value.trim();
      scheduleAnalysis();
    });

    els.searchInput.addEventListener("input", () => {
      state.search = els.searchInput.value.trim();
      renderRows();
    });

    els.sortSelect.addEventListener("change", () => {
      state.sortBy = els.sortSelect.value;
      renderRows();
    });

    els.filterSelect.addEventListener("change", () => {
      state.filterTag = els.filterSelect.value;
      renderRows();
    });

    window.addEventListener("scroll", hidePreview, { passive: true });
  }

  function initAuthenticated(user) {
    const label = user.email || user.name || user.id || "Authenticated User";
    els.userLabel.textContent = `Signed in as ${label}`;
    bindEvents();
    renderTagFilter();
    renderRows();
    renderAnalysis();
    setStatus("Ready.", "success");
  }

  async function init() {
    try {
      const session = await getSession();
      if (!session.authenticated) {
        loginShell.hidden = false;
        app.hidden = true;
        return;
      }

      loginShell.hidden = true;
      app.hidden = false;
      initAuthenticated(session.user || {});
    } catch (err) {
      reportError("init failed", err);
      loginShell.hidden = false;
      app.hidden = true;
      setStatus("Initialization failed.", "error");
    }
  }

  init();
})();
