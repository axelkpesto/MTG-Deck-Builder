(function () {
  "use strict";

  const loginShell = document.getElementById("loginShell");
  const app = document.getElementById("app");
  if (!app || !loginShell) return;

  const state = {
	cards: [],
	sortBy: "tag",
	filterTag: "",
	search: "",
	commander: "",
	loading: false,
  };

  const TAGS = [
	"Commander",
	"Ramp",
	"Draw",
	"Removal",
	"Protection",
	"Board Wipe",
	"Recursion",
	"Token",
	"Land",
	"Utility",
	"Untagged",
  ];

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

  const els = {
	userLabel: document.getElementById("userLabel"),
	deckName: document.getElementById("deckName"),
	commander: document.getElementById("commander"),
	generateBtn: document.getElementById("generateBtn"),
	completeBtn: document.getElementById("completeBtn"),
	analyzeBtn: document.getElementById("analyzeBtn"),
	statusBtn: document.getElementById("statusBtn"),
	importCards: document.getElementById("importCards"),
	importBtn: document.getElementById("importBtn"),
	autotagBtn: document.getElementById("autotagBtn"),
	searchInput: document.getElementById("searchInput"),
	sortSelect: document.getElementById("sortSelect"),
	filterSelect: document.getElementById("filterSelect"),
	cardRows: document.getElementById("cardRows"),
	cardCount: document.getElementById("cardCount"),
	statusText: document.getElementById("statusText"),
	output: document.getElementById("output"),
  };

  async function getSession() {
	const res = await fetch("/session.php", { method: "GET" });
	if (!res.ok) {
	  throw new Error(`Session check failed: ${res.status}`);
	}
	return res.json();
  }

  function setBusy(value, text) {
	state.loading = value;
	els.statusText.textContent = text || "";
	[
	  els.generateBtn,
	  els.completeBtn,
	  els.analyzeBtn,
	  els.statusBtn,
	  els.importBtn,
	  els.autotagBtn,
	].forEach((btn) => {
	  btn.disabled = value;
	});
  }

  function setOutput(value) {
	els.output.textContent =
	  typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  function uid() {
	return (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}_${Math.random().toString(16).slice(2)}`;
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

  function mergeCards(items) {
	for (const item of items) {
	  const existing = state.cards.find(
		(c) => c.name.toLowerCase() === item.name.toLowerCase()
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

  function getDisplayCards() {
	let list = state.cards.slice();

	if (state.search) {
	  const q = state.search.toLowerCase();
	  list = list.filter((c) => c.name.toLowerCase().includes(q));
	}

	if (state.filterTag) {
	  list = list.filter((c) => (c.primaryTag || "Untagged") === state.filterTag);
	}

	if (state.sortBy === "name") {
	  list.sort((a, b) => a.name.localeCompare(b.name));
	} else if (state.sortBy === "quantity") {
	  list.sort((a, b) => b.quantity - a.quantity);
	} else if (state.sortBy === "tag") {
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
	all.textContent = "Filter: All Tags";
	els.filterSelect.appendChild(all);

	tags.forEach((tag) => {
	  const opt = document.createElement("option");
	  opt.value = tag;
	  opt.textContent = `Tag: ${tag}`;
	  if (state.filterTag === tag) opt.selected = true;
	  els.filterSelect.appendChild(opt);
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
	  });
	  qtyTd.appendChild(qtyInput);

	  const nameTd = document.createElement("td");
	  nameTd.textContent = card.name;

	  const tagTd = document.createElement("td");
	  const wrap = document.createElement("div");
	  wrap.style.display = "flex";
	  wrap.style.alignItems = "center";
	  wrap.style.gap = "8px";

	  const dot = document.createElement("span");
	  dot.className = "tag-dot";
	  dot.style.background = TAG_COLORS[card.primaryTag || "Untagged"] || "#6f8099";

	  const tagSelect = document.createElement("select");
	  tagSelect.className = "input tag-select";
	  TAGS.forEach((tag) => {
		const opt = document.createElement("option");
		opt.value = tag;
		opt.textContent = tag;
		if ((card.primaryTag || "Untagged") === tag) opt.selected = true;
		tagSelect.appendChild(opt);
	  });
	  tagSelect.addEventListener("change", () => {
		const next = tagSelect.value;
		card.primaryTag = next === "Untagged" ? "" : next;
		renderTagFilter();
		renderRows();
	  });

	  wrap.appendChild(dot);
	  wrap.appendChild(tagSelect);
	  tagTd.appendChild(wrap);

	  const actionTd = document.createElement("td");
	  const removeBtn = document.createElement("button");
	  removeBtn.className = "button danger";
	  removeBtn.textContent = "Remove";
	  removeBtn.addEventListener("click", () => {
		state.cards = state.cards.filter((c) => c.id !== card.id);
		renderTagFilter();
		renderRows();
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

  function updateCounts() {
	const total = state.cards.reduce((sum, c) => sum + c.quantity, 0);
	els.cardCount.textContent = `${total} cards`;
  }

  async function autoTagUntagged() {
	const targets = state.cards.filter((c) => !c.primaryTag);
	if (targets.length === 0) {
	  setOutput("No untagged cards.");
	  return;
	}

	setBusy(true, "Auto-tagging...");
	try {
	  const results = await Promise.allSettled(
		targets.map(async (card) => {
		  const data = await callApi({
			path: `/get_tags/${encodeURIComponent(card.name)}`,
			method: "GET",
			query: { threshold: 0.5, top_k: 8 },
		  });
		  return { card, predicted: Array.isArray(data.predicted) ? data.predicted : [] };
		})
	  );

	  results.forEach((result) => {
		if (result.status === "fulfilled") {
		  const top = result.value.predicted[0] || "";
		  result.value.card.tags = result.value.predicted;
		  result.value.card.primaryTag = top;
		}
	  });

	  renderTagFilter();
	  renderRows();
	  setOutput({ tagged: targets.length });
	} catch (err) {
	  setOutput(String(err));
	} finally {
	  setBusy(false, "");
	}
  }

  async function generateDeck() {
	const commander = els.commander.value.trim();
	if (!commander) {
	  setOutput("Commander is required.");
	  return;
	}
	state.commander = commander;
	setBusy(true, "Generating deck...");
	try {
	  const data = await callApi({
		path: `/generate_deck/${encodeURIComponent(commander)}`,
		method: "GET",
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

	  Object.entries(deckCounts).forEach(([name, quantity]) => {
		if (name.toLowerCase() === commander.toLowerCase()) return;
		const q = Number(quantity);
		if (!Number.isFinite(q) || q <= 0) return;
		state.cards.push({
		  id: uid(),
		  name,
		  quantity: Math.floor(q),
		  tags: [],
		  primaryTag: "",
		});
	  });

	  renderTagFilter();
	  renderRows();
	  setOutput(data);
	  await autoTagUntagged();
	} catch (err) {
	  setOutput(String(err));
	} finally {
	  setBusy(false, "");
	}
  }

  async function completeDeck() {
	const commander =
	  els.commander.value.trim() ||
	  state.commander ||
	  state.cards.find((c) => c.primaryTag === "Commander")?.name ||
	  "";

	if (!commander) {
	  setOutput("Set a commander first.");
	  return;
	}

	setBusy(true, "Completing deck...");
	try {
	  const data = await callApi({
		path: `/get_similar_vectors/${encodeURIComponent(commander)}`,
		method: "GET",
		query: { num_vectors: 30 },
	  });

	  const existing = new Set(state.cards.map((c) => c.name.toLowerCase()));
	  const names = Object.keys(data || {});
	  const additions = [];
	  for (const name of names) {
		if (existing.has(name.toLowerCase())) continue;
		additions.push({ name, quantity: 1 });
		if (additions.length >= 10) break;
	  }

	  mergeCards(additions);
	  renderTagFilter();
	  renderRows();
	  setOutput({ added: additions.length, sample: additions.slice(0, 5) });
	  await autoTagUntagged();
	} catch (err) {
	  setOutput(String(err));
	} finally {
	  setBusy(false, "");
	}
  }

  async function analyzeDeck() {
	const commander =
	  els.commander.value.trim() ||
	  state.commander ||
	  state.cards.find((c) => c.primaryTag === "Commander")?.name ||
	  "";

	if (!commander || state.cards.length === 0) {
	  setOutput("Need commander and cards.");
	  return;
	}

	setBusy(true, "Analyzing...");
	try {
	  const data = await callApi({
		path: "/analyze_deck",
		method: "POST",
		body: {
		  commander,
		  cards: state.cards.map((c) => c.name),
		},
	  });
	  setOutput(data);
	} catch (err) {
	  setOutput(String(err));
	} finally {
	  setBusy(false, "");
	}
  }

  async function checkStatus() {
	setBusy(true, "Checking status...");
	try {
	  const data = await callApi({ path: "/status", method: "GET" });
	  setOutput(data);
	} catch (err) {
	  setOutput(String(err));
	} finally {
	  setBusy(false, "");
	}
  }

  function importCards() {
	const lines = els.importCards.value.split(/\r?\n/);
	const parsed = lines.map(parseImportLine).filter(Boolean);
	if (parsed.length === 0) {
	  setOutput("No cards to import.");
	  return;
	}
	mergeCards(parsed);
	els.importCards.value = "";
	renderTagFilter();
	renderRows();
	setOutput({ imported: parsed.length });
  }

  function bindEvents() {
	els.generateBtn.addEventListener("click", generateDeck);
	els.completeBtn.addEventListener("click", completeDeck);
	els.analyzeBtn.addEventListener("click", analyzeDeck);
	els.statusBtn.addEventListener("click", checkStatus);
	els.importBtn.addEventListener("click", importCards);
	els.autotagBtn.addEventListener("click", autoTagUntagged);

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
  }

  function initAuthenticated(user) {
	const label = user.email || user.name || user.id || "Authenticated User";
	els.userLabel.textContent = `Signed in as ${label}`;
	bindEvents();
	renderTagFilter();
	renderRows();
	setOutput("Ready.");
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
	  loginShell.hidden = false;
	  app.hidden = true;
	  setOutput(String(err));
	}
  }

  init();
})();
