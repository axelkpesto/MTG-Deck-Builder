<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';
app_start_session();
$user = $_SESSION['user'] ?? null;
?>
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MTG Deck Builder</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body data-page="builder">
  <section id="loginShell" class="login-shell"<?= $user ? ' hidden' : '' ?>>
    <div class="login-card">
      <div class="login-mark">MTG</div>
      <h1>Deck Builder</h1>
      <p>Generate, tune, analyze, and now save commander decks to your account.</p>
      <p><a class="button primary" href="/login.php">Login with OAuth</a></p>
    </div>
  </section>

  <div id="appShell" class="app-shell"<?= $user ? '' : ' hidden' ?>>
    <header class="site-header">
      <div class="site-brand">
        <div class="site-mark">MTG</div>
        <div>
          <div class="site-title">Deck Builder</div>
          <div class="site-subtitle">An AI-Powered Deck Builder</div>
        </div>
      </div>
      <nav class="site-nav">
        <a class="nav-link active" href="/">Builder</a>
        <a class="nav-link" href="/saved/">Saved</a>
      </nav>
      <div class="site-account">
        <div id="userLabel" class="account-label">Authenticated</div>
        <a class="button ghost" href="/logout.php">Logout</a>
      </div>
    </header>

    <main id="app" class="builder-layout">
      <section class="workspace">
        <div class="hero-panel">
          <div class="hero-copy">
            <div class="eyebrow">Commander Workspace</div>
            <h1 id="pageDeckTitle">Commander Builder</h1>
            <p id="heroSummary"></p>
          </div>
          <div class="hero-actions">
            <button id="saveBtn" class="button primary" type="button">Save Deck</button>
          </div>
        </div>

        <section class="panel builder-controls">
          <div class="controls-grid">
            <div class="group">
              <label for="commander">Commander</label>
              <input id="commander" class="input" type="text">
            </div>
            <div class="group">
              <label for="filterSelect">Filter</label>
              <select id="filterSelect" class="input"></select>
            </div>
            <div class="group">
              <label for="sortSelect">Sort</label>
              <select id="sortSelect" class="input">
                <option value="category">Group</option>
                <option value="name">Name</option>
                <option value="quantity">Quantity</option>
                <option value="tag">Primary Tag</option>
              </select>
            </div>
          </div>
          <div class="action-row">
            <button id="generateBtn" class="button primary" type="button">Generate Deck</button>
            <button id="completeBtn" class="button" type="button">Add Similar Cards</button>
            <button id="importBtn" class="button" type="button">Import Cards</button>
            <div class="deck-badge"><span id="cardCount">0 cards</span></div>
          </div>
          <div class="status-line">
            <span id="statusText"></span>
          </div>
        </section>

        <section class="panel import-panel">
          <div class="panel-heading">
            <div>
              <strong>Quick Import</strong>
            </div>
          </div>
          <div class="group">
            <label for="importCards">Import Cards</label>
            <textarea id="importCards" class="textarea"></textarea>
          </div>
        </section>

        <section class="panel deck-panel">
          <div class="panel-heading">
            <div>
              <strong>Deck Stacks</strong>
            </div>
          </div>
          <div class="deck-toolbar">
            <div class="group deck-search">
              <label for="searchInput">Search Deck</label>
              <input id="searchInput" class="input" type="text">
            </div>
          </div>
          <div id="deckBoard" class="deck-board"></div>
        </section>

        <section class="panel stats-panel">
          <div class="panel-heading">
            <div>
              <strong>Deck Stats</strong>
              <div class="subtle" id="analysisMeta"></div>
            </div>
          </div>
          <div id="analysisEmpty" class="empty-state"></div>
          <div id="analysisPanel" hidden>
            <div id="analysisSummary" class="metric-grid stats-summary-grid"></div>
            <div class="stats-layout">
              <div class="stats-main">
                <div class="stats-bar-row">
                  <span class="stats-bar-label">Cost</span>
                  <div class="stats-bar-track" id="costBar"></div>
                </div>
                <div class="stats-bar-row">
                  <span class="stats-bar-label">Production</span>
                  <div class="stats-bar-track" id="productionBar"></div>
                </div>
                <div class="color-stat-grid" id="colorGrid"></div>
                <div class="analysis-section-heading">Mana Curve</div>
                <div id="analysisCurve" class="curve-grid"></div>
                <div class="analysis-section-heading">Primary Tags</div>
                <div id="analysisTags" class="chip-wrap"></div>
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  </div>

  <dialog id="cardModal" class="card-modal">
    <div class="card-modal-inner">
      <button id="cardModalClose" class="card-modal-close" type="button" aria-label="Close">×</button>
      <div class="card-modal-body">
        <div class="card-modal-image-wrap">
          <img id="cardModalImage" class="card-modal-image" alt="">
        </div>
        <div class="card-modal-info">
          <div class="card-modal-header">
            <h2 id="cardModalName" class="card-modal-name"></h2>
            <span id="cardModalCmc" class="card-modal-cmc"></span>
          </div>
          <div id="cardModalTypeline" class="card-modal-typeline"></div>
          <div id="cardModalBadges" class="card-modal-badges"></div>
          <div id="cardModalOracle" class="card-modal-oracle"></div>
          <div id="cardModalFlavor" class="card-modal-flavor"></div>
          <div id="cardModalPT" class="card-modal-pt"></div>
          <div class="card-modal-actions">
            <button id="cardModalRemove" class="button ghost" type="button">Remove from deck</button>
          </div>
        </div>
      </div>
    </div>
  </dialog>

  <script src="/app.js"></script>
</body>
</html>
