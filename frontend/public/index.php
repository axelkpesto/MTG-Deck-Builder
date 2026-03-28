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
<body>
  <section id="loginShell" class="login-shell"<?= $user ? ' hidden' : '' ?>>
    <div class="card">
      <h1>MTG Deck Builder</h1>
      <p>Sign in to generate, tag, and analyze commander decks.</p>
      <p><a class="button primary" href="/login.php">Login with OAuth</a></p>
    </div>
  </section>

  <main id="app"<?= $user ? '' : ' hidden' ?>>
    <div class="topbar">
      <div class="brand">
        <h1>MTG Deck Builder</h1>
        <p id="userLabel">Authenticated</p>
      </div>
      <div class="top-actions">
        <a class="button ghost" href="/logout.php">Logout</a>
      </div>
    </div>

    <section class="panel controls">
      <div class="row split">
        <div class="group flex-1">
          <label for="deckName">Deck Name</label>
          <input id="deckName" class="input" type="text" placeholder="Atraxa Counters">
        </div>
        <div class="group flex-1">
          <label for="commander">Commander</label>
          <input id="commander" class="input" type="text" placeholder="Atraxa, Praetors' Voice">
        </div>
      </div>
      <div class="row">
        <button id="generateBtn" class="button primary" type="button">Generate Deck</button>
        <button id="completeBtn" class="button" type="button">Add Similar Cards</button>
      </div>
      <div class="status-line">
        <span id="statusText"></span>
      </div>
    </section>

    <section class="panel">
      <div class="group">
        <label for="importCards">Import Cards</label>
        <textarea id="importCards" class="textarea" placeholder="Sol Ring&#10;Arcane Signet&#10;3 Island"></textarea>
      </div>
      <div class="row end">
        <button id="importBtn" class="button" type="button">Add Cards</button>
      </div>
    </section>

    <section class="panel">
      <div class="table-header">
        <div>
          <strong>Deck Analysis</strong>
          <div class="subtle" id="analysisMeta">Analysis updates automatically as the deck changes.</div>
        </div>
      </div>
      <div id="analysisEmpty" class="empty-state">Add cards or generate a deck to see analysis.</div>
      <div id="analysisPanel" class="analysis-grid" hidden>
        <article class="analysis-card">
          <h3>Summary</h3>
          <div id="analysisSummary" class="metric-grid"></div>
        </article>
        <article class="analysis-card">
          <h3>Primary Tags</h3>
          <div id="analysisTags" class="chip-wrap"></div>
        </article>
        <article class="analysis-card">
          <h3>Mana Curve</h3>
          <div id="analysisCurve" class="curve-grid"></div>
        </article>
        <article class="analysis-card">
          <h3>Color Distribution</h3>
          <div id="analysisColors" class="stack-list"></div>
        </article>
      </div>
    </section>

    <section class="panel">
      <div class="table-header">
        <strong>Deck List</strong>
        <span id="cardCount">0 cards</span>
      </div>
      <div class="controls compact">
        <div class="group">
          <label for="searchInput">Search</label>
          <input id="searchInput" class="input" type="text" placeholder="Search cards">
        </div>
        <div class="group">
          <label for="sortSelect">Sort</label>
          <select id="sortSelect" class="input">
            <option value="tag">Tag</option>
            <option value="name">Name</option>
            <option value="quantity">Quantity</option>
          </select>
        </div>
        <div class="group">
          <label for="filterSelect">Filter</label>
          <select id="filterSelect" class="input"></select>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Qty</th>
              <th>Card</th>
              <th>Primary Tag</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="cardRows"></tbody>
        </table>
      </div>
    </section>
  </main>

  <div id="cardPreview" class="card-preview" hidden>
    <div id="cardPreviewName" class="card-preview-name"></div>
    <img id="cardPreviewImage" class="card-preview-image" alt="">
  </div>

  <script src="/app.js"></script>
</body>
</html>
