<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';
app_start_session();
$user = $_SESSION['user'] ?? null;
session_write_close();
$initialSession = json_encode([
    'authenticated' => is_array($user),
    'user'          => $user,
]);
?>
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MTG Deck Builder</title>
  <link rel="icon" href="/assets/color-wheel.png" type="image/png">
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
          </div>
          <div class="action-row">
            <button id="generateBtn" class="button primary" type="button">Generate Deck</button>
            <button id="completeBtn" class="button" type="button">Add Similar Cards</button>
            <div class="deck-badge"><span id="cardCount">0 cards</span></div>
          </div>
          <div class="status-line">
            <span id="statusText"></span>
          </div>
        </section>


        <section class="panel deck-panel">
          <div class="panel-heading">
            <div>
              <strong>Deck Stacks</strong>
            </div>
            <div class="deck-panel-actions">
              <button id="importToggleBtn" class="button ghost" type="button">Import</button>
              <button id="downloadBtn" class="button ghost" type="button">Download</button>
            </div>
          </div>
          <div class="deck-toolbar">
            <div class="group deck-search">
              <label for="searchInput">Search Deck</label>
              <input id="searchInput" class="input" type="text">
            </div>
            <div class="group">
              <label for="filterSelect">Filter</label>
              <select id="filterSelect" class="input"></select>
            </div>
            <div class="group">
              <label>View</label>
              <div class="view-toggle">
                <button id="viewStack" class="view-toggle-btn is-active" type="button" title="Stack view">
                  <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2" y="2" width="7" height="9" rx="1.5" fill="currentColor"/>
                    <rect x="11" y="2" width="7" height="9" rx="1.5" fill="currentColor"/>
                    <rect x="2" y="13" width="7" height="5" rx="1.5" fill="currentColor" opacity=".4"/>
                    <rect x="11" y="13" width="7" height="5" rx="1.5" fill="currentColor" opacity=".4"/>
                  </svg>
                  Stacks
                </button>
                <button id="viewList" class="view-toggle-btn" type="button" title="List view">
                  <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2" y="3" width="16" height="2.5" rx="1.25" fill="currentColor"/>
                    <rect x="2" y="8.75" width="16" height="2.5" rx="1.25" fill="currentColor"/>
                    <rect x="2" y="14.5" width="16" height="2.5" rx="1.25" fill="currentColor"/>
                  </svg>
                  List
                </button>
              </div>
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

  <dialog id="importModal" class="deck-modal">
    <div class="deck-modal-inner">
      <div class="deck-modal-header">
        <strong>Import Cards</strong>
        <button id="importCloseBtn" class="card-modal-close" type="button" aria-label="Close">×</button>
      </div>
      <div class="group">
        <label for="importFile">Upload a text file (.txt)</label>
        <div class="file-input-wrap">
          <input id="importFile" class="input" type="file" accept=".txt,.csv">
          <button id="importFileClear" class="file-clear-btn" type="button" aria-label="Remove file" hidden>×</button>
        </div>
      </div>
      <div class="group">
        <label for="importCards">Or paste a deck list below</label>
        <textarea id="importCards" class="textarea" placeholder="1 Sol Ring&#10;4x Lightning Bolt&#10;..."></textarea>
      </div>
      <div class="action-row">
        <button id="importBtn" class="button primary" type="button">Import Cards</button>
      </div>
      <div id="importFailed" class="import-failed" hidden></div>
    </div>
  </dialog>

  <dialog id="exportModal" class="deck-modal export-modal">
    <div class="deck-modal-inner">
      <div class="deck-modal-header">
        <strong>Download Deck</strong>
        <button id="exportModalClose" class="card-modal-close" type="button" aria-label="Close">×</button>
      </div>
      <div class="export-options">
        <button id="exportCsvBtn" class="export-option-card" type="button">
          <svg class="export-option-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="4" y="4" width="40" height="40" rx="4" stroke="currentColor" stroke-width="3"/>
            <line x1="4" y1="17" x2="44" y2="17" stroke="currentColor" stroke-width="3"/>
            <line x1="4" y1="31" x2="44" y2="31" stroke="currentColor" stroke-width="3"/>
            <line x1="19" y1="4" x2="19" y2="44" stroke="currentColor" stroke-width="3"/>
          </svg>
          <span class="export-option-label">Download CSV</span>
        </button>
        <button id="exportTxtBtn" class="export-option-card" type="button">
          <svg class="export-option-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 4h20l10 10v30H10V4z" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/>
            <path d="M30 4v10h10" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/>
            <line x1="16" y1="22" x2="32" y2="22" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            <line x1="16" y1="30" x2="32" y2="30" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            <line x1="16" y1="38" x2="24" y2="38" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
          </svg>
          <span class="export-option-label">Download TXT</span>
        </button>
      </div>
      <button id="exportCloseFooter" class="button" type="button" style="align-self:center">Close</button>
    </div>
  </dialog>

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

  <meta name="app-session" content="<?= htmlspecialchars($initialSession, ENT_QUOTES, 'UTF-8') ?>">
  <script src="/app.js?v=<?= filemtime(__DIR__.'/app.js') ?>"></script>
</body>
</html>
