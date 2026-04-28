<?php
declare(strict_types=1);

require_once __DIR__ . '/../../config.php';
app_start_session();
$user = $_SESSION['user'] ?? null;
?>
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Saved Decks</title>
  <link rel="icon" href="/assets/color-wheel.png" type="image/png">
  <link rel="stylesheet" href="/styles.css">
</head>
<body data-page="saved">
  <section id="loginShell" class="login-shell"<?= $user ? ' hidden' : '' ?>>
    <div class="login-card">
      <div class="login-mark">MTG</div>
      <h1>Saved Decks</h1>
      <p>Sign in to view and reload your saved commander decks.</p>
      <p><a class="button primary" href="/login.php">Login with OAuth</a></p>
    </div>
  </section>

  <div id="appShell" class="app-shell"<?= $user ? '' : ' hidden' ?>>
    <header class="site-header">
      <div class="site-brand">
        <div class="site-mark">MTG</div>
        <div>
          <div class="site-title">Deck Builder</div>
          <div class="site-subtitle">Saved commander libraries</div>
        </div>
      </div>
      <nav class="site-nav">
        <a class="nav-link" href="/">Builder</a>
        <a class="nav-link active" href="/saved/">Saved</a>
      </nav>
      <div class="site-account">
        <div id="userLabel" class="account-label">Authenticated</div>
        <a class="button ghost" href="/logout.php">Logout</a>
      </div>
    </header>

    <main id="savedApp" class="saved-layout">
      <section class="panel saved-hero">
        <div>
          <div class="eyebrow">Your Library</div>
          <h1>Saved Decks</h1>
          <p class="subtle">Open any saved list to jump back into the builder with its current cards and commander.</p>
        </div>
        <a class="button primary" href="/">Back to Builder</a>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <div>
            <strong>Deck Collection</strong>
            <div class="subtle" id="savedStatus">Loading saved decks...</div>
          </div>
        </div>
        <div id="savedDecksEmpty" class="empty-state" hidden>No saved decks yet.</div>
        <div id="savedDecksList" class="saved-grid"></div>
      </section>
    </main>
  </div>

  <script src="/app.js"></script>
</body>
</html>
