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
  <title>MTG Deck Builder (PHP OAuth Gateway)</title>
  <style>
	body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
	.row { margin-bottom: 12px; }
	input, button, textarea { font-size: 14px; padding: 8px; }
	textarea { width: 100%; height: 220px; }
	code { background: #f5f5f5; padding: 2px 4px; }
  </style>
</head>
<body>
  <h1>MTG Deck Builder</h1>
  <p>This frontend talks only to <code>/api.php</code>. The global API key stays in backend env vars.</p>

  <?php if (!$user): ?>
	<p><a href="/login.php">Login with OAuth</a></p>
  <?php else: ?>
	<p>Signed in as <strong><?= htmlspecialchars((string)($user['email'] ?: $user['name'] ?: $user['id'])) ?></strong> | <a href="/logout.php">Logout</a></p>

	<div class="row">
	  <label for="commander">Commander</label><br>
	  <input id="commander" type="text" placeholder="Atraxa, Praetors' Voice" size="40">
	  <button id="generateBtn">Generate Deck</button>
	</div>

	<div class="row">
	  <label for="analysisCommander">Analyze Commander</label><br>
	  <input id="analysisCommander" type="text" placeholder="Atraxa, Praetors' Voice" size="40">
	  <button id="analyzeBtn">Analyze Deck</button>
	</div>

	<div class="row">
	  <label for="cards">Cards (one per line)</label><br>
	  <textarea id="cards" placeholder="Sol Ring&#10;Arcane Signet"></textarea>
	</div>

	<div class="row">
	  <button id="statusBtn">Check API Status</button>
	</div>

	<h3>Response</h3>
	<textarea id="output" readonly></textarea>

	<script>
	  async function callApi(payload) {
		const res = await fetch('/api.php', {
		  method: 'POST',
		  headers: { 'Content-Type': 'application/json' },
		  body: JSON.stringify(payload)
		});
		const text = await res.text();
		let parsed;
		try {
		  parsed = JSON.parse(text);
		} catch (e) {
		  parsed = { raw: text };
		}
		if (!res.ok) {
		  throw new Error(JSON.stringify(parsed, null, 2));
		}
		return parsed;
	  }

	  function setOutput(value) {
		document.getElementById('output').value = typeof value === 'string'
		  ? value
		  : JSON.stringify(value, null, 2);
	  }

	  document.getElementById('statusBtn').addEventListener('click', async () => {
		try {
		  const data = await callApi({ path: '/status', method: 'GET' });
		  setOutput(data);
		} catch (err) {
		  setOutput(String(err));
		}
	  });

	  document.getElementById('generateBtn').addEventListener('click', async () => {
		const commander = document.getElementById('commander').value.trim();
		if (!commander) {
		  setOutput('Commander is required');
		  return;
		}
		try {
		  const data = await callApi({
			path: '/generate_deck/' + encodeURIComponent(commander),
			method: 'GET'
		  });
		  setOutput(data);
		} catch (err) {
		  setOutput(String(err));
		}
	  });

	  document.getElementById('analyzeBtn').addEventListener('click', async () => {
		const commander = document.getElementById('analysisCommander').value.trim();
		const cards = document.getElementById('cards').value
		  .split('\n')
		  .map(s => s.trim())
		  .filter(Boolean);
		if (!commander || cards.length === 0) {
		  setOutput('Commander and at least one card are required');
		  return;
		}
		try {
		  const data = await callApi({
			path: '/analyze_deck',
			method: 'POST',
			body: { commander, cards }
		  });
		  setOutput(data);
		} catch (err) {
		  setOutput(String(err));
		}
	  });
	</script>
  <?php endif; ?>
</body>
</html>
