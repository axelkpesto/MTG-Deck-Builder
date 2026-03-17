<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

app_start_session();

if (!isset($_SESSION['user'])) {
    app_json(['error' => 'Unauthorized'], 401);
}

function load_card_image_dataset(): array
{
    static $cache = null;
    if ($cache !== null) {
        return $cache;
    }

    $cfgPath = dirname(__DIR__, 2) . '/backend/config/config.json';
    $cfgRaw = file_get_contents($cfgPath);
    $cfg = json_decode((string)$cfgRaw, true);
    $datasetPath = dirname(__DIR__, 2) . '/' . ($cfg['datasets']['CARD_IMAGE_DATASET_PATH'] ?? '');
    if ($datasetPath === '' || !is_file($datasetPath)) {
        app_json(['error' => 'Card image dataset not found'], 500);
    }

    $raw = file_get_contents($datasetPath);
    $parsed = json_decode((string)$raw, true);
    if (!is_array($parsed)) {
        app_json(['error' => 'Card image dataset is invalid'], 500);
    }

    $cache = $parsed;
    return $cache;
}

function normalize_card_list(array $input): array
{
    $cards = $input['cards'] ?? null;
    if (!is_array($cards)) {
        app_json(['error' => "JSON body must include 'cards': ['Card Name', ...]"], 400);
    }

    $cleaned = [];
    foreach ($cards as $card) {
        if (!is_string($card)) {
            continue;
        }
        $trimmed = trim($card);
        if ($trimmed !== '') {
            $cleaned[] = $trimmed;
        }
    }

    if ($cleaned === []) {
        app_json(['error' => 'cards list cannot be empty'], 400);
    }

    return $cleaned;
}

$dataset = load_card_image_dataset();
$method = strtoupper($_SERVER['REQUEST_METHOD'] ?? 'GET');

if ($method === 'GET') {
    $name = trim((string)($_GET['name'] ?? ''));
    if ($name === '') {
        app_json(['error' => "Query parameter 'name' is required"], 400);
    }

    app_json([
        'name' => $name,
        'image_urls' => isset($dataset[$name]) && is_array($dataset[$name]) ? $dataset[$name] : [],
    ]);
}

if ($method !== 'POST') {
    app_json(['error' => 'Method not allowed'], 405);
}

$raw = file_get_contents('php://input');
$input = json_decode((string)$raw, true);
if (!is_array($input)) {
    app_json(['error' => 'Invalid JSON body'], 400);
}

$cards = normalize_card_list($input);
$found = [];
$missing = [];

foreach ($cards as $name) {
    if (isset($dataset[$name]) && is_array($dataset[$name])) {
        $found[$name] = $dataset[$name];
        continue;
    }
    $missing[$name] = [];
}

app_json(['found' => $found, 'missing' => $missing]);
