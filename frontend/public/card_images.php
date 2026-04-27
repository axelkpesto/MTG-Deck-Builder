<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

app_start_session();

if (!isset($_SESSION['user'])) {
    app_json(['error' => 'Unauthorized'], 401);
}

function card_images_dataset_path(): string
{
    $env = getenv('CARD_IMAGES_PATH');
    if ($env !== false && $env !== '') {
        return $env;
    }
    return '/var/www/data/card_images.json';
}

function load_card_image_dataset(): ?array
{
    static $cache = null;
    static $loaded = false;
    if ($loaded) {
        return $cache;
    }
    $loaded = true;

    $path = card_images_dataset_path();
    if (!is_file($path)) {
        return null;
    }

    $raw = file_get_contents($path);
    $parsed = json_decode((string)$raw, true);
    $cache = is_array($parsed) ? $parsed : null;
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

/**
 * Extract image URLs from a Scryfall card object.
 * Returns an array with one URL for single-faced cards,
 * or one URL per face for double-faced cards.
 */
function scryfall_image_urls(array $card): array
{
    if (isset($card['image_uris']['normal'])) {
        return [$card['image_uris']['normal']];
    }
    if (isset($card['card_faces']) && is_array($card['card_faces'])) {
        $urls = [];
        foreach ($card['card_faces'] as $face) {
            if (isset($face['image_uris']['normal'])) {
                $urls[] = $face['image_uris']['normal'];
            }
        }
        return $urls;
    }
    return [];
}

/**
 * Fetch images for the given card names via the Scryfall collection endpoint.
 * Scryfall supports up to 75 identifiers per request.
 */
function fetch_images_from_scryfall(array $names): array
{
    $found   = [];
    $missing = array_fill_keys($names, []);

    foreach (array_chunk($names, 75) as $batch) {
        $identifiers = array_values(array_map(fn(string $n) => ['name' => $n], $batch));
        $body = json_encode(['identifiers' => $identifiers]);
        if ($body === false) {
            continue;
        }

        $ch = curl_init('https://api.scryfall.com/cards/collection');
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => $body,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER     => [
                'Content-Type: application/json',
                'Accept: application/json',
                'User-Agent: MTGDeckBuilder/1.0 (local-dev)',
            ],
            CURLOPT_TIMEOUT        => 15,
            CURLOPT_CONNECTTIMEOUT => 5,
        ]);

        $response = curl_exec($ch);
        curl_close($ch);

        if ($response === false || $response === '') {
            continue;
        }

        $data = json_decode((string)$response, true);
        if (!is_array($data) || !is_array($data['data'] ?? null)) {
            continue;
        }

        foreach ($data['data'] as $card) {
            $cardName = $card['name'] ?? null;
            if (!is_string($cardName)) {
                continue;
            }
            $urls = scryfall_image_urls($card);
            if ($urls === []) {
                continue;
            }
            $found[$cardName] = $urls;
            unset($missing[$cardName]);

            // Also index by the requested name in case casing differed.
            foreach ($batch as $requested) {
                if (strcasecmp($requested, $cardName) === 0 && $requested !== $cardName) {
                    $found[$requested] = $urls;
                    unset($missing[$requested]);
                }
            }
        }
    }

    return ['found' => $found, 'missing' => $missing];
}

/**
 * Fetch a single card's images via Scryfall named lookup.
 */
function fetch_single_image_from_scryfall(string $name): array
{
    $url = 'https://api.scryfall.com/cards/named?fuzzy=' . urlencode($name);
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER     => [
            'Accept: application/json',
            'User-Agent: MTGDeckBuilder/1.0 (local-dev)',
        ],
        CURLOPT_TIMEOUT        => 10,
        CURLOPT_CONNECTTIMEOUT => 5,
    ]);
    $response = curl_exec($ch);
    curl_close($ch);

    if ($response === false || $response === '') {
        return [];
    }

    $card = json_decode((string)$response, true);
    if (!is_array($card)) {
        return [];
    }

    return scryfall_image_urls($card);
}

// ----- request handling -----

$dataset = load_card_image_dataset();
$method  = strtoupper($_SERVER['REQUEST_METHOD'] ?? 'GET');

if ($method === 'GET') {
    $name = trim((string)($_GET['name'] ?? ''));
    if ($name === '') {
        app_json(['error' => "Query parameter 'name' is required"], 400);
    }

    if ($dataset !== null) {
        $urls = (isset($dataset[$name]) && is_array($dataset[$name])) ? $dataset[$name] : [];
    } else {
        $urls = fetch_single_image_from_scryfall($name);
    }

    app_json(['name' => $name, 'image_urls' => $urls]);
}

if ($method !== 'POST') {
    app_json(['error' => 'Method not allowed'], 405);
}

$raw   = file_get_contents('php://input');
$input = json_decode((string)$raw, true);
if (!is_array($input)) {
    app_json(['error' => 'Invalid JSON body'], 400);
}

$cards = normalize_card_list($input);

if ($dataset !== null) {
    $found   = [];
    $missing = [];
    foreach ($cards as $name) {
        if (isset($dataset[$name]) && is_array($dataset[$name])) {
            $found[$name] = $dataset[$name];
        } else {
            $missing[$name] = [];
        }
    }
    app_json(['found' => $found, 'missing' => $missing]);
}

// Dataset not available locally — fall back to Scryfall.
app_json(fetch_images_from_scryfall($cards));
