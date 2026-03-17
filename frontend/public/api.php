<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

ini_set('max_execution_time', '120');
set_time_limit(120);
app_start_session();
$cfg = app_config();

if (!isset($_SESSION['user'])) {
    app_json(['error' => 'Unauthorized'], 401);
}

$raw = file_get_contents('php://input');
$input = json_decode((string)$raw, true);

if (!is_array($input)) {
    app_json(['error' => 'Invalid JSON body'], 400);
}

$path = (string)($input['path'] ?? '');
$method = strtoupper((string)($input['method'] ?? 'GET'));
$query = isset($input['query']) && is_array($input['query']) ? $input['query'] : [];
$body = isset($input['body']) && is_array($input['body']) ? $input['body'] : null;

$allowedPrefixes = [
    '/status',
    '/get_vector',
    '/get_vector_description',
    '/get_vector_descriptions',
    '/get_random_vector',
    '/get_random_vector_description',
    '/get_similar_vectors',
    '/get_tags',
    '/get_tag_list',
    '/get_tags_from_vector',
    '/generate_deck',
    '/analyze_deck',
];

$allowed = false;
foreach ($allowedPrefixes as $prefix) {
    if (str_starts_with($path, $prefix)) {
        $allowed = true;
        break;
    }
}

if (!$allowed) {
    app_json(['error' => 'Endpoint not allowed'], 403);
}

if (!in_array($method, ['GET', 'POST'], true)) {
    app_json(['error' => 'Method not allowed'], 405);
}

$queryString = http_build_query($query);
$url = $cfg['mtg_api_base_url'] . $path . ($queryString !== '' ? '?' . $queryString : '');

$headers = [
    'Accept: application/json',
    'Content-Type: application/json',
    'X-API-KEY: ' . $cfg['mtg_global_api_key'],
];

$ch = curl_init($url);
curl_setopt_array($ch, [
    CURLOPT_CUSTOMREQUEST => $method,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => $headers,
    CURLOPT_IPRESOLVE => CURL_IPRESOLVE_V4,
    CURLOPT_CONNECTTIMEOUT => 15,
    CURLOPT_TIMEOUT => 60,
]);

if ($method === 'POST') {
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($body ?? []));
}

$response = curl_exec($ch);
$status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
$curlError = curl_error($ch);
$curlErrno = curl_errno($ch);
curl_close($ch);

if ($response === false) {
    app_json([
        'error' => 'Upstream request failed',
        'details' => [
            'message' => $curlError,
            'errno' => $curlErrno,
            'url' => $url,
            'method' => $method,
        ],
    ], 502);
}

http_response_code($status > 0 ? $status : 502);
header('Content-Type: application/json');
echo $response;
