<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

function decks_project_id(): string
{
    return env_required('PROJECT_ID');
}

function decks_credentials_path(): string
{
    $value = getenv('GOOGLE_APPLICATION_CREDENTIALS');
    return $value === false ? '' : trim($value);
}

function decks_firestore_base_url(): string
{
    $projectId = rawurlencode(decks_project_id());
    return "https://firestore.googleapis.com/v1/projects/{$projectId}/databases/(default)/documents";
}

function decks_firestore_access_token(): string
{
    static $cachedToken = null;
    static $cachedExpiry = 0;

    if (is_string($cachedToken) && $cachedExpiry > time() + 60) {
        return $cachedToken;
    }

    $credentialsPath = decks_credentials_path();
    if ($credentialsPath !== '') {
        $cached = decks_access_token_from_service_account($credentialsPath);
        $cachedToken = $cached['access_token'];
        $cachedExpiry = (int)$cached['expires_at'];
        return $cachedToken;
    }

    $cached = decks_access_token_from_metadata_server();
    $cachedToken = $cached['access_token'];
    $cachedExpiry = (int)$cached['expires_at'];
    return $cachedToken;
}

function decks_access_token_from_service_account(string $path): array
{
    if (!is_file($path)) {
        app_json(['error' => 'Missing service account credentials file'], 500);
    }

    $raw = file_get_contents($path);
    $json = json_decode((string)$raw, true);
    if (!is_array($json)) {
        app_json(['error' => 'Invalid service account credentials JSON'], 500);
    }

    $clientEmail = (string)($json['client_email'] ?? '');
    $privateKey = (string)($json['private_key'] ?? '');
    $tokenUri = (string)($json['token_uri'] ?? 'https://oauth2.googleapis.com/token');
    if ($clientEmail === '' || $privateKey === '') {
        app_json(['error' => 'Service account credentials missing key fields'], 500);
    }

    $now = time();
    $header = ['alg' => 'RS256', 'typ' => 'JWT'];
    $claims = [
        'iss' => $clientEmail,
        'scope' => 'https://www.googleapis.com/auth/datastore',
        'aud' => $tokenUri,
        'iat' => $now,
        'exp' => $now + 3600,
    ];

    $unsigned = decks_base64url_encode(json_encode($header, JSON_UNESCAPED_SLASHES))
        . '.'
        . decks_base64url_encode(json_encode($claims, JSON_UNESCAPED_SLASHES));

    $signature = '';
    $ok = openssl_sign($unsigned, $signature, $privateKey, OPENSSL_ALGO_SHA256);
    if (!$ok) {
        app_json(['error' => 'Failed to sign service account JWT'], 500);
    }

    $jwt = $unsigned . '.' . decks_base64url_encode($signature);

    $response = decks_http_request(
        'POST',
        $tokenUri,
        [
            'Content-Type: application/x-www-form-urlencoded',
        ],
        http_build_query([
            'grant_type' => 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion' => $jwt,
        ]),
    );

    $payload = json_decode($response['body'], true);
    if (!is_array($payload) || !isset($payload['access_token'])) {
        app_json(['error' => 'Failed to obtain Firestore access token', 'raw' => $response['body']], 500);
    }

    return [
        'access_token' => (string)$payload['access_token'],
        'expires_at' => time() + (int)($payload['expires_in'] ?? 3600),
    ];
}

function decks_access_token_from_metadata_server(): array
{
    $response = decks_http_request(
        'GET',
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
        ['Metadata-Flavor: Google'],
        null,
        false,
    );

    $payload = json_decode($response['body'], true);
    if (!is_array($payload) || !isset($payload['access_token'])) {
        app_json(['error' => 'Failed to obtain metadata server access token', 'raw' => $response['body']], 500);
    }

    return [
        'access_token' => (string)$payload['access_token'],
        'expires_at' => time() + (int)($payload['expires_in'] ?? 3600),
    ];
}

function decks_base64url_encode(string $value): string
{
    return rtrim(strtr(base64_encode($value), '+/', '-_'), '=');
}

function decks_http_request(string $method, string $url, array $headers, ?string $body, bool $expectJson = true): array
{
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_CUSTOMREQUEST => $method,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_TIMEOUT => 60,
    ]);
    if ($body !== null) {
        curl_setopt($ch, CURLOPT_POSTFIELDS, $body);
    }

    $response = curl_exec($ch);
    $status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    $curlError = curl_error($ch);
    $curlErrno = curl_errno($ch);
    curl_close($ch);

    if ($response === false) {
        app_json([
            'error' => 'HTTP request failed',
            'details' => ['message' => $curlError, 'errno' => $curlErrno],
        ], 502);
    }

    if ($status < 200 || $status >= 300) {
        $payload = $expectJson ? json_decode((string)$response, true) : null;
        app_json([
            'error' => 'Firestore request failed',
            'status' => $status,
            'details' => is_array($payload) ? $payload : ['raw' => $response],
        ], $status > 0 ? $status : 502);
    }

    return ['status' => $status, 'body' => (string)$response];
}

function decks_firestore_request(string $method, string $path, ?array $payload = null): array
{
    $token = decks_firestore_access_token();
    $headers = [
        'Authorization: Bearer ' . $token,
        'Accept: application/json',
        'Content-Type: application/json',
    ];
    $body = $payload === null ? null : json_encode($payload, JSON_UNESCAPED_SLASHES);
    $response = decks_http_request($method, decks_firestore_base_url() . $path, $headers, $body);
    $decoded = json_decode($response['body'], true);
    if (!is_array($decoded)) {
        app_json(['error' => 'Invalid Firestore response', 'raw' => $response['body']], 502);
    }
    return $decoded;
}

function decks_firestore_encode_value($value): array
{
    if (is_string($value)) {
        return ['stringValue' => $value];
    }
    if (is_int($value)) {
        return ['integerValue' => (string)$value];
    }
    if (is_float($value)) {
        return ['doubleValue' => $value];
    }
    if (is_bool($value)) {
        return ['booleanValue' => $value];
    }
    if ($value === null) {
        return ['nullValue' => null];
    }
    if (is_array($value)) {
        $isList = array_keys($value) === range(0, count($value) - 1);
        if ($isList) {
            return [
                'arrayValue' => [
                    'values' => array_map('decks_firestore_encode_value', $value),
                ],
            ];
        }

        $fields = [];
        foreach ($value as $key => $inner) {
            $fields[(string)$key] = decks_firestore_encode_value($inner);
        }
        return ['mapValue' => ['fields' => $fields]];
    }

    return ['stringValue' => (string)$value];
}

function decks_firestore_decode_value(array $value)
{
    if (array_key_exists('stringValue', $value)) {
        return (string)$value['stringValue'];
    }
    if (array_key_exists('integerValue', $value)) {
        return (int)$value['integerValue'];
    }
    if (array_key_exists('doubleValue', $value)) {
        return (float)$value['doubleValue'];
    }
    if (array_key_exists('booleanValue', $value)) {
        return (bool)$value['booleanValue'];
    }
    if (array_key_exists('timestampValue', $value)) {
        return (string)$value['timestampValue'];
    }
    if (array_key_exists('nullValue', $value)) {
        return null;
    }
    if (isset($value['arrayValue']['values']) && is_array($value['arrayValue']['values'])) {
        return array_map('decks_firestore_decode_value', $value['arrayValue']['values']);
    }
    if (isset($value['mapValue']['fields']) && is_array($value['mapValue']['fields'])) {
        return decks_firestore_decode_fields($value['mapValue']['fields']);
    }
    return null;
}

function decks_firestore_decode_fields(array $fields): array
{
    $decoded = [];
    foreach ($fields as $key => $value) {
        if (!is_array($value)) {
            continue;
        }
        $decoded[(string)$key] = decks_firestore_decode_value($value);
    }
    return $decoded;
}

function decks_firestore_document_to_saved_deck(array $document): array
{
    $name = (string)($document['name'] ?? '');
    $parts = explode('/', $name);
    $docId = end($parts) ?: '';
    $fields = decks_firestore_decode_fields(is_array($document['fields'] ?? null) ? $document['fields'] : []);
    $cards = $fields['cards'] ?? ['commander' => '', 'cards' => []];
    $cardRows = is_array($cards['cards'] ?? null) ? $cards['cards'] : [];
    return [
        'id' => $docId,
        'title' => (string)($fields['title'] ?? ''),
        'owner_id' => (string)($fields['owner_id'] ?? ''),
        'owner_email' => (string)($fields['owner_email'] ?? ''),
        'format' => (string)($fields['format'] ?? 'commander'),
        'cards' => [
            'commander' => (string)($cards['commander'] ?? ''),
            'cards' => $cardRows,
        ],
        'card_count' => array_reduce(
            $cardRows,
            static fn(int $sum, array $card): int => $sum + max(1, (int)($card['quantity'] ?? 1)),
            0
        ),
        'created_at' => (string)($fields['created_at'] ?? ''),
        'updated_at' => (string)($fields['updated_at'] ?? ''),
    ];
}

function decks_random_id(): string
{
    return bin2hex(random_bytes(16));
}

function decks_validate_payload(array $deck): array
{
    $title = trim((string)($deck['title'] ?? ''));
    $format = trim((string)($deck['format'] ?? 'commander')) ?: 'commander';
    $commander = trim((string)($deck['commander'] ?? ''));
    $cardsWrapper = isset($deck['cards']) && is_array($deck['cards']) ? $deck['cards'] : null;
    if ($commander === '' || $cardsWrapper === null) {
        app_json(['error' => 'Invalid saved deck payload'], 400);
    }

    $wrapperCommander = trim((string)($cardsWrapper['commander'] ?? ''));
    $cards = isset($cardsWrapper['cards']) && is_array($cardsWrapper['cards']) ? $cardsWrapper['cards'] : null;
    if ($wrapperCommander !== $commander || $cards === null) {
        app_json(['error' => 'Invalid saved deck cards payload'], 400);
    }

    $normalizedCards = [];
    foreach ($cards as $row) {
        if (!is_array($row)) {
            continue;
        }
        $name = trim((string)($row['name'] ?? ''));
        if ($name === '') {
            continue;
        }
        $normalizedCards[] = [
            'name' => $name,
            'quantity' => max(1, (int)($row['quantity'] ?? 1)),
        ];
    }
    if ($commander === '' || count($normalizedCards) === 0) {
        app_json(['error' => 'Saved deck requires commander and cards'], 400);
    }

    return [
        'id' => trim((string)($deck['id'] ?? '')),
        'title' => $title !== '' ? $title : "{$commander} Deck",
        'format' => $format,
        'cards' => [
            'commander' => $commander,
            'cards' => $normalizedCards,
        ],
    ];
}

function decks_get_document(string $deckId): ?array
{
    $token = decks_firestore_access_token();
    $headers = [
        'Authorization: Bearer ' . $token,
        'Accept: application/json',
    ];
    $url = decks_firestore_base_url() . '/saved_decks/' . rawurlencode($deckId);
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_CUSTOMREQUEST => 'GET',
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_TIMEOUT => 60,
    ]);
    $response = curl_exec($ch);
    $status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    $curlError = curl_error($ch);
    $curlErrno = curl_errno($ch);
    curl_close($ch);

    if ($response === false) {
        app_json(['error' => 'Firestore request failed', 'details' => ['message' => $curlError, 'errno' => $curlErrno]], 502);
    }
    if ($status === 404) {
        return null;
    }
    if ($status < 200 || $status >= 300) {
        $decoded = json_decode((string)$response, true);
        app_json(['error' => 'Firestore request failed', 'details' => is_array($decoded) ? $decoded : ['raw' => $response]], $status);
    }
    $decoded = json_decode((string)$response, true);
    if (!is_array($decoded)) {
        app_json(['error' => 'Invalid Firestore response', 'raw' => $response], 502);
    }
    return $decoded;
}

function decks_list_for_user(string $ownerId): array
{
    $result = decks_firestore_request('POST', ':runQuery', [
        'structuredQuery' => [
            'from' => [['collectionId' => 'saved_decks']],
            'where' => [
                'fieldFilter' => [
                    'field' => ['fieldPath' => 'owner_id'],
                    'op' => 'EQUAL',
                    'value' => ['stringValue' => $ownerId],
                ],
            ],
        ],
    ]);

    $decks = [];
    foreach ($result as $row) {
        if (!is_array($row) || !isset($row['document']) || !is_array($row['document'])) {
            continue;
        }
        $decks[] = decks_firestore_document_to_saved_deck($row['document']);
    }

    usort($decks, static function (array $a, array $b): int {
        return strtotime((string)($b['updated_at'] ?? '')) <=> strtotime((string)($a['updated_at'] ?? ''));
    });

    return $decks;
}

function decks_save_for_user(array $user, array $deck): array
{
    $normalized = decks_validate_payload($deck);
    $deckId = $normalized['id'] !== '' ? $normalized['id'] : decks_random_id();
    $existing = $normalized['id'] !== '' ? decks_get_document($deckId) : null;

    if ($existing !== null) {
        $existingDeck = decks_firestore_document_to_saved_deck($existing);
        if (($existingDeck['owner_id'] ?? '') !== (string)($user['id'] ?? '')) {
            app_json(['error' => 'Cannot overwrite a deck owned by another user'], 403);
        }
    }

    $now = gmdate('c');
    $payload = [
        'title' => $normalized['title'],
        'owner_id' => (string)($user['id'] ?? ''),
        'owner_email' => (string)($user['email'] ?? ''),
        'format' => $normalized['format'],
        'cards' => $normalized['cards'],
        'created_at' => $existing !== null
            ? (string)(decks_firestore_document_to_saved_deck($existing)['created_at'] ?? $now)
            : $now,
        'updated_at' => $now,
    ];

    $fields = [];
    foreach ($payload as $key => $value) {
        if ($key === 'created_at' || $key === 'updated_at') {
            $fields[$key] = ['timestampValue' => $value];
            continue;
        }
        $fields[$key] = decks_firestore_encode_value($value);
    }

    decks_firestore_request(
        'PATCH',
        '/saved_decks/' . rawurlencode($deckId),
        ['fields' => $fields],
    );

    return array_merge(
        ['id' => $deckId],
        $payload,
        [
            'card_count' => array_reduce(
                $normalized['cards']['cards'],
                static fn(int $sum, array $card): int => $sum + max(1, (int)($card['quantity'] ?? 1)),
                0
            ),
        ]
    );
}

$user = app_require_user();
$method = strtoupper($_SERVER['REQUEST_METHOD'] ?? 'GET');

if ($method === 'GET') {
    $deckId = isset($_GET['id']) ? trim((string)$_GET['id']) : '';
    if ($deckId !== '' && !preg_match('/^[a-f0-9]{32}$/', $deckId)) {
        app_json(['error' => 'Invalid deck id'], 400);
    }
    if ($deckId !== '') {
        $document = decks_get_document($deckId);
        if ($document === null) {
            app_json(['error' => 'saved deck not found'], 404);
        }
        $deck = decks_firestore_document_to_saved_deck($document);
        if (($deck['owner_id'] ?? '') !== (string)($user['id'] ?? '')) {
            app_json(['error' => 'saved deck not found'], 404);
        }
        app_json(['deck' => $deck]);
    }

    app_json(['decks' => decks_list_for_user((string)($user['id'] ?? ''))]);
}

if ($method === 'POST') {
    $input = app_json_input();
    $deck = isset($input['deck']) && is_array($input['deck']) ? $input['deck'] : $input;
    app_json(['deck' => decks_save_for_user($user, $deck)]);
}

if ($method === 'DELETE') {
    $deckId = isset($_GET['id']) ? trim((string)$_GET['id']) : '';
    if ($deckId === '') {
        app_json(['error' => 'Missing deck id'], 400);
    }
    if (!preg_match('/^[a-f0-9]{32}$/', $deckId)) {
        app_json(['error' => 'Invalid deck id'], 400);
    }
    $document = decks_get_document($deckId);
    if ($document === null) {
        app_json(['error' => 'Deck not found'], 404);
    }
    $deck = decks_firestore_document_to_saved_deck($document);
    if (($deck['owner_id'] ?? '') !== (string)($user['id'] ?? '')) {
        app_json(['error' => 'Cannot delete a deck owned by another user'], 403);
    }
    decks_firestore_request('DELETE', '/saved_decks/' . rawurlencode($deckId));
    app_json(['deleted' => true]);
}

app_json(['error' => 'Method not allowed'], 405);
