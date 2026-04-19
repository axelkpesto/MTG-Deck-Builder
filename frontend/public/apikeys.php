<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

function apikeys_project_id(): string
{
    return env_required('PROJECT_ID');
}

function apikeys_firestore_base_url(): string
{
    $projectId = rawurlencode(apikeys_project_id());
    return "https://firestore.googleapis.com/v1/projects/{$projectId}/databases/(default)/documents";
}

function apikeys_base64url_encode(string $value): string
{
    return rtrim(strtr(base64_encode($value), '+/', '-_'), '=');
}

function apikeys_access_token(): string
{
    static $cachedToken = null;
    static $cachedExpiry = 0;

    if (is_string($cachedToken) && $cachedExpiry > time() + 60) {
        return $cachedToken;
    }

    $path = (string)(getenv('GOOGLE_APPLICATION_CREDENTIALS') ?: '');
    if ($path !== '' && is_file($path)) {
        [$cachedToken, $cachedExpiry] = apikeys_token_from_service_account($path);
        return $cachedToken;
    }

    [$cachedToken, $cachedExpiry] = apikeys_token_from_metadata_server();
    return $cachedToken;
}

function apikeys_token_from_service_account(string $path): array
{
    $raw = file_get_contents($path);
    $json = json_decode((string)$raw, true);
    if (!is_array($json)) {
        app_json(['error' => 'Invalid service account credentials JSON'], 500);
    }

    $clientEmail = (string)($json['client_email'] ?? '');
    $privateKey  = (string)($json['private_key'] ?? '');
    $tokenUri    = (string)($json['token_uri'] ?? 'https://oauth2.googleapis.com/token');
    if ($clientEmail === '' || $privateKey === '') {
        app_json(['error' => 'Service account credentials missing required fields'], 500);
    }

    $now     = time();
    $header  = apikeys_base64url_encode((string)json_encode(['alg' => 'RS256', 'typ' => 'JWT'], JSON_UNESCAPED_SLASHES));
    $claims  = apikeys_base64url_encode((string)json_encode([
        'iss'   => $clientEmail,
        'scope' => 'https://www.googleapis.com/auth/datastore',
        'aud'   => $tokenUri,
        'iat'   => $now,
        'exp'   => $now + 3600,
    ], JSON_UNESCAPED_SLASHES));
    $unsigned = $header . '.' . $claims;

    $signature = '';
    $ok = openssl_sign($unsigned, $signature, $privateKey, OPENSSL_ALGO_SHA256);
    if (!$ok) {
        app_json(['error' => 'Failed to sign service account JWT'], 500);
    }

    $jwt = $unsigned . '.' . apikeys_base64url_encode($signature);

    $ch = curl_init($tokenUri);
    curl_setopt_array($ch, [
        CURLOPT_POST            => true,
        CURLOPT_RETURNTRANSFER  => true,
        CURLOPT_HTTPHEADER      => ['Content-Type: application/x-www-form-urlencoded'],
        CURLOPT_POSTFIELDS      => http_build_query([
            'grant_type' => 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion'  => $jwt,
        ]),
        CURLOPT_TIMEOUT => 20,
    ]);
    $tokenRaw = curl_exec($ch);
    curl_close($ch);

    $payload = json_decode((string)$tokenRaw, true);
    if (!is_array($payload) || empty($payload['access_token'])) {
        app_json(['error' => 'Failed to obtain Firestore access token from service account'], 500);
    }

    return [(string)$payload['access_token'], $now + (int)($payload['expires_in'] ?? 3600)];
}

function apikeys_token_from_metadata_server(): array
{
    $ch = curl_init('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token');
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER     => ['Metadata-Flavor: Google'],
        CURLOPT_TIMEOUT        => 10,
    ]);
    $raw = curl_exec($ch);
    curl_close($ch);

    $payload = json_decode((string)$raw, true);
    if (!is_array($payload) || empty($payload['access_token'])) {
        app_json(['error' => 'Failed to obtain Firestore access token from metadata server'], 500);
    }

    return [(string)$payload['access_token'], time() + (int)($payload['expires_in'] ?? 3600)];
}

function apikeys_firestore_patch(string $docPath, array $fields, array $updateFields = []): void
{
    $token = apikeys_access_token();
    $url   = apikeys_firestore_base_url() . $docPath;

    if ($updateFields !== []) {
        $qs = implode('&', array_map(
            static fn(string $f): string => 'updateMask.fieldPaths=' . rawurlencode($f),
            $updateFields
        ));
        $url .= '?' . $qs;
    }

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_CUSTOMREQUEST  => 'PATCH',
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER     => [
            'Authorization: Bearer ' . $token,
            'Content-Type: application/json',
        ],
        CURLOPT_POSTFIELDS => (string)json_encode(['fields' => $fields], JSON_UNESCAPED_SLASHES),
        CURLOPT_TIMEOUT    => 15,
    ]);
    curl_exec($ch);
    curl_close($ch);
}

function apikeys_firestore_query(string $collectionId, array $where): array
{
    $token      = apikeys_access_token();
    $projectId  = rawurlencode(apikeys_project_id());
    $url        = "https://firestore.googleapis.com/v1/projects/{$projectId}/databases/(default)/documents:runQuery";

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_POST           => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER     => [
            'Authorization: Bearer ' . $token,
            'Content-Type: application/json',
        ],
        CURLOPT_POSTFIELDS => (string)json_encode([
            'structuredQuery' => [
                'from'  => [['collectionId' => $collectionId]],
                'where' => $where,
            ],
        ], JSON_UNESCAPED_SLASHES),
        CURLOPT_TIMEOUT => 15,
    ]);
    $raw = curl_exec($ch);
    curl_close($ch);

    return json_decode((string)$raw, true) ?: [];
}

function apikeys_generate_raw(): string
{
    return 'mtg_' . rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
}

function apikeys_hmac(string $data, string $pepper): string
{
    return hash_hmac('sha256', $data, $pepper);
}

function apikeys_deactivate_for_user(string $userId): void
{
    $results = apikeys_firestore_query('api_keys', [
        'fieldFilter' => [
            'field' => ['fieldPath' => 'user_id'],
            'op'    => 'EQUAL',
            'value' => ['stringValue' => $userId],
        ],
    ]);

    foreach ($results as $row) {
        if (!is_array($row) || !isset($row['document']) || !is_array($row['document'])) {
            continue;
        }
        $doc    = $row['document'];
        $fields = $doc['fields'] ?? [];

        $isActive = ($fields['is_active']['booleanValue'] ?? false) === true;
        if (!$isActive) {
            continue;
        }

        $nameParts = explode('/', (string)($doc['name'] ?? ''));
        $docId     = end($nameParts) ?: '';
        if ($docId === '') {
            continue;
        }

        apikeys_firestore_patch(
            '/api_keys/' . rawurlencode($docId),
            ['is_active' => ['booleanValue' => false]],
            ['is_active'],
        );
    }
}

function apikeys_register(string $email, string $pepper): string
{
    $userId  = apikeys_hmac($email, $pepper);

    $raw     = apikeys_generate_raw();
    $prefix  = substr($raw, 0, 8);
    $keyHash = apikeys_hmac($raw, $pepper);

    apikeys_deactivate_for_user($userId);

    $now     = gmdate('Y-m-d\TH:i:s\Z');
    $expires = gmdate('Y-m-d\TH:i:s\Z', strtotime('+365 days'));

    apikeys_firestore_patch(
        '/api_keys/' . rawurlencode($prefix),
        [
            'user_id'      => ['stringValue'  => $userId],
            'key_hash'     => ['stringValue'  => $keyHash],
            'is_active'    => ['booleanValue' => true],
            'created_at'   => ['timestampValue' => $now],
            'last_used_at' => ['nullValue'    => null],
            'expires_at'   => ['timestampValue' => $expires],
            'rate_limit'   => ['stringValue'  => '60/minute'],
        ],
    );

    return $raw;
}
