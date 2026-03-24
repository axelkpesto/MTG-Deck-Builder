<?php
declare(strict_types=1);

function app_fail(string $message, int $status = 500, array $details = []): void
{
    $context = $details === [] ? '' : ' ' . json_encode($details);
    error_log(sprintf('[mtg-app] %s%s', $message, $context));
    app_json(array_merge(['error' => $message], $details), $status);
}

function env_required(string $name): string
{
    $value = getenv($name);
    if ($value === false || $value === '') {
        app_fail("Missing required env var: {$name}", 500);
    }
    return $value;
}

function app_config(): array
{
    return [
        'mtg_api_base_url' => env_required('FLASK_API_PATH'),
        'mtg_global_api_key' => env_required('FIREBASE_API_KEY'),
        'redis_url' => env_required('REDIS_URL'),
        'oauth_client_id' => env_required('OAUTH_CLIENT_ID'),
        'oauth_client_secret' => env_required('OAUTH_CLIENT_SECRET'),
        'oauth_redirect_uri' => env_required('OAUTH_REDIRECT_URI'),
        'oauth_authorize_url' => env_required('OAUTH_AUTHORIZE_URL'),
        'oauth_token_url' => env_required('OAUTH_TOKEN_URL'),
        'oauth_userinfo_url' => env_required('OAUTH_USERINFO_URL'),
        'oauth_scopes' => 'openid profile email',
    ];
}

function app_start_session(): void
{
    if (session_status() === PHP_SESSION_ACTIVE) {
        return;
    }

    session_set_cookie_params([
        'lifetime' => 0,
        'path' => '/',
        'domain' => '',
        'secure' => isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
        'httponly' => true,
        'samesite' => 'Lax',
    ]);

    session_start();
}

function app_json(array $payload, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($payload);
    exit;
}

function app_json_input(): array
{
    $raw = file_get_contents('php://input');
    if ($raw === false || $raw === '') {
        return [];
    }

    $decoded = json_decode((string)$raw, true);
    if (!is_array($decoded)) {
        app_fail('Invalid JSON body', 400);
    }

    return $decoded;
}

function app_require_user(): array
{
    app_start_session();
    if (!isset($_SESSION['user']) || !is_array($_SESSION['user'])) {
        app_fail('Unauthorized', 401);
    }

    $user = $_SESSION['user'];
    session_write_close();
    return $user;
}

function app_redis(): Redis
{
    static $client = null;
    if ($client instanceof Redis) {
        return $client;
    }

    if (!class_exists('Redis')) {
        app_fail('Redis PHP extension is not installed', 500);
    }

    $cfg = app_config();
    $parsed = parse_url($cfg['redis_url']);
    if ($parsed === false || !isset($parsed['host'])) {
        app_fail('Invalid REDIS_URL configuration', 500, ['redis_url' => $cfg['redis_url']]);
    }

    $scheme = $parsed['scheme'] ?? 'redis';
    $host = (string)$parsed['host'];
    $port = (int)($parsed['port'] ?? 6379);
    $password = $parsed['pass'] ?? null;
    $username = $parsed['user'] ?? null;
    $path = $parsed['path'] ?? '';
    $database = 0;
    if ($path !== '') {
        $database = (int)ltrim($path, '/');
    }

    $targetHost = $scheme === 'rediss' ? 'tls://' . $host : $host;
    $redis = new Redis();

    try {
        $connected = $redis->connect($targetHost, $port, 2.5);
        if ($connected !== true) {
            app_fail('Failed to connect to Redis', 500, ['host' => $host, 'port' => $port]);
        }

        if (is_string($password) && $password !== '') {
            if (is_string($username) && $username !== '') {
                $redis->auth([$username, $password]);
            } else {
                $redis->auth($password);
            }
        }

        if ($database > 0) {
            $redis->select($database);
        }
    } catch (RedisException $e) {
        app_fail('Redis connection failed', 500, [
            'details' => $e->getMessage(),
            'host' => $host,
            'port' => $port,
            'scheme' => $scheme,
        ]);
    }

    $client = $redis;
    return $client;
}

function app_deckgen_queue_key(): string
{
    return 'deckgen:queue';
}

function app_deckgen_job_key(string $jobId): string
{
    return 'deckgen:job:' . $jobId;
}

function app_deckgen_job_ttl_seconds(): int
{
    return 86400;
}

function app_generate_job_id(): string
{
    try {
        return bin2hex(random_bytes(16));
    } catch (Exception) {
        return uniqid('deckgen_', true);
    }
}

function app_enqueue_deckgen_job(array $user, string $commander): array
{
    $redis = app_redis();
    $jobId = app_generate_job_id();
    $now = (string)time();
    $payload = [
        'kind' => 'generate_deck',
        'job_id' => $jobId,
        'user_id' => (string)($user['id'] ?? ''),
        'commander' => $commander,
        'status' => 'queued',
        'created_at' => $now,
        'updated_at' => $now,
        'result' => '',
        'error' => '',
    ];

    $jobKey = app_deckgen_job_key($jobId);
    try {
        $redis->hMSet($jobKey, $payload);
        $redis->expire($jobKey, app_deckgen_job_ttl_seconds());
        $redis->rPush(app_deckgen_queue_key(), $jobId);
    } catch (RedisException $e) {
        app_fail('Failed to enqueue deck job', 500, [
            'details' => $e->getMessage(),
            'job_id' => $jobId,
            'commander' => $commander,
        ]);
    }

    return $payload;
}

function app_get_deckgen_job(string $jobId): ?array
{
    $job = app_redis()->hGetAll(app_deckgen_job_key($jobId));
    if (!is_array($job) || $job === []) {
        return null;
    }

    return $job;
}
