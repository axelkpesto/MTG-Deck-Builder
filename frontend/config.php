<?php
declare(strict_types=1);

function env_required(string $name): string
{
    $value = getenv($name);
    if ($value === false || $value === '') {
        http_response_code(500);
        header('Content-Type: application/json');
        echo json_encode(['error' => "Missing required env var: {$name}"]);
        exit;
    }
    return $value;
}

function app_config(): array
{
    return [
        'mtg_api_base_url' => rtrim(env_required('MTG_API_BASE_URL'), '/'),
        'mtg_global_api_key' => env_required('MTG_GLOBAL_API_KEY'),
        'oauth_client_id' => env_required('OAUTH_CLIENT_ID'),
        'oauth_client_secret' => env_required('OAUTH_CLIENT_SECRET'),
        'oauth_redirect_uri' => env_required('OAUTH_REDIRECT_URI'),
        'oauth_authorize_url' => env_required('OAUTH_AUTHORIZE_URL'),
        'oauth_token_url' => env_required('OAUTH_TOKEN_URL'),
        'oauth_userinfo_url' => env_required('OAUTH_USERINFO_URL'),
        'oauth_scopes' => getenv('OAUTH_SCOPES') ?: 'openid profile email',
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
