<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';
require_once __DIR__ . '/apikeys.php';

app_start_session();
$cfg = app_config();

$code = $_GET['code'] ?? '';
$state = $_GET['state'] ?? '';

if ($code === '' || $state === '' || !hash_equals($_SESSION['oauth_state'] ?? '', (string)$state)) {
    app_json(['error' => 'Invalid OAuth callback state or code'], 400);
}

unset($_SESSION['oauth_state']);

$tokenPayload = http_build_query([
    'grant_type' => 'authorization_code',
    'code' => (string)$code,
    'redirect_uri' => $cfg['oauth_redirect_uri'],
    'client_id' => $cfg['oauth_client_id'],
    'client_secret' => $cfg['oauth_client_secret'],
]);

$ch = curl_init($cfg['oauth_token_url']);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/x-www-form-urlencoded'],
    CURLOPT_POSTFIELDS => $tokenPayload,
    CURLOPT_TIMEOUT => 20,
]);
$tokenRaw = curl_exec($ch);
$tokenCode = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
$tokenCurlErr = curl_error($ch);
curl_close($ch);

$tokenDetails = $tokenRaw;
if ($tokenDetails === false) {
    $tokenDetails = ['curl_error' => $tokenCurlErr, 'http_code' => $tokenCode];
}

$tokenJson = json_decode((string)$tokenRaw, true);
if ($tokenCode < 200 || $tokenCode >= 300 || !is_array($tokenJson) || empty($tokenJson['access_token'])) {
    error_log('Token exchange failed: ' . print_r($tokenDetails, true));
    app_json(['error' => 'Token exchange failed'], 401);
}

$accessToken = (string)$tokenJson['access_token'];

$ch = curl_init($cfg['oauth_userinfo_url']);
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Authorization: Bearer ' . $accessToken],
    CURLOPT_TIMEOUT => 20,
]);
$userRaw = curl_exec($ch);
$userCode = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
curl_close($ch);

$user = json_decode((string)$userRaw, true);
if ($userCode < 200 || $userCode >= 300 || !is_array($user)) {
    error_log('User profile fetch failed: ' . print_r($userRaw, true));
    app_json(['error' => 'User profile fetch failed'], 401);
}

$_SESSION['user'] = [
    'id' => $user['sub'] ?? ($user['id'] ?? ''),
    'email' => $user['email'] ?? '',
    'name' => $user['name'] ?? '',
];
$_SESSION['oauth_access_token'] = $accessToken;

$userEmail = (string)($user['email'] ?? '');
$pepper    = (string)(getenv('API_KEY_PEPPER') ?: '');
if ($userEmail !== '' && $pepper !== '') {
    try {
        $userApiKey = apikeys_register($userEmail, $pepper);
        $_SESSION['user_api_key'] = $userApiKey;
    } catch (Throwable $e) {
        error_log('API key registration failed: ' . $e->getMessage());
        // Non-fatal: user can still browse without a personal API key.
    }
}
unset($userEmail, $pepper);

session_regenerate_id(true);
session_write_close();

header('Location: /');
exit;
