<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

app_start_session();
$user = $_SESSION['user'] ?? null;

$payload = ['authenticated' => is_array($user), 'user' => $user];
if (isset($_GET['debug']) && $_GET['debug'] === '1') {
    $payload['debug'] = [
        'session_id' => session_id(),
        'cookie_present' => isset($_COOKIE[session_name()]),
        'session_name' => session_name(),
        'session_keys' => array_keys($_SESSION),
    ];
}

app_json($payload);
