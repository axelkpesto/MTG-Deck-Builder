<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

app_start_session();
$cfg = app_config();

$state = bin2hex(random_bytes(32));
$_SESSION['oauth_state'] = $state;

$params = http_build_query([
    'response_type' => 'code',
    'client_id' => $cfg['oauth_client_id'],
    'redirect_uri' => $cfg['oauth_redirect_uri'],
    'scope' => $cfg['oauth_scopes'],
    'state' => $state,
    'prompt' => 'select_account consent',
]);

header('Location: ' . $cfg['oauth_authorize_url'] . '?' . $params);
exit;
