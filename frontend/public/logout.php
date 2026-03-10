<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

app_start_session();
$_SESSION = [];
session_destroy();

header('Location: /');
exit;
