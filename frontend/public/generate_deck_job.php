<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

$user = app_require_user();
$input = app_json_input();

$commander = trim((string)($input['commander'] ?? $input['id'] ?? ''));
if ($commander === '') {
    app_json(['error' => "JSON body must include 'commander': 'Card Name'"], 400);
}

$job = app_enqueue_deckgen_job($user, $commander);

app_json([
    'job_id' => $job['job_id'],
    'status' => $job['status'],
    'commander' => $job['commander'],
    'created_at' => (int)$job['created_at'],
]);
