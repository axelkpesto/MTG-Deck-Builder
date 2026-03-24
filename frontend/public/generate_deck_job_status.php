<?php
declare(strict_types=1);

require_once __DIR__ . '/../config.php';

$user = app_require_user();
$jobId = trim((string)($_GET['id'] ?? ''));
if ($jobId === '') {
    app_json(['error' => "Missing required query parameter: id"], 400);
}

$job = app_get_deckgen_job($jobId);
if ($job === null) {
    app_json(['error' => 'Job not found'], 404);
}

if (($job['user_id'] ?? '') !== (string)($user['id'] ?? '')) {
    app_json(['error' => 'Forbidden'], 403);
}

app_json([
    'job_id' => $jobId,
    'status' => $job['status'] ?? 'unknown',
    'commander' => $job['commander'] ?? '',
    'created_at' => isset($job['created_at']) ? (int)$job['created_at'] : null,
    'updated_at' => isset($job['updated_at']) ? (int)$job['updated_at'] : null,
    'error' => $job['error'] ?? '',
]);
