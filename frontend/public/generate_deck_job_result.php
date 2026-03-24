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

$status = $job['status'] ?? 'unknown';
if ($status !== 'done') {
    app_json([
        'job_id' => $jobId,
        'status' => $status,
        'error' => $job['error'] ?? '',
    ], 409);
}

$result = json_decode((string)($job['result'] ?? ''), true);
if (!is_array($result)) {
    app_json(['error' => 'Stored job result is invalid'], 500);
}

app_json([
    'job_id' => $jobId,
    'status' => $status,
    'result' => $result,
]);
