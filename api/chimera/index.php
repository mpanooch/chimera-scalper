<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, X-Requested-With');

// Handle pre-flight OPTIONS request for CORS
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

$data_file = 'chimera_data.json';
$log_file = 'chimera_log.txt';

// --- Utility Functions ---

function logRequest($message) {
    global $log_file;
    $timestamp = date('Y-m-d H:i:s');
    file_put_contents($log_file, "[$timestamp] $message\n", FILE_APPEND | LOCK_EX);
}

function saveData($data) {
    global $data_file;
    $data['last_updated'] = date('c');
    // Atomically write the file
    return file_put_contents($data_file, json_encode($data, JSON_PRETTY_PRINT), LOCK_EX);
}

function loadData() {
    global $data_file;
    if (file_exists($data_file) && filesize($data_file) > 0) {
        $content = file_get_contents($data_file);
        return json_decode($content, true);
    }
    return null;
}

function sendJsonResponse($data, $statusCode = 200) {
    http_response_code($statusCode);
    echo json_encode($data);
    exit();
}

// --- Default Data Structure ---

function getDefaultData() {
    return [
        'status' => 'no_data',
        'message' => 'No CHIMERA data available yet.',
        'performance' => [
            'total_pnl' => 0, 'pnl_percent' => 0, 'win_rate' => 0,
            'total_trades' => 0, 'balance' => 10000
        ],
        'regime' => ['current' => 'UNKNOWN', 'confidence' => 0],
        'market' => ['BTCUSDT' => 0, 'ETHUSDT' => 0, 'SOLUSDT' => 0],
        'system' => [
            'processing_speed' => 0, 'total_ticks' => 0,
            'signals_generated' => 0, 'model_accuracy' => 0
        ],
        'recent_trades' => []
    ];
}

// --- Main API Logic ---

$action = $_GET['action'] ?? '';

// POST request to update data
if ($_SERVER['REQUEST_METHOD'] === 'POST' && $action === 'update') {
    $input = file_get_contents('php://input');
    $data = json_decode($input, true);
    
    if (json_last_error() === JSON_ERROR_NONE && is_array($data)) {
        if (saveData($data)) {
            logRequest("Data updated successfully via POST.");
            sendJsonResponse(['status' => 'success', 'message' => 'Data updated']);
        } else {
            logRequest("Error: Failed to save data.");
            sendJsonResponse(['status' => 'error', 'message' => 'Failed to save data'], 500);
        }
    } else {
        logRequest("Error: Invalid JSON received.");
        sendJsonResponse(['status' => 'error', 'message' => 'Invalid JSON data'], 400);
    }
}
// GET request to get stats
elseif ($_SERVER['REQUEST_METHOD'] === 'GET' && $action === 'stats') {
    $data = loadData();
    
    if ($data) {
        $last_updated = strtotime($data['last_updated'] ?? 'now');
        $data['status'] = (time() - $last_updated > 30) ? 'stale' : 'active';
        logRequest("Stats requested. Data is {$data['status']}.");
        sendJsonResponse($data);
    } else {
        logRequest("No data available. Returning default structure.");
        sendJsonResponse(getDefaultData(), 404);
    }
}
// GET request for health check
elseif ($_SERVER['REQUEST_METHOD'] === 'GET' && $action === 'health') {
    sendJsonResponse([
        'status' => 'healthy',
        'timestamp' => date('c'),
        'endpoints' => [
            'POST ?action=update' => 'Receives data from CHIMERA system',
            'GET ?action=stats' => 'Returns current trading stats',
            'GET ?action=health' => 'Performs a health check'
        ]
    ]);
}
// Catch-all for unknown actions
else {
    logRequest("Invalid endpoint requested: " . $_SERVER['REQUEST_METHOD'] . " " . $_SERVER['REQUEST_URI']);
    sendJsonResponse([
        'status' => 'error',
        'message' => 'Endpoint not found. Use ?action=stats, ?action=update, or ?action=health.'
    ], 404);
}
?>