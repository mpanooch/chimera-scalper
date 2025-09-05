<?php
// Simple signal receiver for CHIMERA monitoring
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

$signal_file = 'latest_signal.json';

// POST: Receive signal from PC
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = file_get_contents('php://input');
    $signal_data = json_decode($input, true);
    
    if (json_last_error() === JSON_ERROR_NONE && is_array($signal_data)) {
        // Just save the latest signal - no history needed
        if (file_put_contents($signal_file, json_encode($signal_data, JSON_PRETTY_PRINT), LOCK_EX)) {
            echo json_encode(['status' => 'success', 'message' => 'Signal received']);
        } else {
            http_response_code(500);
            echo json_encode(['status' => 'error', 'message' => 'Failed to save signal']);
        }
    } else {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Invalid signal data']);
    }
}
// GET: Return latest signal for website display
elseif ($_SERVER['REQUEST_METHOD'] === 'GET') {
    if (file_exists($signal_file) && filesize($signal_file) > 0) {
        $signal_data = json_decode(file_get_contents($signal_file), true);
        
        // Check if signal is recent (within last 5 minutes)
        $last_update = strtotime($signal_data['timestamp'] ?? 'now');
        $signal_data['status'] = (time() - $last_update > 300) ? 'stale' : 'active';
        
        echo json_encode($signal_data);
    } else {
        http_response_code(404);
        echo json_encode([
            'status' => 'no_signal',
            'message' => 'No signals received yet',
            'timestamp' => date('c')
        ]);
    }
}
else {
    http_response_code(405);
    echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
}
?>