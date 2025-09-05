<?php

error_reporting(E_ALL);
ini_set('display_errors', 1);

header('Content-Type: application/json');

$response = [
    'status' => 'ok',
    'timestamp' => date('c'),
    'php_version' => phpversion(),
    'server_software' => $_SERVER['SERVER_SOFTWARE'] ?? 'N/A',
    'file_permissions' => [],
    'post_test' => 'pending'
];

$test_file = 'diag_test.txt';
$test_data = 'Hello from CHIMERA diagnostics!';


// 1. Test file writing
$write_result = file_put_contents($test_file, $test_data);
if ($write_result !== false) {
    $response['file_permissions']['write_test'] = "SUCCESS: Wrote " . $write_result . " bytes to " . $test_file;
    
    // 2. Test file reading
    $read_result = file_get_contents($test_file);
    if ($read_result === $test_data) {
        $response['file_permissions']['read_test'] = "SUCCESS: Read back the correct data from " . $test_file;
    } else {
        $response['file_permissions']['read_test'] = "ERROR: Could not read back the correct data.";
    }
    
    // 3. Clean up the test file
    unlink($test_file);
    
} else {
    $response['file_permissions']['write_test'] = "ERROR: Could not write to " . $test_file . ". Please check directory permissions.";
    $response['status'] = 'error';
}


// 4. Check if this script was accessed via POST
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $input = file_get_contents('php:
    $response['post_test'] = [
        'status' => 'SUCCESS: Received POST request.',
        'payload_received' => json_decode($input, true)
    ];
} else {
    $response['post_test'] = 'INFO: This script should be tested with a POST request as well.';
}

echo json_encode($response, JSON_PRETTY_PRINT);

?>