#!/bin/bash

echo "🔥 Starting CHIMERA with Bybit Integration"
echo "========================================="

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Shutting down..."
    kill $FEEDER_PID 2>/dev/null
    kill $CHIMERA_PID 2>/dev/null
    rm -f /tmp/chimera_orderbook.dat
    echo "✓ Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Test Bybit connection first
echo "Testing Bybit connection..."
python test_bybit_connection.py

if [ $? -ne 0 ]; then
    echo "❌ Bybit connection test failed. Check your internet connection."
    exit 1
fi

# Start Bybit feeder in background
echo -e "\nStarting Bybit L2 feeder..."
python bybit_integration.py &
FEEDER_PID=$!

# Wait for feeder to initialize
sleep 3

# Check if shared memory file exists
if [ ! -f /tmp/chimera_orderbook.dat ]; then
    echo "⚠️  Waiting for order book data..."
    sleep 5
fi

# Start CHIMERA
echo -e "\nStarting CHIMERA scalper..."
./build/bin/chimera_scalper &
CHIMERA_PID=$!

# Monitor both processes
echo -e "\n✅ Systems running!"
echo "   Bybit Feeder PID: $FEEDER_PID"
echo "   CHIMERA PID: $CHIMERA_PID"
echo -e "\nPress Ctrl+C to stop\n"

# Keep script running
while true; do
    # Check if processes are still running
    if ! kill -0 $FEEDER_PID 2>/dev/null; then
        echo "⚠️  Bybit feeder stopped. Restarting..."
        python bybit_integration.py &
        FEEDER_PID=$!
    fi

    if ! kill -0 $CHIMERA_PID 2>/dev/null; then
        echo "⚠️  CHIMERA stopped. Restarting..."
        ./build/bin/chimera_scalper &
        CHIMERA_PID=$!
    fi

    sleep 5
done
