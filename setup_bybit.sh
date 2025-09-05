#!/bin/bash

echo "🔥 Setting up CHIMERA-Bybit Integration"
echo "======================================="

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install pybit websockets websocket-client numpy

# Create Bybit configuration
cat > bybit_config.json << 'EOF'
{
    "testnet": true,
    "symbols": [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "SHIBUSDT"
    ],
    "depth": 50,
    "api_key": "",
    "api_secret": "",
    "trade_settings": {
        "max_position_size_usd": 1000,
        "max_daily_loss_pct": 2.0,
        "default_leverage": 1,
        "use_testnet_only": true
    }
}
EOF

echo "✓ Created bybit_config.json"

# Create a test script for Bybit connection
cat > test_bybit_connection.py << 'EOF'
#!/usr/bin/env python3
"""Test Bybit WebSocket connection"""

import asyncio
import websockets
import json

async def test_connection():
    url = "wss://stream-testnet.bybit.com/v5/public/linear"

    print("Testing Bybit testnet connection...")

    try:
        async with websockets.connect(url) as ws:
            print("✓ Connected to Bybit testnet")

            # Subscribe to BTC orderbook
            subscribe_msg = {
                "op": "subscribe",
                "args": ["orderbook.50.BTCUSDT"]
            }

            await ws.send(json.dumps(subscribe_msg))
            print("✓ Subscription sent")

            # Wait for a few messages
            for i in range(5):
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(message)

                if 'topic' in data:
                    print(f"✓ Received orderbook data: {data['type']}")
                    if data['type'] == 'snapshot':
                        ob = data['data']
                        if 'b' in ob and 'a' in ob:
                            best_bid = ob['b'][0][0] if ob['b'] else 0
                            best_ask = ob['a'][0][0] if ob['a'] else 0
                            print(f"  Best Bid: {best_bid}")
                            print(f"  Best Ask: {best_ask}")
                            break

            print("\n✅ Bybit connection test successful!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

    return True

if __name__ == "__main__":
    asyncio.run(test_connection())
EOF

echo "✓ Created test_bybit_connection.py"

# Create the main run script
cat > run_chimera_bybit.sh << 'EOF'
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
EOF

chmod +x run_chimera_bybit.sh
chmod +x test_bybit_connection.py

echo ""
echo "✅ Bybit integration setup complete!"
echo ""
echo "📋 Quick Start Guide:"
echo "===================="
echo ""
echo "1. Test Bybit connection:"
echo "   python test_bybit_connection.py"
echo ""
echo "2. Run CHIMERA with live Bybit data:"
echo "   ./run_chimera_bybit.sh"
echo ""
echo "3. Configure trading (optional):"
echo "   - Edit bybit_config.json"
echo "   - Add your API keys (testnet first!)"
echo "   - Get testnet keys at: https://testnet.bybit.com/user/api-management"
echo ""
echo "4. Monitor performance:"
echo "   - Watch the console for real-time stats"
echo "   - Order book imbalance, spread, signals"
echo ""
echo "⚠️  IMPORTANT: Always test on testnet first!"
echo "   The system is configured for testnet by default."
echo ""
echo "📊 Supported Bybit symbols:"
echo "   BTC, ETH, SOL, BNB, XRP, ADA, DOGE, SHIB (all USDT pairs)"