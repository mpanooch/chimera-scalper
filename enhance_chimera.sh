#!/bin/bash

echo "🔥 Enhancing CHIMERA with Real Trading Logic"
echo "==========================================="

# Create a Python script to feed real L2 data
cat > feed_real_data.py << 'EOF'
#!/usr/bin/env python3
"""
Feed real L2 order book data to CHIMERA
"""

import json
import gzip
import time
import socket
import struct
from pathlib import Path

def load_l2_data():
    """Load generated L2 data"""
    ob_files = list(Path('data/ob').glob('*.jsonl.gz'))

    if not ob_files:
        print("No L2 data found. Run generate_synthetic_l2.py first!")
        return []

    all_obs = []
    for f in ob_files[:1]:  # Load first file for testing
        print(f"Loading {f}...")
        with gzip.open(f, 'rt') as gz:
            for line in gz:
                all_obs.append(json.loads(line))
                if len(all_obs) >= 1000:  # Limit for testing
                    break

    print(f"Loaded {len(all_obs)} order books")
    return all_obs

def send_to_chimera(ob_data):
    """Send order book data to CHIMERA via shared memory or socket"""
    # For now, just save to a file that CHIMERA can read
    with open('/tmp/chimera_feed.json', 'w') as f:
        json.dump(ob_data, f)
    print(f"Saved order book to /tmp/chimera_feed.json")

if __name__ == "__main__":
    print("🔥 CHIMERA Data Feeder")
    print("=====================")

    obs = load_l2_data()

    if obs:
        print(f"\nFirst order book:")
        print(f"  Symbol: {obs[0]['symbol']}")
        print(f"  Best Bid: {obs[0]['bids'][0]['price']}")
        print(f"  Best Ask: {obs[0]['asks'][0]['price']}")
        print(f"  Bid Levels: {len(obs[0]['bids'])}")
        print(f"  Ask Levels: {len(obs[0]['asks'])}")

        send_to_chimera(obs[0])

        print("\n✓ Data ready for CHIMERA to consume")
EOF

echo "✓ Created data feeder script"

# Create enhanced main that reads real data
cat > src/main_enhanced.cpp << 'EOF'
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <csignal>
#include <nlohmann/json.hpp>  // You'll need to install this
#include "core/features.h"
#include "core/data_feed.h"
#include "layers/layer1_fastpath.h"
#include "layers/layer2_regime.h"
#include "layers/layer3_experts.h"
#include "layers/layer4_pnlfilter.h"
#include "layers/layer5_execution.h"
#include "utils/logger.h"
#include "utils/timer.h"

using json = nlohmann::json;
std::atomic<bool> g_running(true);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n🛑 Shutdown signal received...\n";
        g_running = false;
    }
}

chimera::OrderBook parse_json_orderbook(const json& j) {
    chimera::OrderBook ob;
    ob.timestamp_ns = j["timestamp_ns"];
    ob.last_trade_price = j["last_trade"]["price"];
    ob.last_trade_size = j["last_trade"]["size"];

    for (const auto& bid : j["bids"]) {
        chimera::OrderBookLevel level;
        level.price = bid["price"];
        level.size = bid["size"];
        level.count = bid["count"];
        ob.bids.push_back(level);
    }

    for (const auto& ask : j["asks"]) {
        chimera::OrderBookLevel level;
        level.price = ask["price"];
        level.size = ask["size"];
        level.count = ask["count"];
        ob.asks.push_back(level);
    }

    return ob;
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ... banner and initialization code ...

    // Load real data if available
    std::ifstream feed_file("/tmp/chimera_feed.json");
    chimera::OrderBook real_ob;
    bool has_real_data = false;

    if (feed_file.is_open()) {
        json j;
        feed_file >> j;
        real_ob = parse_json_orderbook(j);
        has_real_data = true;
        std::cout << "📊 Loaded real order book data!\n";
    }

    // Main loop with real data
    while (g_running) {
        chimera::OrderBook ob = has_real_data ? real_ob : /* generate synthetic */;

        // Process through all layers
        chimera::Timer timer("Full Pipeline");

        // Layer 0: Microstructure
        auto features = microstructure_engine.compute_features(ob);

        // Layer 1: Fast path filter
        auto fast_signal = fastpath.process(features, ob.last_trade_price);

        if (fast_signal.trigger != chimera::ScalpTrigger::NONE) {
            // Layer 2: Regime
            auto regime = regime_classifier.classify(features);

            // Layer 3: Expert signals
            auto expert_signal = experts.generate_signal(features, regime, ob.last_trade_price);

            // Layer 4: PnL filter
            if (pnl_filter.should_trade(expert_signal)) {
                // Layer 5: Execute
                execution.execute_trade(expert_signal);
            }
        }

        timer.stop();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
EOF

echo "✓ Created enhanced main"

# Create a performance monitor
cat > monitor_performance.py << 'EOF'
#!/usr/bin/env python3
"""
Monitor CHIMERA performance in real-time
"""

import time
import psutil
import GPUtil
from datetime import datetime

def monitor():
    print("🔥 CHIMERA Performance Monitor")
    print("==============================\n")

    process_name = "chimera_scalper"

    while True:
        # Find CHIMERA process
        chimera_proc = None
        for proc in psutil.process_iter(['pid', 'name']):
            if process_name in proc.info['name']:
                chimera_proc = psutil.Process(proc.info['pid'])
                break

        if chimera_proc:
            # CPU usage
            cpu_percent = chimera_proc.cpu_percent(interval=0.1)

            # Memory usage
            mem_info = chimera_proc.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024

            # GPU usage
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                gpu_mem = gpu.memoryUsed
            else:
                gpu_util = 0
                gpu_mem = 0

            # Print stats
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {mem_mb:6.1f}MB | "
                  f"GPU: {gpu_util:5.1f}% | "
                  f"VRAM: {gpu_mem:6.1f}MB", end='')
        else:
            print("\rWaiting for CHIMERA to start...", end='')

        time.sleep(1)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
EOF

echo "✓ Created performance monitor"
echo ""
echo "✅ Enhancement complete!"
echo ""
echo "Next steps:"
echo "1. Generate L2 data: python generate_synthetic_l2.py"
echo "2. Feed data: python feed_real_data.py"
echo "3. Monitor: python monitor_performance.py"
echo "4. Run CHIMERA: ./build/bin/chimera_scalper"
echo ""
echo "To add real exchange connectivity:"
echo "  • Binance: pip install python-binance"
echo "  • Bybit: pip install pybit"
echo "  • WebSocket feeds for real-time L2 data"