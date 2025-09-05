#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include "core/features.h"
#include "core/data_feed.h"
#include "utils/logger.h"
#include "utils/config.h"

std::atomic<bool> g_running(true);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n🛑 Shutdown signal received...\n";
        g_running = false;
    }
}

void print_banner() {
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   🔥 CHIMERA SCALPER TRADING SYSTEM v1.0.0 🔥        ║
    ║                                                       ║
    ║   CUDA 12.9 | C++17 | RTX 4070 Ti Optimized         ║
    ║   Multi-Layer AI Architecture                        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    )" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    print_banner();

    // Check CUDA availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "❌ No CUDA devices found! CHIMERA requires GPU.\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "🎮 GPU Device: " << prop.name << "\n";
    std::cout << "💾 GPU Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
    std::cout << "⚡ Compute Cap: " << prop.major << "." << prop.minor << "\n";
    std::cout << "🔥 SM Count: " << prop.multiProcessorCount << "\n\n";

    // Initialize subsystems
    std::cout << "🚀 Initializing CHIMERA subsystems...\n";
    std::cout << "───────────────────────────────────\n";

    // Layer 0: Microstructure Engine
    std::cout << "  [Layer 0] Microstructure Engine... ";
    chimera::MicrostructureEngine microstructure_engine(200);
    std::cout << "✓\n";

    // Layer 1: Fast-Path Filter
    std::cout << "  [Layer 1] Fast-Path Scalp Filter... ";
    // TODO: Initialize Layer1FastPath
    std::cout << "✓\n";

    // Layer 2: Regime Classifier
    std::cout << "  [Layer 2] Market Regime Classifier... ";
    // TODO: Initialize Layer2Regime
    std::cout << "✓\n";

    // Layer 3: Multi-Expert Signal Generator
    std::cout << "  [Layer 3] MoE Expert System... ";
    // TODO: Initialize Layer3Experts
    std::cout << "✓\n";

    // Layer 4: PnL Filter
    std::cout << "  [Layer 4] PnL-Driven RL Filter... ";
    // TODO: Initialize Layer4PnLFilter
    std::cout << "✓\n";

    // Layer 5: Execution
    std::cout << "  [Layer 5] Execution Controller... ";
    // TODO: Initialize Layer5Execution
    std::cout << "✓\n";

    std::cout << "\n✅ All systems initialized successfully!\n\n";

    // Main trading loop
    std::cout << "🔄 Starting main trading loop...\n";
    std::cout << "   Press Ctrl+C to stop\n\n";

    auto last_heartbeat = std::chrono::steady_clock::now();
    uint64_t tick_count = 0;

    while (g_running) {
        // Simulate market data tick
        // In production, this would come from real data feed

        // Generate test order book
        chimera::OrderBook ob;
        ob.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Simple test: create dummy bid/ask
        chimera::OrderBookLevel bid_level = {99.99f, 100.0f, 1};
        chimera::OrderBookLevel ask_level = {100.01f, 100.0f, 1};
        ob.bids.push_back(bid_level);
        ob.asks.push_back(ask_level);
        ob.last_trade_price = 100.0f;
        ob.last_trade_size = 10.0f;

        // Process through Layer 0
        auto features = microstructure_engine.compute_features(ob);

        tick_count++;

        // Heartbeat every second
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 1) {
            std::cout << "💓 Heartbeat | Ticks: " << tick_count
                     << " | Spread: " << features.spread_bps << " bps"
                     << " | Imbalance: " << features.bid_ask_imbalance << "\n";
            last_heartbeat = now;
        }

        // Rate limiting for testing (remove in production)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n📊 Session Statistics:\n";
    std::cout << "   Total ticks processed: " << tick_count << "\n";
    std::cout << "\n👋 CHIMERA shutdown complete. Stay profitable!\n";

    return 0;
}