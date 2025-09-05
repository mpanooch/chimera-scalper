#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <iomanip>
#include "core/features.h"
#include "core/bybit_feed.h"
#include "layers/layer1_fastpath.h"
#include "layers/layer2_regime.h"
#include "layers/layer3_experts.h"
#include "layers/layer4_pnlfilter.h"
#include "layers/layer5_execution.h"
#include "utils/logger.h"
#include "utils/timer.h"

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
    ║   🔥 BYBIT REAL-TIME INTEGRATION 🔥                  ║
    ║                                                       ║
    ║   CUDA 12.9 | C++17 | RTX 4070 Ti Optimized         ║
    ║   Multi-Layer AI Architecture                        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    )" << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    print_banner();

    // Check CUDA
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

    std::cout << "  [Layer 0] Microstructure Engine... ";
    chimera::MicrostructureEngine microstructure_engine(200);
    std::cout << "✓\n";

    std::cout << "  [Layer 1] Fast-Path Scalp Filter... ";
    chimera::Layer1FastPath fastpath;
    std::cout << "✓\n";

    std::cout << "  [Layer 2] Market Regime Classifier... ";
    chimera::Layer2Regime regime_classifier;
    std::cout << "✓\n";

    std::cout << "  [Layer 3] MoE Expert System... ";
    chimera::Layer3Experts experts;
    std::cout << "✓\n";

    std::cout << "  [Layer 4] PnL-Driven RL Filter... ";
    chimera::Layer4PnLFilter pnl_filter;
    std::cout << "✓\n";

    std::cout << "  [Layer 5] Execution Controller... ";
    chimera::Layer5Execution execution;
    std::cout << "✓\n";

    // Initialize Bybit feed
    std::cout << "  [Bybit] Real-time L2 Feed... ";
    chimera::BybitFeed bybit_feed;
    bool bybit_connected = bybit_feed.connect("/tmp/chimera_orderbook.dat");
    if (bybit_connected) {
        std::cout << "✓\n";
    } else {
        std::cout << "⚠️ (using synthetic data)\n";
    }

    std::cout << "\n✅ All systems initialized successfully!\n\n";

    std::cout << "🔄 Starting main trading loop...\n";
    std::cout << "   Press Ctrl+C to stop\n\n";

    auto last_heartbeat = std::chrono::steady_clock::now();
    uint64_t tick_count = 0;
    uint64_t real_data_count = 0;

    // Track stats per symbol
    std::map<std::string, int> symbol_counts;
    std::map<std::string, int> signal_counts;

    while (g_running) {
        chimera::OrderBook ob;
        std::string symbol;
        bool has_real_data = false;

        // Try to get real data from Bybit
        if (bybit_connected && bybit_feed.has_new_data()) {
            if (bybit_feed.read_orderbook(ob, symbol)) {
                has_real_data = true;
                real_data_count++;
                symbol_counts[symbol]++;
            }
        }

        // Fall back to synthetic data if no real data
        if (!has_real_data) {
            ob.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            chimera::OrderBookLevel bid_level = {99.99f, 100.0f, 1};
            chimera::OrderBookLevel ask_level = {100.01f, 100.0f, 1};
            ob.bids.push_back(bid_level);
            ob.asks.push_back(ask_level);
            ob.last_trade_price = 100.0f;
            ob.last_trade_size = 10.0f;
            symbol = "SYNTHETIC";
        }

        // Process through all layers with timing
        chimera::Timer pipeline_timer("Pipeline");

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
                // Layer 5: Execute (paper trading for now)
                if (execution.execute_trade(expert_signal)) {
                    signal_counts[symbol]++;

                    std::cout << "\n🎯 SIGNAL: " << symbol
                             << " | " << expert_signal.expert_name
                             << " | " << (expert_signal.is_long ? "LONG" : "SHORT")
                             << " @ " << std::fixed << std::setprecision(2)
                             << expert_signal.entry_price
                             << " | Confidence: " << expert_signal.confidence << "\n";
                }
            }
        }

        tick_count++;

        // Heartbeat with enhanced stats
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 1) {
            std::cout << "💓 Heartbeat | Ticks: " << tick_count
                     << " | Real: " << real_data_count
                     << " | Spread: " << std::fixed << std::setprecision(2)
                     << features.spread_bps << " bps"
                     << " | Imbalance: " << std::setprecision(3)
                     << features.bid_ask_imbalance;

            if (has_real_data) {
                std::cout << " | " << symbol;
            }

            std::cout << "\n";
            last_heartbeat = now;
        }

        // Don't sleep if we have real data flowing
        if (!has_real_data) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            // Process as fast as possible for real data
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Print session statistics
    std::cout << "\n📊 Session Statistics:\n";
    std::cout << "   Total ticks: " << tick_count << "\n";
    std::cout << "   Real data ticks: " << real_data_count << "\n";
    std::cout << "   Symbols processed:\n";
    for (const auto& [sym, count] : symbol_counts) {
        std::cout << "     " << sym << ": " << count << " ticks";
        if (signal_counts[sym] > 0) {
            std::cout << " (" << signal_counts[sym] << " signals)";
        }
        std::cout << "\n";
    }

    std::cout << "\n👋 CHIMERA shutdown complete. Stay profitable!\n";

    return 0;
}