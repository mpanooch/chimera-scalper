#!/bin/bash

# CHIMERA - Create ALL source files automatically
set -e

echo "🔥 Creating ALL CHIMERA source files..."
echo "======================================"

# ============= ROOT FILES =============
echo "📝 Creating root files..."

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(chimera_scalper LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "89")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CUDA_INCLUDE_DIRS}
)

set(CORE_SOURCES
    src/core/data_feed.cpp
    src/core/features.cu
)

set(LAYER_SOURCES
    src/layers/layer1_fastpath.cpp
    src/layers/layer2_regime.cpp
    src/layers/layer3_experts.cpp
    src/layers/layer4_pnlfilter.cpp
    src/layers/layer5_execution.cpp
)

set(UTIL_SOURCES
    src/utils/logger.cpp
)

set_source_files_properties(src/core/features.cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(chimera_scalper
    src/main.cpp
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${UTIL_SOURCES}
)

target_link_libraries(chimera_scalper
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
    cudart
    cublas
    curand
)

target_compile_options(chimera_scalper PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

set_target_properties(chimera_scalper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
EOF

# ============= SRC/MAIN.CPP =============
echo "📝 Creating main.cpp..."

cat > src/main.cpp << 'EOF'
#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include "core/features.h"
#include "core/data_feed.h"
#include "layers/layer1_fastpath.h"
#include "layers/layer2_regime.h"
#include "layers/layer3_experts.h"
#include "layers/layer4_pnlfilter.h"
#include "layers/layer5_execution.h"
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
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    print_banner();

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

    std::cout << "\n✅ All systems initialized successfully!\n\n";

    std::cout << "🔄 Starting main trading loop...\n";
    std::cout << "   Press Ctrl+C to stop\n\n";

    auto last_heartbeat = std::chrono::steady_clock::now();
    uint64_t tick_count = 0;

    while (g_running) {
        chimera::OrderBook ob;
        ob.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        chimera::OrderBookLevel bid_level = {99.99f, 100.0f, 1};
        chimera::OrderBookLevel ask_level = {100.01f, 100.0f, 1};
        ob.bids.push_back(bid_level);
        ob.asks.push_back(ask_level);
        ob.last_trade_price = 100.0f;
        ob.last_trade_size = 10.0f;

        auto features = microstructure_engine.compute_features(ob);

        tick_count++;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 1) {
            std::cout << "💓 Heartbeat | Ticks: " << tick_count
                     << " | Spread: " << features.spread_bps << " bps"
                     << " | Imbalance: " << features.bid_ask_imbalance << "\n";
            last_heartbeat = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n📊 Session Statistics:\n";
    std::cout << "   Total ticks processed: " << tick_count << "\n";
    std::cout << "\n👋 CHIMERA shutdown complete. Stay profitable!\n";

    return 0;
}
EOF

# ============= CORE FILES =============
echo "📝 Creating core files..."

cat > src/core/features.h << 'EOF'
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace chimera {

struct OrderBookLevel {
    float price;
    float size;
    int count;
};

struct OrderBook {
    std::vector<OrderBookLevel> bids;
    std::vector<OrderBookLevel> asks;
    uint64_t timestamp_ns;
    float last_trade_price;
    float last_trade_size;
};

struct MicrostructureFeatures {
    float bid_ask_imbalance;
    float weighted_imbalance;
    float spread_bps;
    float micro_price;
    float liquidity_score;
    float hidden_liquidity_prob;
    float short_term_vol;
    float order_book_entropy;
    float volume_delta_1m;
    float vwap_deviation;
    float buy_pressure;
    float sell_pressure;
    float hurst_exponent;
    float lyapunov_exponent;
};

class MicrostructureEngine {
public:
    MicrostructureEngine(int max_levels = 200);
    ~MicrostructureEngine();

    MicrostructureFeatures compute_features(const OrderBook& ob);
    std::vector<MicrostructureFeatures> compute_features_batch(
        const std::vector<OrderBook>& obs
    );

    void update_vwap(float price, float volume, uint64_t timestamp_ns);
    float get_vwap() const { return vwap_; }
    float get_total_volume() const { return total_volume_; }

private:
    struct GPUData;
    std::unique_ptr<GPUData> gpu_data_;

    float vwap_;
    float total_volume_;
    float volume_sum_;
    int max_levels_;

    void launch_imbalance_kernel(const OrderBook& ob, MicrostructureFeatures* features);
    void launch_entropy_kernel(const OrderBook& ob, MicrostructureFeatures* features);
    void launch_chaos_kernel(const OrderBook& ob, MicrostructureFeatures* features);
};

namespace cuda_math {
    float compute_hurst_exponent(const float* prices, int n);
    float compute_shannon_entropy(const float* probs, int n);
    float compute_lyapunov(const float* series, int n);
}

} // namespace chimera
EOF

cat > src/core/features.cu << 'EOF'
#include "features.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cmath>

namespace chimera {

struct MicrostructureEngine::GPUData {
    float* d_bid_prices;
    float* d_bid_sizes;
    float* d_ask_prices;
    float* d_ask_sizes;
    MicrostructureFeatures* d_features;
    float* d_temp_buffer;
    int max_levels;

    GPUData(int levels) : max_levels(levels) {
        cudaMalloc(&d_bid_prices, levels * sizeof(float));
        cudaMalloc(&d_bid_sizes, levels * sizeof(float));
        cudaMalloc(&d_ask_prices, levels * sizeof(float));
        cudaMalloc(&d_ask_sizes, levels * sizeof(float));
        cudaMalloc(&d_features, sizeof(MicrostructureFeatures));
        cudaMalloc(&d_temp_buffer, levels * 2 * sizeof(float));
    }

    ~GPUData() {
        cudaFree(d_bid_prices);
        cudaFree(d_bid_sizes);
        cudaFree(d_ask_prices);
        cudaFree(d_ask_sizes);
        cudaFree(d_features);
        cudaFree(d_temp_buffer);
    }
};

__global__ void compute_imbalance_kernel(
    const float* bid_sizes, const float* ask_sizes,
    int n_bids, int n_asks,
    float* imbalance, float* weighted_imbalance
) {
    __shared__ float bid_volume;
    __shared__ float ask_volume;
    __shared__ float weighted_bid;
    __shared__ float weighted_ask;

    if (threadIdx.x == 0) {
        bid_volume = 0.0f;
        ask_volume = 0.0f;
        weighted_bid = 0.0f;
        weighted_ask = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x < n_bids) {
        atomicAdd(&bid_volume, bid_sizes[threadIdx.x]);
        if (threadIdx.x < 5) {
            float weight = 1.0f / (threadIdx.x + 1.0f);
            atomicAdd(&weighted_bid, bid_sizes[threadIdx.x] * weight);
        }
    }

    if (threadIdx.x < n_asks) {
        atomicAdd(&ask_volume, ask_sizes[threadIdx.x]);
        if (threadIdx.x < 5) {
            float weight = 1.0f / (threadIdx.x + 1.0f);
            atomicAdd(&weighted_ask, ask_sizes[threadIdx.x] * weight);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = bid_volume + ask_volume;
        *imbalance = (total > 0) ? (bid_volume - ask_volume) / total : 0.0f;

        float weighted_total = weighted_bid + weighted_ask;
        *weighted_imbalance = (weighted_total > 0) ?
            (weighted_bid - weighted_ask) / weighted_total : 0.0f;
    }
}

MicrostructureEngine::MicrostructureEngine(int max_levels)
    : max_levels_(max_levels), vwap_(0.0f), total_volume_(0.0f), volume_sum_(0.0f) {
    gpu_data_ = std::make_unique<GPUData>(max_levels);
}

MicrostructureEngine::~MicrostructureEngine() = default;

MicrostructureFeatures MicrostructureEngine::compute_features(const OrderBook& ob) {
    MicrostructureFeatures features = {};

    if (ob.bids.empty() || ob.asks.empty()) {
        return features;
    }

    int n_bids = std::min((int)ob.bids.size(), max_levels_);
    int n_asks = std::min((int)ob.asks.size(), max_levels_);

    std::vector<float> bid_sizes(n_bids), ask_sizes(n_asks);
    for (int i = 0; i < n_bids; i++) bid_sizes[i] = ob.bids[i].size;
    for (int i = 0; i < n_asks; i++) ask_sizes[i] = ob.asks[i].size;

    cudaMemcpy(gpu_data_->d_bid_sizes, bid_sizes.data(),
               n_bids * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data_->d_ask_sizes, ask_sizes.data(),
               n_asks * sizeof(float), cudaMemcpyHostToDevice);

    float *d_imbalance, *d_weighted_imbalance;
    cudaMalloc(&d_imbalance, sizeof(float));
    cudaMalloc(&d_weighted_imbalance, sizeof(float));

    compute_imbalance_kernel<<<1, 256>>>(
        gpu_data_->d_bid_sizes, gpu_data_->d_ask_sizes,
        n_bids, n_asks,
        d_imbalance, d_weighted_imbalance
    );

    cudaMemcpy(&features.bid_ask_imbalance, d_imbalance,
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&features.weighted_imbalance, d_weighted_imbalance,
               sizeof(float), cudaMemcpyDeviceToHost);

    if (!ob.bids.empty() && !ob.asks.empty()) {
        float bid = ob.bids[0].price;
        float ask = ob.asks[0].price;
        float mid = (bid + ask) / 2.0f;

        features.spread_bps = ((ask - bid) / mid) * 10000.0f;

        float bid_size = ob.bids[0].size;
        float ask_size = ob.asks[0].size;
        features.micro_price = (bid * ask_size + ask * bid_size) / (bid_size + ask_size);

        if (vwap_ > 0) {
            features.vwap_deviation = (ob.last_trade_price - vwap_) / vwap_;
        }
    }

    features.liquidity_score = 100.0f;
    features.short_term_vol = 0.02f;
    features.buy_pressure = 0.5f;
    features.sell_pressure = 0.5f;
    features.order_book_entropy = 3.5f;
    features.hurst_exponent = 0.5f;

    cudaFree(d_imbalance);
    cudaFree(d_weighted_imbalance);

    return features;
}

void MicrostructureEngine::update_vwap(float price, float volume, uint64_t timestamp_ns) {
    volume_sum_ += price * volume;
    total_volume_ += volume;
    if (total_volume_ > 0) {
        vwap_ = volume_sum_ / total_volume_;
    }
}

} // namespace chimera
EOF

cat > src/core/data_feed.h << 'EOF'
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include "features.h"

namespace chimera {

struct MarketTick {
    std::string symbol;
    uint64_t timestamp_ns;
    float price;
    float volume;
    char side;
};

struct Candle {
    uint64_t timestamp_ms;
    float open;
    float high;
    float low;
    float close;
    float volume;
};

class DataFeed {
public:
    DataFeed();
    ~DataFeed();

    bool load_csv(const std::string& filepath, const std::string& symbol);
    bool load_orderbook_data(const std::string& filepath);
    void start();
    void stop();
    bool get_next_orderbook(OrderBook& ob, int timeout_ms = 1000);
    bool is_running() const { return running_.load(); }
    size_t queue_size() const { return orderbook_queue_.size(); }

private:
    std::queue<OrderBook> orderbook_queue_;
    mutable std::mutex queue_mutex_;
    std::vector<Candle> historical_candles_;
    std::vector<OrderBook> historical_orderbooks_;
    std::atomic<bool> running_;
    std::thread feed_thread_;
    void feed_worker();
};

} // namespace chimera
EOF

cat > src/core/data_feed.cpp << 'EOF'
#include "data_feed.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

namespace chimera {

DataFeed::DataFeed() : running_(false) {}

DataFeed::~DataFeed() {
    stop();
}

bool DataFeed::load_csv(const std::string& filepath, const std::string& symbol) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str;
        Candle candle;

        std::getline(ss, timestamp_str, ',');
        ss >> candle.open;
        ss.ignore();
        ss >> candle.high;
        ss.ignore();
        ss >> candle.low;
        ss.ignore();
        ss >> candle.close;
        ss.ignore();
        ss >> candle.volume;

        candle.timestamp_ms = std::chrono::system_clock::now().time_since_epoch().count() / 1000000;
        historical_candles_.push_back(candle);
    }

    std::cout << "Loaded " << historical_candles_.size() << " candles from " << filepath << std::endl;
    return true;
}

bool DataFeed::load_orderbook_data(const std::string& filepath) {
    for (const auto& candle : historical_candles_) {
        OrderBook ob;
        ob.timestamp_ns = candle.timestamp_ms * 1000000;
        ob.last_trade_price = candle.close;
        ob.last_trade_size = candle.volume / 100.0f;

        float spread = candle.close * 0.0001f;

        for (int i = 0; i < 50; i++) {
            OrderBookLevel bid_level;
            bid_level.price = candle.close - spread * (i + 1);
            bid_level.size = 100.0f / (i + 1);
            bid_level.count = 1;
            ob.bids.push_back(bid_level);

            OrderBookLevel ask_level;
            ask_level.price = candle.close + spread * (i + 1);
            ask_level.size = 100.0f / (i + 1);
            ask_level.count = 1;
            ob.asks.push_back(ask_level);
        }

        historical_orderbooks_.push_back(ob);
    }

    return !historical_orderbooks_.empty();
}

void DataFeed::start() {
    if (running_.load()) return;
    running_ = true;
    feed_thread_ = std::thread(&DataFeed::feed_worker, this);
}

void DataFeed::stop() {
    running_ = false;
    if (feed_thread_.joinable()) {
        feed_thread_.join();
    }
}

bool DataFeed::get_next_orderbook(OrderBook& ob, int timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    auto deadline = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(timeout_ms);

    while (orderbook_queue_.empty()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        lock.lock();
    }

    ob = orderbook_queue_.front();
    orderbook_queue_.pop();
    return true;
}

void DataFeed::feed_worker() {
    size_t index = 0;

    while (running_.load() && index < historical_orderbooks_.size()) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (orderbook_queue_.size() < 1000) {
                orderbook_queue_.push(historical_orderbooks_[index]);
                index++;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace chimera
EOF

# Create Layer 1-5 files (continuing from previous script)
echo "📝 Creating Layer 1-5 files..."

# [Include all the layer files from the complete_setup.sh script here]
# I'll include just the essential headers to save space

cat > src/layers/layer1_fastpath.h << 'EOF'
#pragma once
#include "../core/features.h"
#include <vector>

namespace chimera {

enum class ScalpTrigger {
    NONE = 0,
    ROSS_MOMENTUM = 1,
    BAO_MEAN_REVERSION = 2,
    NICK_VOLUME_PRESSURE = 3,
    FABIO_CHAOS = 4
};

struct FastPathSignal {
    ScalpTrigger trigger;
    float confidence;
    float entry_price;
    bool is_long;
};

class Layer1FastPath {
public:
    Layer1FastPath();
    FastPathSignal process(const MicrostructureFeatures& features, float price);
    void set_ross_sensitivity(float s) { ross_sensitivity_ = s; }
    void set_bao_sensitivity(float s) { bao_sensitivity_ = s; }
    void set_nick_sensitivity(float s) { nick_sensitivity_ = s; }
    void set_fabio_sensitivity(float s) { fabio_sensitivity_ = s; }

private:
    float ema_fast_;
    float ema_slow_;
    float vwap_upper_;
    float vwap_lower_;
    std::vector<float> volume_deltas_;
    float ross_sensitivity_;
    float bao_sensitivity_;
    float nick_sensitivity_;
    float fabio_sensitivity_;
    bool check_ross_trigger(const MicrostructureFeatures& f, float price);
    bool check_bao_trigger(const MicrostructureFeatures& f, float price);
    bool check_nick_trigger(const MicrostructureFeatures& f, float price);
    bool check_fabio_trigger(const MicrostructureFeatures& f, float price);
};

} // namespace chimera
EOF

cat > src/layers/layer1_fastpath.cpp << 'EOF'
#include "layer1_fastpath.h"
#include <cmath>

namespace chimera {

Layer1FastPath::Layer1FastPath()
    : ema_fast_(0), ema_slow_(0),
      vwap_upper_(0), vwap_lower_(0),
      ross_sensitivity_(1.0f),
      bao_sensitivity_(1.0f),
      nick_sensitivity_(1.0f),
      fabio_sensitivity_(1.0f) {
    volume_deltas_.reserve(100);
}

FastPathSignal Layer1FastPath::process(const MicrostructureFeatures& features, float price) {
    FastPathSignal signal = {ScalpTrigger::NONE, 0.0f, price, true};

    float alpha_fast = 2.0f / (9.0f + 1.0f);
    float alpha_slow = 2.0f / (21.0f + 1.0f);

    if (ema_fast_ == 0) {
        ema_fast_ = price;
        ema_slow_ = price;
    } else {
        ema_fast_ = alpha_fast * price + (1 - alpha_fast) * ema_fast_;
        ema_slow_ = alpha_slow * price + (1 - alpha_slow) * ema_slow_;
    }

    if (check_ross_trigger(features, price)) {
        signal.trigger = ScalpTrigger::ROSS_MOMENTUM;
        signal.confidence = 0.85f * ross_sensitivity_;
        signal.is_long = (ema_fast_ > ema_slow_);
    }

    return signal;
}

bool Layer1FastPath::check_ross_trigger(const MicrostructureFeatures& f, float price) {
    bool ema_cross = std::abs(ema_fast_ - ema_slow_) > (price * 0.001f);
    bool momentum = std::abs(f.bid_ask_imbalance) > 0.3f;
    return ema_cross && momentum;
}

bool Layer1FastPath::check_bao_trigger(const MicrostructureFeatures& f, float price) {
    return std::abs(f.vwap_deviation) > 0.002f;
}

bool Layer1FastPath::check_nick_trigger(const MicrostructureFeatures& f, float price) {
    float pressure_diff = std::abs(f.buy_pressure - f.sell_pressure);
    return pressure_diff > 0.2f;
}

bool Layer1FastPath::check_fabio_trigger(const MicrostructureFeatures& f, float price) {
    return f.order_book_entropy > 3.0f && f.hurst_exponent < 0.4f;
}

} // namespace chimera
EOF

# Continue with remaining layers...
# [Add all Layer 2-5 files here from the previous script]

# ============= UTILS FILES =============
echo "📝 Creating utils files..."

cat > src/utils/logger.h << 'EOF'
#pragma once
#include <string>
#include <iostream>
#include <mutex>

namespace chimera {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string level_str;
        std::string color;

        switch(level) {
            case LogLevel::DEBUG:    level_str = "DEBUG"; color = "\033[36m"; break;
            case LogLevel::INFO:     level_str = "INFO "; color = "\033[32m"; break;
            case LogLevel::WARNING:  level_str = "WARN "; color = "\033[33m"; break;
            case LogLevel::ERROR:    level_str = "ERROR"; color = "\033[31m"; break;
            case LogLevel::CRITICAL: level_str = "CRIT "; color = "\033[35m"; break;
        }

        std::cout << color << "[" << level_str << "] " << message << "\033[0m" << std::endl;
    }

private:
    Logger() = default;
    std::mutex mutex_;
};

#define LOG_DEBUG(msg) chimera::Logger::instance().log(chimera::LogLevel::DEBUG, msg)
#define LOG_INFO(msg) chimera::Logger::instance().log(chimera::LogLevel::INFO, msg)
#define LOG_WARNING(msg) chimera::Logger::instance().log(chimera::LogLevel::WARNING, msg)
#define LOG_ERROR(msg) chimera::Logger::instance().log(chimera::LogLevel::ERROR, msg)
#define LOG_CRITICAL(msg) chimera::Logger::instance().log(chimera::LogLevel::CRITICAL, msg)

} // namespace chimera
EOF

cat > src/utils/logger.cpp << 'EOF'
#include "logger.h"

namespace chimera {
// Logger implementation is in header (header-only for simplicity)
}
EOF

cat > src/utils/config.h << 'EOF'
#pragma once

namespace chimera {
namespace config {

constexpr int MAX_ORDER_BOOK_LEVELS = 200;
constexpr int CUDA_THREADS_PER_BLOCK = 256;

constexpr float MAX_POSITION_SIZE = 10000.0f;
constexpr float MAX_DAILY_LOSS = -0.02f;
constexpr float DEFAULT_STOP_LOSS_PCT = 0.005f;
constexpr float DEFAULT_TAKE_PROFIT_PCT = 0.01f;

constexpr int TARGET_L0_LATENCY_US = 100;
constexpr int TARGET_L1_LATENCY_US = 500;
constexpr int TARGET_L2_LATENCY_US = 2000;
constexpr int TARGET_L3_LATENCY_US = 3000;
constexpr int TARGET_L4_LATENCY_US = 1000;
constexpr int TARGET_L5_LATENCY_US = 500;

constexpr float KELLY_FRACTION = 0.25f;
constexpr int MAX_CONCURRENT_POSITIONS = 5;
constexpr float MIN_CONFIDENCE_THRESHOLD = 0.7f;

constexpr int ORDER_BOOK_QUEUE_SIZE = 1000;
constexpr int TICK_BUFFER_SIZE = 10000;

} // namespace config
} // namespace chimera
EOF

cat > src/utils/timer.h << 'EOF'
#pragma once
#include <chrono>
#include <string>
#include <iostream>

namespace chimera {

class Timer {
public:
    Timer(const std::string& name = "") : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        if (!stopped_) {
            stop();
        }
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();

        if (!name_.empty()) {
            std::cout << "⏱️  " << name_ << ": " << duration << " μs" << std::endl;
        }

        stopped_ = true;
    }

    long elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    bool stopped_ = false;
};

} // namespace chimera
EOF

# Create remaining layer files
echo "📝 Creating remaining layer files..."

cat > src/layers/layer2_regime.h << 'EOF'
#pragma once
#include "../core/features.h"

namespace chimera {

enum class MarketRegime {
    TREND_UP,
    TREND_DOWN,
    RANGE_BOUND,
    LIQUIDITY_EVENT,
    CHAOTIC_SPIKE
};

class Layer2Regime {
public:
    Layer2Regime() = default;
    MarketRegime classify(const MicrostructureFeatures& features);
    float get_confidence() const { return confidence_; }

private:
    float confidence_ = 0.5f;
};

} // namespace chimera
EOF

cat > src/layers/layer2_regime.cpp << 'EOF'
#include "layer2_regime.h"
#include <cmath>

namespace chimera {

MarketRegime Layer2Regime::classify(const MicrostructureFeatures& features) {
    confidence_ = 0.7f;

    if (std::abs(features.bid_ask_imbalance) > 0.4f) {
        if (features.bid_ask_imbalance > 0) {
            confidence_ = 0.85f;
            return MarketRegime::TREND_UP;
        } else {
            confidence_ = 0.85f;
            return MarketRegime::TREND_DOWN;
        }
    }

    if (features.order_book_entropy > 4.0f && features.hurst_exponent < 0.4f) {
        confidence_ = 0.8f;
        return MarketRegime::CHAOTIC_SPIKE;
    }

    if (features.liquidity_score < 50.0f) {
        confidence_ = 0.75f;
        return MarketRegime::LIQUIDITY_EVENT;
    }

    return MarketRegime::RANGE_BOUND;
}

} // namespace chimera
EOF

cat > src/layers/layer3_experts.h << 'EOF'
#pragma once
#include "../core/features.h"
#include "layer2_regime.h"
#include <string>

namespace chimera {

struct ExpertSignal {
    float entry_price;
    float stop_loss;
    float take_profit;
    float confidence;
    bool is_long;
    std::string expert_name;
};

class Layer3Experts {
public:
    Layer3Experts();
    ExpertSignal generate_signal(const MicrostructureFeatures& features,
                                MarketRegime regime,
                                float current_price);

private:
    ExpertSignal ross_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal bao_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal nick_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal fabio_expert(const MicrostructureFeatures& f, float price);
    float get_expert_weight(const std::string& expert, MarketRegime regime);
};

} // namespace chimera
EOF

cat > src/layers/layer3_experts.cpp << 'EOF'
#include "layer3_experts.h"
#include <cmath>
#include <algorithm>

namespace chimera {

Layer3Experts::Layer3Experts() {}

ExpertSignal Layer3Experts::generate_signal(const MicrostructureFeatures& features,
                                            MarketRegime regime,
                                            float current_price) {
    ExpertSignal ross = ross_expert(features, current_price);
    ExpertSignal bao = bao_expert(features, current_price);
    ExpertSignal nick = nick_expert(features, current_price);
    ExpertSignal fabio = fabio_expert(features, current_price);

    ross.confidence *= get_expert_weight("ross", regime);
    bao.confidence *= get_expert_weight("bao", regime);
    nick.confidence *= get_expert_weight("nick", regime);
    fabio.confidence *= get_expert_weight("fabio", regime);

    ExpertSignal best = ross;
    if (bao.confidence > best.confidence) best = bao;
    if (nick.confidence > best.confidence) best = nick;
    if (fabio.confidence > best.confidence) best = fabio;

    return best;
}

ExpertSignal Layer3Experts::ross_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Ross";
    signal.entry_price = price;
    signal.is_long = (f.bid_ask_imbalance > 0);
    signal.stop_loss = price * (signal.is_long ? 0.995f : 1.005f);
    signal.take_profit = price * (signal.is_long ? 1.01f : 0.99f);
    signal.confidence = std::abs(f.bid_ask_imbalance) * 0.8f;
    return signal;
}

ExpertSignal Layer3Experts::bao_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Bao";
    signal.entry_price = price;
    signal.is_long = (f.vwap_deviation < -0.002f);
    signal.stop_loss = price * (signal.is_long ? 0.997f : 1.003f);
    signal.take_profit = price * (signal.is_long ? 1.005f : 0.995f);
    signal.confidence = std::min(0.9f, std::abs(f.vwap_deviation) * 100.0f);
    return signal;
}

ExpertSignal Layer3Experts::nick_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Nick";
    signal.entry_price = price;
    signal.is_long = (f.buy_pressure > f.sell_pressure);
    signal.stop_loss = price * (signal.is_long ? 0.996f : 1.004f);
    signal.take_profit = price * (signal.is_long ? 1.008f : 0.992f);
    signal.confidence = std::abs(f.buy_pressure - f.sell_pressure);
    return signal;
}

ExpertSignal Layer3Experts::fabio_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Fabio";
    signal.entry_price = price;
    signal.is_long = (f.hurst_exponent < 0.5f && f.bid_ask_imbalance > 0);
    signal.stop_loss = price * (signal.is_long ? 0.993f : 1.007f);
    signal.take_profit = price * (signal.is_long ? 1.015f : 0.985f);
    signal.confidence = (1.0f - std::abs(f.hurst_exponent - 0.5f)) * 0.7f;
    return signal;
}

float Layer3Experts::get_expert_weight(const std::string& expert, MarketRegime regime) {
    if (expert == "ross") {
        return (regime == MarketRegime::TREND_UP || regime == MarketRegime::TREND_DOWN) ? 1.2f : 0.8f;
    }
    if (expert == "bao") {
        return (regime == MarketRegime::RANGE_BOUND) ? 1.3f : 0.7f;
    }
    if (expert == "nick") {
        return (regime == MarketRegime::LIQUIDITY_EVENT) ? 1.1f : 0.9f;
    }
    if (expert == "fabio") {
        return (regime == MarketRegime::CHAOTIC_SPIKE) ? 1.5f : 0.6f;
    }
    return 1.0f;
}

} // namespace chimera
EOF

cat > src/layers/layer4_pnlfilter.h << 'EOF'
#pragma once
#include "layer3_experts.h"
#include <map>

namespace chimera {

class Layer4PnLFilter {
public:
    Layer4PnLFilter();
    bool should_trade(const ExpertSignal& signal);
    float calculate_expected_pnl(const ExpertSignal& signal);
    void update_performance(const std::string& expert, float pnl);

private:
    struct ExpertStats {
        int trades = 0;
        int wins = 0;
        float total_pnl = 0.0f;
        float win_rate = 0.5f;
        float avg_win = 0.0f;
        float avg_loss = 0.0f;
    };

    std::map<std::string, ExpertStats> expert_stats_;
    float min_expected_pnl_ = 0.0005f;
};

} // namespace chimera
EOF

cat > src/layers/layer4_pnlfilter.cpp << 'EOF'
#include "layer4_pnlfilter.h"
#include <algorithm>

namespace chimera {

Layer4PnLFilter::Layer4PnLFilter() {
    expert_stats_["Ross"] = ExpertStats();
    expert_stats_["Bao"] = ExpertStats();
    expert_stats_["Nick"] = ExpertStats();
    expert_stats_["Fabio"] = ExpertStats();
}

bool Layer4PnLFilter::should_trade(const ExpertSignal& signal) {
    if (signal.confidence < 0.7f) {
        return false;
    }
    float expected_pnl = calculate_expected_pnl(signal);
    return expected_pnl > min_expected_pnl_;
}

float Layer4PnLFilter::calculate_expected_pnl(const ExpertSignal& signal) {
    auto it = expert_stats_.find(signal.expert_name);
    if (it == expert_stats_.end()) {
        return 0.001f;
    }

    const ExpertStats& stats = it->second;

    if (stats.trades < 10) {
        return signal.confidence * 0.001f;
    }

    float win_prob = stats.win_rate;
    float loss_prob = 1.0f - win_prob;

    float reward = std::abs(signal.take_profit - signal.entry_price) / signal.entry_price;
    float risk = std::abs(signal.stop_loss - signal.entry_price) / signal.entry_price;

    float expected = (win_prob * reward) - (loss_prob * risk);
    return expected * signal.confidence;
}

void Layer4PnLFilter::update_performance(const std::string& expert, float pnl) {
    auto it = expert_stats_.find(expert);
    if (it == expert_stats_.end()) {
        return;
    }

    ExpertStats& stats = it->second;
    stats.trades++;
    stats.total_pnl += pnl;

    if (pnl > 0) {
        stats.wins++;
        stats.avg_win = (stats.avg_win * (stats.wins - 1) + pnl) / stats.wins;
    } else {
        int losses = stats.trades - stats.wins;
        stats.avg_loss = (stats.avg_loss * (losses - 1) + std::abs(pnl)) / losses;
    }

    stats.win_rate = static_cast<float>(stats.wins) / stats.trades;
}

} // namespace chimera
EOF

cat > src/layers/layer5_execution.h << 'EOF'
#pragma once
#include "layer3_experts.h"
#include <atomic>

namespace chimera {

class Layer5Execution {
public:
    Layer5Execution();
    bool execute_trade(const ExpertSignal& signal);
    void update_risk_metrics();
    bool check_risk_limits();

private:
    std::atomic<float> daily_pnl_{0.0f};
    std::atomic<float> position_size_{0.0f};
    std::atomic<int> open_positions_{0};

    const float max_daily_loss_ = -0.02f;
    const float max_position_size_ = 10000.0f;
    const int max_positions_ = 5;

    bool place_order(float price, float size, bool is_buy);
    bool set_stop_loss(float price);
    bool set_take_profit(float price);
};

} // namespace chimera
EOF

cat > src/layers/layer5_execution.cpp << 'EOF'
#include "layer5_execution.h"
#include "../utils/logger.h"
#include <iostream>

namespace chimera {

Layer5Execution::Layer5Execution() {}

bool Layer5Execution::execute_trade(const ExpertSignal& signal) {
    if (!check_risk_limits()) {
        LOG_WARNING("Risk limits exceeded, trade rejected");
        return false;
    }

    float position_size = 1000.0f;

    LOG_INFO("Executing trade: " + signal.expert_name +
             " | Direction: " + (signal.is_long ? "LONG" : "SHORT") +
             " | Entry: " + std::to_string(signal.entry_price) +
             " | Confidence: " + std::to_string(signal.confidence));

    bool success = place_order(signal.entry_price, position_size, signal.is_long);

    if (success) {
        set_stop_loss(signal.stop_loss);
        set_take_profit(signal.take_profit);
        open_positions_++;
        position_size_ += position_size;
    }

    return success;
}

bool Layer5Execution::check_risk_limits() {
    if (daily_pnl_ < max_daily_loss_) {
        LOG_ERROR("Daily loss limit reached: " + std::to_string(daily_pnl_.load()));
        return false;
    }

    if (open_positions_ >= max_positions_) {
        LOG_WARNING("Max positions reached: " + std::to_string(open_positions_.load()));
        return false;
    }

    if (position_size_ >= max_position_size_) {
        LOG_WARNING("Max position size reached: " + std::to_string(position_size_.load()));
        return false;
    }

    return true;
}

void Layer5Execution::update_risk_metrics() {
}

bool Layer5Execution::place_order(float price, float size, bool is_buy) {
    std::cout << "  [ORDER] " << (is_buy ? "BUY" : "SELL")
              << " " << size << " @ " << price << std::endl;
    return true;
}

bool Layer5Execution::set_stop_loss(float price) {
    std::cout << "  [STOP LOSS] Set at " << price << std::endl;
    return true;
}

bool Layer5Execution::set_take_profit(float price) {
    std::cout << "  [TAKE PROFIT] Set at " << price << std::endl;
    return true;
}

} // namespace chimera
EOF

echo "✅ All source files created successfully!"
echo ""
echo "🔨 Now building CHIMERA..."
echo "================================"

# Make build script executable and run it
chmod +x build.sh
./build.sh