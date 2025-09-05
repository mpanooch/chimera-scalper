#!/bin/bash

echo "🔧 Fixing CHIMERA Compilation Issues"
echo "===================================="

# First, let's fix the CMakeLists.txt with proper modern CUDA setup
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(chimera_scalper LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 4070 Ti
set(CMAKE_CUDA_ARCHITECTURES "89")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Find packages
find_package(Threads REQUIRED)

# Find Eigen3 (optional for now)
find_package(Eigen3 3.3 QUIET NO_MODULE)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# If Eigen3 found, add its include directory
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_definitions(-DHAS_EIGEN)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/core/data_feed.cpp
    src/core/features.cu
    src/layers/layer1_fastpath.cpp
    src/layers/layer2_regime.cpp
    src/layers/layer3_experts.cpp
    src/layers/layer4_pnlfilter.cpp
    src/layers/layer5_execution.cpp
    src/utils/logger.cpp
)

# Create executable
add_executable(chimera_scalper ${SOURCES})

# Link libraries
target_link_libraries(chimera_scalper
    ${CMAKE_THREAD_LIBS_INIT}
    cudart
)

# If Eigen3 found, link it
if(Eigen3_FOUND)
    target_link_libraries(chimera_scalper Eigen3::Eigen)
endif()

# Set output directory
set_target_properties(chimera_scalper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CUDA separable compilation if needed
set_property(TARGET chimera_scalper PROPERTY CUDA_SEPARABLE_COMPILATION ON)
EOF

echo "✓ Updated CMakeLists.txt"

# Fix potential std::min/max issues in layer files
echo "Fixing std::min/max issues..."

# Fix layer3_experts.cpp
sed -i 's/std::min(/fmin(/g' src/layers/layer3_experts.cpp
sed -i 's/std::max(/fmax(/g' src/layers/layer3_experts.cpp

# Add missing headers to layer3_experts.cpp
cat > src/layers/layer3_experts.cpp.fixed << 'EOF'
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
    signal.confidence = fabsf(f.bid_ask_imbalance) * 0.8f;
    return signal;
}

ExpertSignal Layer3Experts::bao_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Bao";
    signal.entry_price = price;
    signal.is_long = (f.vwap_deviation < -0.002f);
    signal.stop_loss = price * (signal.is_long ? 0.997f : 1.003f);
    signal.take_profit = price * (signal.is_long ? 1.005f : 0.995f);
    float conf = fabsf(f.vwap_deviation) * 100.0f;
    signal.confidence = (conf < 0.9f) ? conf : 0.9f;  // min replacement
    return signal;
}

ExpertSignal Layer3Experts::nick_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Nick";
    signal.entry_price = price;
    signal.is_long = (f.buy_pressure > f.sell_pressure);
    signal.stop_loss = price * (signal.is_long ? 0.996f : 1.004f);
    signal.take_profit = price * (signal.is_long ? 1.008f : 0.992f);
    signal.confidence = fabsf(f.buy_pressure - f.sell_pressure);
    return signal;
}

ExpertSignal Layer3Experts::fabio_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Fabio";
    signal.entry_price = price;
    signal.is_long = (f.hurst_exponent < 0.5f && f.bid_ask_imbalance > 0);
    signal.stop_loss = price * (signal.is_long ? 0.993f : 1.007f);
    signal.take_profit = price * (signal.is_long ? 1.015f : 0.985f);
    signal.confidence = (1.0f - fabsf(f.hurst_exponent - 0.5f)) * 0.7f;
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

mv src/layers/layer3_experts.cpp.fixed src/layers/layer3_experts.cpp

# Fix layer4_pnlfilter.cpp
cat > src/layers/layer4_pnlfilter.cpp.fixed << 'EOF'
#include "layer4_pnlfilter.h"
#include <cmath>

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

    float reward = fabsf(signal.take_profit - signal.entry_price) / signal.entry_price;
    float risk = fabsf(signal.stop_loss - signal.entry_price) / signal.entry_price;

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
        stats.avg_loss = (stats.avg_loss * (losses - 1) + fabsf(pnl)) / losses;
    }

    stats.win_rate = static_cast<float>(stats.wins) / stats.trades;
}

} // namespace chimera
EOF

mv src/layers/layer4_pnlfilter.cpp.fixed src/layers/layer4_pnlfilter.cpp

echo "✓ Fixed potential std::min/max issues"

# Try building with the fixed version
echo ""
echo "🔨 Building with fixes..."
rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10

echo ""
echo "Making project..."
make -j$(nproc) 2>&1 | tail -20

if [ -f "bin/chimera_scalper" ]; then
    echo ""
    echo "✅ BUILD SUCCESSFUL!"
    echo "Run with: ./bin/chimera_scalper"
else
    echo ""
    echo "❌ Build still failing. Trying simplified version..."

    # Create a super simple version that will definitely work
    cd ..
    cat > simple_test.cpp << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "CHIMERA Test\n";

    int devices = 0;
    cudaGetDeviceCount(&devices);
    std::cout << "CUDA Devices: " << devices << "\n";

    if (devices > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << "\n";
    }

    return 0;
}
EOF

    echo "Trying simple compilation..."
    g++ -o simple_test simple_test.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

    if [ -f "simple_test" ]; then
        echo "✓ Simple test compiled!"
        ./simple_test
    fi
fi