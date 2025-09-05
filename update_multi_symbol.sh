#!/bin/bash

echo "🔥 Updating CHIMERA for Multi-Symbol Trading"
echo "==========================================="

# Update the Bybit feed header for multiple symbols
cat > src/core/bybit_multi_feed.h << 'EOF'
#pragma once

#include <string>
#include <map>
#include <memory>
#include "features.h"

namespace chimera {

class BybitMultiFeed {
public:
    BybitMultiFeed();
    ~BybitMultiFeed();

    bool connect(const std::string& shm_path = "/tmp/chimera_multi_ob.dat");
    void disconnect();

    // Read all available orderbooks
    std::map<std::string, OrderBook> read_all_orderbooks();

    // Get specific symbol
    bool read_orderbook(const std::string& symbol, OrderBook& ob);

    // Check for new data
    bool has_new_data();

private:
    void* shm_ptr_;
    size_t shm_size_;
    int shm_fd_;
    std::map<std::string, uint64_t> last_timestamps_;
    bool connected_;
};

} // namespace chimera
EOF

# Create the implementation
cat > src/core/bybit_multi_feed.cpp << 'EOF'
#include "bybit_multi_feed.h"
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace chimera {

BybitMultiFeed::BybitMultiFeed()
    : shm_ptr_(nullptr), shm_size_(4 * 1024 * 1024),  // 4MB
      shm_fd_(-1), connected_(false) {
}

BybitMultiFeed::~BybitMultiFeed() {
    disconnect();
}

bool BybitMultiFeed::connect(const std::string& shm_path) {
    shm_fd_ = open(shm_path.c_str(), O_RDONLY);
    if (shm_fd_ == -1) {
        std::cerr << "Failed to open shared memory: " << shm_path << std::endl;
        return false;
    }

    shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    connected_ = true;
    std::cout << "✓ Connected to multi-symbol Bybit feed" << std::endl;
    return true;
}

void BybitMultiFeed::disconnect() {
    if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
        munmap(shm_ptr_, shm_size_);
        shm_ptr_ = nullptr;
    }

    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }

    connected_ = false;
}

std::map<std::string, OrderBook> BybitMultiFeed::read_all_orderbooks() {
    std::map<std::string, OrderBook> orderbooks;

    if (!connected_ || !shm_ptr_) {
        return orderbooks;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(shm_ptr_);
    size_t offset = 0;

    // Read number of symbols
    uint32_t num_symbols;
    memcpy(&num_symbols, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Read each symbol's orderbook
    for (uint32_t sym_idx = 0; sym_idx < num_symbols; sym_idx++) {
        // Read symbol name
        uint32_t symbol_len;
        memcpy(&symbol_len, ptr + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        std::string symbol(symbol_len, '\0');
        memcpy(&symbol[0], ptr + offset, symbol_len);
        offset += symbol_len;

        // Read timestamp
        uint64_t timestamp_ns;
        memcpy(&timestamp_ns, ptr + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        // Check if new data
        if (last_timestamps_[symbol] >= timestamp_ns) {
            // Skip this orderbook
            uint32_t n_bids, n_asks;
            memcpy(&n_bids, ptr + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            memcpy(&n_asks, ptr + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            offset += (n_bids + n_asks) * 2 * sizeof(float);
            continue;
        }

        last_timestamps_[symbol] = timestamp_ns;

        OrderBook ob;
        ob.timestamp_ns = timestamp_ns;

        // Read bid/ask counts
        uint32_t n_bids, n_asks;
        memcpy(&n_bids, ptr + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&n_asks, ptr + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // Read bids
        for (uint32_t i = 0; i < n_bids; i++) {
            OrderBookLevel level;
            memcpy(&level.price, ptr + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&level.size, ptr + offset, sizeof(float));
            offset += sizeof(float);
            level.count = 1;
            ob.bids.push_back(level);
        }

        // Read asks
        for (uint32_t i = 0; i < n_asks; i++) {
            OrderBookLevel level;
            memcpy(&level.price, ptr + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&level.size, ptr + offset, sizeof(float));
            offset += sizeof(float);
            level.count = 1;
            ob.asks.push_back(level);
        }

        // Set last trade price
        if (!ob.bids.empty() && !ob.asks.empty()) {
            ob.last_trade_price = (ob.bids[0].price + ob.asks[0].price) / 2.0f;
            ob.last_trade_size = 1.0f;
        }

        orderbooks[symbol] = ob;
    }

    return orderbooks;
}

bool BybitMultiFeed::has_new_data() {
    if (!connected_ || !shm_ptr_) {
        return false;
    }

    // Quick check: read first symbol's timestamp
    const uint8_t* ptr = static_cast<const uint8_t*>(shm_ptr_);
    size_t offset = sizeof(uint32_t);  // Skip num_symbols

    if (offset < shm_size_) {
        uint32_t symbol_len;
        memcpy(&symbol_len, ptr + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t) + symbol_len;

        uint64_t timestamp_ns;
        memcpy(&timestamp_ns, ptr + offset, sizeof(uint64_t));

        // Check any symbol for new data
        for (const auto& [symbol, last_ts] : last_timestamps_) {
            if (timestamp_ns > last_ts) {
                return true;
            }
        }
    }

    return false;
}

} // namespace chimera
EOF

# Create run script for multi-symbol
cat > run_multi_symbol.sh << 'EOF'
#!/bin/bash

echo "🔥 Starting CHIMERA Multi-Symbol Trading"
echo "========================================"

cleanup() {
    echo -e "\n🛑 Shutting down..."
    kill $FEEDER_PID 2>/dev/null
    kill $CHIMERA_PID 2>/dev/null
    rm -f /tmp/chimera_multi_ob.dat
    echo "✓ Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start multi-symbol Bybit feeder
echo "Starting multi-symbol Bybit feeder..."
python enhanced_bybit_multi.py &
FEEDER_PID=$!

sleep 3

# Start CHIMERA
echo -e "\nStarting CHIMERA..."
./build/bin/chimera_scalper &
CHIMERA_PID=$!

echo -e "\n✅ Systems running!"
echo "   Multi-Symbol Feeder PID: $FEEDER_PID"
echo "   CHIMERA PID: $CHIMERA_PID"
echo -e "\nMonitoring all symbols from your data files..."
echo "Press Ctrl+C to stop\n"

# Monitor
while true; do
    if ! kill -0 $FEEDER_PID 2>/dev/null; then
        echo "⚠️  Feeder stopped. Restarting..."
        python enhanced_bybit_multi.py &
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

chmod +x run_multi_symbol.sh

# Update CMakeLists.txt to include new files
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(chimera_scalper LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89")

find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(SOURCES
    src/main.cpp
    src/core/data_feed.cpp
    src/core/bybit_feed.cpp
    src/core/bybit_multi_feed.cpp
    src/core/features.cu
    src/layers/layer1_fastpath.cpp
    src/layers/layer2_regime.cpp
    src/layers/layer3_experts.cpp
    src/layers/layer4_pnlfilter.cpp
    src/layers/layer5_execution.cpp
    src/utils/logger.cpp
)

add_executable(chimera_scalper ${SOURCES})

target_link_libraries(chimera_scalper
    CUDA::cudart
    pthread
)

set_target_properties(chimera_scalper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
EOF

echo "✅ Multi-symbol setup complete!"
echo ""
echo "📋 To run multi-symbol trading:"
echo "   1. ./run_multi_symbol.sh"
echo ""
echo "This will monitor ALL symbols from your CSV files:"
echo "   • BTC, ETH, SOL, BNB, XRP, ADA, DOGE, SHIB, MEME"
echo "   • Both 1m and 5m timeframes"
echo "   • Real-time L2 order books (50 levels)"
echo ""
echo "Rebuild CHIMERA first:"
echo "   cd build && make clean && cmake .. && make -j\$(nproc)"