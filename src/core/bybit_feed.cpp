// ===== src/core/bybit_feed.cpp =====
#include "bybit_feed.h"
#include <iostream>
#include <cstring>
#include <chrono>

namespace chimera {

BybitFeed::BybitFeed()
    : shm_ptr_(nullptr), shm_size_(1024 * 1024),
      shm_fd_(-1), last_timestamp_(0), connected_(false) {
}

BybitFeed::~BybitFeed() {
    disconnect();
}

bool BybitFeed::connect(const std::string& shm_path) {
    // Open shared memory file
    shm_fd_ = open(shm_path.c_str(), O_RDONLY);
    if (shm_fd_ == -1) {
        std::cerr << "Failed to open shared memory: " << shm_path << std::endl;
        return false;
    }

    // Map the file into memory
    shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    connected_ = true;
    std::cout << "✓ Connected to Bybit feed at " << shm_path << std::endl;
    return true;
}

void BybitFeed::disconnect() {
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

bool BybitFeed::read_orderbook(OrderBook& ob, std::string& symbol) {
    if (!connected_ || !shm_ptr_) {
        return false;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(shm_ptr_);
    size_t offset = 0;

    // Read timestamp
    uint64_t timestamp_ns;
    memcpy(&timestamp_ns, ptr + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    // Check if new data
    if (timestamp_ns <= last_timestamp_) {
        return false;  // No new data
    }

    last_timestamp_ = timestamp_ns;
    ob.timestamp_ns = timestamp_ns;

    // Read symbol
    uint32_t symbol_len;
    memcpy(&symbol_len, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    symbol.resize(symbol_len);
    memcpy(&symbol[0], ptr + offset, symbol_len);
    offset += symbol_len;

    // Read bid/ask counts
    uint32_t n_bids, n_asks;
    memcpy(&n_bids, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(&n_asks, ptr + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Clear existing levels
    ob.bids.clear();
    ob.asks.clear();

    // Read bid levels
    for (uint32_t i = 0; i < n_bids; i++) {
        OrderBookLevel level;
        memcpy(&level.price, ptr + offset, sizeof(float));
        offset += sizeof(float);
        memcpy(&level.size, ptr + offset, sizeof(float));
        offset += sizeof(float);
        level.count = 1;  // Default
        ob.bids.push_back(level);
    }

    // Read ask levels
    for (uint32_t i = 0; i < n_asks; i++) {
        OrderBookLevel level;
        memcpy(&level.price, ptr + offset, sizeof(float));
        offset += sizeof(float);
        memcpy(&level.size, ptr + offset, sizeof(float));
        offset += sizeof(float);
        level.count = 1;  // Default
        ob.asks.push_back(level);
    }

    // Set last trade price (use mid price as approximation)
    if (!ob.bids.empty() && !ob.asks.empty()) {
        ob.last_trade_price = (ob.bids[0].price + ob.asks[0].price) / 2.0f;
        ob.last_trade_size = 1.0f;  // Default
    }

    return true;
}

bool BybitFeed::has_new_data() {
    if (!connected_ || !shm_ptr_) {
        return false;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(shm_ptr_);
    uint64_t timestamp_ns;
    memcpy(&timestamp_ns, ptr, sizeof(uint64_t));

    return timestamp_ns > last_timestamp_;
}

} // namespace chimera