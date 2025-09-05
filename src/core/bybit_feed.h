// ===== src/core/bybit_feed.h =====
#pragma once

#include <string>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "features.h"

namespace chimera {

class BybitFeed {
public:
    BybitFeed();
    ~BybitFeed();

    // Initialize shared memory connection
    bool connect(const std::string& shm_path = "/tmp/chimera_orderbook.dat");
    void disconnect();

    // Read latest order book from shared memory
    bool read_orderbook(OrderBook& ob, std::string& symbol);

    // Check if new data is available
    bool has_new_data();

private:
    void* shm_ptr_;
    size_t shm_size_;
    int shm_fd_;
    uint64_t last_timestamp_;

    bool connected_;
};

} // namespace chimera