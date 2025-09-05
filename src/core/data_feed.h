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
