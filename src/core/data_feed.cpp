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
