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
