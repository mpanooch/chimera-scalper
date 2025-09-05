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
