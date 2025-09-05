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
