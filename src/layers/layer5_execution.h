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
