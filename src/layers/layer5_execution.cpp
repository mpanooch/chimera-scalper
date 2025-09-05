#include "layer5_execution.h"
#include "../utils/logger.h"
#include <iostream>

namespace chimera {

Layer5Execution::Layer5Execution() {}

bool Layer5Execution::execute_trade(const ExpertSignal& signal) {
    if (!check_risk_limits()) {
        LOG_WARNING("Risk limits exceeded, trade rejected");
        return false;
    }

    float position_size = 1000.0f;

    LOG_INFO("Executing trade: " + signal.expert_name +
             " | Direction: " + (signal.is_long ? "LONG" : "SHORT") +
             " | Entry: " + std::to_string(signal.entry_price) +
             " | Confidence: " + std::to_string(signal.confidence));

    bool success = place_order(signal.entry_price, position_size, signal.is_long);

    if (success) {
        set_stop_loss(signal.stop_loss);
        set_take_profit(signal.take_profit);

        // Fix: atomic<float> doesn't support +=, use fetch_add or store
        open_positions_.fetch_add(1);

        // For atomic<float>, we need to use compare_exchange or direct load/store
        float current_size = position_size_.load();
        position_size_.store(current_size + position_size);
    }

    return success;
}

bool Layer5Execution::check_risk_limits() {
    if (daily_pnl_.load() < max_daily_loss_) {
        LOG_ERROR("Daily loss limit reached: " + std::to_string(daily_pnl_.load()));
        return false;
    }

    if (open_positions_.load() >= max_positions_) {
        LOG_WARNING("Max positions reached: " + std::to_string(open_positions_.load()));
        return false;
    }

    if (position_size_.load() >= max_position_size_) {
        LOG_WARNING("Max position size reached: " + std::to_string(position_size_.load()));
        return false;
    }

    return true;
}

void Layer5Execution::update_risk_metrics() {
    // TODO: Implement risk metrics update
}

bool Layer5Execution::place_order(float price, float size, bool is_buy) {
    std::cout << "  [ORDER] " << (is_buy ? "BUY" : "SELL")
              << " " << size << " @ " << price << std::endl;
    return true;
}

bool Layer5Execution::set_stop_loss(float price) {
    std::cout << "  [STOP LOSS] Set at " << price << std::endl;
    return true;
}

bool Layer5Execution::set_take_profit(float price) {
    std::cout << "  [TAKE PROFIT] Set at " << price << std::endl;
    return true;
}

} // namespace chimera
