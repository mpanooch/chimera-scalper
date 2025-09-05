#include "layer4_pnlfilter.h"
#include <cmath>

namespace chimera {

Layer4PnLFilter::Layer4PnLFilter() {
    expert_stats_["Ross"] = ExpertStats();
    expert_stats_["Bao"] = ExpertStats();
    expert_stats_["Nick"] = ExpertStats();
    expert_stats_["Fabio"] = ExpertStats();
}

bool Layer4PnLFilter::should_trade(const ExpertSignal& signal) {
    if (signal.confidence < 0.7f) {
        return false;
    }
    float expected_pnl = calculate_expected_pnl(signal);
    return expected_pnl > min_expected_pnl_;
}

float Layer4PnLFilter::calculate_expected_pnl(const ExpertSignal& signal) {
    auto it = expert_stats_.find(signal.expert_name);
    if (it == expert_stats_.end()) {
        return 0.001f;
    }

    const ExpertStats& stats = it->second;

    if (stats.trades < 10) {
        return signal.confidence * 0.001f;
    }

    float win_prob = stats.win_rate;
    float loss_prob = 1.0f - win_prob;

    float reward = fabsf(signal.take_profit - signal.entry_price) / signal.entry_price;
    float risk = fabsf(signal.stop_loss - signal.entry_price) / signal.entry_price;

    float expected = (win_prob * reward) - (loss_prob * risk);
    return expected * signal.confidence;
}

void Layer4PnLFilter::update_performance(const std::string& expert, float pnl) {
    auto it = expert_stats_.find(expert);
    if (it == expert_stats_.end()) {
        return;
    }

    ExpertStats& stats = it->second;
    stats.trades++;
    stats.total_pnl += pnl;

    if (pnl > 0) {
        stats.wins++;
        stats.avg_win = (stats.avg_win * (stats.wins - 1) + pnl) / stats.wins;
    } else {
        int losses = stats.trades - stats.wins;
        stats.avg_loss = (stats.avg_loss * (losses - 1) + fabsf(pnl)) / losses;
    }

    stats.win_rate = static_cast<float>(stats.wins) / stats.trades;
}

} // namespace chimera
