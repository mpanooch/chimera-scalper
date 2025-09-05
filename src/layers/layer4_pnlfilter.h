#pragma once
#include "layer3_experts.h"
#include <map>

namespace chimera {

class Layer4PnLFilter {
public:
    Layer4PnLFilter();
    bool should_trade(const ExpertSignal& signal);
    float calculate_expected_pnl(const ExpertSignal& signal);
    void update_performance(const std::string& expert, float pnl);

private:
    struct ExpertStats {
        int trades = 0;
        int wins = 0;
        float total_pnl = 0.0f;
        float win_rate = 0.5f;
        float avg_win = 0.0f;
        float avg_loss = 0.0f;
    };

    std::map<std::string, ExpertStats> expert_stats_;
    float min_expected_pnl_ = 0.0005f;
};

} // namespace chimera
