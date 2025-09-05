#pragma once
#include "../core/features.h"
#include "layer2_regime.h"
#include <string>

namespace chimera {

struct ExpertSignal {
    float entry_price;
    float stop_loss;
    float take_profit;
    float confidence;
    bool is_long;
    std::string expert_name;
};

class Layer3Experts {
public:
    Layer3Experts();
    ExpertSignal generate_signal(const MicrostructureFeatures& features,
                                MarketRegime regime,
                                float current_price);

private:
    ExpertSignal ross_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal bao_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal nick_expert(const MicrostructureFeatures& f, float price);
    ExpertSignal fabio_expert(const MicrostructureFeatures& f, float price);
    float get_expert_weight(const std::string& expert, MarketRegime regime);
};

} // namespace chimera
