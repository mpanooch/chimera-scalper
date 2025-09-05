#pragma once
#include "../core/features.h"

namespace chimera {

enum class MarketRegime {
    TREND_UP,
    TREND_DOWN,
    RANGE_BOUND,
    LIQUIDITY_EVENT,
    CHAOTIC_SPIKE
};

class Layer2Regime {
public:
    Layer2Regime() = default;
    MarketRegime classify(const MicrostructureFeatures& features);
    float get_confidence() const { return confidence_; }

private:
    float confidence_ = 0.5f;
};

} // namespace chimera
