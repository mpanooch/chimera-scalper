#include "layer2_regime.h"
#include <cmath>

namespace chimera {

MarketRegime Layer2Regime::classify(const MicrostructureFeatures& features) {
    confidence_ = 0.7f;

    if (std::abs(features.bid_ask_imbalance) > 0.4f) {
        if (features.bid_ask_imbalance > 0) {
            confidence_ = 0.85f;
            return MarketRegime::TREND_UP;
        } else {
            confidence_ = 0.85f;
            return MarketRegime::TREND_DOWN;
        }
    }

    if (features.order_book_entropy > 4.0f && features.hurst_exponent < 0.4f) {
        confidence_ = 0.8f;
        return MarketRegime::CHAOTIC_SPIKE;
    }

    if (features.liquidity_score < 50.0f) {
        confidence_ = 0.75f;
        return MarketRegime::LIQUIDITY_EVENT;
    }

    return MarketRegime::RANGE_BOUND;
}

} // namespace chimera
