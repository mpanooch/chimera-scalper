#include "layer3_experts.h"
#include <cmath>
#include <algorithm>

namespace chimera {

Layer3Experts::Layer3Experts() {}

ExpertSignal Layer3Experts::generate_signal(const MicrostructureFeatures& features,
                                            MarketRegime regime,
                                            float current_price) {
    ExpertSignal ross = ross_expert(features, current_price);
    ExpertSignal bao = bao_expert(features, current_price);
    ExpertSignal nick = nick_expert(features, current_price);
    ExpertSignal fabio = fabio_expert(features, current_price);

    ross.confidence *= get_expert_weight("ross", regime);
    bao.confidence *= get_expert_weight("bao", regime);
    nick.confidence *= get_expert_weight("nick", regime);
    fabio.confidence *= get_expert_weight("fabio", regime);

    ExpertSignal best = ross;
    if (bao.confidence > best.confidence) best = bao;
    if (nick.confidence > best.confidence) best = nick;
    if (fabio.confidence > best.confidence) best = fabio;

    return best;
}

ExpertSignal Layer3Experts::ross_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Ross";
    signal.entry_price = price;
    signal.is_long = (f.bid_ask_imbalance > 0);
    signal.stop_loss = price * (signal.is_long ? 0.995f : 1.005f);
    signal.take_profit = price * (signal.is_long ? 1.01f : 0.99f);
    signal.confidence = fabsf(f.bid_ask_imbalance) * 0.8f;
    return signal;
}

ExpertSignal Layer3Experts::bao_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Bao";
    signal.entry_price = price;
    signal.is_long = (f.vwap_deviation < -0.002f);
    signal.stop_loss = price * (signal.is_long ? 0.997f : 1.003f);
    signal.take_profit = price * (signal.is_long ? 1.005f : 0.995f);
    float conf = fabsf(f.vwap_deviation) * 100.0f;
    signal.confidence = (conf < 0.9f) ? conf : 0.9f;  // min replacement
    return signal;
}

ExpertSignal Layer3Experts::nick_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Nick";
    signal.entry_price = price;
    signal.is_long = (f.buy_pressure > f.sell_pressure);
    signal.stop_loss = price * (signal.is_long ? 0.996f : 1.004f);
    signal.take_profit = price * (signal.is_long ? 1.008f : 0.992f);
    signal.confidence = fabsf(f.buy_pressure - f.sell_pressure);
    return signal;
}

ExpertSignal Layer3Experts::fabio_expert(const MicrostructureFeatures& f, float price) {
    ExpertSignal signal;
    signal.expert_name = "Fabio";
    signal.entry_price = price;
    signal.is_long = (f.hurst_exponent < 0.5f && f.bid_ask_imbalance > 0);
    signal.stop_loss = price * (signal.is_long ? 0.993f : 1.007f);
    signal.take_profit = price * (signal.is_long ? 1.015f : 0.985f);
    signal.confidence = (1.0f - fabsf(f.hurst_exponent - 0.5f)) * 0.7f;
    return signal;
}

float Layer3Experts::get_expert_weight(const std::string& expert, MarketRegime regime) {
    if (expert == "ross") {
        return (regime == MarketRegime::TREND_UP || regime == MarketRegime::TREND_DOWN) ? 1.2f : 0.8f;
    }
    if (expert == "bao") {
        return (regime == MarketRegime::RANGE_BOUND) ? 1.3f : 0.7f;
    }
    if (expert == "nick") {
        return (regime == MarketRegime::LIQUIDITY_EVENT) ? 1.1f : 0.9f;
    }
    if (expert == "fabio") {
        return (regime == MarketRegime::CHAOTIC_SPIKE) ? 1.5f : 0.6f;
    }
    return 1.0f;
}

} // namespace chimera
