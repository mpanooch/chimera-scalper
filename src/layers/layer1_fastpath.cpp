#include "layer1_fastpath.h"
#include <cmath>

namespace chimera {

Layer1FastPath::Layer1FastPath()
    : ema_fast_(0), ema_slow_(0),
      vwap_upper_(0), vwap_lower_(0),
      ross_sensitivity_(1.0f),
      bao_sensitivity_(1.0f),
      nick_sensitivity_(1.0f),
      fabio_sensitivity_(1.0f) {
    volume_deltas_.reserve(100);
}

FastPathSignal Layer1FastPath::process(const MicrostructureFeatures& features, float price) {
    FastPathSignal signal = {ScalpTrigger::NONE, 0.0f, price, true};

    float alpha_fast = 2.0f / (9.0f + 1.0f);
    float alpha_slow = 2.0f / (21.0f + 1.0f);

    if (ema_fast_ == 0) {
        ema_fast_ = price;
        ema_slow_ = price;
    } else {
        ema_fast_ = alpha_fast * price + (1 - alpha_fast) * ema_fast_;
        ema_slow_ = alpha_slow * price + (1 - alpha_slow) * ema_slow_;
    }

    if (check_ross_trigger(features, price)) {
        signal.trigger = ScalpTrigger::ROSS_MOMENTUM;
        signal.confidence = 0.85f * ross_sensitivity_;
        signal.is_long = (ema_fast_ > ema_slow_);
    }

    return signal;
}

bool Layer1FastPath::check_ross_trigger(const MicrostructureFeatures& f, float price) {
    bool ema_cross = std::abs(ema_fast_ - ema_slow_) > (price * 0.001f);
    bool momentum = std::abs(f.bid_ask_imbalance) > 0.3f;
    return ema_cross && momentum;
}

bool Layer1FastPath::check_bao_trigger(const MicrostructureFeatures& f, float price) {
    return std::abs(f.vwap_deviation) > 0.002f;
}

bool Layer1FastPath::check_nick_trigger(const MicrostructureFeatures& f, float price) {
    float pressure_diff = std::abs(f.buy_pressure - f.sell_pressure);
    return pressure_diff > 0.2f;
}

bool Layer1FastPath::check_fabio_trigger(const MicrostructureFeatures& f, float price) {
    return f.order_book_entropy > 3.0f && f.hurst_exponent < 0.4f;
}

} // namespace chimera
