#pragma once
#include "../core/features.h"
#include <vector>

namespace chimera {

enum class ScalpTrigger {
    NONE = 0,
    ROSS_MOMENTUM = 1,
    BAO_MEAN_REVERSION = 2,
    NICK_VOLUME_PRESSURE = 3,
    FABIO_CHAOS = 4
};

struct FastPathSignal {
    ScalpTrigger trigger;
    float confidence;
    float entry_price;
    bool is_long;
};

class Layer1FastPath {
public:
    Layer1FastPath();
    FastPathSignal process(const MicrostructureFeatures& features, float price);
    void set_ross_sensitivity(float s) { ross_sensitivity_ = s; }
    void set_bao_sensitivity(float s) { bao_sensitivity_ = s; }
    void set_nick_sensitivity(float s) { nick_sensitivity_ = s; }
    void set_fabio_sensitivity(float s) { fabio_sensitivity_ = s; }

private:
    float ema_fast_;
    float ema_slow_;
    float vwap_upper_;
    float vwap_lower_;
    std::vector<float> volume_deltas_;
    float ross_sensitivity_;
    float bao_sensitivity_;
    float nick_sensitivity_;
    float fabio_sensitivity_;
    bool check_ross_trigger(const MicrostructureFeatures& f, float price);
    bool check_bao_trigger(const MicrostructureFeatures& f, float price);
    bool check_nick_trigger(const MicrostructureFeatures& f, float price);
    bool check_fabio_trigger(const MicrostructureFeatures& f, float price);
};

} // namespace chimera
