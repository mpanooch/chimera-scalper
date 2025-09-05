# orderbook_engine/features.py

import numpy as np
from orderbook_engine.bybit_orderbook import BybitOrderBook

def compute_features(orderbook: BybitOrderBook, top_n=10):
    """Compute microstructure features from L2 order book"""

    bids = orderbook.top_levels("b", top_n)
    asks = orderbook.top_levels("a", top_n)

    if not bids or not asks:
        return {}

    bid_vol = sum(s for _, s in bids)
    ask_vol = sum(s for _, s in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    spread = asks[0][0] - bids[0][0]
    micro_price = (asks[0][0] * bid_vol + bids[0][0] * ask_vol) / (bid_vol + ask_vol + 1e-9)

    # Entropy of volume distribution
    volumes = np.array([s for _, s in bids + asks])
    probs = volumes / (volumes.sum() + 1e-9)
    entropy = -(probs * np.log(probs + 1e-9)).sum()

    return {
        "imbalance": imbalance,
        "spread": spread,
        "micro_price": micro_price,
        "entropy": entropy,
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
    }
