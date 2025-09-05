# tests/test_orderbook.py

import json
from orderbook_engine.bybit_orderbook import BybitOrderBook
from orderbook_engine.features import compute_features

def test_snapshot_and_features():
    snapshot = {
        "b": [["100.0", "2"], ["99.5", "3"]],
        "a": [["100.5", "1"], ["101.0", "4"]],
    }
    ob = BybitOrderBook()
    ob.load_snapshot(snapshot)

    feats = compute_features(ob, top_n=2)
    assert "imbalance" in feats
    assert feats["spread"] > 0
