# orderbook_engine/simulator.py

import json
from orderbook_engine.bybit_orderbook import BybitOrderBook
from orderbook_engine.features import compute_features

class ReplaySimulator:
    """
    Replays Bybit historical order book data (snapshot + updates).
    """

    def __init__(self, snapshot_file, updates_file, depth=200):
        self.snapshot_file = snapshot_file
        self.updates_file = updates_file
        self.ob = BybitOrderBook(depth=depth)

    def run(self):
        """Yield feature dicts for each incremental update"""
        with open(self.snapshot_file) as f:
            snapshot = json.load(f)
        self.ob.load_snapshot(snapshot)

        with open(self.updates_file) as f:
            for line in f:
                delta = json.loads(line.strip())
                self.ob.apply_delta(delta)
                yield compute_features(self.ob)
