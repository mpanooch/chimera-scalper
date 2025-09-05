# orderbook_engine/bybit_orderbook.py

import numpy as np

class BybitOrderBook:
    """
    Bybit L2 Order Book (200 levels).
    Handles snapshot loading and incremental updates.
    """

    def __init__(self, depth=200):
        self.depth = depth
        self.bids = {}  # {price: size}
        self.asks = {}

    def load_snapshot(self, snapshot: dict):
        """Initialize from Bybit L2 snapshot JSON"""
        self.bids = {float(p): float(s) for p, s in snapshot.get("b", [])}
        self.asks = {float(p): float(s) for p, s in snapshot.get("a", [])}

    def apply_delta(self, delta: dict):
        """
        Apply incremental updates:
        {"b": [["43000.5", "0.25"], ...], "a": [["43001.0", "0.20"], ...]}
        """
        for side, updates in delta.items():
            book = self.bids if side == "b" else self.asks
            for price, size in updates:
                price, size = float(price), float(size)
                if size == 0:
                    book.pop(price, None)
                else:
                    book[price] = size

    def top_levels(self, side: str, n: int):
        """Return top N price levels for bids or asks"""
        book = self.bids if side == "b" else self.asks
        if not book:
            return []
        if side == "b":
            return sorted(book.items(), key=lambda x: -x[0])[:n]
        else:
            return sorted(book.items(), key=lambda x: x[0])[:n]
