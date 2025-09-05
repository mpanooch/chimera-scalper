#!/usr/bin/env python3
"""
Generate synthetic L2 order book data from OHLCV CSVs
Simulates realistic bid/ask levels based on volatility and volume patterns
"""

import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from typing import List, Tuple
import sys

class L2Generator:
    def __init__(self, symbol: str, csv_path: str):
        self.symbol = symbol
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Calculate metrics for realistic spread generation
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(20).std()
        self.df['typical_price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3

    def generate_order_book(self, row, num_levels: int = 50) -> dict:
        """Generate synthetic L2 order book for a given OHLCV row"""

        mid_price = row['Close']
        volatility = row.get('volatility', 0.001) or 0.001
        volume = row['Volume']

        # Base spread depends on volatility (more volatile = wider spread)
        base_spread_bps = max(1, min(20, volatility * 10000))  # 1-20 bps
        spread = mid_price * base_spread_bps / 10000

        best_bid = mid_price - spread/2
        best_ask = mid_price + spread/2

        # Generate bid levels
        bids = []
        for i in range(num_levels):
            # Price decreases exponentially from best bid
            price_decay = 1 + (i * 0.0001 * (1 + volatility * 10))
            price = best_bid / price_decay

            # Size follows power law with volume influence
            base_size = volume / 1000 * np.random.pareto(2.0)
            size_multiplier = 1 / (i + 1) ** 0.5
            size = max(0.01, base_size * size_multiplier * np.random.uniform(0.5, 1.5))

            bids.append({
                'price': round(price, 8),
                'size': round(size, 4),
                'count': np.random.randint(1, 5)
            })

        # Generate ask levels
        asks = []
        for i in range(num_levels):
            # Price increases exponentially from best ask
            price_growth = 1 + (i * 0.0001 * (1 + volatility * 10))
            price = best_ask * price_growth

            # Similar size distribution
            base_size = volume / 1000 * np.random.pareto(2.0)
            size_multiplier = 1 / (i + 1) ** 0.5
            size = max(0.01, base_size * size_multiplier * np.random.uniform(0.5, 1.5))

            asks.append({
                'price': round(price, 8),
                'size': round(size, 4),
                'count': np.random.randint(1, 5)
            })

        # Add liquidity holes/gaps randomly (10% chance)
        if np.random.random() < 0.1:
            # Remove some mid-levels to create gaps
            gap_start = np.random.randint(5, 20)
            gap_size = np.random.randint(2, 5)
            for i in range(gap_start, min(gap_start + gap_size, len(bids))):
                bids[i]['size'] *= 0.1
                asks[i]['size'] *= 0.1

        # Add iceberg orders (5% chance)
        if np.random.random() < 0.05:
            iceberg_level = np.random.randint(0, 5)
            bids[iceberg_level]['size'] *= 10  # Hidden liquidity

        return {
            'symbol': self.symbol,
            'timestamp': row['timestamp'].isoformat(),
            'timestamp_ns': int(row['timestamp'].timestamp() * 1e9),
            'bids': bids,
            'asks': asks,
            'last_trade': {
                'price': mid_price,
                'size': volume / np.random.randint(100, 500),
                'side': 'buy' if np.random.random() > 0.5 else 'sell'
            }
        }

    def generate_l2_file(self, output_dir: str, sample_rate: int = 100):
        """Generate L2 data file sampling every Nth row"""

        output_path = Path(output_dir) / f"{self.symbol}_l2.jsonl.gz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(output_path, 'wt') as f:
            for idx in range(0, len(self.df), sample_rate):
                if idx >= len(self.df):
                    break

                row = self.df.iloc[idx]
                if pd.isna(row['volatility']):
                    continue

                ob = self.generate_order_book(row)
                f.write(json.dumps(ob) + '\n')

                if idx % 1000 == 0:
                    print(f"  Generated {idx}/{len(self.df)} order books")

        print(f"✓ Saved L2 data to {output_path}")
        return output_path

def main():
    # Define assets and timeframes
    assets = ['btc', 'eth', 'sol', 'bnb', 'xrp', 'ada', 'doge', 'shib', 'meme']
    timeframes = ['1m', '5m']

    data_dir = Path('data/raw')
    output_dir = Path('data/ob')

    print("🔥 CHIMERA L2 Order Book Generator")
    print("=" * 50)

    for asset in assets:
        for tf in timeframes:
            csv_file = data_dir / f"{asset}_usdt_{tf}.csv"

            if not csv_file.exists():
                print(f"⚠️  Skipping {csv_file} (not found)")
                continue

            print(f"\n📊 Processing {asset.upper()}/USDT {tf}...")

            symbol = f"{asset.upper()}USDT"
            generator = L2Generator(symbol, str(csv_file))

            # Sample more frequently for 5m (every 5 rows = every 25min)
            # Sample less for 1m (every 60 rows = every hour)
            sample_rate = 60 if tf == '1m' else 5

            generator.generate_l2_file(str(output_dir), sample_rate)

    print("\n✅ L2 generation complete!")
    print(f"📁 Files saved to: {output_dir}")

if __name__ == "__main__":
    main()