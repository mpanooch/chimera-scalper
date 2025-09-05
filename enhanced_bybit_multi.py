#!/usr/bin/env python3
"""
Enhanced Bybit integration for multiple symbols and timeframes
"""

import json
import asyncio
import websockets
from datetime import datetime
import mmap
import struct
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

class MultiSymbolBybitFeeder:
    def __init__(self, testnet=True):
        if testnet:
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        self.orderbooks = {}
        self.klines = defaultdict(dict)  # symbol -> timeframe -> data
        self.running = True
        self.shm_size = 4 * 1024 * 1024  # 4MB for multiple symbols
        self.setup_shared_memory()

        # Track performance per symbol
        self.stats = defaultdict(lambda: {
            'updates': 0,
            'signals': 0,
            'last_price': 0,
            'best_spread': float('inf'),
            'worst_spread': 0
        })

    def setup_shared_memory(self):
        try:
            self.shm_file = open('/tmp/chimera_multi_ob.dat', 'w+b')
            self.shm_file.write(b'\x00' * self.shm_size)
            self.shm_file.flush()
            self.shm = mmap.mmap(self.shm_file.fileno(), self.shm_size)
            print("✓ Shared memory initialized (4MB)")
        except Exception as e:
            print(f"❌ Failed to setup shared memory: {e}")

    def write_multi_orderbooks(self):
        """Write all orderbooks to shared memory"""
        try:
            self.shm.seek(0)

            # Write header: number of symbols
            self.shm.write(struct.pack('I', len(self.orderbooks)))

            for symbol, ob in self.orderbooks.items():
                # Write symbol
                symbol_bytes = symbol.encode('utf-8')
                self.shm.write(struct.pack('I', len(symbol_bytes)))
                self.shm.write(symbol_bytes)

                # Write timestamp
                timestamp_ns = int(time.time() * 1e9)
                self.shm.write(struct.pack('Q', timestamp_ns))

                # Get bid/ask data
                bids = ob.get('b', [])[:50]
                asks = ob.get('a', [])[:50]

                # Write counts
                self.shm.write(struct.pack('II', len(bids), len(asks)))

                # Write bids
                for bid in bids:
                    price, size = float(bid[0]), float(bid[1])
                    self.shm.write(struct.pack('ff', price, size))

                # Write asks
                for ask in asks:
                    price, size = float(ask[0]), float(ask[1])
                    self.shm.write(struct.pack('ff', price, size))

            self.shm.flush()

        except Exception as e:
            print(f"Error writing to shared memory: {e}")

    async def handle_orderbook_message(self, data: dict):
        if 'topic' not in data:
            return

        topic = data['topic']
        symbol = topic.split('.')[-1]

        if data['type'] == 'snapshot':
            self.orderbooks[symbol] = data['data']
            self.stats[symbol]['updates'] += 1

        elif data['type'] == 'delta':
            if symbol not in self.orderbooks:
                return

            ob = self.orderbooks[symbol]

            if 'b' in data['data']:
                ob['b'] = self.apply_delta(ob.get('b', []), data['data']['b'], True)

            if 'a' in data['data']:
                ob['a'] = self.apply_delta(ob.get('a', []), data['data']['a'], False)

            self.orderbooks[symbol] = ob
            self.stats[symbol]['updates'] += 1

        # Write all orderbooks to shared memory
        self.write_multi_orderbooks()

        # Update statistics
        self.update_stats(symbol)

    def update_stats(self, symbol):
        """Update and display statistics"""
        if symbol not in self.orderbooks:
            return

        ob = self.orderbooks[symbol]
        if 'b' in ob and 'a' in ob and len(ob['b']) > 0 and len(ob['a']) > 0:
            best_bid = float(ob['b'][0][0])
            best_ask = float(ob['a'][0][0])
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000

            # Update stats
            stats = self.stats[symbol]
            stats['last_price'] = (best_bid + best_ask) / 2
            stats['best_spread'] = min(stats['best_spread'], spread_bps)
            stats['worst_spread'] = max(stats['worst_spread'], spread_bps)

            # Calculate order book metrics
            bid_vol = sum(float(b[1]) for b in ob['b'][:5])
            ask_vol = sum(float(a[1]) for a in ob['a'][:5])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

            # Display rotating stats
            if stats['updates'] % 10 == 0:  # Every 10 updates
                print(f"\r{symbol:12} | Price: {stats['last_price']:.2f} | "
                      f"Spread: {spread_bps:.2f}bps | Imbalance: {imbalance:+.3f} | "
                      f"Updates: {stats['updates']}", end='')

    def apply_delta(self, levels: list, updates: list, is_bid: bool) -> list:
        level_dict = {level[0]: level for level in levels}

        for update in updates:
            price = update[0]
            size = update[1]

            if float(size) == 0:
                level_dict.pop(price, None)
            else:
                level_dict[price] = update

        sorted_levels = sorted(level_dict.items(),
                               key=lambda x: float(x[0]),
                               reverse=is_bid)

        return [level[1] for level in sorted_levels]

    async def subscribe_multiple(self, ws, symbols, depths=[50]):
        """Subscribe to multiple symbols and depths"""
        topics = []

        # Order book subscriptions
        for symbol in symbols:
            for depth in depths:
                topics.append(f"orderbook.{depth}.{symbol}")

        # Also subscribe to 1m and 5m klines for each symbol
        for symbol in symbols:
            topics.append(f"kline.1.{symbol}")  # 1 minute
            topics.append(f"kline.5.{symbol}")  # 5 minutes

        # Split into chunks (Bybit has subscription limits)
        chunk_size = 10
        for i in range(0, len(topics), chunk_size):
            chunk = topics[i:i+chunk_size]
            subscribe_msg = {
                "op": "subscribe",
                "args": chunk
            }
            await ws.send(json.dumps(subscribe_msg))
            await asyncio.sleep(0.1)  # Small delay between subscriptions

        print(f"📡 Subscribed to {len(symbols)} symbols with {len(depths)} depth levels")
        print(f"   Total subscriptions: {len(topics)}")

    async def handle_kline_message(self, data: dict):
        """Handle kline (candlestick) data"""
        if 'topic' not in data:
            return

        topic = data['topic']
        parts = topic.split('.')
        interval = parts[1]  # 1 or 5 (minutes)
        symbol = parts[2]

        if data['type'] == 'snapshot':
            kline_data = data['data'][0] if data['data'] else None
            if kline_data:
                self.klines[symbol][f"{interval}m"] = {
                    'timestamp': kline_data['timestamp'],
                    'open': float(kline_data['open']),
                    'high': float(kline_data['high']),
                    'low': float(kline_data['low']),
                    'close': float(kline_data['close']),
                    'volume': float(kline_data['volume'])
                }

    async def run(self, symbols, depth=50):
        print(f"🔥 Multi-Symbol Bybit Feed Starting...")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Depth: {depth}")
        print(f"   Mode: {'TESTNET' if 'testnet' in self.ws_url else 'MAINNET'}\n")

        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print("✓ Connected to Bybit")

                    await self.subscribe_multiple(ws, symbols, [depth])

                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(message)

                            if 'topic' in data:
                                if 'orderbook' in data['topic']:
                                    await self.handle_orderbook_message(data)
                                elif 'kline' in data['topic']:
                                    await self.handle_kline_message(data)
                            elif data.get('op') == 'subscribe':
                                if data.get('success'):
                                    print(f"✓ Subscription confirmed: {data.get('req_id', 'batch')}")
                            elif data.get('op') == 'ping':
                                await ws.send(json.dumps({"op": "pong"}))

                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"op": "ping"}))
                        except Exception as e:
                            print(f"\n⚠️ Error: {e}")

            except Exception as e:
                print(f"\n❌ Connection error: {e}")
                print("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def print_summary(self):
        """Print trading summary"""
        print("\n\n📊 Trading Session Summary")
        print("=" * 60)

        for symbol, stats in self.stats.items():
            if stats['updates'] > 0:
                print(f"\n{symbol}:")
                print(f"  Updates: {stats['updates']}")
                print(f"  Last Price: ${stats['last_price']:.2f}")
                print(f"  Best Spread: {stats['best_spread']:.2f} bps")
                print(f"  Worst Spread: {stats['worst_spread']:.2f} bps")

    def stop(self):
        self.running = False
        self.print_summary()
        if hasattr(self, 'shm'):
            self.shm.close()
        if hasattr(self, 'shm_file'):
            self.shm_file.close()
        print("\n✓ Feeder stopped")

def get_symbols_from_csv_files():
    """Extract symbols from CSV files in data/raw"""
    symbols = set()
    data_path = Path('data/raw')

    if data_path.exists():
        for csv_file in data_path.glob('*_usdt_*.csv'):
            # Extract symbol from filename (e.g., btc_usdt_1m.csv -> BTCUSDT)
            parts = csv_file.stem.split('_')
            if len(parts) >= 2:
                symbol = f"{parts[0].upper()}USDT"
                symbols.add(symbol)

    return sorted(list(symbols))

def signal_handler(signum, frame):
    print("\n\n🛑 Shutdown signal received...")
    if 'feeder' in globals():
        feeder.stop()
    sys.exit(0)

def main():
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   🔥 CHIMERA MULTI-SYMBOL BYBIT FEEDER 🔥     ║
    ║   All Symbols | 1m & 5m | Real-time L2        ║
    ╚═══════════════════════════════════════════════╝
    """)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get symbols from your CSV files
    symbols = get_symbols_from_csv_files()

    if not symbols:
        # Default symbols if no CSVs found
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
                   'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'SHIBUSDT']

    print(f"\n📋 Monitoring {len(symbols)} symbols:")
    for i, symbol in enumerate(symbols, 1):
        print(f"   {i:2}. {symbol}")

    # Configuration
    TESTNET = True
    DEPTH = 50

    global feeder
    feeder = MultiSymbolBybitFeeder(testnet=TESTNET)

    try:
        asyncio.run(feeder.run(symbols, DEPTH))
    except KeyboardInterrupt:
        pass
    finally:
        feeder.stop()

if __name__ == "__main__":
    main()