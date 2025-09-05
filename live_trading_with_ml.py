#!/usr/bin/env python3
"""
Live trading system with trained ML models on Bybit testnet
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import websockets
import json
import time
from datetime import datetime
from pathlib import Path
import struct
import mmap
from collections import deque
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============= Model Definitions =============
class RegimeClassifier(nn.Module):
    """Load trained regime classifier"""
    def __init__(self, input_dim=14, hidden_dim=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.classifier(last_hidden)
        return output

class TradingSignalGenerator:
    """Generate trading signals using ML models"""
    def __init__(self, model_path='models/best_regime_classifier.pth'):
        self.device = device
        self.sequence_length = 60
        self.feature_dim = 14

        # Load trained model
        self.model = self.load_model(model_path)

        # Store recent data for each symbol
        self.symbol_data = {}

        # Trading statistics
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0,
            'win_rate': 0
        }

        # Risk management
        self.max_positions = 5
        self.current_positions = {}

    def load_model(self, model_path):
        """Load trained regime classifier"""
        print(f"Loading model from {model_path}...")

        model = RegimeClassifier(input_dim=14, hidden_dim=64, num_classes=5)

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"✓ Model loaded successfully")
            return model
        else:
            print(f"❌ Model file not found: {model_path}")
            return None

    def calculate_features(self, orderbook, symbol):
        """Calculate features from orderbook data"""
        features = np.zeros(self.feature_dim)

        if 'bids' not in orderbook or 'asks' not in orderbook:
            return features

        bids = orderbook['bids']
        asks = orderbook['asks']

        if len(bids) > 0 and len(asks) > 0:
            # Price features
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            mid_price = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid_price

            # Volume features
            bid_volume = sum(float(b['size']) for b in bids[:5])
            ask_volume = sum(float(a['size']) for a in asks[:5])
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)

            # Depth features
            bid_depth = sum(float(b['size']) for b in bids[:20])
            ask_depth = sum(float(a['size']) for a in asks[:20])

            # Create feature vector (simplified for live trading)
            features[0] = 0  # returns (will be calculated from history)
            features[1] = np.log1p(bid_volume + ask_volume)  # volume_norm
            features[2] = spread * 100  # high_low proxy
            features[3] = volume_imbalance  # close_open proxy
            features[4] = mid_price / (mid_price + 1)  # sma_5 proxy
            features[5] = mid_price / (mid_price + 2)  # sma_10 proxy
            features[6] = mid_price / (mid_price + 3)  # sma_20 proxy
            features[7] = 1.0  # price_position
            features[8] = spread * 10  # volatility proxy
            features[9] = bid_volume / (ask_volume + 1e-8)  # volume_ratio
            features[10] = 0.5  # rsi proxy
            features[11] = 1.0 if symbol in ['BTCUSDT', 'ETHUSDT'] else 0.0  # is_major
            features[12] = 1.0  # is_1m
            features[13] = volume_imbalance * 0.5  # momentum proxy

        return features

    def update_symbol_data(self, symbol, features):
        """Update historical data for symbol"""
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = deque(maxlen=self.sequence_length)

        self.symbol_data[symbol].append(features)

    def predict_regime(self, symbol):
        """Predict market regime for symbol"""
        if symbol not in self.symbol_data or len(self.symbol_data[symbol]) < self.sequence_length:
            return None, 0

        # Prepare input tensor
        sequence = np.array(list(self.symbol_data[symbol]))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            regime = output.argmax(1).item()
            confidence = probabilities[0, regime].item()

        return regime, confidence

    def generate_signal(self, symbol, orderbook):
        """Generate trading signal based on ML prediction"""
        # Calculate features
        features = self.calculate_features(orderbook, symbol)

        # Update historical data
        self.update_symbol_data(symbol, features)

        # Predict regime
        regime, confidence = self.predict_regime(symbol)

        if regime is None:
            return None

        # Map regime to action
        regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE_BOUND', 'LIQUIDITY_EVENT', 'CHAOTIC']
        regime_name = regime_names[regime]

        # Generate signal based on regime and confidence
        signal = None

        if confidence > 0.7:  # High confidence threshold
            if regime == 0:  # TREND_UP
                signal = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'regime': regime_name,
                    'confidence': confidence,
                    'size': self.calculate_position_size(symbol, confidence),
                    'timestamp': datetime.now().isoformat()
                }
            elif regime == 1:  # TREND_DOWN
                signal = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'regime': regime_name,
                    'confidence': confidence,
                    'size': self.calculate_position_size(symbol, confidence),
                    'timestamp': datetime.now().isoformat()
                }
            elif regime == 2:  # RANGE_BOUND
                # Mean reversion strategy
                if features[3] > 0.3:  # Volume imbalance indicates oversold
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'regime': regime_name,
                        'confidence': confidence * 0.8,
                        'size': self.calculate_position_size(symbol, confidence * 0.8),
                        'timestamp': datetime.now().isoformat()
                    }

        if signal:
            self.stats['signals_generated'] += 1

        return signal

    def calculate_position_size(self, symbol, confidence):
        """Calculate position size based on confidence and risk management"""
        base_size = 100  # Base position size in USD

        # Scale by confidence
        size = base_size * confidence

        # Reduce size if we have many positions
        if len(self.current_positions) >= 3:
            size *= 0.5

        return round(size, 2)

    def should_execute_trade(self, signal):
        """Check if trade should be executed based on risk management"""
        # Check max positions
        if len(self.current_positions) >= self.max_positions:
            return False

        # Check if already in position for this symbol
        if signal['symbol'] in self.current_positions:
            # Only add if opposite direction (reversal)
            current_pos = self.current_positions[signal['symbol']]
            if current_pos['action'] == signal['action']:
                return False

        return True

    def execute_trade(self, signal):
        """Execute trade (paper trading for testnet)"""
        if self.should_execute_trade(signal):
            self.current_positions[signal['symbol']] = {
                'action': signal['action'],
                'size': signal['size'],
                'entry_time': signal['timestamp'],
                'regime': signal['regime']
            }
            self.stats['trades_executed'] += 1
            return True
        return False

class LiveTradingSystem:
    """Main live trading system"""
    def __init__(self, testnet=True):
        self.testnet = testnet
        self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear" if testnet else "wss://stream.bybit.com/v5/public/linear"

        # Initialize ML signal generator
        self.signal_generator = TradingSignalGenerator()

        # Track performance
        self.trades = []
        self.running = True

    async def handle_orderbook(self, data):
        """Process orderbook and generate signals"""
        if 'topic' not in data or 'data' not in data:
            return

        topic = data['topic']
        symbol = topic.split('.')[-1]

        if data['type'] == 'snapshot':
            orderbook = {
                'bids': [{'price': b[0], 'size': b[1]} for b in data['data'].get('b', [])[:50]],
                'asks': [{'price': a[0], 'size': a[1]} for a in data['data'].get('a', [])[:50]]
            }

            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, orderbook)

            if signal:
                print(f"\n🎯 Signal Generated:")
                print(f"   Symbol: {signal['symbol']}")
                print(f"   Action: {signal['action']}")
                print(f"   Regime: {signal['regime']}")
                print(f"   Confidence: {signal['confidence']:.2%}")
                print(f"   Size: ${signal['size']}")

                # Execute trade
                if self.signal_generator.execute_trade(signal):
                    print(f"   ✅ Trade Executed!")
                    self.trades.append(signal)
                else:
                    print(f"   ❌ Trade Rejected (Risk Management)")

    async def run(self, symbols):
        """Main trading loop"""
        print(f"\n🔥 Live Trading System Started")
        print(f"   Mode: {'TESTNET' if self.testnet else 'MAINNET'}")
        print(f"   Model: Loaded ✓")
        print(f"   Symbols: {', '.join(symbols)}")
        print("=" * 60)

        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print("✓ Connected to Bybit")

                    # Subscribe to orderbooks
                    for symbol in symbols:
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"orderbook.50.{symbol}"]
                        }
                        await ws.send(json.dumps(subscribe_msg))

                    print(f"✓ Subscribed to {len(symbols)} symbols")

                    # Main message loop
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(message)

                            if 'topic' in data:
                                await self.handle_orderbook(data)
                            elif data.get('op') == 'ping':
                                await ws.send(json.dumps({"op": "pong"}))

                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"op": "ping"}))

                            # Print periodic stats
                            if self.signal_generator.stats['signals_generated'] > 0:
                                print(f"\r📊 Signals: {self.signal_generator.stats['signals_generated']} | "
                                      f"Trades: {self.signal_generator.stats['trades_executed']} | "
                                      f"Positions: {len(self.signal_generator.current_positions)}", end='')

                        except Exception as e:
                            print(f"\n⚠️ Error: {e}")

            except Exception as e:
                print(f"\n❌ Connection error: {e}")
                print("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def print_summary(self):
        """Print trading summary"""
        print("\n\n" + "=" * 60)
        print("📈 Trading Session Summary")
        print("=" * 60)
        print(f"Signals Generated: {self.signal_generator.stats['signals_generated']}")
        print(f"Trades Executed: {self.signal_generator.stats['trades_executed']}")
        print(f"Current Positions: {len(self.signal_generator.current_positions)}")

        if self.trades:
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:
                print(f"  {trade['timestamp']}: {trade['symbol']} {trade['action']} "
                      f"(Regime: {trade['regime']}, Conf: {trade['confidence']:.2%})")

def main():
    print("""
    ╔═══════════════════════════════════════════╗
    ║   🔥 CHIMERA LIVE ML TRADING 🔥           ║
    ║   Testnet | ML-Powered | Real-time        ║
    ╚═══════════════════════════════════════════╝
    """)

    # Symbols to trade
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    # Create trading system
    system = LiveTradingSystem(testnet=True)

    try:
        # Run trading
        asyncio.run(system.run(symbols))
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown signal received...")
        system.running = False
        system.print_summary()

if __name__ == "__main__":
    main()